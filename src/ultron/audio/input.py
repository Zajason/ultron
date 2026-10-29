"""Microphone capture: native-rate PortAudio callback in, 16 kHz linear16 frames out.

The callback only hands blocks to the event loop. Decimation (48 kHz -> 16 kHz) runs in the
consumer, with an anti-alias filter that keeps state between blocks. Capturing at the device's
native rate avoids CoreAudio's resampler and the latency it adds.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
import sounddevice as sd
from pedalboard import LowpassFilter, Pedalboard


class Decimator:
    """Integer-factor downsampler with an anti-alias low-pass and exact chunk continuity."""

    def __init__(self, in_rate: int, out_rate: int) -> None:
        if in_rate % out_rate:
            raise ValueError(f"{in_rate} Hz is not an integer multiple of {out_rate} Hz")
        self.factor = in_rate // out_rate
        self.in_rate = in_rate
        self.lp = Pedalboard([LowpassFilter(cutoff_frequency_hz=0.45 * out_rate) for _ in range(4)])
        self._phase = 0  # index of the next kept sample within the incoming block

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        y = self.lp(x[np.newaxis, :].astype(np.float32, copy=False), self.in_rate, reset=False)[0]
        out = y[self._phase :: self.factor]
        self._phase = (self._phase - x.size) % self.factor
        return out


def to_linear16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767).astype("<i2").tobytes()


class MicInput:
    """Async microphone source. start()/stop() bracket one capture; frames() yields fixed-size
    linear16 frames, and the final partial frame immediately when stopped (push-to-talk release
    must not wait for a frame to fill)."""

    def __init__(self, out_rate: int = 16000, frame_ms: int = 80, latency_ms: float = 10,
                 device: str | int | None = None) -> None:
        dev = sd.query_devices(device or None, kind="input")
        self.name = dev["name"]
        self.capture_rate = int(dev["default_samplerate"])
        self.out_rate = out_rate
        self.frame_samples = out_rate * frame_ms // 1000
        self._q: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self.level = 0.0  # RMS of the last block, for a meter
        self.t_started: float | None = None
        self._stream = sd.InputStream(
            samplerate=self.capture_rate, channels=1, dtype="float32", blocksize=0,
            latency=latency_ms / 1000, device=device or None, callback=self._callback,
        )

    @property
    def device_latency_ms(self) -> float:
        return self._stream.latency * 1000

    def _callback(self, indata, frames, time_info, status) -> None:
        block = indata[:, 0].copy()
        self.level = float(np.sqrt(np.mean(block**2)))
        if self._loop:
            self._loop.call_soon_threadsafe(self._q.put_nowait, block)

    def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._q = asyncio.Queue()
        self.t_started = time.perf_counter()
        self._stream.start()

    def stop(self) -> None:
        """Stop capturing; frames() drains what's left and ends."""
        self._stream.stop()
        if self._loop:
            self._q.put_nowait(None)

    async def frames(self):
        dec = Decimator(self.capture_rate, self.out_rate)
        pending = np.zeros(0, np.float32)
        while (block := await self._q.get()) is not None:
            pending = np.concatenate([pending, dec.process(block)])
            while pending.size >= self.frame_samples:
                yield to_linear16(pending[: self.frame_samples])
                pending = pending[self.frame_samples :]
        if pending.size:
            yield to_linear16(pending)

    def close(self) -> None:
        self._stream.close()
