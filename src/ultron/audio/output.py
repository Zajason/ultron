"""Low-latency speaker output: a PortAudio callback stream fed from asyncio.

The callback thread only copies samples out of a queue; nothing else happens on it. Writers
append float32 mono blocks; clear() drops everything queued (barge-in).
"""

from __future__ import annotations

import collections
import threading
import time

import numpy as np
import sounddevice as sd


class AudioOutput:
    def __init__(self, sample_rate: int = 0, latency_ms: float = 10, device: str | int | None = None) -> None:
        # 0 = the device's native rate, so CoreAudio doesn't resample (and add latency) behind us.
        self.sample_rate = sample_rate or int(sd.query_devices(device or None, kind="output")["default_samplerate"])
        self._q: collections.deque[np.ndarray] = collections.deque()
        self._offset = 0  # samples already played from _q[0]
        self._lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()
        self._armed = False
        self.first_audio_at: float | None = None  # perf_counter time the first armed sample hits the DAC
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=0,  # let CoreAudio pick per-callback sizes; with a small latency hint that is lowest
            latency=latency_ms / 1000,
            device=device or None,
            callback=self._callback,
        )

    def start(self) -> None:
        self._stream.start()

    @property
    def device_latency_ms(self) -> float:
        return self._stream.latency * 1000

    def arm(self) -> None:
        """Record when the next audible sample actually plays (for voice-to-voice timing)."""
        self.first_audio_at = None
        self._armed = True

    def write(self, samples: np.ndarray) -> None:
        if samples.size:
            with self._lock:
                self._q.append(samples.astype(np.float32, copy=False))
                self._idle.clear()

    def clear(self) -> None:
        with self._lock:
            self._q.clear()
            self._offset = 0
            self._idle.set()

    def wait_idle(self, timeout: float | None = None) -> bool:
        return self._idle.wait(timeout)

    @property
    def busy(self) -> bool:
        return not self._idle.is_set()

    def _callback(self, outdata, frames, time_info, status) -> None:
        out = outdata[:, 0]
        filled = 0
        with self._lock:
            while filled < frames and self._q:
                head = self._q[0]
                n = min(frames - filled, head.size - self._offset)
                out[filled : filled + n] = head[self._offset : self._offset + n]
                filled += n
                self._offset += n
                if self._offset >= head.size:
                    self._q.popleft()
                    self._offset = 0
            if not self._q:
                self._idle.set()
        out[filled:] = 0.0
        if self._armed and filled:
            # Time until this buffer reaches the DAC; CoreAudio reports it via the stream clock.
            dac_delay = time_info.outputBufferDacTime - time_info.currentTime
            if not 0 < dac_delay < 1:
                dac_delay = self._stream.latency
            self.first_audio_at = time.perf_counter() + dac_delay
            self._armed = False

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()
