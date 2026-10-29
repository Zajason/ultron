"""Acoustic echo cancellation (WebRTC AEC3 via livekit's AudioProcessingModule).

Without it, open speakers feed Ultron's own voice back into the mic: VAD fires on it (he
interrupts himself) and STT transcribes it. AEC needs two streams at the same rate in 10 ms
frames: the mic (near end) and exactly what was played (far end, the "reference"). The output
callback copies every block it plays into a queue; this module downsamples it and feeds it to the
canceller just before the mic frames that may contain its echo.
"""

from __future__ import annotations

import collections

import numpy as np
from livekit import rtc

from .input import Downsampler


class EchoCanceller:
    def __init__(self, rate: int, render_rate: int, delay_ms: int, noise_suppression: bool = False) -> None:
        self.rate = rate
        self.frame = rate // 100  # 10 ms
        self.apm = rtc.AudioProcessingModule(echo_cancellation=True, high_pass_filter=True,
                                             noise_suppression=noise_suppression)
        self.apm.set_stream_delay_ms(delay_ms)
        self._render_ds = Downsampler(render_rate, rate)
        self._render_q: collections.deque[np.ndarray] = collections.deque()
        self._render_buf = np.zeros(0, np.float32)
        self._near_buf = np.zeros(0, np.float32)

    # Called from the output callback thread: just a copy into a deque.
    def on_render(self, block: np.ndarray) -> None:
        self._render_q.append(block.copy())

    def _feed_render(self) -> None:
        while self._render_q:
            self._render_buf = np.concatenate([self._render_buf, self._render_ds.process(self._render_q.popleft())])
        while self._render_buf.size >= self.frame:
            f = _frame(self._render_buf[: self.frame], self.rate)
            self.apm.process_reverse_stream(f)
            self._render_buf = self._render_buf[self.frame :]

    def process(self, near: np.ndarray) -> np.ndarray:
        """Mic audio at `rate` in, echo-cancelled audio out (whole 10 ms frames; remainder carried)."""
        self._near_buf = np.concatenate([self._near_buf, near.astype(np.float32, copy=False)])
        n = self._near_buf.size // self.frame
        if n == 0:
            return np.zeros(0, np.float32)
        out = np.empty(n * self.frame, np.float32)
        for i in range(n):
            self._feed_render()
            f = _frame(self._near_buf[i * self.frame : (i + 1) * self.frame], self.rate)
            self.apm.process_stream(f)
            out[i * self.frame : (i + 1) * self.frame] = np.frombuffer(f.data, np.int16) / 32768.0
        self._near_buf = self._near_buf[n * self.frame :]
        return out


def _frame(x: np.ndarray, rate: int) -> rtc.AudioFrame:
    data = (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()
    return rtc.AudioFrame(data, rate, 1, x.size)
