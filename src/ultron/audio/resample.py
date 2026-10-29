"""Streaming resampler with varispeed.

TTS audio arrives at 24 kHz; the Mac's output runs natively at 48 kHz. Letting CoreAudio convert
costs 15-30 ms of extra output latency, so we convert here instead (linear interpolation; the
upsampling images sit above 12 kHz and are inaudible on speech).

Varispeed: treating the input as slightly slower than it really is lowers pitch and formants
together, which gives the voice more size and weight at zero added latency. The TTS speed setting
is raised to compensate, so the speaking rate stays where it was configured.
"""

from __future__ import annotations

import numpy as np


def varispeed_ratio(semitones: float) -> float:
    """Playback-speed factor for a pitch change in semitones (negative = lower and slower)."""
    return 2.0 ** (semitones / 12.0)


class Resampler:
    def __init__(self, in_rate: int, out_rate: int, semitones: float = 0.0) -> None:
        self.step = in_rate * varispeed_ratio(semitones) / out_rate  # input samples per output sample
        self._prev = np.zeros(1, np.float32)
        self._pos = 1.0  # read position in [prev, *chunk]; starts at the first new sample

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        buf = np.concatenate([self._prev, x.astype(np.float32, copy=False)])
        last = buf.size - 1
        if self._pos > last:
            self._pos -= x.size
            self._prev = buf[-1:]
            return np.zeros(0, np.float32)
        n = int((last - self._pos) // self.step) + 1
        idx = self._pos + self.step * np.arange(n)
        i0 = idx.astype(np.int64)
        frac = (idx - i0).astype(np.float32)
        i1 = np.minimum(i0 + 1, last)
        y = buf[i0] * (1 - frac) + buf[i1] * frac
        self._pos = idx[-1] + self.step - last
        self._prev = buf[-1:]
        return y.astype(np.float32, copy=False)

    def reset(self) -> None:
        self._prev = np.zeros(1, np.float32)
        self._pos = 1.0
