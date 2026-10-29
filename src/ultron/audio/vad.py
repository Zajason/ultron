"""Local voice activity detection (Silero VAD v6, ONNX) for barge-in.

Waiting for STT to recognise words before stopping Ultron takes hundreds of ms; Silero decides on
32 ms windows locally (~0.1 ms of CPU each), so his audio can stop within the 150 ms budget.
It runs on echo-cancelled audio, otherwise his own voice would count as the user speaking.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

MODEL = Path(__file__).resolve().parent.parent / "models" / "silero_vad.onnx"
WINDOW = 512  # samples at 16 kHz = 32 ms
CONTEXT = 64  # the v5+ model expects the previous 64 samples prepended


@dataclass
class VADEvent:
    kind: str  # "start" | "stop"
    t: float  # perf_counter estimate of when the speech actually started/stopped at the mic
    prob: float


class SileroVAD:
    def __init__(self, threshold: float = 0.5, start_ms: int = 64, stop_ms: int = 320,
                 input_latency_ms: float = 0.0, sample_rate: int = 16000) -> None:
        if sample_rate != 16000:
            raise ValueError("SileroVAD runs at 16 kHz")
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(MODEL), sess_options=opts, providers=["CPUExecutionProvider"])
        self.threshold = threshold
        self.release = max(0.15, threshold - 0.15)  # hysteresis
        self.start_windows = max(1, round(start_ms / 32))
        self.stop_windows = max(1, round(stop_ms / 32))
        self.input_latency = input_latency_ms / 1000
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), np.float32)
        self._context = np.zeros(CONTEXT, np.float32)
        self._buf = np.zeros(0, np.float32)
        self.speaking = False
        self._run = 0  # consecutive windows agreeing with a state change
        self.prob = 0.0
        self.last_voiced: float | None = None  # when the most recent voiced window ended, at the mic

    def window_prob(self, x: np.ndarray) -> float:
        inp = np.concatenate([self._context, x])[np.newaxis, :].astype(np.float32)
        out, self._state = self.session.run(None, {"input": inp, "state": self._state, "sr": np.array(16000, np.int64)})
        self._context = x[-CONTEXT:]
        return float(out[0][0])

    def process(self, block: np.ndarray, now: float | None = None) -> list[VADEvent]:
        """Feed 16 kHz float audio; returns state changes. `now` is when the block arrived."""
        now = time.perf_counter() if now is None else now
        self._buf = np.concatenate([self._buf, block.astype(np.float32, copy=False)])
        events = []
        n = self._buf.size // WINDOW
        for i in range(n):
            self.prob = p = self.window_prob(self._buf[i * WINDOW : (i + 1) * WINDOW])
            # Audio age of the end of this window: what's still buffered after it, plus device latency.
            age = (self._buf.size - (i + 1) * WINDOW) / 16000 + self.input_latency
            if p >= self.threshold:
                self.last_voiced = now - age
            if not self.speaking:
                self._run = self._run + 1 if p >= self.threshold else 0
                if self._run >= self.start_windows:
                    self.speaking, self._run = True, 0
                    onset = now - age - self.start_windows * WINDOW / 16000
                    events.append(VADEvent("start", onset, p))
            else:
                self._run = self._run + 1 if p < self.release else 0
                if self._run >= self.stop_windows:
                    self.speaking, self._run = False, 0
                    end = now - age - self.stop_windows * WINDOW / 16000
                    events.append(VADEvent("stop", end, p))
        self._buf = self._buf[n * WINDOW :]
        return events
