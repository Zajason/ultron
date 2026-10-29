"""Wake word: openWakeWord's feature pipeline on onnxruntime, plus our "Ultron" classifier.

16 kHz audio -> mel spectrogram (8 frames per 80 ms) -> Google speech-embedding over the last 76
mel frames -> one 96-dim embedding per 80 ms -> classifier over the last 16 embeddings (1.28 s).
Only the classifier is ours; the two feature models are frozen and shared with openWakeWord, so
its large precomputed negative set can be used for training as-is.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

MODELS = Path(__file__).resolve().parent / "models"
CHUNK = 1280  # 80 ms at 16 kHz: one embedding per chunk
WINDOW = 16  # embeddings per classifier input (1.28 s)
MEL_WINDOW = 76
# The mel model has long memory: a frame's value depends on audio well before it (an adaptive
# normaliser, it seems). Live, each 80 ms step computes mel over this much preceding audio; training
# clips get the same lead-in. Measured feature error vs. full history: 0.1 s -> 0.50, 2 s -> 0.12
# (|e| ~ 15), at ~1.4% of one core either way.
HISTORY = 32000


def _session(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = opts.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


class Features:
    def __init__(self) -> None:
        self.mel = _session(MODELS / "melspectrogram.onnx")
        self.emb = _session(MODELS / "embedding_model.onnx")

    def melspec(self, pcm: np.ndarray) -> np.ndarray:
        """int16-valued audio (any float/int dtype) -> (frames, 32)."""
        x = np.asarray(pcm, dtype=np.float32)[None, :]
        spec = np.squeeze(self.mel.run(None, {"input": x})[0])
        return spec / 10 + 2  # matches Google's TF implementation (openWakeWord's transform)

    def embed_windows(self, mel: np.ndarray) -> np.ndarray:
        """(n, 76, 32) mel windows -> (n, 96)."""
        out = self.emb.run(None, {"input_1": mel[..., None].astype(np.float32)})[0]
        return out.reshape(out.shape[0], 96)

    def embed_clip(self, pcm: np.ndarray, last: int | None = None) -> np.ndarray:
        """Whole clip -> (n, 96), one embedding per 80 ms, windows counted back from the clip's end.

        End-aligned so that, for a clip ending on an 80 ms boundary, these are exactly the features
        StreamingFeatures produces live (openWakeWord counts from the start, which lands one 10 ms
        mel frame off the streaming grid: ~16% feature error between training and inference).
        """
        spec = self.melspec(pcm)
        starts = list(range(spec.shape[0] - MEL_WINDOW, -1, -8))[:last]
        if not starts:
            return np.zeros((0, 96), np.float32)
        return self.embed_windows(np.stack([spec[i : i + MEL_WINDOW] for i in reversed(starts)]))


class StreamingFeatures:
    """Same features computed incrementally, 80 ms at a time (openWakeWord's streaming scheme)."""

    def __init__(self, features: Features) -> None:
        self.f = features
        self.raw = np.zeros(0, np.float32)  # recent raw audio (int16 scale)
        self.pending = np.zeros(0, np.float32)
        self.mel = np.ones((MEL_WINDOW, 32), np.float32)
        self.emb = np.zeros((WINDOW, 96), np.float32)

    def push(self, pcm16: np.ndarray) -> int:
        """Feed int16-scaled audio; returns how many new embeddings were produced."""
        self.pending = np.concatenate([self.pending, pcm16.astype(np.float32, copy=False)])
        new = 0
        while self.pending.size >= CHUNK:
            chunk, self.pending = self.pending[:CHUNK], self.pending[CHUNK:]
            # Mel over the new chunk plus HISTORY of context; keep only the newest 8 frames.
            self.raw = np.concatenate([self.raw, chunk])[-(CHUNK + HISTORY):]
            self.mel = np.vstack([self.mel, self.f.melspec(self.raw)[-8:]])[-(MEL_WINDOW + 24):]
            e = self.f.embed_windows(self.mel[-MEL_WINDOW:][None])
            self.emb = np.vstack([self.emb, e])[-WINDOW * 4:]
            new += 1
        return new

    def window(self) -> np.ndarray:
        return self.emb[-WINDOW:][None].astype(np.float32)  # (1, 16, 96)


def stream_embed(features: Features, pcm16: np.ndarray, last: int = WINDOW) -> np.ndarray:
    """Training features computed by the live streaming code itself, so training and inference
    see identical numbers. pcm16 should start at least HISTORY + ~0.8 s before the scored part."""
    s = StreamingFeatures(features)
    n = (pcm16.size // CHUNK) * CHUNK
    s.push(pcm16[pcm16.size - n :])
    return s.emb[-last:].copy()


@dataclass
class Detection:
    score: float
    t: float  # perf_counter when detected


class WakeWordDetector:
    """Always-on detector. push() 16 kHz float audio; returns a Detection when "Ultron" is heard."""

    def __init__(self, model: Path | None = None, threshold: float = 0.5, patience: int = 1,
                 refractory_s: float = 2.0, features: Features | None = None) -> None:
        self.features = features or Features()
        self.stream = StreamingFeatures(self.features)
        self.clf = _session(model or MODELS / "ultron_wakeword.onnx")
        self.input_name = self.clf.get_inputs()[0].name
        self.threshold = threshold
        self.patience = patience  # consecutive frames above threshold
        self.refractory = refractory_s
        self._above = 0
        self._last = -1e9
        self.score = 0.0

    def reset(self) -> None:
        """Forget buffered audio. Must happen whenever detection resumes after a pause, or the
        'Ultron' that woke us last time is still in the window and wakes us again."""
        self.stream = StreamingFeatures(self.features)
        self._above = 0
        self.score = 0.0

    def push(self, audio: np.ndarray) -> Detection | None:
        hit = None
        n = self.stream.push(audio * 32767)
        for _ in range(n):
            self.score = float(self.clf.run(None, {self.input_name: self.stream.window()})[0].ravel()[0])
            self._above = self._above + 1 if self.score >= self.threshold else 0
            now = time.perf_counter()
            if self._above >= self.patience and now - self._last > self.refractory:
                self._last = now
                self._above = 0
                hit = Detection(self.score, now)
        return hit
