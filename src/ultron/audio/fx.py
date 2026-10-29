"""Real-time voice FX: warmth, a short plate, and a faint metallic layer.

Streaming-safe: every stage keeps its state between chunks (pedalboard reset=False, and the ring
modulator carries its phase), so chunk boundaries are inaudible. Nothing here looks ahead, so the
only added latency is compute time per chunk.
"""

from __future__ import annotations

import numpy as np
from pedalboard import Chorus, HighpassFilter, LowShelfFilter, Pedalboard, Reverb

from ..config import FXConfig


class VoiceFX:
    def __init__(self, cfg: FXConfig, sample_rate: int) -> None:
        self.sr = sample_rate
        self.mix = cfg.metal_mix
        self.ring_hz = cfg.metal_hz
        self._phase = 0.0
        self.dry = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=70),
            LowShelfFilter(cutoff_frequency_hz=220, gain_db=cfg.warmth_db, q=0.7),
            Reverb(room_size=0.2, damping=0.6, wet_level=cfg.reverb_wet, dry_level=1.0, width=0.0),
        ])
        self.metal = Pedalboard([
            Chorus(rate_hz=0.7, depth=0.35, centre_delay_ms=6.0, feedback=0.25, mix=1.0),
            HighpassFilter(cutoff_frequency_hz=300),
        ])

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        block = x[np.newaxis, :]
        out = self.dry(block, self.sr, reset=False)[0]
        if self.mix > 0:
            t = self._phase + 2 * np.pi * self.ring_hz * np.arange(x.size) / self.sr
            self._phase = float((t[-1] + 2 * np.pi * self.ring_hz / self.sr) % (2 * np.pi))
            ring = (x * np.sin(t)).astype(np.float32)[np.newaxis, :]
            out = out + self.mix * self.metal(ring, self.sr, reset=False)[0]
        return np.clip(out, -1.0, 1.0).astype(np.float32, copy=False)

    def reset(self) -> None:
        self.dry.reset()
        self.metal.reset()
        self._phase = 0.0
