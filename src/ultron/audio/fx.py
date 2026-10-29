"""Real-time voice FX.

Chain: high-pass → compressor (close-mic density) → low-shelf warmth → optional rasp (parallel
saturation) → short plate. Two parallel layers mixed underneath:
  - metal: ring modulator + chorus, a faint machine edge
  - sub:   analog-style octave divider, an inhuman low double one octave down

Streaming-safe: every stage keeps state between chunks (pedalboard reset=False; the ring
modulator's phase and the divider's flip-flop carry over), and nothing looks ahead, so the only
cost is compute time per chunk.
"""

from __future__ import annotations

import numpy as np
from pedalboard import (
    Chorus,
    Compressor,
    Distortion,
    HighpassFilter,
    LowpassFilter,
    LowShelfFilter,
    Pedalboard,
    Reverb,
)

from ..config import FXConfig


class OctaveDown:
    """Frequency divider: a square wave that flips on every other upward zero crossing of the
    low-passed voice, shaped by its envelope. Same trick as a guitar octave pedal."""

    def __init__(self, sr: int) -> None:
        self.sr = sr
        self.track = Pedalboard([LowpassFilter(cutoff_frequency_hz=300)])
        self.env = Pedalboard([LowpassFilter(cutoff_frequency_hz=25)])
        self.smooth = Pedalboard([LowpassFilter(cutoff_frequency_hz=220), LowpassFilter(cutoff_frequency_hz=220)])
        self._last_pos = False
        self._flips = 0

    def process(self, x: np.ndarray) -> np.ndarray:
        lp = self.track(x[np.newaxis, :], self.sr, reset=False)[0]
        pos = lp > 0
        prev = np.concatenate([[self._last_pos], pos[:-1]])
        ups = pos & ~prev
        flips = self._flips + np.cumsum(ups)
        self._flips = int(flips[-1]) % 2
        self._last_pos = bool(pos[-1])
        square = np.where(flips % 2 == 0, 1.0, -1.0).astype(np.float32)
        env = self.env(np.abs(lp)[np.newaxis, :], self.sr, reset=False)[0]
        return self.smooth((square * env * 2.0)[np.newaxis, :], self.sr, reset=False)[0]

    def reset(self) -> None:
        for b in (self.track, self.env, self.smooth):
            b.reset()
        self._last_pos, self._flips = False, 0


class VoiceFX:
    def __init__(self, cfg: FXConfig, sample_rate: int) -> None:
        self.sr = sample_rate
        self.cfg = cfg
        self._phase = 0.0
        stages = [HighpassFilter(cutoff_frequency_hz=70)]
        if cfg.compress:
            stages.append(Compressor(threshold_db=-24, ratio=3.0, attack_ms=4, release_ms=80))
        stages.append(LowShelfFilter(cutoff_frequency_hz=cfg.warmth_hz, gain_db=cfg.warmth_db, q=0.7))
        self.pre = Pedalboard(stages)
        self.rasp = Pedalboard([HighpassFilter(cutoff_frequency_hz=900), Distortion(drive_db=18), LowpassFilter(cutoff_frequency_hz=5000)])
        self.space = Pedalboard([Reverb(room_size=0.18, damping=0.6, wet_level=cfg.reverb_wet, dry_level=1.0, width=0.0)])
        self.metal = Pedalboard([
            Chorus(rate_hz=0.7, depth=0.35, centre_delay_ms=6.0, feedback=0.25, mix=1.0),
            HighpassFilter(cutoff_frequency_hz=300),
        ])
        self.sub = OctaveDown(sample_rate)

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        c = self.cfg
        voice = self.pre(x[np.newaxis, :], self.sr, reset=False)[0]
        if c.rasp > 0:
            voice = voice + c.rasp * self.rasp(voice[np.newaxis, :], self.sr, reset=False)[0]
        out = self.space(voice[np.newaxis, :], self.sr, reset=False)[0]
        if c.metal_mix > 0:
            t = self._phase + 2 * np.pi * c.metal_hz * np.arange(x.size) / self.sr
            self._phase = float((t[-1] + 2 * np.pi * c.metal_hz / self.sr) % (2 * np.pi))
            ring = (voice * np.sin(t)).astype(np.float32)[np.newaxis, :]
            out = out + c.metal_mix * self.metal(ring, self.sr, reset=False)[0]
        if c.sub_mix > 0:
            out = out + c.sub_mix * self.sub.process(voice)
        return np.clip(out * c.gain, -1.0, 1.0).astype(np.float32, copy=False)

    def reset(self) -> None:
        for b in (self.pre, self.rasp, self.space, self.metal):
            b.reset()
        self.sub.reset()
        self._phase = 0.0
