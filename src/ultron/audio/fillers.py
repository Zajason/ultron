"""Filler lines, synthesised once and cached on disk so they play with zero network latency."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import numpy as np

from ..backends import TTSBackend
from ..config import TTSConfig
from ..text.normalize import normalize, pause_markup


def _key(cfg: TTSConfig, text: str) -> str:
    ident = f"{cfg.model}|{cfg.voice_id}|{cfg.sample_rate}|{cfg.speed}|{cfg.pitch_semitones}|{cfg.emotion}|{cfg.ellipsis_ms}|{text}"
    return hashlib.sha1(ident.encode()).hexdigest()[:16]


async def load_fillers(tts: TTSBackend, cfg: TTSConfig, lines: list[str], cache_dir: Path) -> dict[str, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, np.ndarray] = {}
    missing = []
    for line in lines:
        f = cache_dir / f"{_key(cfg, line)}.npy"
        if f.exists():
            out[line] = np.load(f)
        else:
            missing.append((line, f))

    # One at a time: the Cartesia plan allows 2 concurrent generations, and a live reply may need one.
    sem = asyncio.Semaphore(1)

    async def render(line: str, f: Path) -> None:
        async with sem:
            clip = await tts.synthesize(pause_markup(normalize(line), cfg.ellipsis_ms))
        np.save(f, clip)
        out[line] = clip

    await asyncio.gather(*(render(line, f) for line, f in missing))
    return out
