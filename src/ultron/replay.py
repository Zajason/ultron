"""Replay harness: stream recorded utterances through STT in real time, without a microphone.

Two modes:
  ptt   send the utterance, then ForceEndTurn at the end of the audio (push-to-talk release)
  auto  send the utterance, then silence, and wait for Flux's own end-of-turn (Milestone 4)
Latency is measured from the moment the last speech frame is sent, i.e. when the speaker stopped.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import wave
from dataclasses import dataclass
from pathlib import Path

from .backends import DeepgramFlux, TurnEvent

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "utterances"


@dataclass
class ReplayResult:
    name: str
    expected: str
    transcript: str
    connect_ms: float
    final_ms: float | None  # speech end -> EndOfTurn
    eager_ms: float | None  # speech end -> EagerEndOfTurn (auto mode, if enabled)
    trigger: str | None
    wer: float


def load_fixtures(directory: Path = FIXTURES) -> list[tuple[str, str, bytes]]:
    index = json.loads((directory / "utterances.json").read_text())
    out = []
    for name, text in index.items():
        with wave.open(str(directory / name)) as w:
            out.append((name, text, w.readframes(w.getnframes())))
    return out


def _words(s: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", s.lower()).split()


def wer(ref: str, hyp: str) -> float:
    r, h = _words(ref), _words(hyp)
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[len(h)] / max(1, len(r))


async def replay_one(stt: DeepgramFlux, name: str, expected: str, pcm: bytes, mode: str,
                     frame_ms: int = 80, sample_rate: int = 16000, tail_s: float = 6.0) -> ReplayResult:
    frame_bytes = sample_rate * frame_ms // 1000 * 2
    t0 = time.perf_counter()
    stream = await stt.open(ptt=mode == "ptt")
    connect_ms = (stream.t_connected - t0) * 1000
    try:
        start = time.perf_counter()
        frames = [pcm[i : i + frame_bytes] for i in range(0, len(pcm), frame_bytes)]
        for i, frame in enumerate(frames):
            await stream.send(frame)
            await asyncio.sleep(max(0.0, start + (i + 1) * frame_ms / 1000 - time.perf_counter()))
        t_speech_end = time.perf_counter()

        if mode == "ptt":
            await stream.end_turn()
            silence = b""
        else:
            silence = bytes(frame_bytes)

        eager: TurnEvent | None = None
        final: TurnEvent | None = None
        early: list[str] = []  # ptt: turns Flux ended on its own before the release
        deadline = t_speech_end + tail_s
        sent = 0
        while final is None and time.perf_counter() < deadline:
            if silence:
                await stream.send(silence)
                sent += 1
            wait_until = t_speech_end + (sent + (1 if silence else 0)) * frame_ms / 1000
            try:
                ev = await asyncio.wait_for(stream.events.get(), timeout=max(0.001, wait_until - time.perf_counter()))
            except TimeoutError:
                continue
            if ev.kind == "eager" and eager is None:
                eager = ev
            elif ev.kind == "end" and mode == "ptt" and ev.trigger != "manual":
                early.append(ev.transcript)
            elif ev.kind == "end":
                final = ev
            elif ev.kind in ("error", "closed"):
                raise RuntimeError(f"STT {ev.kind}: {ev.detail}")
            elif ev.kind == "warning" and mode == "ptt":
                # No active turn: Flux already closed it before the release. That transcript stands.
                final = TurnEvent("end", "", ev.t, "manual-empty")
        transcript = " ".join(t for t in [*early, final.transcript if final else ""] if t)
        return ReplayResult(
            name, expected, transcript, connect_ms,
            (final.t - t_speech_end) * 1000 if final else None,
            (eager.t - t_speech_end) * 1000 if eager else None,
            final.trigger if final else None,
            wer(expected, transcript),
        )
    finally:
        await stream.close()
