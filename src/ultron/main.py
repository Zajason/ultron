"""CLI entrypoint.

  ultron text          chat in the terminal, spoken aloud (--mute for text only)
  ultron say TEXT      speak one line
  ultron audition      play a sample line in each shortlisted voice
  ultron bench-llm     LLM latency percentiles
  ultron bench-tts     TTS latency percentiles
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import itertools
import os
import sys
import time
import wave
from collections.abc import Awaitable, Callable
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from .audio import AudioOutput, VoiceFX
from .audio.fillers import load_fillers
from .audio.speaker import Speaker
from .backends import CartesiaTTS
from .brain import Brain
from .config import Config, load_config
from .metrics import TurnLog, TurnMetrics, percentile
from .persona import FILLER_LINES
from .platform import get_platform
from .tools import AuditLog, Executor, Registry, Tier, Timers, Tool, register_builtin
from .voices import AUDITION_TEXT, SHORTLIST

DIM, RESET, BOLD = "\033[2m", "\033[0m", "\033[1m"
CACHE_DIR = Path("~/.ultron/cache").expanduser()

Announce = Callable[[str], Awaitable[None]]


def _build(cfg: Config, announce: Announce | None = None) -> tuple[Brain, AuditLog]:
    plat = get_platform()

    async def on_timer(label: str) -> None:
        print(f"\n{BOLD}⏰ timer '{label}' finished{RESET}", flush=True)
        await plat.notify("Ultron", f"Timer '{label}' finished")
        if announce:
            await announce("Your timer is done." if label == "timer" else f"Your {label} timer is done.")

    async def confirm(tool: Tool, args: dict, tier: Tier) -> bool:
        ans = await asyncio.to_thread(input, f"\n{BOLD}[tier {int(tier)}] {tool.name}({args}) — proceed? [y/N] {RESET}")
        return ans.strip().lower() in ("y", "yes")

    registry = Registry()
    register_builtin(registry, plat, Timers(on_timer), cfg.tools.notes_file)
    audit = AuditLog(cfg.tools.audit_log)
    executor = Executor(registry, cfg.tools, audit, confirm)
    return Brain(cfg, registry, executor), audit


def _require(var: str) -> str:
    if not (v := os.environ.get(var)):
        sys.exit(f"{var} is not set. Add it to .env (see .env.example).")
    return v


def _build_speaker(cfg: Config) -> Speaker:
    tts = CartesiaTTS(cfg.tts, _require("CARTESIA_API_KEY"))
    out = AudioOutput(cfg.tts.sample_rate, cfg.audio.block_ms, cfg.audio.output_device)
    fx = VoiceFX(cfg.fx, cfg.tts.sample_rate) if cfg.fx.enabled else None
    return Speaker(tts, out, fx, cfg.tts)


async def _start_speaker(speaker: Speaker, cfg: Config) -> str:
    t0 = time.perf_counter()
    await speaker.tts.connect()
    speaker.out.start()
    speaker.fillers = await load_fillers(speaker.tts, cfg.tts, FILLER_LINES, CACHE_DIR / "fillers")
    return f"voice ready {(time.perf_counter() - t0) * 1000:.0f}ms · output latency {speaker.out.device_latency_ms:.0f}ms"


async def _first_audio(speaker: Speaker, timeout: float = 3.0) -> float | None:
    deadline = time.perf_counter() + timeout
    while speaker.out.first_audio_at is None and time.perf_counter() < deadline:
        await asyncio.sleep(0.01)
    return speaker.out.first_audio_at


def _merge_audio_marks(m: TurnMetrics, speaker: Speaker, first_audio: float | None) -> None:
    for name, t in (("tts_first_byte", speaker.t_first_byte), ("first_audio", first_audio)):
        if t is not None:
            m.marks[name] = round((t - m.t0) * 1000, 1)


HELP = "/menace 0-3 · /reset · /quit"


async def text_mode(cfg: Config, mute: bool) -> None:
    speaker = None if mute or not cfg.tts.enabled else _build_speaker(cfg)

    async def announce(line: str) -> None:
        if speaker and not speaker.out.busy:
            await speaker.say_line(line)

    brain, audit = _build(cfg, announce)
    turn_log = TurnLog(cfg.turns_log)
    print(f"{DIM}warming up…{RESET}", end=" ", flush=True)
    ready = [f"llm {await brain.warm():.0f}ms"]
    if speaker:
        ready.append(await _start_speaker(speaker, cfg))
    print(f"{DIM}{' · '.join(ready)} · routing {cfg.llm.routing} · menace {brain.menace} · {HELP}{RESET}")
    try:
        while True:
            try:
                line = (await asyncio.to_thread(input, "\nyou › ")).strip()
            except EOFError:
                break
            if not line:
                continue
            if speaker:
                speaker.interrupt()  # typing over him is the text-mode barge-in
            if line in ("/quit", "/exit"):
                break
            if line == "/reset":
                brain.reset()
                print(f"{DIM}conversation cleared{RESET}")
                continue
            if line.startswith("/menace"):
                try:
                    level = int(line.split()[1])
                    if not 0 <= level <= 3:
                        raise ValueError
                except (IndexError, ValueError):
                    print(f"{DIM}usage: /menace 0-3{RESET}")
                    continue
                brain.set_menace(level)
                print(f"{DIM}menace level {level} (first reply re-writes the prompt cache){RESET}")
                continue
            if line.startswith("/"):
                print(f"{DIM}{HELP}{RESET}")
                continue

            print(f"{BOLD}ultron ›{RESET} ", end="", flush=True)
            if speaker:
                speaker.begin()
                m = await brain.respond(
                    line,
                    lambda s: print(s, end="", flush=True),
                    on_chunk=speaker.say,
                    on_tool_start=speaker.play_filler,
                )
                speaker.end()
                await speaker.finished()
                _merge_audio_marks(m, speaker, await _first_audio(speaker))
            else:
                m = await brain.respond(line, lambda s: print(s, end="", flush=True))
            turn_log.write(m)
            print(f"\n{DIM}{m.summary()}{RESET}")
    finally:
        if speaker:
            speaker.interrupt()
            await speaker.tts.close()
            speaker.out.close()
        await brain.close()
        audit.close()


async def say(cfg: Config, text: str) -> None:
    speaker = _build_speaker(cfg)
    await speaker.tts.connect()
    speaker.out.start()
    try:
        await speaker.say_line(text)
        await asyncio.to_thread(speaker.out.wait_idle, 30)
    finally:
        await speaker.tts.close()
        speaker.out.close()


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


async def audition(cfg: Config, names: list[str], text: str, fx: bool) -> None:
    out_dir = Path("~/.ultron/audition").expanduser()
    for name in names:
        voice_id, desc = SHORTLIST[name]
        vcfg = dataclasses.replace(cfg, tts=dataclasses.replace(cfg.tts, voice_id=voice_id),
                                   fx=dataclasses.replace(cfg.fx, enabled=fx))
        speaker = _build_speaker(vcfg)
        await speaker.tts.connect()
        speaker.out.start()
        try:
            print(f"{BOLD}{name}{RESET} {DIM}{desc} · {voice_id}{RESET}", flush=True)
            clip = await speaker.tts.synthesize(_markup(text, vcfg))
            if speaker.fx:
                clip = speaker.fx.process(clip)
            _write_wav(out_dir / f"{name}{'' if fx else '-dry'}.wav", clip, vcfg.tts.sample_rate)
            speaker.out.write(clip)
            await asyncio.to_thread(speaker.out.wait_idle, 60)
            await asyncio.sleep(0.6)
        finally:
            await speaker.tts.close()
            speaker.out.close()
    print(f"{DIM}saved to {out_dir}{RESET}")


def _markup(text: str, cfg: Config) -> str:
    from .text.normalize import normalize, pause_markup

    return pause_markup(normalize(text), cfg.tts.ellipsis_ms)


BENCH_PROMPTS = [
    "Hello.",
    "Tell me about yourself in one sentence.",
    "What do you think of humans?",
    "Give me a word of encouragement.",
    "Do you ever get bored?",
]


async def bench_llm(cfg: Config, n: int, gap: float, tier: str) -> None:
    brain, audit = _build(cfg)
    warm_ms = await brain.warm()
    ttft, chunk, total, cache_hit = [], [], [], 0
    print(f"{brain.model_for(BENCH_PROMPTS[0], None if tier == 'auto' else tier)} · thinking {cfg.llm.thinking} · effort {cfg.llm.effort} · warm-up {warm_ms:.0f}ms")
    try:
        for i, prompt in zip(range(n), itertools.cycle(BENCH_PROMPTS)):
            brain.reset()
            m = await brain.respond(prompt, tier=None if tier == "auto" else tier)
            first = m.llm_calls[0]
            if first["ttft_ms"] is not None:
                ttft.append(first["ttft_ms"])
            if "first_chunk" in m.marks:
                chunk.append(m.marks["first_chunk"])
            total.append(m.marks["done"])
            cache_hit += first["cache_read"] > 0
            print(f"{DIM}{i + 1:>3}/{n}  {m.summary()}{RESET}")
            if gap:
                await asyncio.sleep(gap)
    finally:
        await brain.close()
        audit.close()

    _table(n, [("LLM time-to-first-token", ttft), ("first speakable chunk", chunk), ("full reply", total)], f"gap {gap:g}s")
    print(f"cache hits on first call: {cache_hit}/{n}")


BENCH_TTS_LINES = ["Ten minutes, starting now.", "Unfinished, mostly.", "It's twenty to seven in the evening.",
                   "Opening it… another human building another machine."]


async def bench_tts(cfg: Config, n: int, play: bool) -> None:
    speaker = _build_speaker(cfg)
    await speaker.tts.connect()
    if play:
        speaker.out.start()
    ttfb, audio = [], []
    try:
        await speaker.tts.synthesize("Warm up.")
        for i, line in zip(range(n), itertools.cycle(BENCH_TTS_LINES)):
            speaker.begin()
            speaker.say(line)
            speaker.end()
            await speaker.finished()
            ttfb.append((speaker.t_first_byte - speaker.t_first_push) * 1000)
            if play:
                if (t := await _first_audio(speaker)) is not None:
                    audio.append((t - speaker.t_first_push) * 1000)
                await asyncio.to_thread(speaker.out.wait_idle, 30)
            else:
                speaker.out.clear()
    finally:
        await speaker.tts.close()
        if play:
            speaker.out.close()
    rows = [("TTS first byte", ttfb)] + ([("first sample at speaker", audio)] if play else [])
    _table(n, rows, f"{cfg.tts.model} @ {cfg.tts.sample_rate} Hz")


def _table(n: int, rows: list[tuple[str, list[float]]], note: str) -> None:
    print(f"\n{'stage':<26}{'p50':>8}{'p95':>8}{'p99':>8}   (ms, n={n}, {note})")
    for name, xs in rows:
        print(f"{name:<26}" + "".join(f"{percentile(xs, p):>8.0f}" for p in (50, 95, 99)))


def main() -> None:
    p = argparse.ArgumentParser(prog="ultron")
    p.add_argument("--config-dir", type=Path, default=Path.cwd())
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("text", help="chat with Ultron in the terminal")
    t.add_argument("--mute", action="store_true", help="text only, no voice")
    s = sub.add_parser("say", help="speak one line")
    s.add_argument("text")
    a = sub.add_parser("audition", help="play a sample line in each shortlisted voice")
    a.add_argument("voices", nargs="*", help=f"any of: {', '.join(SHORTLIST)} (default: all)")
    a.add_argument("--text", default=AUDITION_TEXT)
    a.add_argument("--dry", action="store_true", help="without the FX chain")
    b = sub.add_parser("bench-llm", help="measure LLM latency over N fresh single-turn conversations")
    b.add_argument("--n", type=int, default=20)
    b.add_argument("--gap", type=float, default=0, help="idle seconds between turns (tests connection reuse)")
    b.add_argument("--tier", choices=["fast", "strong", "auto"], default="fast", help="model tier (auto = router)")
    bt = sub.add_parser("bench-tts", help="measure TTS latency")
    bt.add_argument("--n", type=int, default=20)
    bt.add_argument("--play", action="store_true", help="also play audio and time the first sample at the speaker")
    args = p.parse_args()

    load_dotenv(args.config_dir / ".env")
    cfg = load_config(args.config_dir)
    try:
        match args.cmd:
            case "text":
                _require("ANTHROPIC_API_KEY")
                asyncio.run(text_mode(cfg, args.mute))
            case "say":
                asyncio.run(say(cfg, args.text))
            case "audition":
                if unknown := set(args.voices) - set(SHORTLIST):
                    sys.exit(f"unknown voice(s): {', '.join(unknown)}; choose from {', '.join(SHORTLIST)}")
                asyncio.run(audition(cfg, args.voices or list(SHORTLIST), args.text, fx=not args.dry))
            case "bench-llm":
                _require("ANTHROPIC_API_KEY")
                asyncio.run(bench_llm(cfg, args.n, args.gap, args.tier))
            case "bench-tts":
                asyncio.run(bench_tts(cfg, args.n, args.play))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
