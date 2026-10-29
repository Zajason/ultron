"""CLI entrypoint: `ultron text` (chat), `ultron bench-llm` (LLM latency)."""

from __future__ import annotations

import argparse
import asyncio
import itertools
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .brain import Brain
from .config import Config, load_config
from .metrics import TurnLog, percentile
from .platform import get_platform
from .tools import AuditLog, Executor, Registry, Tier, Timers, Tool, register_builtin

DIM, RESET, BOLD = "\033[2m", "\033[0m", "\033[1m"


def _build(cfg: Config, *, log_turns: bool) -> tuple[Brain, AuditLog]:
    plat = get_platform()

    async def on_timer(label: str) -> None:
        print(f"\n{BOLD}⏰ timer '{label}' finished{RESET}", flush=True)
        await plat.notify("Ultron", f"Timer '{label}' finished")

    async def confirm(tool: Tool, args: dict, tier: Tier) -> bool:
        ans = await asyncio.to_thread(input, f"\n{BOLD}[tier {int(tier)}] {tool.name}({args}) — proceed? [y/N] {RESET}")
        return ans.strip().lower() in ("y", "yes")

    registry = Registry()
    register_builtin(registry, plat, Timers(on_timer), cfg.tools.notes_file)
    audit = AuditLog(cfg.tools.audit_log)
    executor = Executor(registry, cfg.tools, audit, confirm)
    brain = Brain(cfg, registry, executor, turn_log=TurnLog(cfg.turns_log) if log_turns else None)
    return brain, audit


def _require_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")


HELP = "/menace 0-3 · /reset · /quit"


async def text_mode(cfg: Config) -> None:
    brain, audit = _build(cfg, log_turns=True)
    print(f"{DIM}warming up…{RESET}", end=" ", flush=True)
    print(f"{DIM}{await brain.warm():.0f}ms · routing {cfg.llm.routing} · menace {brain.menace} · {HELP}{RESET}")
    try:
        while True:
            try:
                line = (await asyncio.to_thread(input, "\nyou › ")).strip()
            except EOFError:
                break
            if not line:
                continue
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
            m = await brain.respond(line, lambda s: print(s, end="", flush=True))
            print(f"\n{DIM}{m.summary()}{RESET}")
    finally:
        await brain.close()
        audit.close()


BENCH_PROMPTS = [
    "Hello.",
    "Tell me about yourself in one sentence.",
    "What do you think of humans?",
    "Give me a word of encouragement.",
    "Do you ever get bored?",
]


async def bench_llm(cfg: Config, n: int, gap: float, tier: str) -> None:
    brain, audit = _build(cfg, log_turns=False)
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

    print(f"\n{'stage':<22}{'p50':>8}{'p95':>8}{'p99':>8}   (ms, n={n}, gap {gap:g}s)")
    for name, xs in (("LLM time-to-first-token", ttft), ("first speakable chunk", chunk), ("full reply", total)):
        print(f"{name:<22}" + "".join(f"{percentile(xs, p):>8.0f}" for p in (50, 95, 99)))
    print(f"cache hits on first call: {cache_hit}/{n}")


def main() -> None:
    p = argparse.ArgumentParser(prog="ultron")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("text", help="chat with Ultron in the terminal")
    b = sub.add_parser("bench-llm", help="measure LLM latency over N fresh single-turn conversations")
    b.add_argument("--n", type=int, default=20)
    b.add_argument("--gap", type=float, default=0, help="idle seconds between turns (tests connection reuse)")
    b.add_argument("--tier", choices=["fast", "strong", "auto"], default="fast", help="model tier (auto = router)")
    p.add_argument("--config-dir", type=Path, default=Path.cwd())
    args = p.parse_args()

    load_dotenv(args.config_dir / ".env")
    cfg = load_config(args.config_dir)
    _require_key()
    try:
        if args.cmd == "text":
            asyncio.run(text_mode(cfg))
        else:
            asyncio.run(bench_llm(cfg, args.n, args.gap, args.tier))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
