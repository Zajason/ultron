"""Tier 0/1 tools for Milestone 1."""

from __future__ import annotations

import asyncio
import datetime as dt
import itertools
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import psutil

from ..platform import PlatformActions
from .registry import Registry, Tier, Tool


@dataclass
class Timer:
    id: int
    label: str
    ends_at: float  # monotonic
    task: asyncio.Task


class Timers:
    def __init__(self, on_fire: Callable[[str], Awaitable[None]]) -> None:
        self.on_fire = on_fire
        self._timers: dict[int, Timer] = {}
        self._ids = itertools.count(1)

    def start(self, seconds: float, label: str) -> Timer:
        tid = next(self._ids)
        task = asyncio.create_task(self._run(tid, seconds, label))
        t = self._timers[tid] = Timer(tid, label, time.monotonic() + seconds, task)
        return t

    async def _run(self, tid: int, seconds: float, label: str) -> None:
        await asyncio.sleep(seconds)
        self._timers.pop(tid, None)
        await self.on_fire(label)

    def active(self) -> list[tuple[str, float]]:
        now = time.monotonic()
        return [(t.label, t.ends_at - now) for t in sorted(self._timers.values(), key=lambda t: t.ends_at)]


def _dur(seconds: float) -> str:
    s = max(0, round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    parts = [f"{h} h"] * bool(h) + [f"{m} min"] * bool(m) + [f"{s} s"] * bool(s or not (h or m))
    return " ".join(parts)


def register_builtin(reg: Registry, plat: PlatformActions, timers: Timers, notes_file: Path) -> None:
    psutil.cpu_percent(None)  # prime: the first non-blocking call always returns 0

    async def get_time() -> str:
        now = dt.datetime.now().astimezone()
        return now.strftime("%A %d %B %Y, %H:%M, timezone %Z")

    async def system_stats() -> str:
        vm = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        out = [
            f"cpu {psutil.cpu_percent(None):.0f}%",
            f"memory {vm.used / 2**30:.1f} of {vm.total / 2**30:.0f} GB used",
            f"disk {disk.free / 2**30:.0f} GB free",
        ]
        if (b := psutil.sensors_battery()) is not None:
            out.append(f"battery {b.percent:.0f}% {'charging' if b.power_plugged else 'on battery'}")
        return ", ".join(out)

    async def open_app(name: str) -> str:
        return await plat.open_app(name)

    async def set_volume(level: int) -> str:
        return await plat.set_volume(level)

    async def set_timer(seconds: int, label: str = "timer") -> str:
        timers.start(seconds, label)
        return f"timer '{label}' set for {_dur(seconds)}"

    async def list_timers() -> str:
        active = timers.active()
        if not active:
            return "no timers running"
        return "; ".join(f"'{label}': {_dur(left)} left" for label, left in active)

    async def create_note(text: str) -> str:
        notes_file.parent.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        await asyncio.to_thread(_append, notes_file, f"- {stamp}: {text}\n")
        return "note saved"

    for tool in [
        Tool("get_time", "Current local date, time and timezone.", {}, (), Tier.READ, get_time),
        Tool("system_stats", "CPU, memory, disk and battery status of this computer.", {}, (), Tier.READ, system_stats),
        Tool(
            "list_timers", "Running timers and the time left on each.", {}, (), Tier.READ, list_timers,
        ),
        Tool(
            "open_app",
            "Open or focus an application by its name, e.g. 'Visual Studio Code', 'Safari', 'Spotify'.",
            {"name": {"type": "string", "description": "Application name as macOS knows it"}},
            ("name",), Tier.LOW_RISK, open_app,
        ),
        Tool(
            "set_volume",
            "Set the speaker output volume.",
            {"level": {"type": "integer", "minimum": 0, "maximum": 100, "description": "0 to 100"}},
            ("level",), Tier.LOW_RISK, set_volume,
        ),
        Tool(
            "set_timer",
            "Start a countdown timer. The user is notified when it ends.",
            {
                "seconds": {"type": "integer", "minimum": 1, "maximum": 86400},
                "label": {"type": "string", "description": "Short name, e.g. 'pasta'"},
            },
            ("seconds",), Tier.LOW_RISK, set_timer,
        ),
        Tool(
            "create_note",
            "Save a short note to the user's notes file.",
            {"text": {"type": "string"}},
            ("text",), Tier.LOW_RISK, create_note,
        ),
    ]:
        reg.add(tool)


def _append(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
