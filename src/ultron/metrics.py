"""Per-turn timing. All marks are time.perf_counter() offsets from the turn start, in ms."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TurnMetrics:
    transcript: str = ""
    model: str = ""
    t0: float = field(default_factory=time.perf_counter)
    marks: dict[str, float] = field(default_factory=dict)
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)

    def mark(self, name: str, *, once: bool = True) -> float:
        ms = (time.perf_counter() - self.t0) * 1000
        if not (once and name in self.marks):
            self.marks[name] = round(ms, 1)
        return ms

    def summary(self) -> str:
        m = self.marks
        parts = [self.model.removeprefix("claude-").split("-")[0]]
        parts.append(f"ttft {m['first_token']:.0f}ms" if "first_token" in m else "no text")
        if "first_chunk" in m:
            parts.append(f"first chunk {m['first_chunk']:.0f}ms")
        if "first_audio" in m:
            filler = " (filler)" if "tool_block" in m and m["first_audio"] < m.get("tts_first_byte", 0) else ""
            parts.append(f"first audio {m['first_audio']:.0f}ms{filler}")
        if self.tools:
            parts.append("tools " + ", ".join(f"{t['name']} {t['ms']:.0f}ms" for t in self.tools))
        parts.append(f"total {m.get('done', 0):.0f}ms")
        cr = sum(c.get("cache_read", 0) for c in self.llm_calls)
        ci = sum(c.get("input", 0) for c in self.llm_calls)
        parts.append(f"cache {cr}/{cr + ci} tok")
        return " · ".join(parts)

    def to_json(self) -> str:
        d = asdict(self)
        d.pop("t0")
        d["ts"] = time.time()
        return json.dumps(d, ensure_ascii=False)


class TurnLog:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = path.open("a", encoding="utf-8")

    def write(self, m: TurnMetrics) -> None:
        self._f.write(m.to_json() + "\n")
        self._f.flush()


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile (no interpolation): with few samples it reports a real observation."""
    if not values:
        return math.nan
    s = sorted(values)
    k = max(0, math.ceil(p / 100 * len(s)) - 1)
    return s[k]
