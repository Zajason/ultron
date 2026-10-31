"""Durable memory: facts the user tells Ultron, kept across sessions in a plain markdown file
(one "- fact" per line, editable by hand)."""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

from .tools.registry import Registry, Tier, Tool


class Memory:
    def __init__(self, path: Path, max_facts: int = 200) -> None:
        self.path = path
        self.max_facts = max_facts
        self.facts: list[str] = []
        if path.exists():
            self.facts = [m.group(1).strip() for line in path.read_text().splitlines()
                          if (m := re.match(r"^\s*-\s+(.+)$", line))]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("# What Ultron remembers\n\n" + "".join(f"- {f}\n" for f in self.facts))

    def add(self, fact: str) -> str:
        fact = " ".join(fact.split())
        if any(fact.lower() == f.lower() for f in self.facts):
            return "already known"
        self.facts.append(fact)
        self.facts = self.facts[-self.max_facts :]
        self._save()
        return "remembered"

    def forget(self, what: str) -> list[str]:
        words = [w for w in re.findall(r"\w+", what.lower()) if len(w) > 2]
        gone = [f for f in self.facts if words and all(w in f.lower() for w in words)]
        self.facts = [f for f in self.facts if f not in gone]
        self._save()
        return gone

    def render(self) -> str:
        return "\n".join(f"- {f}" for f in self.facts)


def register_memory(reg: Registry, memory: Memory) -> None:
    async def remember(fact: str) -> str:
        stamp = dt.date.today().isoformat()
        return memory.add(f"{fact} (noted {stamp})")

    async def forget(what: str) -> str:
        gone = memory.forget(what)
        return f"forgot {len(gone)}: " + "; ".join(gone) if gone else "nothing matched"

    reg.add(Tool("remember", "Save a durable fact about the user (preferences, people, plans, routines) so you know "
                 "it in future conversations. Write it as a short third-person statement.",
                 {"fact": {"type": "string"}}, ("fact",), Tier.LOW_RISK, remember))
    reg.add(Tool("forget", "Remove remembered facts that match these words, when the user asks you to forget something.",
                 {"what": {"type": "string"}}, ("what",), Tier.LOW_RISK, forget))
