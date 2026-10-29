"""Config loading: config.toml, then per-platform and per-host overrides."""

from __future__ import annotations

import socket
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class LLMConfig:
    fast_model: str = "claude-haiku-5-5"
    strong_model: str = "claude-sonnet-5-5"
    max_tokens: int = 1024
    thinking: str = "disabled"
    effort: str = "low"
    cache_ttl: str = "1h"
    max_turns: int = 12
    max_tool_rounds: int = 5
    keepalive_expiry_s: float = 300


@dataclass(frozen=True)
class ToolsConfig:
    max_auto_tier: int = 1
    tier3_enabled: bool = False
    timeout_s: float = 10
    max_output_chars: int = 2000
    notes_file: Path = Path("~/.ultron/notes.md")
    audit_log: Path = Path("~/.ultron/audit.jsonl")
    tier_overrides: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Config:
    user_name: str = "Iason"
    menace_level: int = 2
    llm: LLMConfig = field(default_factory=LLMConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    turns_log: Path = Path("~/.ultron/turns.jsonl")


def platform_name() -> str:
    return {"darwin": "macos", "linux": "linux"}.get(sys.platform, sys.platform)


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _path(p: str | Path) -> Path:
    return Path(p).expanduser()


def load_config(root: Path | None = None) -> Config:
    root = root or Path.cwd()
    raw: dict = {}
    host = socket.gethostname().split(".")[0]
    for name in ("config.toml", f"config.{platform_name()}.toml", f"config.{host}.toml"):
        f = root / name
        if f.exists():
            raw = _merge(raw, tomllib.loads(f.read_text()))

    llm = LLMConfig(**raw.get("llm", {}))
    tools = ToolsConfig(**raw.get("tools", {}))
    tools = ToolsConfig(**{**tools.__dict__, "notes_file": _path(tools.notes_file), "audit_log": _path(tools.audit_log)})

    menace = int(raw.get("persona", {}).get("menace_level", 2))
    if not 0 <= menace <= 3:
        raise ValueError(f"persona.menace_level must be 0..3, got {menace}")
    return Config(
        user_name=raw.get("user", {}).get("name", "Iason"),
        menace_level=menace,
        llm=llm,
        tools=tools,
        turns_log=_path(raw.get("metrics", {}).get("turns_log", "~/.ultron/turns.jsonl")),
    )
