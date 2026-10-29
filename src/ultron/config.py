"""Config loading: config.toml, then per-platform and per-host overrides."""

from __future__ import annotations

import socket
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class LLMConfig:
    routing: str = "hybrid"
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
class TTSConfig:
    enabled: bool = True
    provider: str = "cartesia"
    model: str = "sonic-3.6-2026-08-27"
    voice_id: str = "d2c66146-c1c8-4c3a-9870-38e5a6b72442"
    sample_rate: int = 24000
    speed: float = 0.9
    emotion: str = ""
    ellipsis_ms: int = 400
    pitch_semitones: float = 0.0


@dataclass(frozen=True)
class STTConfig:
    provider: str = "deepgram"
    model: str = "flux-general-en"
    region: str = "eu"
    sample_rate: int = 16000
    frame_ms: int = 80
    eot_threshold: float = 0.7
    ptt_eot_threshold: float = 1.0
    eager_eot_threshold: float = 0.0
    eot_timeout_ms: int = 5000
    keyterms: tuple[str, ...] = ("Ultron", "Iason")


@dataclass(frozen=True)
class AudioConfig:
    input_device: str = ""
    output_device: str = ""
    output_rate: int = 48000
    latency_ms: float = 10


@dataclass(frozen=True)
class FXConfig:
    enabled: bool = True
    compress: bool = False
    warmth_db: float = 3.0
    warmth_hz: float = 220.0
    rasp: float = 0.0
    reverb_wet: float = 0.08
    metal_mix: float = 0.12
    metal_hz: float = 60.0
    sub_mix: float = 0.0
    gain: float = 1.0


@dataclass(frozen=True)
class Config:
    user_name: str = "Iason"
    menace_level: int = 2
    llm: LLMConfig = field(default_factory=LLMConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    fx: FXConfig = field(default_factory=FXConfig)
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
    if llm.routing not in ("hybrid", "fast", "strong"):
        raise ValueError(f"llm.routing must be hybrid, fast or strong, got {llm.routing!r}")
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
        tts=TTSConfig(**raw.get("tts", {})),
        stt=STTConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.get("stt", {}).items()}),
        audio=AudioConfig(**raw.get("audio", {})),
        fx=FXConfig(**raw.get("fx", {})),
        turns_log=_path(raw.get("metrics", {}).get("turns_log", "~/.ultron/turns.jsonl")),
    )
