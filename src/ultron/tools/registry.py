from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class Tier(IntEnum):
    READ = 0
    LOW_RISK = 1
    RISKY = 2
    DESTRUCTIVE = 3


Handler = Callable[..., Awaitable[str]]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    params: dict[str, dict[str, Any]]  # JSON-schema properties
    required: tuple[str, ...]
    tier: Tier
    handler: Handler

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.params,
                "required": list(self.required),
                "additionalProperties": False,
            },
        }


_JSON_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool}


def validate_args(tool: Tool, args: Any) -> str | None:
    """Cheap structural check of model-supplied args. Returns an error message or None."""
    if not isinstance(args, dict):
        return "arguments must be an object"
    for k in tool.required:
        if k not in args:
            return f"missing required argument '{k}'"
    for k, v in args.items():
        spec = tool.params.get(k)
        if spec is None:
            return f"unknown argument '{k}'"
        py = _JSON_TYPES.get(spec.get("type", ""))
        if py and (not isinstance(v, py) or (py is int and isinstance(v, bool))):
            return f"argument '{k}' must be {spec['type']}"
        if "enum" in spec and v not in spec["enum"]:
            return f"argument '{k}' must be one of {spec['enum']}"
        if "minimum" in spec and v < spec["minimum"] or "maximum" in spec and v > spec["maximum"]:
            return f"argument '{k}' out of range"
    return None


class Registry:
    """Tools in registration order. The order is part of the cached prompt prefix, so it's fixed."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def add(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def __iter__(self):
        return iter(self._tools.values())
