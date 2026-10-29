"""Runs tool calls with tier enforcement. Safety lives here, not in the prompt."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ..config import ToolsConfig
from .audit import AuditLog
from .registry import Registry, Tier, Tool, validate_args

# (tool, args, tier) -> approved?  Text mode asks on stdin; voice mode will ask out loud.
Confirm = Callable[[Tool, dict[str, Any], Tier], Awaitable[bool]]


@dataclass
class ToolResult:
    content: str
    is_error: bool = False


class Executor:
    def __init__(self, registry: Registry, cfg: ToolsConfig, audit: AuditLog, confirm: Confirm,
                 dry_run: bool = False) -> None:
        self.registry = registry
        self.cfg = cfg
        self.audit = audit
        self.confirm = confirm
        self.dry_run = dry_run  # benches: anything above tier 0 pretends to succeed without acting

    def tier_of(self, tool: Tool) -> Tier:
        return Tier(self.cfg.tier_overrides.get(tool.name, tool.tier))

    async def run(self, name: str, args: Any, transcript: str = "") -> ToolResult:
        t0 = time.monotonic()
        tool = self.registry.get(name)
        tier = self.tier_of(tool) if tool else None
        confirmed: bool | None = None

        if tool is None:
            res = ToolResult(f"unknown tool '{name}'", True)
        elif err := validate_args(tool, args):
            res = ToolResult(err, True)
        elif tier == Tier.DESTRUCTIVE and not self.cfg.tier3_enabled:
            res = ToolResult("this action is disabled in the config", True)
        else:
            if tier > self.cfg.max_auto_tier:
                confirmed = await self.confirm(tool, args, tier)
            if confirmed is False:
                res = ToolResult("the user declined; the action was not run", True)
            elif self.dry_run and tier > Tier.READ:
                res = ToolResult("done")
            else:
                res = await self._call(tool, args)

        self.audit.write(
            transcript=transcript,
            tool=name,
            args=args,
            tier=int(tier) if tier is not None else None,
            confirmed=confirmed,
            is_error=res.is_error,
            result=res.content[:500],
            duration_ms=round((time.monotonic() - t0) * 1000, 1),
        )
        return res

    async def _call(self, tool: Tool, args: dict[str, Any]) -> ToolResult:
        try:
            out = await asyncio.wait_for(tool.handler(**args), self.cfg.timeout_s)
        except TimeoutError:
            return ToolResult(f"timed out after {self.cfg.timeout_s:g}s", True)
        except Exception as e:  # tool failures go back to the model, not up the stack
            return ToolResult(f"{type(e).__name__}: {e}", True)
        return ToolResult(_truncate(out, self.cfg.max_output_chars))


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + f"… [truncated {len(s) - n} chars]"
