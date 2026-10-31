"""MCP client: tools from configured MCP servers (Notion, GitHub, filesystem, calendar...) appear as
Ultron's own tools, behind the same tiers, confirmations and audit log.

    [[mcp.servers]]
    name = "files"
    command = "npx"
    args = ["-y", "@modelcontextprotocol/server-filesystem", "~/Documents"]
    tier = 2                      # default for its tools (2 = ask first)
    tiers = { read_file = 0 }     # per tool

    [[mcp.servers]]
    name = "remote"
    url = "https://example.com/mcp"

Tier per tool: `tiers` entry, else the server's own hints (read-only -> 0, destructive -> 3), else
`tier` (default 2). Each server lives in its own task: MCP sessions must be entered and left in
the same task.
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass, field
from typing import Any

from .registry import Registry, Tier, Tool


@dataclass
class ServerHandle:
    name: str
    task: asyncio.Task
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    stop: asyncio.Event = field(default_factory=asyncio.Event)
    tools: list[str] = field(default_factory=list)
    error: str = ""


def _tool_name(server: str, tool: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", f"{server}__{tool}")[:64]


def _tier_for(t: Any, spec: dict) -> Tier:
    if t.name in spec.get("tiers", {}):
        return Tier(int(spec["tiers"][t.name]))
    hints = t.annotations
    if hints is not None and hints.destructive_hint:
        return Tier.DESTRUCTIVE
    if hints is not None and hints.read_only_hint:
        return Tier.READ
    return Tier(int(spec.get("tier", 2)))


def _text(result: Any) -> str:
    parts = []
    for c in getattr(result, "content", None) or []:
        if getattr(c, "type", "") == "text":
            parts.append(c.text)
        else:
            parts.append(f"[{getattr(c, 'type', 'content')}]")
    if not parts and getattr(result, "structured_content", None):
        parts.append(str(result.structured_content))
    return "\n".join(parts) or "(no output)"


class MCPHub:
    def __init__(self, servers: list[dict], timeout_s: float = 30.0) -> None:
        self.specs = [s for s in servers if s.get("enabled", True)]
        self.timeout_s = timeout_s
        self.handles: list[ServerHandle] = []

    async def connect(self, reg: Registry, connect_timeout_s: float = 20.0) -> list[str]:
        """Start every server, register its tools. Returns one status line per server."""
        for spec in self.specs:
            h = ServerHandle(spec["name"], None)  # type: ignore[arg-type]
            h.task = asyncio.create_task(self._serve(spec, h, reg))
            self.handles.append(h)
        lines = []
        for h in self.handles:
            try:
                await asyncio.wait_for(h.ready.wait(), connect_timeout_s)
            except TimeoutError:
                h.error = h.error or f"no answer within {connect_timeout_s:g}s"
            lines.append(f"mcp {h.name}: {len(h.tools)} tools" if not h.error else f"mcp {h.name}: failed ({h.error})")
        return lines

    async def _serve(self, spec: dict, h: ServerHandle, reg: Registry) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from mcp.client.streamable_http import streamable_http_client

        try:
            if "url" in spec:
                transport = streamable_http_client(spec["url"])
            else:
                args = [os.path.expanduser(a) for a in spec.get("args", [])]
                env = {**os.environ, **spec.get("env", {})}
                transport = stdio_client(StdioServerParameters(command=spec["command"], args=args, env=env))
            async with transport as streams:
                read, write = streams[0], streams[1]
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    listed = await session.list_tools()
                    for t in listed.tools:
                        self._register(reg, spec, h, session, t)
                    h.ready.set()
                    await h.stop.wait()
        except Exception as e:
            h.error = f"{type(e).__name__}: {e}"
            h.ready.set()

    def _register(self, reg: Registry, spec: dict, h: ServerHandle, session: Any, t: Any) -> None:
        name = _tool_name(spec["name"], t.name)

        async def call(**args: Any) -> str:
            result = await session.call_tool(t.name, args, read_timeout_seconds=self.timeout_s)
            text = _text(result)
            if getattr(result, "is_error", False):
                raise RuntimeError(text)
            return text

        schema = dict(t.input_schema or {"type": "object", "properties": {}})
        reg.add(Tool(
            name, (t.description or t.title or t.name)[:1000], schema.get("properties", {}),
            tuple(schema.get("required", [])), _tier_for(t, spec), call,
            describe=lambda a, s=spec["name"], n=t.name: f"use {n} on {s} with {a}",
            input_schema=schema,
        ))
        h.tools.append(name)

    async def close(self) -> None:
        for h in self.handles:
            h.stop.set()
        for h in self.handles:
            if h.task is not None:
                try:
                    await asyncio.wait_for(h.task, 5)
                except (TimeoutError, Exception):
                    h.task.cancel()
