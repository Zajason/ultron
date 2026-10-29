from __future__ import annotations

import asyncio
from typing import Protocol


class PlatformActions(Protocol):
    async def open_app(self, name: str) -> str: ...
    async def set_volume(self, level: int) -> str: ...
    async def get_volume(self) -> int: ...
    async def notify(self, title: str, message: str) -> None: ...


class CommandError(RuntimeError):
    pass


async def run(*argv: str, timeout: float = 5.0) -> str:
    """Run a command without a shell; return stdout or raise CommandError with stderr."""
    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
    except TimeoutError:
        proc.kill()
        raise CommandError(f"{argv[0]} timed out after {timeout:g}s")
    if proc.returncode != 0:
        raise CommandError(err.decode(errors="replace").strip() or f"{argv[0]} exited {proc.returncode}")
    return out.decode(errors="replace").strip()
