from __future__ import annotations

from .base import run


def _as_str(s: str) -> str:
    """Quote a Python string as an AppleScript string literal."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


class MacOSActions:
    async def open_app(self, name: str) -> str:
        await run("open", "-a", name)
        return f"opened {name}"

    async def set_volume(self, level: int) -> str:
        level = max(0, min(100, int(level)))
        await run("osascript", "-e", f"set volume output volume {level}")
        return f"volume set to {level}"

    async def get_volume(self) -> int:
        return int(await run("osascript", "-e", "output volume of (get volume settings)"))

    async def notify(self, title: str, message: str) -> None:
        await run("osascript", "-e", f"display notification {_as_str(message)} with title {_as_str(title)}")
