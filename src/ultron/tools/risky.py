"""Tier 2 (confirm) and tier 3 (passphrase, off by default) tools: shell, writing, moving, deleting.

The executor asks before any of these run; the checks here are a second, independent fence for
things that should never happen whatever the user says yes to.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
from pathlib import Path

from ..platform.base import run
from .practical import _is_secret
from .registry import Registry, Tier, Tool

# Refused outright, confirmed or not: privilege escalation, disk wiping, power, fork bombs,
# piping the internet into a shell.
DEFAULT_DENY = [
    r"\bsudo\b", r"\bsu\b", r"\bdoas\b", r"\bmkfs", r"\bdiskutil\s+(erase|zero|secure)", r"\bdd\b.*\bof=/dev/",
    r"\brm\s+(-\w*\s+)*(/|~|\$HOME)\s*$", r"\brm\s+-\w*r\w*f?\w*\s+(/|~|\$HOME)(\s|$)", r"\b(shutdown|reboot|halt)\b",
    r":\(\)\s*\{", r"(curl|wget)[^|]*\|\s*(ba|z)?sh\b", r"\bchmod\s+-R\s+777\s+/", r"\blaunchctl\s+(unload|bootout)\s+system",
    r"\bcsrutil\b", r"\bspctl\b.*--master-disable",
]


def _home_path(p: str) -> Path:
    path = Path(p).expanduser().resolve()
    if not path.is_relative_to(Path.home()):
        raise PermissionError("only paths inside your home folder")
    if _is_secret(path):
        raise PermissionError("that path holds credentials")
    return path


def register_risky(reg: Registry, timeout_s: float = 20.0, allow: list[str] | None = None,
                   deny: list[str] | None = None) -> None:
    deny_re = [re.compile(p) for p in (deny if deny is not None else DEFAULT_DENY)]
    allow = allow or []

    async def run_shell(command: str) -> str:
        if os.geteuid() == 0:
            return "refusing: running as root"
        for rx in deny_re:
            if rx.search(command):
                return f"refusing: that command matches a blocked pattern ({rx.pattern})"
        if allow and command.split()[0] not in allow:
            return f"refusing: {command.split()[0]} is not in the allowed commands {allow}"
        proc = await asyncio.create_subprocess_exec(
            "/bin/zsh", "-c", command, cwd=str(Path.home()),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout_s)
        except TimeoutError:
            proc.kill()
            return f"timed out after {timeout_s:g}s"
        text = out.decode(errors="replace").strip()
        return f"exit {proc.returncode}\n{text}" if text else f"exit {proc.returncode}, no output"

    async def write_file(path: str, text: str, mode: str = "overwrite") -> str:
        p = _home_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a" if mode == "append" else "w", encoding="utf-8") as f:
            f.write(text)
        return f"{'appended to' if mode == 'append' else 'wrote'} {p.name} ({len(text)} characters)"

    async def move_file(src: str, dst: str) -> str:
        s, d = _home_path(src), _home_path(dst)
        if not s.exists():
            return f"{src} doesn't exist"
        if d.exists() and not d.is_dir():
            return f"{dst} already exists; not overwriting"
        await asyncio.to_thread(shutil.move, str(s), str(d))
        return f"moved {s.name} to {d}"

    async def trash(path: str) -> str:
        p = _home_path(path)
        if p == Path.home():
            return "refusing to delete your home folder"
        if not p.exists():
            return f"{path} doesn't exist"
        await run("osascript", "-e", f'tell application "Finder" to delete (POSIX file "{p}" as alias)', timeout=15)
        return f"moved {p.name} to the Trash"

    for tool in [
        Tool("run_shell", "Run a shell command (zsh, in the home folder, as the user) and return its output. "
             "The user is asked to confirm before it runs.",
             {"command": {"type": "string"}}, ("command",), Tier.RISKY, run_shell,
             describe=lambda a: f"run the command {a.get('command', '')!r}"),
        Tool("write_file", "Create or change a text file in the home folder. Asks the user first.",
             {"path": {"type": "string"}, "text": {"type": "string"}, "mode": {"type": "string", "enum": ["overwrite", "append"]}},
             ("path", "text"), Tier.RISKY, write_file,
             describe=lambda a: f"{'append to' if a.get('mode') == 'append' else 'write'} the file {a.get('path', '')}"),
        Tool("move_file", "Move or rename a file or folder in the home folder. Asks the user first.",
             {"src": {"type": "string"}, "dst": {"type": "string"}}, ("src", "dst"), Tier.RISKY, move_file,
             describe=lambda a: f"move {a.get('src', '')} to {a.get('dst', '')}"),
        Tool("delete", "Move a file or folder in the home folder to the Trash. Needs the user's passphrase.",
             {"path": {"type": "string"}}, ("path",), Tier.DESTRUCTIVE, trash,
             describe=lambda a: f"move {a.get('path', '')} to the Trash"),
    ]:
        reg.add(tool)
