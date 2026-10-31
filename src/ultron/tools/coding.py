"""Coding jobs for Claude Code (the `claude` CLI, on the user's own Claude login): Ultron writes the
brief, Claude Code does the work in a project folder in the background, and Ultron says when it's
done. Follow-up requests continue the same Claude Code session, so it remembers the project.

Claude Code runs headless (`claude -p`) with `acceptEdits`: it may create and edit files inside the
project folder. Shell commands are refused unless listed in tools.code_allowed_tools, and it never
touches anything outside the folder without asking (which, headless, means it doesn't).
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from .registry import Registry, Tier, Tool

OnDone = Callable[["CodeJob"], Awaitable[None]]


@dataclass
class CodeJob:
    n: int
    project: str
    folder: Path
    task: str
    started: float = field(default_factory=time.monotonic)
    status: str = "running"  # running / done / failed
    result: str = ""
    session_id: str = ""
    cost_usd: float = 0.0
    turns: int = 0
    denied: list[str] = field(default_factory=list)
    ended: float | None = None
    proc: asyncio.subprocess.Process | None = None

    def minutes(self) -> float:
        return ((self.ended or time.monotonic()) - self.started) / 60


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s[:40] or "project"


class CodeJobs:
    def __init__(self, root: Path, claude: str = "claude", allowed_tools: tuple[str, ...] = (),
                 on_done: OnDone | None = None, timeout_s: float = 3600) -> None:
        self.root = root.expanduser()
        self.claude = claude
        self.allowed_tools = allowed_tools
        self.on_done = on_done
        self.timeout_s = timeout_s
        self.jobs: list[CodeJob] = []
        self._tasks: set[asyncio.Task] = set()

    def find_claude(self) -> str | None:
        return shutil.which(self.claude) or next(
            (str(p) for p in (Path("~/.local/bin/claude").expanduser(), Path("~/.claude/local/claude").expanduser())
             if p.exists()), None)

    def last_session(self, folder: Path) -> str:
        return next((j.session_id for j in reversed(self.jobs) if j.folder == folder and j.session_id), "")

    def command(self, exe: str, job: CodeJob, resume: str) -> list[str]:
        cmd = [exe, "-p", job.task, "--output-format", "json", "--permission-mode", "acceptEdits"]
        if self.allowed_tools:
            cmd += ["--allowedTools", *self.allowed_tools]
        if resume:
            cmd += ["--resume", resume]
        return cmd

    async def start(self, task: str, project: str, continue_previous: bool) -> CodeJob:
        exe = self.find_claude()
        if exe is None:
            raise RuntimeError("Claude Code isn't installed: the `claude` command was not found")
        if any(j.status == "running" and j.folder == self.root / slug(project) for j in self.jobs):
            raise RuntimeError(f"a job is already running on {project}; wait for it or ask for its status")
        folder = self.root / slug(project)
        folder.mkdir(parents=True, exist_ok=True)
        job = CodeJob(len(self.jobs) + 1, project, folder, task)
        resume = self.last_session(folder) if continue_previous else ""
        job.proc = await asyncio.create_subprocess_exec(
            *self.command(exe, job, resume), cwd=folder,
            stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        self.jobs.append(job)
        t = asyncio.create_task(self._wait(job))
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)
        return job

    async def _wait(self, job: CodeJob) -> None:
        try:
            out, err = await asyncio.wait_for(job.proc.communicate(), self.timeout_s)
            parse_result(job, out.decode(errors="replace"), err.decode(errors="replace"), job.proc.returncode)
        except TimeoutError:
            job.proc.kill()
            job.status, job.result = "failed", f"stopped after {self.timeout_s / 60:.0f} minutes"
        except asyncio.CancelledError:
            if job.proc.returncode is None:
                job.proc.kill()
            raise
        job.ended = time.monotonic()
        if self.on_done is not None:
            try:
                await self.on_done(job)
            except Exception:
                pass

    async def close(self) -> None:
        for j in self.jobs:
            if j.proc is not None and j.proc.returncode is None:
                j.proc.kill()
        for t in list(self._tasks):
            t.cancel()


def parse_result(job: CodeJob, stdout: str, stderr: str, code: int | None) -> None:
    """`claude -p --output-format json` prints one JSON object with the final report."""
    try:
        d = json.loads(stdout.strip().splitlines()[-1]) if stdout.strip() else {}
    except (json.JSONDecodeError, IndexError):
        d = {}
    job.result = str(d.get("result") or stderr.strip() or stdout.strip() or f"exited with code {code}")
    job.session_id = d.get("session_id", "") or job.session_id
    job.cost_usd = float(d.get("total_cost_usd") or 0)
    job.turns = int(d.get("num_turns") or 0)
    job.denied = [str(p.get("tool_name", p)) for p in d.get("permission_denials") or []]
    job.status = "failed" if d.get("is_error") or code not in (0, None) or not d else "done"


def describe(job: CodeJob, full: bool = False) -> str:
    head = f"job {job.n} on {job.project} ({job.folder}): {job.status}, {job.minutes():.0f} min"
    if job.status == "running":
        return head
    parts = [head]
    if job.denied:
        parts.append(f"refused for safety: {', '.join(sorted(set(job.denied)))}")
    report = job.result if full else job.result[:600] + ("…" if len(job.result) > 600 else "")
    parts.append(f"Claude Code's report: {report}")
    return "\n".join(parts)


def _describe_call(jobs: CodeJobs, a: dict) -> str:
    folder = jobs.root / slug(a.get("project", ""))
    where = f"the existing {folder.name} folder" if folder.exists() else f"a new folder called {folder.name}"
    return f"have Claude Code work in {where} in your projects, where it can create and edit files"


def register_coding(reg: Registry, jobs: CodeJobs) -> None:
    async def code_task(task: str, project: str, continue_previous: bool = False) -> str:
        job = await jobs.start(task, project, continue_previous)
        how = "continuing its earlier session" if continue_previous and jobs.last_session(job.folder) else "new session"
        return (f"started job {job.n} in {job.folder} ({how}). Claude Code works in the background, usually "
                f"a few minutes; the user will be told when it finishes.")

    async def code_status(job: int = 0) -> str:
        if not jobs.jobs:
            return "no coding jobs yet"
        if job:
            match = [j for j in jobs.jobs if j.n == job]
            return describe(match[0], full=True) if match else f"no job {job}"
        return "\n\n".join(describe(j) for j in jobs.jobs[-5:])

    reg.add(Tool(
        "code_task",
        "Have Claude Code (a coding agent) build or change software: apps, scripts, websites, fixes. It "
        "works on its own in a project folder for minutes and reports back. Write `task` as a complete "
        "brief: what to build, every detail the user gave, the stack if known, and to finish with a short "
        "summary of what it made and how to run it. Ask the user first if the request is too vague to "
        "brief (what kind of app, for what). For a follow-up on the same project, set continue_previous.",
        {"task": {"type": "string", "description": "The full brief for Claude Code"},
         "project": {"type": "string", "description": "Short project name; it becomes the folder name"},
         "continue_previous": {"type": "boolean", "description": "Continue the last Claude Code session on this project"}},
        ("task", "project"), Tier.RISKY, code_task,
        describe=lambda a: _describe_call(jobs, a),
    ))
    reg.add(Tool(
        "code_status",
        "Progress and reports of Claude Code jobs. With `job`, that job's full report.",
        {"job": {"type": "integer", "description": "Job number; 0 or omitted = the recent ones"}},
        (), Tier.READ, code_status,
    ))
