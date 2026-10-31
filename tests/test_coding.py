"""Claude Code jobs against a fake `claude` that records its arguments and prints a result."""

import asyncio
import json
import stat

import pytest

from ultron.tools import Registry
from ultron.tools.coding import CodeJobs, parse_result, register_coding, slug


def fake_claude(tmp_path, result="Built a to-do app in index.html. Open it in a browser.", code=0, denied=()):
    log = tmp_path / "calls.jsonl"
    out = {"type": "result", "subtype": "success", "is_error": code != 0, "result": result,
           "session_id": "sess-123", "total_cost_usd": 0.42, "num_turns": 7,
           "permission_denials": [{"tool_name": d} for d in denied]}
    exe = tmp_path / "claude"
    exe.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        f"open({str(log)!r}, 'a').write(json.dumps({{'argv': sys.argv[1:], 'cwd': os.getcwd()}}) + '\\n')\n"
        f"print({json.dumps(json.dumps(out))})\n"
        f"sys.exit({code})\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return str(exe), log


async def run_job(jobs, reg, **args):
    done = asyncio.Event()
    jobs.on_done = lambda job: _set(done)
    out = await reg.get("code_task").handler(**args)
    await asyncio.wait_for(done.wait(), 10)
    return out


async def _set(ev):
    ev.set()


async def test_job_runs_in_its_folder_and_reports(tmp_path):
    exe, log = fake_claude(tmp_path, denied=["Bash"])
    jobs, reg = CodeJobs(tmp_path / "projects", exe), Registry()
    register_coding(reg, jobs)
    out = await run_job(jobs, reg, task="Build a to-do app", project="To-Do App!")
    assert "started job 1" in out
    call = json.loads(log.read_text().splitlines()[0])
    assert call["cwd"].endswith("projects/to-do-app")
    assert call["argv"][:2] == ["-p", "Build a to-do app"]
    assert "acceptEdits" in call["argv"] and "--resume" not in call["argv"]
    job = jobs.jobs[0]
    assert job.status == "done" and job.session_id == "sess-123" and job.cost_usd == 0.42
    status = await reg.get("code_status").handler()
    assert "done" in status and "index.html" in status and "refused for safety: Bash" in status


async def test_follow_up_resumes_the_same_session(tmp_path):
    exe, log = fake_claude(tmp_path)
    jobs, reg = CodeJobs(tmp_path / "projects", exe), Registry()
    register_coding(reg, jobs)
    await run_job(jobs, reg, task="Build it", project="todo")
    await run_job(jobs, reg, task="Make the buttons blue", project="todo", continue_previous=True)
    argv = json.loads(log.read_text().splitlines()[1])["argv"]
    assert argv[argv.index("--resume") + 1] == "sess-123"


async def test_allowed_tools_are_passed_and_missing_cli_is_an_error(tmp_path):
    exe, log = fake_claude(tmp_path)
    jobs, reg = CodeJobs(tmp_path / "p", exe, ("Bash(npm install:*)",)), Registry()
    register_coding(reg, jobs)
    await run_job(jobs, reg, task="x", project="y")
    argv = json.loads(log.read_text())["argv"]
    assert argv[argv.index("--allowedTools") + 1] == "Bash(npm install:*)"
    with pytest.raises(RuntimeError, match="isn't installed"):
        await CodeJobs(tmp_path, "/nonexistent/claude").start("x", "y", False)


def test_parse_result_handles_failures_and_garbage():
    from ultron.tools.coding import CodeJob
    from pathlib import Path
    j = CodeJob(1, "p", Path("."), "t")
    parse_result(j, "not json", "boom", 1)
    assert j.status == "failed" and j.result == "boom"
    assert slug("  My Cool App  ") == "my-cool-app" and slug("!!!") == "project"


def test_confirmation_names_the_folder(tmp_path):
    jobs, reg = CodeJobs(tmp_path, "claude"), Registry()
    register_coding(reg, jobs)
    (tmp_path / "ultron").mkdir()
    tool = reg.get("code_task")
    assert "existing ultron folder" in tool.describe_call({"project": "Ultron", "task": "x"})
    assert "new folder called todo" in tool.describe_call({"project": "todo", "task": "x"})
