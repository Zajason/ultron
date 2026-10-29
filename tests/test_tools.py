import asyncio
import json

import pytest

from ultron.config import ToolsConfig
from ultron.tools import AuditLog, Executor, Registry, Tier, Tool
from ultron.tools.registry import validate_args


async def echo(**kw) -> str:
    return json.dumps(kw)


async def slow() -> str:
    await asyncio.sleep(5)
    return "late"


async def boom() -> str:
    raise RuntimeError("hardware")


def make(tmp_path, approve=True, **cfg):
    reg = Registry()
    reg.add(Tool("read", "", {"n": {"type": "integer", "minimum": 0, "maximum": 10}}, ("n",), Tier.READ, echo))
    reg.add(Tool("write", "", {}, (), Tier.LOW_RISK, echo))
    reg.add(Tool("risky", "", {}, (), Tier.RISKY, echo))
    reg.add(Tool("nuke", "", {}, (), Tier.DESTRUCTIVE, echo))
    reg.add(Tool("slow", "", {}, (), Tier.READ, slow))
    reg.add(Tool("boom", "", {}, (), Tier.READ, boom))
    asked = []

    async def confirm(tool, args, tier):
        asked.append(tool.name)
        return approve

    log = tmp_path / "audit.jsonl"
    ex = Executor(reg, ToolsConfig(audit_log=log, **cfg), AuditLog(log), confirm)
    return ex, asked, log


def audit_lines(log):
    return [json.loads(line) for line in log.read_text().splitlines()]


async def test_tier_0_and_1_auto_run(tmp_path):
    ex, asked, _ = make(tmp_path)
    assert not (await ex.run("read", {"n": 3})).is_error
    assert not (await ex.run("write", {})).is_error
    assert asked == []


async def test_tier_2_needs_confirmation(tmp_path):
    ex, asked, log = make(tmp_path, approve=False)
    res = await ex.run("risky", {}, transcript="do the risky thing")
    assert res.is_error and "declined" in res.content
    assert asked == ["risky"]
    rec = audit_lines(log)[-1]
    assert rec["tool"] == "risky" and rec["tier"] == 2 and rec["confirmed"] is False
    assert rec["transcript"] == "do the risky thing"


async def test_tier_2_runs_when_confirmed(tmp_path):
    ex, asked, _ = make(tmp_path, approve=True)
    assert not (await ex.run("risky", {})).is_error


async def test_tier_3_disabled_by_default_and_never_asks(tmp_path):
    ex, asked, _ = make(tmp_path)
    res = await ex.run("nuke", {})
    assert res.is_error and "disabled" in res.content
    assert asked == []


async def test_tier_3_enabled_still_confirms(tmp_path):
    ex, asked, _ = make(tmp_path, tier3_enabled=True)
    await ex.run("nuke", {})
    assert asked == ["nuke"]


async def test_tier_override_from_config(tmp_path):
    ex, asked, _ = make(tmp_path, tier_overrides={"write": 2})
    await ex.run("write", {})
    assert asked == ["write"]


async def test_lowering_max_auto_tier_confirms_writes(tmp_path):
    ex, asked, _ = make(tmp_path, max_auto_tier=0)
    await ex.run("write", {})
    assert asked == ["write"]


@pytest.mark.parametrize(
    "args, err",
    [({}, "missing"), ({"n": "3"}, "must be integer"), ({"n": True}, "must be integer"), ({"n": 11}, "range"), ({"n": 1, "x": 2}, "unknown")],
)
async def test_bad_args_rejected_and_logged(tmp_path, args, err):
    ex, _, log = make(tmp_path)
    res = await ex.run("read", args)
    assert res.is_error and err in res.content
    assert audit_lines(log)[-1]["is_error"] is True


async def test_unknown_tool(tmp_path):
    ex, _, _ = make(tmp_path)
    assert (await ex.run("nope", {})).is_error


async def test_timeout_and_exception_become_errors(tmp_path):
    ex, _, _ = make(tmp_path, timeout_s=0.05)
    assert "timed out" in (await ex.run("slow", {})).content
    res = await ex.run("boom", {})
    assert res.is_error and "hardware" in res.content


async def test_output_truncated(tmp_path):
    ex, _, _ = make(tmp_path, max_output_chars=4)
    assert "truncated" in (await ex.run("read", {"n": 1})).content


def test_validate_args_non_dict():
    t = Tool("t", "", {}, (), Tier.READ, echo)
    assert validate_args(t, "x")
