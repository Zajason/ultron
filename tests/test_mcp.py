import sys
from pathlib import Path

from ultron.tools import Registry, Tier
from ultron.tools.mcp_client import MCPHub

SERVER = str(Path(__file__).parent / "fixtures" / "mcp_test_server.py")


async def test_mcp_tools_register_with_tiers_and_run():
    reg = Registry()
    hub = MCPHub([{"name": "test", "command": sys.executable, "args": [SERVER], "tiers": {"shout": 1}}])
    try:
        status = await hub.connect(reg)
        assert status == ["mcp test: 3 tools"]
        tiers = {t.name: t.tier for t in reg}
        assert tiers == {"test__add": Tier.READ, "test__wipe": Tier.DESTRUCTIVE, "test__shout": Tier.LOW_RISK}
        assert await reg.get("test__add").handler(a=2, b=3) == "5"
        assert reg.get("test__add").schema()["input_schema"]["properties"]["a"]["type"] == "integer"
        try:
            await reg.get("test__shout").handler(text="")
        except RuntimeError as e:
            assert "shout" in str(e)  # MCP 2.x reports a generic error, not the exception text
        else:
            raise AssertionError("expected the server's error")
    finally:
        await hub.close()


async def test_broken_server_reports_instead_of_hanging():
    reg = Registry()
    hub = MCPHub([{"name": "nope", "command": "/nonexistent/binary"}])
    status = await hub.connect(reg, connect_timeout_s=5)
    await hub.close()
    assert status[0].startswith("mcp nope: failed") and list(reg) == []
