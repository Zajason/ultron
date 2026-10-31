"""A tiny MCP server for tests: one read-only tool, one destructive, one plain."""

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

server = MCPServer("test")


@server.tool(annotations=ToolAnnotations(read_only_hint=True))
def add(a: int, b: int) -> str:
    """Add two numbers."""
    return str(a + b)


@server.tool(annotations=ToolAnnotations(destructive_hint=True))
def wipe(target: str) -> str:
    """Pretend to destroy something."""
    return f"wiped {target}"


@server.tool()
def shout(text: str) -> str:
    """Upper-case some text."""
    if not text:
        raise ValueError("nothing to shout")
    return text.upper()


if __name__ == "__main__":
    server.run("stdio")
