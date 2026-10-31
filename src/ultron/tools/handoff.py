"""Hand a big request to the Claude or ChatGPT app for the user to send.

Ultron writes the prompt, puts it on the clipboard and brings the app to the front; the user pastes
and presses send. Nothing is automated inside the app (that would break both apps' terms and
fight the user for the screen): it's ordinary use of their subscription, with Ultron as the
prompt writer.
"""

from __future__ import annotations

import asyncio

from ..platform.base import run
from .registry import Registry, Tier, Tool

APPS = {"claude": ("Claude", "https://claude.ai/new"), "chatgpt": ("ChatGPT", "https://chatgpt.com/")}


async def _copy(text: str) -> None:
    proc = await asyncio.create_subprocess_exec("pbcopy", stdin=asyncio.subprocess.PIPE)
    await proc.communicate(text.encode())


def register_handoff(reg: Registry, default_app: str = "claude") -> None:
    async def hand_off(prompt: str, app: str = "") -> str:
        name, url = APPS[app or default_app]
        await _copy(prompt)
        try:
            await run("open", "-a", name)
            where = f"the {name} app"
        except Exception:
            await run("open", url)  # app not installed: the website, same clipboard
            where = url
        return (f"prompt ({len(prompt)} characters) is on the clipboard and {where} is in front; "
                f"the user pastes it into a new chat and sends it")

    reg.add(Tool(
        "hand_off",
        "For big jobs better done in a full chat app than by voice: long documents, research, plans, "
        "code, anything the user will want to read rather than hear. Write a complete, self-contained "
        "prompt for a top model (Claude Opus or GPT): the goal, every relevant detail the user gave, "
        "constraints, and the format wanted. Then tell the user briefly that it's ready to paste and send.",
        {"prompt": {"type": "string", "description": "The full prompt, written for the other model"},
         "app": {"type": "string", "enum": ["claude", "chatgpt"], "description": "Default: claude"}},
        ("prompt",), Tier.LOW_RISK, hand_off,
        describe=lambda a: f"open {APPS[a.get('app') or default_app][0]} with a prepared prompt",
    ))
