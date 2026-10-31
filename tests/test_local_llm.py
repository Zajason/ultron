"""Local backend: tool-call parsing, history conversion, and the brain's local turn path against a
scripted fake model (no MLX needed)."""

import asyncio

from test_brain import make_brain, reply, text

from ultron.backends.local_llm import held_back, parse_calls, to_chat
from ultron.brain import LOCAL

TOOLS = [{"type": "function", "function": {"name": "set_timer", "parameters": {
    "properties": {"seconds": {"type": "integer"}, "label": {"type": "string"}}}}}]

CALL = ("<tool_call>\n<function=set_timer>\n<parameter=seconds>\n600\n</parameter>\n"
        "<parameter=label>\ntea\n</parameter>\n</function>\n</tool_call>")


def test_parse_calls_coerces_schema_types():
    [c] = parse_calls("On it. " + CALL, TOOLS)
    assert c.name == "set_timer" and c.args == {"seconds": 600, "label": "tea"}
    assert parse_calls("just words", TOOLS) == []


def test_held_back_only_for_a_possible_tool_tag():
    assert held_back("Done <tool") == 5
    assert held_back("Done <") == 1
    assert held_back("Done.") == 0
    assert held_back("a < b") == 0


def test_to_chat_converts_tool_rounds_and_system_notes():
    msgs = [
        {"role": "user", "content": "[Sat 10:00] tea timer"},
        {"role": "assistant", "content": [text("On it."), {"type": "tool_use", "id": "t1", "name": "set_timer",
                                                            "input": {"seconds": 600}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "set"}]},
        {"role": "assistant", "content": [text("Ten minutes.")]},
        {"role": "system", "content": "Plain mode is on."},
    ]
    chat = to_chat(msgs)
    assert chat[1] == {"role": "assistant", "content": "On it.",
                       "tool_calls": [{"function": {"name": "set_timer", "arguments": {"seconds": 600}}}]}
    assert chat[2] == {"role": "tool", "content": "set"}
    assert chat[4]["role"] == "user" and "Plain mode" in chat[4]["content"]


class FakeLocal:
    def __init__(self, rounds):
        self.rounds = [list(r) for r in rounds]
        self.tools = TOOLS
        self.seen = []
        self.synced = 0

    async def stream(self, messages):
        self.seen.append(len(messages))
        for piece in self.rounds.pop(0):
            await asyncio.sleep(0)
            yield piece

    def sync(self, messages):
        self.synced += 1
        f = asyncio.get_running_loop().create_future()
        f.set_result(None)
        return f


def local_brain(tmp_path, rounds, cloud=()):
    brain, client = make_brain(tmp_path, list(cloud), routing="hybrid")
    brain.local, brain._local_ready = FakeLocal(rounds), True
    ran = []

    async def set_timer(seconds, label="timer"):
        ran.append((seconds, label))
        return "set"

    from ultron.tools import Tier, Tool
    brain.registry.add(Tool("set_timer", "", {"seconds": {"type": "integer"}, "label": {"type": "string"}},
                            ("seconds",), Tier.LOW_RISK, set_timer))
    return brain, client, ran


async def test_local_turn_hides_stamp_and_tool_xml_and_runs_the_tool(tmp_path):
    brain, client, ran = local_brain(tmp_path, [
        ["[Sat 10:", "00] On it.", " <tool", "_call>\n<function=set_timer>", CALL.split("set_timer>")[1], "ignored"],
        ["Ten minutes", ". Tea awaits."],
    ])
    out, fillers = [], []
    m = await brain.respond("set a timer for ten minutes", out.append, on_tool_start=lambda: fillers.append(1))
    said = "".join(out)
    assert m.model == LOCAL and not client.calls
    assert ran == [(600, "tea")]
    assert "[" not in said and "<" not in said and "ignored" not in said
    assert said == "On it. Ten minutes. Tea awaits."
    assert not fillers  # the answer came fast: no filler line in front of it
    assistant = brain.messages[1]["content"]
    assert assistant[0].text == "On it." and assistant[1].name == "set_timer"
    assert brain.messages[2]["content"][0]["tool_use_id"] == assistant[1].id
    assert brain.local.synced == 1 and m.llm_calls[0]["usd"] == 0.0


async def test_local_escalates_to_the_strong_model(tmp_path):
    esc = "<tool_call>\n<function=escalate>\n<parameter=reason>\ndepth\n</parameter>\n</function>\n</tool_call>"
    brain, client, _ = local_brain(tmp_path, [[esc]], cloud=[reply(text("Bach, then."))])
    out = []
    m = await brain.respond("set a timer for the bread", out.append)
    assert m.model == brain.llm.strong_model and "escalated" in m.marks
    assert client.calls[0]["model"] == brain.llm.strong_model
    assert "Bach, then." in "".join(out)


async def test_conversation_turns_skip_the_local_model(tmp_path):
    brain, client, _ = local_brain(tmp_path, [], cloud=[reply(text("A long answer."))])
    m = await brain.respond("what do you think of humans")
    assert m.model == brain.llm.strong_model and brain.local.seen == []
