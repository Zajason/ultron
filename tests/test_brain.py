"""Brain tool loop and history handling, against a scripted fake of the Anthropic client."""

import copy
from types import SimpleNamespace as NS

from ultron.brain import Brain
from ultron.config import Config, LLMConfig, ToolsConfig
from ultron.persona import PLAIN_ON
from ultron.tools import AuditLog, Executor, Registry, Tier, Tool


def text(t):
    return NS(type="text", text=t)


def tool_use(id, name, input):
    return NS(type="tool_use", id=id, name=name, input=input)


def reply(*blocks, stop="end_turn"):
    return NS(content=list(blocks), stop_reason=stop,
              usage=NS(input_tokens=10, cache_read_input_tokens=900, cache_creation_input_tokens=0, output_tokens=5))


class FakeStream:
    def __init__(self, final):
        self.final = final

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def __aiter__(self):
        async def gen():
            for b in self.final.content:
                if b.type == "text":
                    for word in b.text.split(" "):
                        yield NS(type="text", text=word + " ")
        return gen()

    async def get_final_message(self):
        return self.final


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.messages = self

    def stream(self, **params):
        self.calls.append(copy.deepcopy(params))
        return FakeStream(self.replies.pop(0))


def make_brain(tmp_path, replies, **llm):
    reg = Registry()

    async def get_time():
        return "12:00"

    reg.add(Tool("get_time", "", {}, (), Tier.READ, get_time))
    cfg = Config(llm=LLMConfig(**llm), tools=ToolsConfig(audit_log=tmp_path / "a.jsonl"))

    async def confirm(*a):
        return True

    ex = Executor(reg, cfg.tools, AuditLog(tmp_path / "a.jsonl"), confirm)
    client = FakeClient(replies)
    return Brain(cfg, reg, ex, client=client), client


async def test_simple_turn_streams_and_records_metrics(tmp_path):
    brain, client = make_brain(tmp_path, [reply(text("Morning. Coffee first."))])
    out = []
    m = await brain.respond("Good morning", out.append)
    assert "".join(out).strip() == "Morning. Coffee first."
    assert {"first_token", "first_chunk", "done"} <= m.marks.keys()
    assert m.llm_calls[0]["cache_read"] == 900
    params = client.calls[0]
    assert params["system"][0]["cache_control"]["ttl"] == "1h"
    assert params["cache_control"] == {"type": "ephemeral"}
    assert params["thinking"] == {"type": "disabled"}
    assert "temperature" not in params
    assert params["messages"][0]["content"].startswith("[") and params["messages"][0]["content"].endswith("] Good morning")


async def test_tool_loop(tmp_path):
    brain, client = make_brain(tmp_path, [
        reply(text("One moment."), tool_use("t1", "get_time", {}), stop="tool_use"),
        reply(text("Noon.")),
    ])
    m = await brain.respond("time?")
    assert [t["name"] for t in m.tools] == ["get_time"]
    roles = [msg["role"] for msg in brain.messages]
    assert roles == ["user", "assistant", "user", "assistant"]
    result = brain.messages[2]["content"][0]
    assert result == {"type": "tool_result", "tool_use_id": "t1", "content": "12:00"}
    assert len(client.calls) == 2


async def test_last_round_disables_tools(tmp_path):
    loop = [reply(tool_use(f"t{i}", "get_time", {}), stop="tool_use") for i in range(2)]
    brain, client = make_brain(tmp_path, [*loop, reply(text("Enough."))], max_tool_rounds=2)
    await brain.respond("time?")
    assert "tool_choice" not in client.calls[0]
    assert client.calls[2]["tool_choice"] == {"type": "none"}


async def test_truncated_tool_call_not_run_but_history_valid(tmp_path):
    brain, _ = make_brain(tmp_path, [reply(tool_use("t1", "get_time", {}), stop="max_tokens")])
    m = await brain.respond("time?")
    assert m.tools == []
    last = brain.messages[-1]
    assert last["role"] == "user" and last["content"][0]["tool_use_id"] == "t1"
    assert last["content"][0]["is_error"] is True


async def test_plain_mode_is_a_system_message_not_a_prompt_change(tmp_path):
    brain, client = make_brain(tmp_path, [reply(text("Understood.")), reply(text("Fine."))])
    system_before = brain._system
    await brain.respond("Ultron, plain mode")
    assert brain.messages[1] == {"role": "system", "content": PLAIN_ON}
    assert brain._system == system_before
    await brain.respond("plain mode again")  # already on: no duplicate
    assert sum(m["role"] == "system" for m in brain.messages) == 1


async def test_trim_in_batches_starting_at_a_user_turn(tmp_path):
    brain, client = make_brain(tmp_path, [reply(text(f"r{i}")) for i in range(7)], max_turns=6)
    for i in range(6):
        await brain.respond(f"q{i}")
    assert len(brain.messages) == 12
    await brain.respond("q6")  # 7 turns > 6: drop to 6 - 6 // 3 = 4 kept
    users = [m["content"].split("] ", 1)[1] for m in brain.messages if m["role"] == "user"]
    assert users == ["q3", "q4", "q5", "q6"]
    assert brain.messages[0]["role"] == "user"


async def test_trim_keeps_plain_mode(tmp_path):
    brain, _ = make_brain(tmp_path, [reply(text("ok")) for _ in range(4)], max_turns=3)
    await brain.respond("plain mode")
    for i in range(3):
        await brain.respond(f"q{i}")
    assert brain.messages[1] == {"role": "system", "content": PLAIN_ON}


async def test_hybrid_routing_picks_model_per_turn(tmp_path):
    brain, client = make_brain(tmp_path, [reply(text("Done.")), reply(text("A long thought."))])
    await brain.respond("set a timer for five minutes")
    await brain.respond("Do you think art outlives the people who make it?")
    assert [c["model"] for c in client.calls] == ["claude-haiku-5-5", "claude-sonnet-5-5"]
    assert client.calls[1]["thinking"] == {"type": "between_tools"}


async def test_forced_tier(tmp_path):
    brain, client = make_brain(tmp_path, [reply(text("x"))], routing="fast")
    await brain.respond("What is the meaning of life?")
    assert client.calls[0]["model"] == "claude-haiku-5-5"
