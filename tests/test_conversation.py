"""Turn-taking logic: replies, early start (commit / rollback), barge-in. Fakes for STT, TTS, output."""

import asyncio

import numpy as np

from test_brain import FakeClient, make_brain, reply, text, tool_use
from test_speaker import FakeOut, FakeTTS

from ultron.audio.speaker import Speaker
from ultron.audio.vad import VADEvent
from ultron.backends.stt import TurnEvent
from ultron.config import TTSConfig
from ultron.conversation import UI, Conversation


class StubVAD:
    last_voiced = None


class FakeMic:
    out_rate = 16000


class SlowClient(FakeClient):
    """Holds each stream open until released, so a reply can be caught mid-flight."""

    def __init__(self, replies):
        super().__init__(replies)
        self.gate = asyncio.Event()

    def stream(self, **params):
        inner = super().stream(**params)
        gate = self.gate

        class Slow:
            async def __aenter__(self):
                await gate.wait()
                return await inner.__aenter__()

            async def __aexit__(self, *a):
                return await inner.__aexit__(*a)

        return Slow()


def make(tmp_path, replies, slow=False):
    brain, client = make_brain(tmp_path, replies)
    if slow:
        brain.client = client = SlowClient(replies)
    out = FakeOut()
    speaker = Speaker(FakeTTS(), out, None, TTSConfig())
    done = []
    convo = Conversation(brain, speaker, mic=FakeMic(), stt=None, vad=StubVAD(), aec=None, eager=True,
                         ui=UI(reply_done=lambda m, i: done.append(m)))
    return convo, brain, client, speaker, out, done


async def settle():
    for _ in range(20):
        await asyncio.sleep(0)


async def wait_turn(convo, n=1):
    for _ in range(200):
        if convo.turns_done >= n:
            return
        await asyncio.sleep(0.005)
    raise AssertionError("turn did not finish")


async def test_end_of_turn_starts_reply(tmp_path):
    convo, brain, client, speaker, out, done = make(tmp_path, [reply(text("Twenty past four."))])
    convo._on_turn(TurnEvent("end", "What time is it?", 0.0, "model"))
    await wait_turn(convo)
    assert len(done) == 1 and len(client.calls) == 1
    assert out.written  # audible
    assert [m["role"] for m in brain.messages] == ["user", "assistant"]


async def test_eager_then_confirmed_plays_the_held_reply(tmp_path):
    convo, brain, client, speaker, out, done = make(tmp_path, [reply(text("Twenty past four."))])
    convo._on_turn(TurnEvent("eager", "What time is it?", 0.0))
    await settle()
    await asyncio.sleep(0.02)
    assert out.written == [] and speaker.holding  # drafted, not heard
    convo._on_turn(TurnEvent("end", "what time is it", 0.1, "model"))  # same words, different case
    await wait_turn(convo)
    assert len(client.calls) == 1  # no second LLM call
    assert out.written and len(done) == 1


async def test_eager_then_resumed_rolls_back(tmp_path):
    convo, brain, client, speaker, out, done = make(tmp_path, [reply(text("Twenty past four."))], slow=True)
    convo._on_turn(TurnEvent("eager", "What time", 0.0))
    await settle()
    convo._on_turn(TurnEvent("resumed", "", 0.1))
    client.gate.set()
    await settle()
    await convo._cleanup
    assert brain.messages == [] and out.written == [] and done == []


async def test_speculative_tool_waits_for_commit(tmp_path):
    convo, brain, client, speaker, out, done = make(tmp_path, [
        reply(tool_use("t1", "get_time", {}), stop="tool_use"), reply(text("Noon.")),
    ])
    ran = []
    original = brain.executor.run

    async def spy(*a, **k):
        ran.append(a[0])
        return await original(*a, **k)

    brain.executor.run = spy
    convo._on_turn(TurnEvent("eager", "what time is it", 0.0))
    await asyncio.sleep(0.05)
    assert ran == []  # held at the gate
    convo._on_turn(TurnEvent("end", "what time is it", 0.1, "model"))
    await wait_turn(convo)
    assert ran == ["get_time"]


async def test_different_final_transcript_replaces_the_draft(tmp_path):
    convo, brain, client, speaker, out, done = make(tmp_path, [reply(text("Draft.")), reply(text("Real."))])
    convo._on_turn(TurnEvent("eager", "open safari", 0.0))
    await asyncio.sleep(0.02)
    convo._on_turn(TurnEvent("end", "open safari and spotify", 0.1, "model"))
    await wait_turn(convo)
    users = [m["content"].split("] ", 1)[1] for m in brain.messages if m["role"] == "user"]
    assert users == ["open safari and spotify"]


async def test_barge_in_keeps_only_what_was_heard(tmp_path):
    convo, brain, client, speaker, out, done = make(tmp_path, [reply(text("Ten minutes, starting now."))], slow=True)
    convo._on_turn(TurnEvent("end", "set a timer", 0.0, "model"))
    await settle()
    out.busy = True
    speaker.spoken_text = lambda: "Ten minutes,"
    convo._on_vad(VADEvent("start", 0.0, 0.9))
    client.gate.set()
    await convo._cleanup
    assert convo.reply is None and done == []
    assert brain.messages[-1]["role"] == "assistant"
    assert brain.messages[-1]["content"][0]["text"] == "Ten minutes, —"
    assert out.cleared >= 1
