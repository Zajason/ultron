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


async def _confirm_with(convo, answer, tier=2):
    from ultron.tools import Tier, Tool

    async def noop():
        return ""

    tool = Tool("run_shell", "", {}, (), Tier(tier), noop, describe=lambda a: "run the command 'ls'")
    task = asyncio.create_task(convo.confirm(tool, {}, Tier(tier)))
    await settle()
    if answer is not None:
        convo._on_turn(TurnEvent("end", answer, 0.0, "model"))
    return await task


async def test_spoken_confirmation_yes_and_no(tmp_path):
    convo, *_ = make(tmp_path, [])
    assert await _confirm_with(convo, "Yes, go ahead.") is True
    assert await _confirm_with(convo, "No, don't.") is False
    assert convo.reply is None  # the answers did not start replies


async def test_confirmation_timeout_declines(tmp_path):
    convo, *_ = make(tmp_path, [])
    convo.confirm_timeout_s = 0.05
    assert await _confirm_with(convo, None) is False


async def test_destructive_needs_the_passphrase(tmp_path):
    convo, *_ = make(tmp_path, [])
    assert await _confirm_with(convo, "yes", tier=3) is False  # no passphrase configured
    convo.passphrase = "Phase Two"
    assert await _confirm_with(convo, "yes", tier=3) is False
    assert await _confirm_with(convo, "Phase two.", tier=3) is True


async def test_answering_does_not_barge_in(tmp_path):
    convo, brain, client, speaker, out, done = make(tmp_path, [])
    convo._confirm = asyncio.get_running_loop().create_future()
    convo.reply = object()  # a reply is in flight, waiting for the answer
    convo._on_vad(VADEvent("start", 0.0, 0.9))
    assert convo.reply is not None and out.cleared >= 1
    convo._confirm = None
    convo.reply = None


def test_is_summons_only_for_a_bare_name():
    from ultron.conversation import is_summons
    for t in ["Ultron.", "Hey, Ultron?", "Okay Ultron", "ultron ultron"]:
        assert is_summons(t), t
    for t in ["Ultron, what time is it?", "What time is it?", "Ultron, hi there friend"]:
        assert not is_summons(t), t


async def test_bare_name_gets_no_reply_and_the_request_after_it_does(tmp_path):
    convo, brain, client, speaker, out, done = make(tmp_path, [reply(text("Twenty past four."))])
    convo._verify = True  # just woken
    convo._on_turn(TurnEvent("eager", "Ultron.", 0.0, ""))
    convo._on_turn(TurnEvent("end", "Ultron.", 0.0, "model"))
    await settle()
    assert convo.reply is None and not client.calls and not convo._verify
    convo._on_turn(TurnEvent("end", "What time is it?", 0.0, "model"))  # no name needed now
    await wait_turn(convo)
    assert len(client.calls) == 1 and brain.messages[0]["content"].endswith("What time is it?")


async def test_dismissal_sends_him_back_to_sleep_after_answering(tmp_path):
    convo, brain, client, speaker, out, done = make(tmp_path, [reply(text("Until next time."))])
    slept = []
    convo.wake, convo.awake = object(), True

    async def sleep(false_wake=False):
        slept.append(false_wake)
    convo._go_to_sleep = sleep
    convo._on_turn(TurnEvent("end", "Thanks, that's all.", 0.0, "model"))
    await wait_turn(convo)
    await settle()
    assert slept == [False] and convo._talked


async def test_follow_up_stays_awake_without_dismissal(tmp_path):
    convo, brain, client, speaker, out, done = make(tmp_path, [reply(text("What kind of app?"))])
    convo.wake, convo.awake = object(), True
    convo._go_to_sleep = None  # must not be called
    convo._on_turn(TurnEvent("end", "I want you to create an app.", 0.0, "model"))
    await wait_turn(convo)
    await settle()
    assert convo._talked and convo.awake
