"""Push-to-talk against a fake mic and a fake STT stream, plus the decimator and WER helper."""

import asyncio

import numpy as np
import pytest

from ultron.audio.input import Downsampler
from ultron.backends.stt import TurnEvent
from ultron.listen import PushToTalk
from ultron.replay import wer


class FakeMic:
    level = 0.0
    peak = 0.0

    def __init__(self, frames):
        self._frames = frames
        self.stopped = asyncio.Event()

    def start(self):
        pass

    def stop(self):
        self.stopped.set()

    async def frames(self):
        for f in self._frames:
            yield f
        await self.stopped.wait()
        yield b"tail"


class FakeStream:
    def __init__(self, script):
        self.events = asyncio.Queue()
        self.t_connected = 0.0
        self.sent, self.ended, self.closed = [], False, False
        self.script = script  # events to emit when ForceEndTurn arrives

    async def send(self, audio):
        self.sent.append(audio)
        if len(self.sent) == 1:
            self.events.put_nowait(TurnEvent("update", "set a", 0.0))

    async def end_turn(self):
        self.ended = True
        for ev in self.script:
            self.events.put_nowait(ev)

    async def close(self):
        self.closed = True


class FakeSTT:
    def __init__(self, stream):
        self.stream = stream
        self.ptt = None

    async def open(self, ptt=False):
        self.ptt = ptt
        return self.stream


async def run(script, frames=(b"a", b"b")):
    stream = FakeStream(script)
    stt = FakeSTT(stream)
    partials = []
    release = asyncio.Event()
    asyncio.get_running_loop().call_later(0.02, release.set)
    heard = await PushToTalk(stt, FakeMic(list(frames))).capture(release.wait, partials.append, final_timeout=0.5)
    return heard, stream, stt, partials


async def test_release_sends_tail_then_force_end_turn():
    heard, stream, stt, partials = await run([TurnEvent("end", "Set a timer.", 1.0, "manual")])
    assert stt.ptt is True
    assert stream.sent == [b"a", b"b", b"tail"] and stream.ended and stream.closed
    assert heard.transcript == "Set a timer."
    assert partials == ["set a"]


async def test_early_model_end_is_joined():
    stream_script = [TurnEvent("end", "for ten minutes.", 1.0, "manual")]
    stream = FakeStream(stream_script)
    stream.events.put_nowait(TurnEvent("end", "Set a timer", 0.5, "model"))
    release = asyncio.Event()
    asyncio.get_running_loop().call_later(0.02, release.set)
    heard = await PushToTalk(FakeSTT(stream), FakeMic([b"a"])).capture(release.wait, final_timeout=0.5)
    assert heard.transcript == "Set a timer for ten minutes."


async def test_nothing_said_returns_empty():
    heard, stream, *_ = await run([TurnEvent("warning", t=1.0, detail="FORCE_END_TURN_NO_ACTIVE_TURN")], frames=())
    assert heard.transcript == "" and stream.closed


@pytest.mark.parametrize("sr_in", [48000, 44100, 24000])
def test_downsampler_keeps_speech_band_and_rejects_aliases(sr_in):
    t = np.arange(sr_in) / sr_in
    def out(hz):
        return Downsampler(sr_in, 16000).process(np.sin(2 * np.pi * hz * t).astype(np.float32))
    y = out(1000)
    assert abs(y.size - 16000) <= 2
    assert np.sqrt(np.mean(y[2000:] ** 2)) > 0.6     # passband
    if sr_in > 20000:
        assert np.sqrt(np.mean(out(11000)[2000:] ** 2)) < 0.05  # would alias into the speech band


def test_downsampler_passthrough_at_16k():
    x = np.random.default_rng(2).standard_normal(1600).astype(np.float32)
    np.testing.assert_array_equal(Downsampler(16000, 16000).process(x), x)


@pytest.mark.parametrize("sr_in", [48000, 44100])
def test_downsampler_chunked_equals_whole(sr_in):
    x = np.random.default_rng(1).standard_normal(sr_in).astype(np.float32)
    whole = Downsampler(sr_in, 16000).process(x)
    d = Downsampler(sr_in, 16000)
    chunked = np.concatenate([d.process(x[i : i + 517]) for i in range(0, x.size, 517)])
    n = min(whole.size, chunked.size)
    assert abs(whole.size - chunked.size) <= 1
    np.testing.assert_allclose(chunked[:n], whole[:n], atol=1e-5)


def test_wer():
    assert wer("What do you think of humans?", "what do you think of humans") == 0
    assert abs(wer("What do you think of humans?", "What do you think of human") - 1 / 6) < 1e-9


async def test_pump_failure_surfaces_before_release():
    class BrokenMic(FakeMic):
        async def frames(self):
            raise ValueError("bad mic rate")
            yield b""

    release = asyncio.Event()  # never set: the error must not wait for the release
    try:
        await asyncio.wait_for(
            PushToTalk(FakeSTT(FakeStream([])), BrokenMic([])).capture(release.wait), timeout=1.0
        )
    except ValueError as e:
        assert "bad mic rate" in str(e)
    else:
        raise AssertionError("expected the mic error")
