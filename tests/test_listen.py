"""Push-to-talk against a fake mic and a fake STT stream, plus the decimator and WER helper."""

import asyncio

import numpy as np

from ultron.audio.input import Decimator
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


def test_decimator_keeps_speech_band_and_rejects_aliases():
    sr_in, sr_out = 48000, 16000
    t = np.arange(sr_in) / sr_in
    def level(hz):
        y = Decimator(sr_in, sr_out).process(np.sin(2 * np.pi * hz * t).astype(np.float32))
        return np.sqrt(np.mean(y[2000:] ** 2))
    assert level(1000) > 0.6    # passband
    assert level(12000) < 0.05  # would alias to 4 kHz without the filter


def test_decimator_chunked_equals_whole():
    x = np.random.default_rng(1).standard_normal(48000).astype(np.float32)
    whole = Decimator(48000, 16000).process(x)
    d = Decimator(48000, 16000)
    chunked = np.concatenate([d.process(x[i : i + 517]) for i in range(0, x.size, 517)])
    np.testing.assert_allclose(chunked, whole, atol=1e-6)


def test_wer():
    assert wer("What do you think of humans?", "what do you think of humans") == 0
    assert abs(wer("What do you think of humans?", "What do you think of human") - 1 / 6) < 1e-9
