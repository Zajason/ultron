"""Speaker against fake TTS and output: what gets sent to TTS, fillers, interruption."""

import asyncio

import numpy as np

from ultron.audio.speaker import Speaker
from ultron.config import TTSConfig


class FakeUtt:
    def __init__(self, log):
        self.log = log
        self.q = asyncio.Queue()

    async def push(self, text):
        self.log.append(("push", text))
        await self.q.put(np.ones(10, np.float32))

    async def end(self):
        self.log.append(("end",))
        await self.q.put(None)

    async def cancel(self):
        self.log.append(("cancel",))

    async def audio(self):
        while (a := await self.q.get()) is not None:
            yield a


class FakeTTS:
    sample_rate = 24000

    def __init__(self):
        self.log = []

    async def utterance(self):
        return FakeUtt(self.log)


class FakeOut:
    sample_rate = 24000

    def __init__(self):
        self.written, self.cleared, self.busy = [], 0, False

    def arm(self):
        pass

    def write(self, a):
        self.written.append(a)

    def clear(self):
        self.cleared += 1


def make():
    tts, out = FakeTTS(), FakeOut()
    return Speaker(tts, out, None, TTSConfig()), tts, out


async def test_chunks_are_normalised_marked_and_streamed():
    sp, tts, out = make()
    sp.begin()
    sp.say("Ten minutes… done.")
    sp.say("About 3.5GB free!")
    sp.end()
    await sp.finished()
    assert tts.log == [
        ("push", 'Ten minutes<break time="400ms"/>done.'),
        ("push", "About 3.5 gigs free."),
        ("end",),
    ]
    assert len(out.written) == 2
    assert sp.spoken == ["Ten minutes… done.", "About 3.5GB free!"]
    assert sp.t_first_byte >= sp.t_first_push


async def test_interrupt_cancels_and_clears():
    sp, tts, out = make()
    sp.begin()
    sp.say("This will be cut off.")
    await asyncio.sleep(0.01)
    sp.interrupt()
    await asyncio.sleep(0.01)
    assert ("cancel",) in tts.log
    assert out.cleared >= 2  # once on begin, once on interrupt


async def test_fillers_dont_repeat_recently():
    sp, _, out = make()
    sp.fillers = {f"line {i}": np.zeros(5, np.float32) for i in range(8)}
    played = [sp.play_filler() for _ in range(6)]
    assert len(set(played)) == 6 and len(out.written) == 6


async def test_no_fillers_is_silent():
    sp, _, out = make()
    assert sp.play_filler() is None and out.written == []
