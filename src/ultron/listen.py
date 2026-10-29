"""Push-to-talk: mic -> streaming STT, with the release as a definitive end of turn.

The STT connection opens at the press, while the user is already speaking: its ~190 ms connect
time is hidden, and the mic frames captured meanwhile are queued and sent as soon as it's up.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from .audio.input import MicInput
from .backends import DeepgramFlux


@dataclass
class Heard:
    transcript: str
    t_release: float
    t_final: float
    connect_ms: float

    @property
    def final_ms(self) -> float:
        return (self.t_final - self.t_release) * 1000


class PushToTalk:
    def __init__(self, stt: DeepgramFlux, mic: MicInput) -> None:
        self.stt = stt
        self.mic = mic

    async def capture(self, wait_release: Callable[[], Awaitable[object]],
                      on_partial: Callable[[str], None] = lambda s: None, final_timeout: float = 3.0) -> Heard:
        t_press = time.perf_counter()
        self.mic.start()
        opening = asyncio.create_task(self.stt.open(ptt=True))
        early: list[str] = []
        final: list[str] = []
        done = asyncio.Event()
        t_final = 0.0

        async def pump():
            stream = await opening
            async for frame in self.mic.frames():
                await stream.send(frame)
            return stream

        async def collect():
            nonlocal t_final
            stream = await opening
            while True:
                ev = await stream.events.get()
                if ev.kind == "update":
                    on_partial(" ".join([*early, ev.transcript]).strip())
                elif ev.kind == "end" and ev.trigger != "manual":
                    early.append(ev.transcript)  # only possible if ptt_eot_threshold < 1
                elif ev.kind == "end" or ev.kind == "warning":
                    final.append(ev.transcript)
                    t_final = ev.t
                    done.set()
                    return
                elif ev.kind in ("error", "closed"):
                    t_final = ev.t
                    done.set()
                    raise RuntimeError(f"STT {ev.kind}: {ev.detail}")

        pumping = asyncio.create_task(pump())
        collecting = asyncio.create_task(collect())
        stream = None
        try:
            # Wait for the release, but fail at once if the mic pump or the STT connection breaks:
            # otherwise the user talks into a dead pipe until they press the key again.
            release = asyncio.ensure_future(wait_release())
            await asyncio.wait({release, pumping, collecting}, return_when=asyncio.FIRST_COMPLETED)
            for task in (pumping, collecting):
                if task.done() and not task.cancelled() and task.exception():
                    self.mic.stop()
                    raise task.exception()
            await release
            t_release = time.perf_counter()
            self.mic.stop()
            stream = await pumping  # the tail of the audio is sent before the end-of-turn signal
            await stream.end_turn()
            try:
                await asyncio.wait_for(done.wait(), final_timeout)
            except TimeoutError:
                t_final = time.perf_counter()
            if collecting.done() and collecting.exception():
                raise collecting.exception()
            transcript = " ".join(t for t in [*early, *final] if t).strip()
            return Heard(transcript, t_release, t_final, (stream.t_connected - t_press) * 1000)
        finally:
            collecting.cancel()
            pumping.cancel()
            if stream is None and opening.done() and not opening.exception():
                stream = opening.result()
            if stream is not None:
                await stream.close()
