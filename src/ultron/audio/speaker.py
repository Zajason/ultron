"""Speaker: LLM text chunks in, audio out.

Per reply: chunks are normalised, pause-marked and pushed into one TTS utterance (one context keeps
prosody continuous across chunks); audio streams back through the FX chain into the output queue.
say() is synchronous so the LLM stream loop never waits on TTS.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections import deque

import numpy as np

from ..backends import TTSBackend
from ..config import TTSConfig
from ..text.normalize import normalize, pause_markup
from .fx import VoiceFX
from .output import AudioOutput

_END = object()


class Speaker:
    def __init__(self, tts: TTSBackend, out: AudioOutput, fx: VoiceFX | None, cfg: TTSConfig) -> None:
        self.tts = tts
        self.out = out
        self.fx = fx
        self.cfg = cfg
        self.fillers: dict[str, np.ndarray] = {}
        self._recent_fillers: deque[str] = deque(maxlen=6)
        self._q: asyncio.Queue | None = None
        self._task: asyncio.Task | None = None
        self.spoken: list[str] = []  # chunks sent to TTS this turn
        # perf_counter timestamps for the current turn
        self.t_first_push: float | None = None
        self.t_first_byte: float | None = None

    # -- per reply ------------------------------------------------------------------------------

    def begin(self) -> None:
        self.interrupt()
        self._q = asyncio.Queue()
        self.spoken = []
        self.t_first_push = self.t_first_byte = None
        self.out.arm()
        self._task = asyncio.create_task(self._run(self._q))

    def say(self, chunk: str) -> None:
        if self._q is not None:
            self._q.put_nowait(chunk)

    def end(self) -> None:
        if self._q is not None:
            self._q.put_nowait(_END)
            self._q = None

    async def finished(self) -> None:
        """Wait until TTS has delivered all audio for this reply (not until it has played)."""
        if self._task:
            await asyncio.shield(self._task)

    def interrupt(self) -> None:
        """Stop speaking now: cancel synthesis and drop queued audio."""
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None
        self._q = None
        self.out.clear()

    async def _run(self, q: asyncio.Queue) -> None:
        utt = await self.tts.utterance()
        recv = asyncio.create_task(self._receive(utt))
        try:
            while (item := await q.get()) is not _END:
                text = pause_markup(normalize(item), self.cfg.ellipsis_ms)
                if not text:
                    continue
                if self.t_first_push is None:
                    self.t_first_push = time.perf_counter()
                self.spoken.append(item)
                await utt.push(text)
            await utt.end()
            await recv
        except asyncio.CancelledError:
            recv.cancel()
            try:
                await utt.cancel()
            except Exception:
                pass
            raise

    async def _receive(self, utt) -> None:
        async for audio in utt.audio():
            if self.t_first_byte is None:
                self.t_first_byte = time.perf_counter()
            self.out.write(self.fx.process(audio) if self.fx else audio)

    # -- one-off lines and fillers --------------------------------------------------------------

    async def say_line(self, text: str) -> None:
        """Speak a complete line outside a reply (e.g. a timer going off)."""
        self.begin()
        self.say(text)
        self.end()
        await self.finished()

    def play_filler(self) -> str | None:
        choices = [t for t in self.fillers if t not in self._recent_fillers] or list(self.fillers)
        if not choices:
            return None
        line = random.choice(choices)
        self._recent_fillers.append(line)
        clip = self.fillers[line]
        self.out.write(self.fx.process(clip) if self.fx else clip)
        return line
