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
from .resample import Resampler, varispeed_ratio

_END = object()


class Speaker:
    def __init__(self, tts: TTSBackend, out: AudioOutput, fx: VoiceFX | None, cfg: TTSConfig) -> None:
        self.tts = tts
        self.out = out
        self.fx = fx
        self.cfg = cfg
        self.resampler = Resampler(tts.sample_rate, out.sample_rate, cfg.pitch_semitones)
        self.fillers: dict[str, np.ndarray] = {}
        self._recent_fillers: deque[str] = deque(maxlen=6)
        self._q: asyncio.Queue | None = None
        self._task: asyncio.Task | None = None
        self.spoken: list[str] = []  # chunks sent to TTS this turn
        self._words: list[tuple[str, float]] = []  # (word, end time in TTS seconds) for this reply
        self._utt_start: int | None = None  # output sample position where this reply's speech starts
        self._held: list[tuple[str, np.ndarray]] | None = None  # speculative reply: rendered, not yet audible
        # perf_counter timestamps for the current turn
        self.t_first_push: float | None = None
        self.t_first_byte: float | None = None

    # -- per reply ------------------------------------------------------------------------------

    def begin(self, hold: bool = False) -> None:
        """Start a reply. hold=True synthesises it without playing (speculative: the user may not
        be done yet); release() makes it audible, interrupt() discards it."""
        self.interrupt()
        self._q = asyncio.Queue()
        self.spoken = []
        self._words = []
        self._utt_start = None
        self._held = [] if hold else None
        self.t_first_push = self.t_first_byte = None
        if not hold:
            self.out.arm()
        self._task = asyncio.create_task(self._run(self._q))

    @property
    def holding(self) -> bool:
        return self._held is not None

    def release(self) -> None:
        if self._held is None:
            return
        held, self._held = self._held, None
        self.out.arm()
        for kind, audio in held:
            self._emit(kind, audio)

    def _emit(self, kind: str, audio: np.ndarray) -> None:
        if self._held is not None:
            self._held.append((kind, audio))
            return
        if kind == "speech" and self._utt_start is None:
            self._utt_start = self.out.queued_until
        self.out.write(audio)

    def spoken_text(self) -> str:
        """The words of the current reply that have actually come out of the speaker."""
        if self._utt_start is None or not self._words:
            return ""
        latency = int(self.out.device_latency_ms / 1000 * self.out.sample_rate)
        played = max(0, self.out.played - latency - self._utt_start)
        tts_s = played / self.out.sample_rate * varispeed_ratio(self.cfg.pitch_semitones)
        return " ".join(w for w, end in self._words if end <= tts_s + 0.05)

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
        self._held = None
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
        async for kind, value in utt.events():
            if kind == "words":
                self._words.extend(value)
                continue
            if self.t_first_byte is None:
                self.t_first_byte = time.perf_counter()
            self._emit("speech", self.render(value))

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
        self._emit("filler", self.render(self.fillers[line]))
        return line

    def render(self, audio: np.ndarray) -> np.ndarray:
        """TTS-rate audio -> output-rate audio, with varispeed and FX."""
        audio = self.resampler.process(audio)
        return self.fx.process(audio) if self.fx else audio
