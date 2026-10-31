"""Hands-free conversation: mic always on, Flux decides when the user has finished, VAD handles
barge-in, and replies may start speculatively on Flux's EagerEndOfTurn.

Audio path (one task): mic 16 kHz -> echo canceller -> VAD (barge-in) -> 80 ms frames -> Flux.
Turn path (one task): Flux events -> start / confirm / cancel replies.

Speculation: on EagerEndOfTurn the reply starts with its audio held and its tools gated. If Flux
then confirms the turn with the same transcript, the held audio plays and tools may run; if the
user keeps talking (TurnResumed) or the final transcript differs, the reply is discarded and the
brain's history rolled back, so nothing the user didn't hear and nothing they didn't ask for
survives.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .audio.aec import EchoCanceller
from .audio.input import MicInput, to_linear16
from .audio.speaker import Speaker
from .audio.vad import SileroVAD, VADEvent
from .backends import DeepgramFlux, STTStream, TurnEvent
from .brain import Brain
from .metrics import TurnLog, TurnMetrics
from .wakeword import WakeWordDetector

_YES_RE = re.compile(r"^\W*(yes|yeah|yep|yup|sure|ok(ay)?|do it|go ahead|proceed|confirm(ed)?|affirmative|please do)\b", re.I)

_WAKE_RE = re.compile(r"\b(ultron|ul ?tron|ultra ?tron|altron)\b", re.I)


@dataclass
class Reply:
    transcript: str
    speculative: bool
    task: asyncio.Task | None = None
    committed: asyncio.Event = field(default_factory=asyncio.Event)
    checkpoint: tuple[int, bool] = (0, False)
    t0: float = 0.0
    metrics: TurnMetrics | None = None
    events: dict[str, float] = field(default_factory=dict)  # perf_counter of turn events (for breakdowns)
    shown: list[str] = field(default_factory=list)  # reply text produced before commit, shown on commit


@dataclass
class UI:
    """Terminal callbacks; replaced by no-ops in benches and tests."""
    partial: Callable[[str], None] = lambda s: None
    heard: Callable[[str], None] = lambda s: None
    reply_start: Callable[[], None] = lambda: None
    reply_text: Callable[[str], None] = lambda s: None
    reply_done: Callable[[TurnMetrics, bool], None] = lambda m, interrupted: None
    note: Callable[[str], None] = lambda s: None
    sleep: Callable[[], None] = lambda: None
    wake: Callable[[float], None] = lambda score: None


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9' ]+", "", s.lower()).strip()


class Conversation:
    def __init__(self, brain: Brain, speaker: Speaker, mic: MicInput, stt: DeepgramFlux,
                 vad: SileroVAD, aec: EchoCanceller | None, eager: bool, frame_ms: int = 80,
                 turn_log: TurnLog | None = None, ui: UI | None = None,
                 wake: WakeWordDetector | None = None, idle_sleep_s: float = 10.0,
                 passphrase: str = "", confirm_timeout_s: float = 20.0) -> None:
        self.brain, self.speaker, self.mic, self.stt = brain, speaker, mic, stt
        self.vad, self.aec, self.eager = vad, aec, eager
        self.frame_samples = mic.out_rate * frame_ms // 1000
        self.turn_log = turn_log
        self.ui = ui or UI()
        self.stream: STTStream | None = None
        self.reply: Reply | None = None
        self._cleanup: asyncio.Task | None = None  # history repair of the last cancelled reply
        self.barge_ins: list[float] = []  # ms from speech onset to Ultron's audio stopping
        self._stop = asyncio.Event()
        self.turns_done = 0
        self.errors: list[BaseException] = []
        # Wake word: while asleep nothing is sent to STT. The last 2 s are kept so the request that
        # follows "Ultron" (often in the same breath) reaches STT once we wake.
        self.wake = wake
        self.awake = wake is None
        self.idle_sleep_s = idle_sleep_s
        self._verify = False  # first turn after a wake must actually contain the wake word
        self._last_activity = time.perf_counter()
        self._ring = np.zeros(0, np.float32)
        self._stream_ready = asyncio.Event()
        self.false_wakes = 0
        # Spoken confirmation for risky tools: the next finished turn is the answer, not a request.
        self.passphrase = passphrase
        self.confirm_timeout_s = confirm_timeout_s
        self._confirm: asyncio.Future | None = None

    # -- lifecycle ------------------------------------------------------------------------------

    async def run(self) -> None:
        if self.awake:
            await self._open_stream()
        else:
            self.ui.sleep()
        self.mic.start()
        tasks = [asyncio.create_task(self._audio_loop()), asyncio.create_task(self._turn_loop())]
        try:
            await self._stop.wait()
        finally:
            for t in tasks:
                t.cancel()
            self._cancel_reply()
            self.mic.stop()
            await self._close_stream()

    def stop(self) -> None:
        self._stop.set()

    async def _open_stream(self) -> None:
        self.stream = await self.stt.open(ptt=False)
        self._stream_ready.set()

    async def _close_stream(self) -> None:
        self._stream_ready.clear()
        stream, self.stream = self.stream, None
        if stream is not None:
            with contextlib.suppress(Exception):
                await stream.close()

    async def _wake_up(self, score: float) -> None:
        self.ui.wake(score)
        self.awake = True
        self._verify = True
        self._last_activity = time.perf_counter()
        await self._open_stream()
        # Send the buffered audio first: "Ultron" itself (for verification) and whatever followed.
        backlog = self._ring[-int(1.5 * self.mic.out_rate):]
        for i in range(0, backlog.size, self.frame_samples):
            await self.stream.send(to_linear16(backlog[i : i + self.frame_samples]))

    async def _go_to_sleep(self, false_wake: bool = False) -> None:
        self.awake = False
        self._verify = False
        if false_wake:
            self.false_wakes += 1
        await self._close_stream()
        self.vad.reset()
        if self.wake is not None:
            self.wake.reset()
        self.ui.sleep()

    # -- audio: AEC -> VAD -> Flux --------------------------------------------------------------

    async def _audio_loop(self) -> None:
        pending = np.zeros(0, np.float32)
        async for block in self.mic.blocks():
            now = time.perf_counter()
            clean = self.aec.process(block) if self.aec else block
            if self.wake is not None:
                self._ring = np.concatenate([self._ring, clean])[-2 * self.mic.out_rate :]
                if not self.awake:
                    if (hit := self.wake.push(clean)) is not None:
                        await self._wake_up(hit.score)
                        pending = np.zeros(0, np.float32)
                    continue
                if self.reply is None and not self.vad.speaking and now - self._last_activity > self.idle_sleep_s:
                    await self._go_to_sleep(false_wake=self._verify)
                    continue
            for ev in self.vad.process(clean, now):
                self._last_activity = now
                self._on_vad(ev)
            if self.vad.speaking or self.reply is not None:
                self._last_activity = now
            pending = np.concatenate([pending, clean])
            while pending.size >= self.frame_samples:
                await self.stream.send(to_linear16(pending[: self.frame_samples]))
                pending = pending[self.frame_samples :]

    def _on_vad(self, ev: VADEvent) -> None:
        if ev.kind == "stop":
            return
        if self._confirm is not None:
            self.speaker.out.clear()  # answering the question: stop asking it, but keep the turn
            return
        r = self.reply
        if r is None:
            return
        if r.speculative and not r.committed.is_set():
            return  # nothing audible yet: Flux's TurnResumed decides whether this draft survives
        # Barge-in: stop the audio now, then cancel the reply and keep only what was heard.
        audible = self.speaker.out.busy
        spoken = self.speaker.spoken_text()
        self.speaker.interrupt()
        if audible:
            stop_ms = (time.perf_counter() - ev.t) * 1000 + self.speaker.out.device_latency_ms
            self.barge_ins.append(stop_ms)
        self._cancel_reply(spoken=spoken, interrupted=True)

    # -- turns: Flux events -> replies ----------------------------------------------------------

    async def _turn_loop(self) -> None:
        while True:
            await self._stream_ready.wait()
            stream = self.stream
            ev = await stream.events.get()
            if stream is not self.stream:
                continue  # closed for sleep while we waited
            if ev.kind in ("error", "closed"):
                if not self.awake:
                    continue
                self.ui.note(f"STT {ev.kind}: {ev.detail}; reconnecting")
                await self._open_stream()
                continue
            self._on_turn(ev)

    def _on_turn(self, ev: TurnEvent) -> None:
        if self._confirm is not None:
            if ev.kind == "end" and not self._confirm.done():
                self.ui.heard(ev.transcript)
                self._confirm.set_result(ev.transcript)
            return
        if self._verify and ev.kind in ("eager", "end") and ev.transcript:
            if not _WAKE_RE.search(ev.transcript):
                if ev.kind == "end":  # the detector misfired: no reply, straight back to sleep
                    self.ui.note(f"(not for me: {ev.transcript!r})")
                    asyncio.create_task(self._go_to_sleep(false_wake=True))
                return
            if ev.kind == "end":
                self._verify = False
        if ev.kind == "update" and ev.transcript:
            self.ui.partial(ev.transcript)
        elif ev.kind == "eager" and self.eager and ev.transcript:
            self._start_reply(ev.transcript, speculative=True)
            self.reply.events["eager"] = ev.t
        elif ev.kind == "resumed":
            if self.reply and self.reply.speculative and not self.reply.committed.is_set():
                self._cancel_reply(rollback=True)
        elif ev.kind == "end" and ev.transcript:
            r = self.reply
            if r and r.speculative and not r.committed.is_set():
                if _norm(r.transcript) == _norm(ev.transcript):
                    self.ui.heard(ev.transcript)
                    r.events["end"] = ev.t
                    self._commit(r)
                    return
                self._cancel_reply(rollback=True)
            self.ui.heard(ev.transcript)
            self._start_reply(ev.transcript, speculative=False)
            self.reply.events["end"] = ev.t

    async def confirm(self, tool, args: dict, tier) -> bool:
        """Executor callback: ask out loud, take the user's next turn as the answer."""
        what = tool.describe_call(args)
        destructive = int(tier) >= 3
        if destructive and not self.passphrase:
            self.ui.note(f"[tier 3] {what}: no passphrase configured, declined")
            return False
        question = (f"This one can't be undone easily: {what}. Say the passphrase to go ahead." if destructive
                    else f"Before I do that: I'm about to {what}. Shall I go ahead?")
        self.ui.note(f"[confirm, tier {int(tier)}] {what}")
        self.speaker.say(question)
        self._confirm = asyncio.get_running_loop().create_future()
        try:
            answer = await asyncio.wait_for(self._confirm, self.confirm_timeout_s)
        except TimeoutError:
            answer = ""
        finally:
            self._confirm = None
        if destructive:
            ok = _norm(self.passphrase) in _norm(answer)
        else:
            ok = bool(_YES_RE.match(answer))
        self.ui.note("confirmed" if ok else "declined")
        self._last_activity = time.perf_counter()
        return ok

    def _turn_t0(self) -> float:
        # Voice-to-voice is timed from the end of the user's last voiced audio (VAD), not from when
        # Flux told us. Flux often confirms the turn before VAD has seen enough silence to say "stop".
        t = self.vad.last_voiced
        now = time.perf_counter()
        return t if t is not None and now - t < 3.0 else now

    def _start_reply(self, transcript: str, speculative: bool) -> None:
        self._cancel_reply()
        r = Reply(transcript, speculative, t0=self._turn_t0())
        if not speculative:
            r.committed.set()
        self.speaker.begin(hold=speculative)
        if not speculative:
            self.ui.reply_start()
        r.task = asyncio.create_task(self._run_reply(r))
        self.reply = r

    def _commit(self, r: Reply) -> None:
        r.t0 = self._turn_t0()
        r.committed.set()
        self.ui.reply_start()
        for s in r.shown:
            self.ui.reply_text(s)
        self.speaker.release()

    async def _run_reply(self, r: Reply) -> None:
        try:
            await self._reply_body(r)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # A failed reply must not wedge the conversation: say so, clean up, keep listening.
            self.ui.note(f"reply failed: {type(e).__name__}: {e}")
            self.errors.append(e)
            self.speaker.interrupt()
            self.brain._repair()
            if self.reply is r:
                self.reply = None
            self.turns_done += 1

    async def _reply_body(self, r: Reply) -> None:
        if self._cleanup is not None:
            await self._cleanup  # the previous reply's history repair must land first
        r.checkpoint = self.brain.checkpoint()

        def show(s: str) -> None:
            (self.ui.reply_text if r.committed.is_set() else r.shown.append)(s)

        m = await self.brain.respond(
            r.transcript,
            on_text=show,
            on_chunk=self.speaker.say,
            on_tool_start=self.speaker.play_filler,
            t0=r.t0,
            tool_gate=r.committed.wait,
        )
        r.metrics = m
        self.speaker.end()
        await r.committed.wait()
        await self.speaker.finished()
        for _ in range(100):  # first audio is stamped by the output callback; give it a moment
            if self.speaker.out.first_audio_at is not None:
                break
            await asyncio.sleep(0.01)
        self._finish(r, interrupted=False)

    def _finish(self, r: Reply, interrupted: bool) -> None:
        m = r.metrics
        if m is not None:
            m.t0 = r.t0  # speculative turns: time from the real end of speech, not the draft start
            for name, t in (("tts_first_byte", self.speaker.t_first_byte), ("first_audio", self.speaker.out.first_audio_at)):
                if t is not None and t >= r.t0:
                    m.marks[name] = round((t - r.t0) * 1000, 1)
            m.marks["speculative"] = float(r.speculative)
            for name, t in r.events.items():
                m.marks[f"{name}_event"] = round((t - r.t0) * 1000, 1)
            if self.turn_log:
                self.turn_log.write(m)
            self.ui.reply_done(m, interrupted)
        self.turns_done += 1
        if self.reply is r:
            self.reply = None

    def _cancel_reply(self, spoken: str = "", interrupted: bool = False, rollback: bool = False) -> None:
        r, self.reply = self.reply, None
        if r is None:
            return
        self.speaker.interrupt()
        if r.task and not r.task.done():
            r.task.cancel()
        previous = self._cleanup

        async def fix_history() -> None:
            # Cancellation lands at the task's next await; the brain repairs its history there.
            # Only after that may we roll back or rewrite the reply.
            if previous is not None:
                await previous
            if r.task is not None:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await r.task
            if rollback:
                self.brain.rollback(r.checkpoint)
            elif interrupted:
                self.brain.set_last_reply(spoken)

        self._cleanup = asyncio.create_task(fix_history())
        if interrupted:
            self.ui.note(f"(interrupted after: {spoken!r})" if spoken else "(interrupted before speaking)")
