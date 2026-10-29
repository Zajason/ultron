"""Streaming STT behind one interface. A stream carries linear16 audio up and turn events down.

Deepgram Flux does turn detection itself (Milestone 4 uses it); for push-to-talk the release is
sent as ForceEndTurn, which closes the turn on the transcript Flux already has, no extra decode.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass
from typing import Any, Protocol

from deepgram import AsyncDeepgramClient
from deepgram.environment import DeepgramClientEnvironment

from ..config import STTConfig

_REGIONS = {"us": "api.deepgram.com", "eu": "api.eu.deepgram.com"}
_EVENTS = {"Update": "update", "StartOfTurn": "start", "EagerEndOfTurn": "eager",
           "TurnResumed": "resumed", "EndOfTurn": "end"}


@dataclass
class TurnEvent:
    kind: str  # update | start | eager | resumed | end | warning | error | closed
    transcript: str = ""
    t: float = 0.0  # perf_counter when received
    trigger: str | None = None  # on "end": model | manual | timeout
    detail: str = ""


class STTStream(Protocol):
    events: asyncio.Queue[TurnEvent]
    t_connected: float

    async def send(self, audio: bytes) -> None: ...
    async def end_turn(self) -> None: ...
    async def close(self) -> None: ...


class FluxStream:
    def __init__(self, conn: Any, stack: contextlib.AsyncExitStack) -> None:
        self._conn = conn
        self._stack = stack
        self.events: asyncio.Queue[TurnEvent] = asyncio.Queue()
        self.t_connected = time.perf_counter()
        self._listener = asyncio.create_task(self._listen())

    async def _listen(self) -> None:
        try:
            async for msg in self._conn:
                now = time.perf_counter()
                kind = getattr(msg, "type", None)
                if kind == "TurnInfo":
                    ev = _EVENTS.get(msg.event, str(msg.event).lower())
                    self.events.put_nowait(TurnEvent(ev, msg.transcript or "", now, getattr(msg, "trigger", None)))
                elif kind == "Warning":
                    self.events.put_nowait(TurnEvent("warning", t=now, detail=str(getattr(msg, "code", msg))))
                elif kind in ("Error", "FatalError"):
                    self.events.put_nowait(TurnEvent("error", t=now, detail=str(getattr(msg, "description", msg))))
        except Exception as e:
            self.events.put_nowait(TurnEvent("error", t=time.perf_counter(), detail=f"{type(e).__name__}: {e}"))
        finally:
            self.events.put_nowait(TurnEvent("closed", t=time.perf_counter()))

    async def send(self, audio: bytes) -> None:
        await self._conn.send_media(audio)

    async def end_turn(self) -> None:
        await self._conn.send_force_end_turn()

    async def close(self) -> None:
        with contextlib.suppress(Exception):
            await self._conn.send_close_stream()
        self._listener.cancel()
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await self._stack.aclose()


class DeepgramFlux:
    def __init__(self, cfg: STTConfig, api_key: str) -> None:
        self.cfg = cfg
        host = _REGIONS.get(cfg.region, cfg.region)
        env = DeepgramClientEnvironment(base=f"https://{host}", production=f"wss://{host}",
                                        agent="wss://agent.deepgram.com", agent_rest="https://agent.deepgram.com")
        self._client = AsyncDeepgramClient(api_key=api_key, environment=env)

    async def open(self, ptt: bool = False) -> FluxStream:
        """ptt=True: only ForceEndTurn ends a turn, so Flux never cuts the user off mid-word."""
        c = self.cfg
        params: dict[str, Any] = {
            "model": c.model, "encoding": "linear16", "sample_rate": c.sample_rate,
            "eot_threshold": c.ptt_eot_threshold if ptt else c.eot_threshold, "eot_timeout_ms": c.eot_timeout_ms,
        }
        if c.eager_eot_threshold and not ptt:
            params["eager_eot_threshold"] = c.eager_eot_threshold
        if c.keyterms:
            params["keyterm"] = list(c.keyterms)
        stack = contextlib.AsyncExitStack()
        conn = await stack.enter_async_context(self._client.listen.v2.connect(**params))
        return FluxStream(conn, stack)
