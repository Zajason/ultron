"""Streaming TTS behind one interface. One utterance = one reply; text arrives in chunks while audio
streams back, so speech starts before the LLM has finished."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, Protocol

import numpy as np
from cartesia import AsyncCartesia

from ..audio.resample import varispeed_ratio
from ..config import TTSConfig


class Utterance(Protocol):
    async def push(self, text: str) -> None: ...
    async def end(self) -> None: ...
    async def cancel(self) -> None: ...
    def audio(self) -> AsyncIterator[np.ndarray]: ...


class TTSBackend(Protocol):
    sample_rate: int

    async def connect(self) -> None: ...
    async def utterance(self) -> Utterance: ...
    async def synthesize(self, text: str) -> np.ndarray: ...
    async def close(self) -> None: ...


def tts_speed(cfg: TTSConfig) -> float:
    """Speed to request so the configured speaking rate survives varispeed playback."""
    return round(min(1.5, max(0.6, cfg.speed / varispeed_ratio(cfg.pitch_semitones))), 3)


class CartesiaUtterance:
    def __init__(self, ctx: Any) -> None:
        self._ctx = ctx

    async def push(self, text: str) -> None:
        # Continuations are concatenated as-is, so chunks need their own separating space.
        await self._ctx.push(text + " ")

    async def end(self) -> None:
        await self._ctx.no_more_inputs()

    async def cancel(self) -> None:
        await self._ctx.cancel()

    async def audio(self) -> AsyncIterator[np.ndarray]:
        async for r in self._ctx.receive():
            if r.type == "chunk" and r.audio:
                yield np.frombuffer(r.audio, dtype=np.float32)
            elif r.type == "error":
                raise RuntimeError(f"Cartesia: {getattr(r, 'message', None) or getattr(r, 'title', r)}")


class CartesiaTTS:
    """One long-lived websocket; each utterance is a context on it."""

    def __init__(self, cfg: TTSConfig, api_key: str) -> None:
        self.cfg = cfg
        self.sample_rate = cfg.sample_rate
        self._client = AsyncCartesia(api_key=api_key)
        self._mgr: Any = None
        self._ws: Any = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        async with self._lock:
            if self._ws is not None:
                return
            self._mgr = self._client.tts.websocket_connect()
            self._ws = await self._mgr.__aenter__()

    async def _reconnect(self) -> None:
        old, self._ws, self._mgr = self._mgr, None, None
        if old is not None:
            try:
                await old.__aexit__(None, None, None)
            except Exception:
                pass
        await self.connect()

    def _context(self) -> Any:
        gen: dict[str, Any] = {"speed": tts_speed(self.cfg)}
        if self.cfg.emotion:
            gen["emotion"] = self.cfg.emotion
        return self._ws.context(
            model_id=self.cfg.model,
            voice={"mode": "id", "id": self.cfg.voice_id},
            output_format={"container": "raw", "encoding": "pcm_f32le", "sample_rate": self.sample_rate},
            language="en",
            generation_config=gen,
            # We send whole clauses ourselves. The default (3000 ms) makes the server wait for more.
            max_buffer_delay_ms=0,
        )

    async def utterance(self) -> CartesiaUtterance:
        await self.connect()
        try:
            return CartesiaUtterance(self._context())
        except Exception:
            # The socket may have been closed while idle; one reconnect, then let it raise.
            await self._reconnect()
            return CartesiaUtterance(self._context())

    async def synthesize(self, text: str) -> np.ndarray:
        """Whole clip, for pre-rendered lines."""
        u = await self.utterance()
        await u.push(text)
        await u.end()
        parts = [a async for a in u.audio()]
        return np.concatenate(parts) if parts else np.zeros(0, np.float32)

    async def close(self) -> None:
        if self._mgr is not None:
            await self._mgr.__aexit__(None, None, None)
            self._mgr = self._ws = None
        await self._client.close()
