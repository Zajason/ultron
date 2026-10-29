"""The conversation engine: streaming Claude calls, the tool loop, prompt caching and history.

Framework-free on purpose: text mode drives it directly, and the Pipecat pipeline (Milestone 4)
will wrap it in a frame processor, so both paths share one LLM/tool code path.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

import httpx2
from anthropic import AsyncAnthropic, DefaultAsyncHttpxClient

from .config import Config, LLMConfig
from .metrics import TurnMetrics
from .persona import PLAIN_OFF, PLAIN_ON, build_system_prompt, detect_mode_switch
from .router import route
from .text import Chunker
from .tools import Executor, Registry

OnText = Callable[[str], None]
OnEvent = Callable[[], None]


def make_client(cfg: LLMConfig) -> AsyncAnthropic:
    http = DefaultAsyncHttpxClient(
        limits=httpx2.Limits(max_connections=20, max_keepalive_connections=10, keepalive_expiry=cfg.keepalive_expiry_s)
    )
    # Fail fast instead of silently retrying behind a user who is waiting on a voice reply.
    return AsyncAnthropic(http_client=http, max_retries=1, timeout=httpx2.Timeout(30.0, connect=5.0))


class Brain:
    def __init__(
        self,
        cfg: Config,
        registry: Registry,
        executor: Executor,
        client: AsyncAnthropic | None = None,
    ) -> None:
        self.cfg = cfg
        self.llm = cfg.llm
        self.client = client or make_client(cfg.llm)
        self.executor = executor
        self.tools = registry.schemas()
        self.messages: list[dict[str, Any]] = []
        self.plain = False
        self.menace = cfg.menace_level
        self._system = self._build_system()

    # -- prompt -------------------------------------------------------------------------------

    def _build_system(self) -> list[dict[str, Any]]:
        # One explicit breakpoint at the end of system caches tools + system (tools render first).
        return [
            {
                "type": "text",
                "text": build_system_prompt(self.menace, self.cfg.user_name),
                "cache_control": {"type": "ephemeral", "ttl": self.llm.cache_ttl},
            }
        ]

    def set_menace(self, level: int) -> None:
        self.menace = level
        self._system = self._build_system()

    def _params(self, model: str) -> dict[str, Any]:
        if self.llm.thinking == "adaptive":
            thinking = {"type": "adaptive"}
        elif model.startswith("claude-sonnet-5-5"):
            thinking = {"type": "between_tools"}  # Sonnet 5.5 rejects "disabled"; this is its off switch
        else:
            thinking = {"type": "disabled"}
        return {
            "model": model,
            "max_tokens": self.llm.max_tokens,
            "system": self._system,
            "tools": self.tools,
            "thinking": thinking,
            "output_config": {"effort": self.llm.effort},
        }

    # -- connection / cache warming --------------------------------------------------------------

    def _models(self) -> list[str]:
        r = self.llm.routing
        return [self.llm.fast_model] * (r != "strong") + [self.llm.strong_model] * (r != "fast")

    def model_for(self, text: str, tier: str | None = None) -> str:
        tier = tier or (route(text) if self.llm.routing == "hybrid" else self.llm.routing)
        return self.llm.fast_model if tier == "fast" else self.llm.strong_model

    async def warm(self) -> float:
        """Open the connection and write the tools+system cache for every model in use (caches are
        per model). Returns elapsed ms."""
        t0 = time.perf_counter()
        await asyncio.gather(*(
            self.client.messages.create(
                **{**self._params(model), "max_tokens": 0},
                messages=[{"role": "user", "content": "warmup"}],
            )
            for model in self._models()
        ))
        return (time.perf_counter() - t0) * 1000

    async def close(self) -> None:
        await self.client.close()

    # -- turns ----------------------------------------------------------------------------------

    def reset(self) -> None:
        self.messages.clear()
        self.plain = False

    def _add_user(self, text: str) -> None:
        # The clock rides on the user message, after the cached prefix: the model knows the time
        # without a get_time round trip, and the system prompt stays byte-identical.
        stamp = datetime.now().astimezone().strftime("%a %d %b %Y %H:%M")
        self.messages.append({"role": "user", "content": f"[{stamp}] {text}"})
        # Mode switches go in as mid-conversation system messages, so the cached system prompt
        # (and everything before this point) stays byte-identical.
        mode = detect_mode_switch(text)
        if mode == "plain" and not self.plain:
            self.plain = True
            self.messages.append({"role": "system", "content": PLAIN_ON})
        elif mode == "normal" and self.plain:
            self.plain = False
            self.messages.append({"role": "system", "content": PLAIN_OFF})

    def _trim(self) -> None:
        """Keep the last max_turns user turns. Trims in batches: every trim changes the cached
        prefix, so dropping several turns at once means fewer cache misses than one per turn."""
        starts = [i for i, m in enumerate(self.messages) if m["role"] == "user" and isinstance(m["content"], str)]
        if len(starts) <= self.llm.max_turns:
            return
        keep = self.llm.max_turns - max(1, self.llm.max_turns // 3)
        self.messages = self.messages[starts[-keep]:]
        if self.plain and not any(m["role"] == "system" for m in self.messages):
            self.messages.insert(1, {"role": "system", "content": PLAIN_ON})

    async def respond(
        self,
        user_text: str,
        on_text: OnText = lambda s: None,
        tier: str | None = None,
        on_chunk: OnText = lambda s: None,
        on_tool_start: OnEvent = lambda: None,
    ) -> TurnMetrics:
        """Run one user turn. on_text gets raw deltas (display), on_chunk gets speakable chunks
        (TTS), on_tool_start fires once if a tool call starts before anything was said (filler)."""
        model = self.model_for(user_text, tier)
        m = TurnMetrics(transcript=user_text, model=model)
        self._add_user(user_text)
        self._trim()
        chunker = Chunker()
        spoke = False

        def emit(delta: str) -> None:
            m.mark("first_token")
            on_text(delta)
            for c in chunker.feed(delta):
                speak(c)

        def speak(c: str | None) -> None:
            if c:
                m.mark("first_chunk")
                on_chunk(c)

        def tool_block() -> None:
            if "first_token" not in m.marks and "tool_block" not in m.marks:
                m.mark("tool_block")
                on_tool_start()

        rounds = self.llm.max_tool_rounds
        for rnd in range(rounds + 1):
            if spoke:
                on_text(" ")
            # Last round: no more tools, so the model has to answer with what it has.
            final = await self._stream_once(model, m, rnd, emit, tool_block, no_tools=rnd == rounds)
            self.messages.append({"role": "assistant", "content": final.content})
            spoke = spoke or any(b.type == "text" and b.text for b in final.content)

            tool_uses = [b for b in final.content if b.type == "tool_use"]
            if not tool_uses:
                break
            if final.stop_reason != "tool_use":
                # Truncated (max_tokens) or refused mid-call: don't run a possibly partial input,
                # but answer every tool_use so the history stays valid for the next turn.
                self.messages.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": b.id, "content": "not run: reply was cut off", "is_error": True}
                    for b in tool_uses
                ]})
                break
            # Text before a tool call ("Opening it.") is speakable now, before the tool runs.
            speak(chunker.flush())
            self.messages.append({"role": "user", "content": await self._run_tools(tool_uses, m, user_text)})

        speak(chunker.flush())
        m.mark("done")
        return m

    async def _stream_once(
        self, model: str, m: TurnMetrics, rnd: int, emit: OnText, tool_block: OnEvent, no_tools: bool = False
    ):
        t_req = time.perf_counter()
        ttft = None
        extra = {"tool_choice": {"type": "none"}} if no_tools else {}
        async with self.client.messages.stream(
            **self._params(model), **extra, messages=self.messages, cache_control={"type": "ephemeral"}
        ) as stream:
            async for event in stream:
                if event.type == "text" and event.text:
                    if ttft is None:
                        ttft = (time.perf_counter() - t_req) * 1000
                    emit(event.text)
                elif event.type == "content_block_start" and event.content_block.type == "tool_use":
                    tool_block()
            final = await stream.get_final_message()
        u = final.usage
        m.llm_calls.append(
            {
                "round": rnd,
                "ttft_ms": round(ttft, 1) if ttft is not None else None,
                "ms": round((time.perf_counter() - t_req) * 1000, 1),
                "input": u.input_tokens,
                "cache_read": u.cache_read_input_tokens or 0,
                "cache_write": u.cache_creation_input_tokens or 0,
                "output": u.output_tokens,
                "stop": final.stop_reason,
            }
        )
        return final

    async def _run_tools(self, tool_uses: list, m: TurnMetrics, transcript: str) -> list[dict[str, Any]]:
        async def one(block) -> dict[str, Any]:
            t0 = time.perf_counter()
            res = await self.executor.run(block.name, block.input, transcript)
            m.tools.append({"name": block.name, "ms": round((time.perf_counter() - t0) * 1000, 1), "error": res.is_error})
            out = {"type": "tool_result", "tool_use_id": block.id, "content": res.content}
            if res.is_error:
                out["is_error"] = True
            return out

        # All results go back in one user message, in call order.
        return list(await asyncio.gather(*(one(b) for b in tool_uses)))
