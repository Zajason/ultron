"""The conversation engine: streaming Claude calls, the tool loop, prompt caching and history.

Framework-free on purpose: text mode drives it directly, and the Pipecat pipeline (Milestone 4)
will wrap it in a frame processor, so both paths share one LLM/tool code path.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from types import SimpleNamespace
from typing import Any

import httpx2
from anthropic import AsyncAnthropic, DefaultAsyncHttpxClient
from anthropic.types import TextBlock, ToolUseBlock

from .config import Config, LLMConfig
from .memory import Memory
from .metrics import TurnMetrics
from .persona import PLAIN_OFF, PLAIN_ON, build_local_prompt, build_system_prompt, detect_mode_switch
from .pricing import usd
from .router import route
from .text import Chunker
from .tools import Executor, Registry

# Handled inside the brain, not by the executor: the rest of the turn runs on the strong model.
ESCALATE = {
    "name": "escalate",
    "description": "Hand this request to your deeper reasoning mode before answering. Use it when the request "
                   "needs multi-step reasoning, careful analysis, nuanced writing, judgement or deep knowledge. "
                   "Never for simple practical requests (time, timers, apps, volume, weather, files).",
    "input_schema": {"type": "object", "properties": {"reason": {"type": "string"}}, "additionalProperties": False},
}

LOCAL = "local"  # model name for turns on the local backend
LOCAL_FILLER_DELAY_S = 1.0

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
        memory: Memory | None = None,
    ) -> None:
        self.cfg = cfg
        self.llm = cfg.llm
        self.client = client or make_client(cfg.llm)
        self.executor = executor
        self.registry = registry
        self.refresh_tools()
        self.messages: list[dict[str, Any]] = []
        self.memory = memory
        self.summary = ""  # of turns trimmed from the rolling window, written in the background
        self._summarising: asyncio.Task | None = None
        self.plain = False
        self.menace = cfg.menace_level
        self._system = self._build_system()
        self.local = None  # LocalLLM once warm() has loaded it
        self._local_ready = False

    def _make_local(self):
        if not self.llm.local_model or self.llm.routing == "strong":
            return None
        try:
            from .backends.local_llm import LocalLLM
            import mlx_lm  # noqa: F401  (Apple Silicon only)
        except ImportError:
            return None
        return LocalLLM(self.llm.local_model, self.llm.local_max_tokens)

    def _local_prompt(self) -> str:
        mem = self.memory.render() if self.memory is not None else ""
        return build_local_prompt(self.menace, self.cfg.user_name, mem)

    def _local_tools(self) -> list[dict[str, Any]]:
        return [t for t in self.tools if "input_schema" in t]  # no server tools (web search)

    def refresh_tools(self) -> None:
        """Rebuild the tool list (after MCP servers registered theirs). Changes the cached prefix."""
        cfg = self.cfg
        self.tools = self.registry.schemas()
        if cfg.tools.web_search:
            # Anthropic-hosted search. The basic variant: the dynamic-filtering one errors on Haiku 5.5
            # and took 13.8 s vs 4.5 s on Sonnet 5.5 (measured).
            self.tools.append({"type": "web_search_20250305", "name": "web_search", "max_uses": cfg.tools.web_search_max_uses})
        if cfg.llm.routing == "hybrid":
            # In both models' lists (same tool set, and history may contain the call): Haiku uses it,
            # Sonnet is told it's already there.
            self.tools.append(ESCALATE)

    # -- prompt -------------------------------------------------------------------------------

    def _build_system(self) -> list[dict[str, Any]]:
        # One explicit breakpoint at the end of the persona caches tools + persona (tools render
        # first). Memory and the running summary come after it: when they change, only the
        # conversation's cache is rewritten, never the persona's.
        return [
            {
                "type": "text",
                "text": build_system_prompt(self.menace, self.cfg.user_name),
                "cache_control": {"type": "ephemeral", "ttl": self.llm.cache_ttl},
            }
        ]

    def _system_blocks(self) -> list[dict[str, Any]]:
        extra = []
        if self.memory is not None and self.memory.facts:
            extra.append(f"What you remember about {self.cfg.user_name} from earlier conversations:\n{self.memory.render()}")
        if self.summary:
            extra.append(f"Earlier in this conversation (summary of turns no longer shown):\n{self.summary}")
        return self._system + ([{"type": "text", "text": "\n\n".join(extra)}] if extra else [])

    def set_menace(self, level: int) -> None:
        self.menace = level
        self._system = self._build_system()
        if self._local_ready:
            self._local_ready = False
            asyncio.ensure_future(self._reload_local())

    async def _reload_local(self) -> None:
        await self.local.load(self._local_prompt(), self._local_tools())
        self._local_ready = True

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
            "system": self._system_blocks(),
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
        if tier == "fast":
            return LOCAL if self._local_ready else self.llm.fast_model
        return self.llm.strong_model

    async def warm(self) -> float:
        """Open the connection and write the tools+system cache for every model in use (caches are
        per model). Returns elapsed ms."""
        t0 = time.perf_counter()
        jobs = [
            self.client.messages.create(
                **{**self._params(model), "max_tokens": 0},
                messages=[{"role": "user", "content": "warmup"}],
            )
            for model in self._models()
        ]
        if self.local is None:
            self.local = self._make_local()
        if self.local is not None and not self._local_ready:
            jobs.append(self._reload_local())
        await asyncio.gather(*jobs)
        return (time.perf_counter() - t0) * 1000

    async def close(self) -> None:
        if self.local is not None:
            self.local.close()
        await self.client.close()

    def _sync_local(self) -> None:
        # Prefill this turn into the local cache now, while the user is listening, not next turn.
        if self._local_ready:
            fut = self.local.sync(self.messages)
            fut.add_done_callback(lambda f: f.exception())  # a failed sync only costs prefill time

    # -- turns ----------------------------------------------------------------------------------

    def reset(self) -> None:
        self.messages.clear()
        self.plain = False
        self.summary = ""

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

    def checkpoint(self) -> tuple[int, bool]:
        return len(self.messages), self.plain

    def rollback(self, cp: tuple[int, bool]) -> None:
        """Undo a turn completely (a speculative reply the user talked over before hearing it)."""
        n, self.plain = cp
        del self.messages[n:]

    def set_last_reply(self, spoken: str) -> None:
        """Replace the text of the last reply with what the user actually heard before cutting in,
        so the history matches the conversation. Tool calls in it are kept."""
        for msg in reversed(self.messages):
            if msg["role"] != "assistant":
                continue
            blocks = msg["content"] if isinstance(msg["content"], list) else []
            kept = [b for b in blocks if _block_type(b) != "text"]
            text = (spoken.strip() + " —") if spoken.strip() else "—"
            msg["content"] = [{"type": "text", "text": text}, *kept]
            return

    def _repair(self) -> None:
        """After a cancelled turn: every tool_use needs a result, and the turn must end with a reply."""
        last = self.messages[-1] if self.messages else None
        if last and last["role"] == "assistant":
            ids = [b.id for b in (last["content"] if isinstance(last["content"], list) else []) if _block_type(b) == "tool_use"]
            if ids:
                self.messages.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": i, "content": "cancelled: the user interrupted", "is_error": True}
                    for i in ids
                ]})
                last = self.messages[-1]
        if last and last["role"] != "assistant":
            self.messages.append({"role": "assistant", "content": [{"type": "text", "text": "—"}]})

    def _trim(self) -> None:
        """Keep the last max_turns user turns. Trims in batches: every trim changes the cached
        prefix, so dropping several turns at once means fewer cache misses than one per turn."""
        starts = [i for i, m in enumerate(self.messages) if m["role"] == "user" and isinstance(m["content"], str)]
        if len(starts) <= self.llm.max_turns:
            return
        keep = self.llm.max_turns - max(1, self.llm.max_turns // 3)
        dropped, self.messages = self.messages[: starts[-keep]], self.messages[starts[-keep]:]
        self._summarising = asyncio.create_task(self._summarise(dropped, self._summarising))
        if self.plain and not any(m["role"] == "system" for m in self.messages):
            self.messages.insert(1, {"role": "system", "content": PLAIN_ON})

    async def _summarise(self, dropped: list[dict[str, Any]], previous: asyncio.Task | None) -> None:
        """Fold trimmed turns into the running summary, off the critical path (Haiku, ~1 s)."""
        if previous is not None:
            await asyncio.gather(previous, return_exceptions=True)
        lines = []
        for msg in dropped:
            if msg["role"] == "user" and isinstance(msg["content"], str):
                lines.append(f"{self.cfg.user_name}: {msg['content'].split('] ', 1)[-1]}")
            elif msg["role"] == "assistant":
                for blk in msg["content"] if isinstance(msg["content"], list) else []:
                    kind = _block_type(blk)
                    if kind == "text":
                        lines.append(f"Ultron: {blk['text'] if isinstance(blk, dict) else blk.text}")
                    elif kind == "tool_use":
                        lines.append(f"(Ultron used {blk['name'] if isinstance(blk, dict) else blk.name})")
        if not lines:
            return
        prompt = (f"Previous summary:\n{self.summary or '(none)'}\n\nNew exchanges:\n" + "\n".join(lines) +
                  "\n\nUpdate the summary: facts, requests, decisions and open threads, in at most 120 words. "
                  "Plain sentences, no preamble.")
        try:
            r = await self.client.messages.create(model=self.llm.fast_model, max_tokens=400, thinking={"type": "disabled"},
                                                  messages=[{"role": "user", "content": prompt}])
            text = "".join(b.text for b in r.content if b.type == "text").strip()
            if text:
                self.summary = text
        except Exception:
            pass  # a missed summary only costs context, never the conversation

    async def respond(
        self,
        user_text: str,
        on_text: OnText = lambda s: None,
        tier: str | None = None,
        on_chunk: OnText = lambda s: None,
        on_tool_start: OnEvent = lambda: None,
        t0: float | None = None,
        tool_gate: Callable[[], Awaitable[object]] | None = None,
    ) -> TurnMetrics:
        """Run one user turn. on_text gets raw deltas (display), on_chunk gets speakable chunks
        (TTS), on_tool_start fires once if a tool call starts before anything was said (filler).
        tool_gate, if given, is awaited before any tool runs: a speculative turn must not act
        until the user has really finished. If the turn is cancelled, the history is repaired."""
        model = self.model_for(user_text, tier)
        m = TurnMetrics(transcript=user_text, model=model)
        if t0 is not None:
            m.t0 = t0  # voice turns are timed from when the user stopped talking
        self._add_user(user_text)
        self._trim()
        chunker = Chunker(first_max_words=self.cfg.tts.first_chunk_words)

        def emit(delta: str) -> None:
            m.mark("first_token")
            on_text(delta)
            for c in chunker.feed(delta):
                speak(c)

        def speak(c: str | None) -> None:
            if c:
                m.mark("first_chunk")
                on_chunk(c)

        late: list[asyncio.TimerHandle] = []

        def late_filler() -> None:
            if "first_token" not in m.marks:
                on_tool_start()

        def tool_block() -> None:
            if "first_token" not in m.marks and "tool_block" not in m.marks:
                m.mark("tool_block")
                if model == LOCAL:
                    # The local model answers ~0.6 s after starting a call, and a filler line (1-3 s)
                    # would delay that answer. Only cover slow tools, or an escalation to the cloud.
                    late.append(asyncio.get_running_loop().call_later(LOCAL_FILLER_DELAY_S, late_filler))
                else:
                    on_tool_start()

        try:
            await self._turn(model, m, user_text, chunker, emit, speak, tool_block, on_text, tool_gate)
        except asyncio.CancelledError:
            self._repair()
            raise
        finally:
            for h in late:
                h.cancel()
        m.mark("done")
        self._sync_local()
        return m

    async def _turn(self, model, m, user_text, chunker, emit, speak, tool_block, on_text, tool_gate) -> None:
        spoke = False
        rounds = self.llm.max_tool_rounds
        for rnd in range(rounds + 1):
            if spoke:
                on_text(" ")
            # Last round: no more tools, so the model has to answer with what it has.
            final = await self._stream_once(model, m, rnd, emit, tool_block, no_tools=rnd == rounds)
            self.messages.append({"role": "assistant", "content": final.content})
            spoke = spoke or any(b.type == "text" and b.text for b in final.content)

            if final.stop_reason == "pause_turn":
                continue  # a server tool (web search) paused mid-turn: send the turn back to resume
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
            if tool_gate is not None:
                await tool_gate()
            results = {}
            for blk in tool_uses:
                if blk.name == "escalate":
                    if model != self.llm.strong_model:
                        model = m.model = self.llm.strong_model
                        m.mark("escalated")
                        note = "Handed to your deeper mode. Now answer the request fully, as yourself."
                    else:
                        note = "You are already at full depth. Answer directly."
                    results[blk.id] = {"type": "tool_result", "tool_use_id": blk.id, "content": note}
            others = [blk for blk in tool_uses if blk.name != "escalate"]
            for r in await self._run_tools(others, m, user_text) if others else []:
                results[r["tool_use_id"]] = r
            self.messages.append({"role": "user", "content": [results[blk.id] for blk in tool_uses]})

        speak(chunker.flush())

    async def _stream_once(
        self, model: str, m: TurnMetrics, rnd: int, emit: OnText, tool_block: OnEvent, no_tools: bool = False
    ):
        if model == LOCAL:
            return await self._stream_local(m, rnd, emit, tool_block, no_tools)
        t_req = time.perf_counter()
        m.mark("llm_request")
        ttft = None
        extra = {"tool_choice": {"type": "none"}} if no_tools else {}
        saw_block = False
        last_char = [" "]
        gap = [False]

        def emit_gap() -> None:
            gap[0] = True  # decided on the next text: it may already start with a space
        async with self.client.messages.stream(
            **self._params(model), **extra, messages=self.messages, cache_control={"type": "ephemeral"}
        ) as stream:
            async for event in stream:
                if event.type == "content_block_start" and event.content_block.type == "text" and saw_block:
                    emit_gap()  # text resumes after a tool/search block: keep words apart
                if event.type == "content_block_start":
                    saw_block = True
                if event.type == "text" and event.text:
                    if ttft is None:
                        ttft = (time.perf_counter() - t_req) * 1000
                    if gap[0] and not last_char[0].isspace() and not event.text[0].isspace():
                        emit(" ")
                    gap[0] = False
                    emit(event.text)
                    last_char[0] = event.text[-1]
                elif event.type == "content_block_start" and event.content_block.type in ("tool_use", "server_tool_use"):
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
                "usd": round(usd(model, u), 6),
                "stop": final.stop_reason,
            }
        )
        return final

    async def _stream_local(self, m: TurnMetrics, rnd: int, emit: OnText, tool_block: OnEvent, no_tools: bool):
        """One round on the local model, returned in the shape of an Anthropic message so the
        tool loop and the history don't care which model answered."""
        from .backends.local_llm import _OPEN, _STAMP, held_back, parse_calls

        t_req = time.perf_counter()
        m.mark("llm_request")
        ttft = None
        buf, sent, skip, call_at = "", 0, None, -1
        async for delta in self.local.stream(self.messages):
            if ttft is None:
                ttft = (time.perf_counter() - t_req) * 1000
            buf += delta
            if call_at >= 0:
                if "</tool_call>" in buf[call_at:]:
                    break  # run the first call now; don't wait for the model to close its turn
                continue
            if skip is None:
                head = buf.lstrip()
                if head.startswith("[") and "]" not in head and len(head) < 40:
                    continue  # maybe the echoed clock stamp: wait to see it whole
                st = _STAMP.match(buf)
                skip = sent = st.end() if st else len(buf) - len(buf.lstrip())
            if (i := buf.find(_OPEN)) >= 0:
                call_at = i
                if buf[sent:i].strip():
                    emit(buf[sent:i].rstrip())
                tool_block()
                continue
            end = len(buf) - held_back(buf)
            while end > sent and buf[end - 1].isspace():
                end -= 1  # trailing space goes out with the next word, or not at all before a call
            if end > sent:
                emit(buf[sent:end])
                sent = end
        text_end = call_at if call_at >= 0 else len(buf)
        if call_at < 0 and sent < text_end and skip is not None:
            emit(buf[sent:text_end])
        text = buf[skip or 0:text_end].strip()
        calls = [] if no_tools else parse_calls(buf, self.local.tools)
        content: list[Any] = [TextBlock(type="text", text=text)] if text else []
        content += [ToolUseBlock(type="tool_use", id=f"toolu_local_{rnd}_{i}_{int(t_req * 1e6) % 10**9}",
                                 name=c.name, input=c.args) for i, c in enumerate(calls)]
        if not content:
            content = [TextBlock(type="text", text="—")]
        m.llm_calls.append({"round": rnd, "model": LOCAL, "ttft_ms": round(ttft, 1) if ttft is not None else None,
                            "ms": round((time.perf_counter() - t_req) * 1000, 1), "input": 0, "cache_read": 0,
                            "cache_write": 0, "output": len(buf), "usd": 0.0, "stop": "tool_use" if calls else "end_turn"})
        return SimpleNamespace(content=content, stop_reason="tool_use" if calls else "end_turn")

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


def _block_type(b: Any) -> str | None:
    return b.get("type") if isinstance(b, dict) else getattr(b, "type", None)
