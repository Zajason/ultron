"""Local LLM for practical turns (MLX on Apple Silicon): no network round trip, so the first token
comes ~75 ms after the request instead of Haiku's ~600-800 ms.

The prompt (system + tools) is prefilled once and kept as a KV cache. After every turn the
conversation so far is prefilled in the background, so a new turn only prefills its own user
message. Qwen3.5's linear-attention layers can't be trimmed, so caches are snapshotted (copied)
rather than rolled back. All MLX work runs on one worker thread: MLX is not thread-safe, and one
thread keeps the background prefill and a live turn from running at once.
"""

from __future__ import annotations

import asyncio
import copy
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

_CALL = re.compile(r"<tool_call>\s*<function=([\w.-]+)>(.*?)</function>\s*</tool_call>", re.S)
_PARAM = re.compile(r"<parameter=([\w.-]+)>\n?(.*?)\n?</parameter>", re.S)
_OPEN = "<tool_call>"
_STAMP = re.compile(r"^\s*\[[A-Z][a-z]{2}[^\]]{0,30}\]\s*")  # the clock stamp, if the model echoes it


@dataclass
class LocalCall:
    name: str
    args: dict[str, Any]


def parse_calls(text: str, tools: list[dict]) -> list[LocalCall]:
    """Qwen's XML tool calls -> name + arguments, coerced to the schema's types."""
    props = {t["function"]["name"]: t["function"]["parameters"].get("properties", {}) for t in tools}
    out = []
    for m in _CALL.finditer(text):
        name = m.group(1)
        args: dict[str, Any] = {}
        for k, v in _PARAM.findall(m.group(2)):
            args[k] = _coerce(v.strip(), props.get(name, {}).get(k, {}).get("type"))
        out.append(LocalCall(name, args))
    return out


def _coerce(v: str, typ: str | None) -> Any:
    try:
        if typ == "integer":
            return int(float(v))
        if typ == "number":
            return float(v)
        if typ == "boolean":
            return v.lower() == "true"
    except ValueError:
        pass
    return v


def held_back(text: str) -> int:
    """How many trailing characters might be the start of '<tool_call>' and must not be spoken yet."""
    for n in range(min(len(_OPEN) - 1, len(text)), 0, -1):
        if _OPEN.startswith(text[-n:]):
            return n
    return 0


def to_chat(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Anthropic-format history -> chat-template messages. Web-search blocks are dropped (their
    text stays); mid-conversation system notes become user notes (the template only allows a
    system message first)."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        role, content = msg["role"], msg["content"]
        if role == "system":
            out.append({"role": "user", "content": f"(System note: {content})"})
        elif role == "user" and isinstance(content, str):
            out.append({"role": "user", "content": content})
        elif role == "user":
            for b in content:
                if _get(b, "type") == "tool_result":
                    c = _get(b, "content")
                    out.append({"role": "tool", "content": c if isinstance(c, str) else str(c)})
        else:
            blocks = content if isinstance(content, list) else [{"type": "text", "text": str(content)}]
            text = "".join(_get(b, "text") or "" for b in blocks if _get(b, "type") == "text")
            calls = [{"function": {"name": _get(b, "name"), "arguments": _get(b, "input") or {}}}
                     for b in blocks if _get(b, "type") == "tool_use"]
            out.append({"role": "assistant", "content": text, **({"tool_calls": calls} if calls else {})})
    return out


def _get(b: Any, key: str) -> Any:
    return b.get(key) if isinstance(b, dict) else getattr(b, key, None)


class LocalLLM:
    def __init__(self, repo: str, max_tokens: int = 300) -> None:
        self.repo = repo
        self.max_tokens = max_tokens
        self._pool = ThreadPoolExecutor(1, thread_name_prefix="mlx")
        self.model = self.tok = None
        self.tools: list[dict] = []
        self._system = ""
        self._base: tuple[str, Any] | None = None  # (prompt text, cache) for system + tools
        self._conv: tuple[str, Any] | None = None  # (prompt text, cache) for the history so far

    # -- worker-thread side ------------------------------------------------------------------

    def _render(self, chat: list[dict], gen: bool) -> str:
        return self.tok.apply_chat_template(chat, tools=self.tools or None, add_generation_prompt=gen,
                                            tokenize=False, enable_thinking=False)

    def _stable(self, chat: list[dict]) -> str:
        """The prompt text up to where the next user message will start: identical whatever that
        message is, so it can be cached ahead of time."""
        full = self._render(chat + [{"role": "user", "content": "\x00"}], gen=False)
        return full[: full.rindex("<|im_start|>user\n\x00")]

    def _extend(self, start: tuple[str, Any], text: str) -> Any:
        """A copy of start's cache, prefilled up to `text` (which must extend start's text)."""
        import mlx.core as mx
        cache = copy.deepcopy(start[1])
        toks = self.tok.encode(text[len(start[0]):], add_special_tokens=False)
        for i in range(0, len(toks), 512):
            self.model(mx.array(toks[i:i + 512])[None], cache=cache)
            mx.eval([c.state for c in cache])
        return cache

    def _start_for(self, text: str) -> tuple[str, Any]:
        for snap in (self._conv, self._base):
            if snap is not None and text.startswith(snap[0]):
                return snap
        raise RuntimeError("local prompt does not extend the cached system prompt")

    def _load(self, system: str, tools: list[dict]) -> float:
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache
        t0 = time.perf_counter()
        if self.model is None:
            self.model, self.tok = load(self.repo)
        self.tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                         "parameters": t["input_schema"]}} for t in tools]
        self._system = system
        text = self._stable([{"role": "system", "content": system}])
        self._base = (text, self._extend(("", make_prompt_cache(self.model)), text))
        self._conv = None
        return (time.perf_counter() - t0) * 1000

    def _sync(self, chat: list[dict]) -> None:
        text = self._stable([{"role": "system", "content": self._system}, *chat])
        start = self._start_for(text)
        if start[0] != text:
            self._conv = (text, self._extend(start, text))

    def _generate(self, chat: list[dict], push, stop: threading.Event) -> None:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        import mlx.core as mx
        try:
            text = self._render([{"role": "system", "content": self._system}, *chat], gen=True)
            start = self._start_for(text)
            cache = copy.deepcopy(start[1])
            toks = self.tok.encode(text[len(start[0]):], add_special_tokens=False)
            for r in stream_generate(self.model, self.tok, mx.array(toks), max_tokens=self.max_tokens,
                                     prompt_cache=cache, sampler=make_sampler(temp=0.0)):
                if stop.is_set():
                    break
                push(r.text)
        finally:
            push(None)  # the reader must never wait forever; an error surfaces through the job

    # -- asyncio side ------------------------------------------------------------------------

    async def load(self, system: str, tools: list[dict]) -> float:
        """Load the model (first call) and prefill system + tools. Returns elapsed ms."""
        return await asyncio.get_running_loop().run_in_executor(self._pool, self._load, system, tools)

    def sync(self, messages: list[dict[str, Any]]) -> asyncio.Future:
        """Prefill the history in the background, ready for the next turn."""
        return asyncio.get_running_loop().run_in_executor(self._pool, self._sync, to_chat(messages))

    async def stream(self, messages: list[dict[str, Any]]):
        """Yield text deltas for a reply to the history. Cancelling stops generation at the next token."""
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[str | None] = asyncio.Queue()
        stop = threading.Event()
        push = lambda s: loop.call_soon_threadsafe(q.put_nowait, s)
        job = loop.run_in_executor(self._pool, self._generate, to_chat(messages), push, stop)
        try:
            while (s := await q.get()) is not None:
                yield s
            await job
        finally:
            stop.set()

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
