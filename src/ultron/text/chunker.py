"""Splits streamed LLM text into speakable chunks, so TTS can start before the reply is finished.

The first chunk is allowed to be short (a clause), because time-to-first-audio is what the user
hears. Later chunks prefer whole sentences, which TTS prosody handles better.
"""

from __future__ import annotations

_SENTENCE = ".!?"
_CLAUSE = ",;:—"
_ABBREV = {"mr", "mrs", "ms", "dr", "e.g", "i.e", "vs", "st", "approx"}


class Chunker:
    def __init__(self, first_min_chars: int = 12, clause_min_chars: int = 60, max_chars: int = 240,
                 first_max_words: int = 0) -> None:
        self.first_min = first_min_chars
        self.first_max_words = first_max_words  # >0: cut the first chunk after this many words even mid-clause
        self.clause_min = clause_min_chars
        self.max_chars = max_chars
        self._buf = ""
        self._pos = 0  # next index of _buf to inspect
        self._emitted = 0

    def feed(self, delta: str) -> list[str]:
        self._buf += delta
        out: list[str] = []
        while (cut := self._find_cut()) is not None:
            chunk, self._buf = self._buf[:cut].strip(), self._buf[cut:]
            self._pos = 0
            if chunk:
                out.append(chunk)
                self._emitted += 1
        return out

    def flush(self) -> str | None:
        chunk, self._buf, self._pos = self._buf.strip(), "", 0
        if chunk:
            self._emitted += 1
            return chunk
        return None

    def _find_cut(self) -> int | None:
        buf = self._buf
        clause_min = self.first_min if self._emitted == 0 else self.clause_min
        # A boundary is only confirmed once the following character has arrived.
        i = self._pos
        while i < len(buf) - 1:
            c, nxt = buf[i], buf[i + 1]
            n = i + 1  # chunk length if we cut after c
            if nxt.isspace():
                if c in _SENTENCE and not self._is_abbrev(i) and not (c == "." and buf[i - 1 : i] == "."):
                    return n
                if c == "…" or (c == "." and buf[i - 2 : i] == ".."):
                    if n >= self.first_min or self._emitted:
                        return n
                if c in _CLAUSE and n >= clause_min:
                    return n
            elif c == "—" and n >= clause_min:
                return n
            i += 1
        if self.first_max_words and not self._emitted:
            # Count only words followed by a space: the last one may still be growing.
            words = buf.lstrip().split(" ")
            if len(words) > self.first_max_words:
                cut = len(buf) - len(buf.lstrip()) + len(" ".join(words[: self.first_max_words]))
                return cut
        self._pos = max(0, len(buf) - 1)
        if len(buf) > self.max_chars:
            sp = buf.rfind(" ", 0, self.max_chars)
            return sp if sp > 0 else self.max_chars
        return None

    def _is_abbrev(self, i: int) -> bool:
        if self._buf[i] != ".":
            return False
        start = i
        while start > 0 and (self._buf[start - 1].isalpha() or self._buf[start - 1] == "."):
            start -= 1
        return self._buf[start:i].lower() in _ABBREV
