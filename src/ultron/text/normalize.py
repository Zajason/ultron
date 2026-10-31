"""Text cleanup before TTS: things the LLM sometimes writes that sound wrong when read aloud.

The persona already asks for spoken-style output, so this is a safety net, not a rewriter. It runs
per chunk, so every rule must work on a clause in isolation. Sonic reads plain numbers, dates and
clock times well on its own; we only fix units, symbols, paths, URLs and markup.
"""

from __future__ import annotations

import re

_UNITS = {
    "TB": "terabytes", "GB": "gigs", "MB": "megs", "KB": "kilobytes", "kB": "kilobytes",
    "GHz": "gigahertz", "MHz": "megahertz", "kHz": "kilohertz", "Hz": "hertz",
    "ms": "milliseconds", "km/h": "kilometres an hour", "mph": "miles an hour",
    "km": "kilometres", "kg": "kilos", "°C": "degrees", "°F": "degrees Fahrenheit", "%": "percent",
}
# Longest first, so "km/h" wins over "km".
_UNIT_RE = re.compile(
    r"(?<![\w.])(\d+(?:[.,]\d+)?)\s?(" + "|".join(re.escape(u) for u in sorted(_UNITS, key=len, reverse=True)) + r")(?![\w/])"
)

_ABBREV = [
    (re.compile(r"\be\.g\.,?", re.I), "for example"),
    (re.compile(r"\bi\.e\.,?", re.I), "that is"),
    (re.compile(r"\betc\.", re.I), "et cetera"),
    (re.compile(r"\bvs\.?(?=\s)", re.I), "versus"),
]

_URL_RE = re.compile(r"\bhttps?://([^/\s]+)\S*|\bwww\.([^/\s]+)\S*")
# Unix paths with at least two components, or ~/...
_PATH_RE = re.compile(r"(?<![\w/])(?:~|\.{1,2})?/(?:[\w.\-]+/)*[\w.\-]+/?")
_MARKUP_RE = re.compile(r"[*_`#]+|^\s*[-•]\s+", re.M)


def _speak_path(m: re.Match) -> str:
    path = m.group(0).rstrip("/")
    parts = [p for p in path.split("/") if p not in ("", "~", ".", "..")]
    if not parts:
        return "your home folder" if path == "~" else path
    name = parts[-1]
    if "." in name.strip("."):
        stem, ext = name.rsplit(".", 1)
        return f"{stem} dot {ext}"
    return f"the {name} folder"


def normalize(text: str) -> str:
    text = _MARKUP_RE.sub("", text)
    text = _URL_RE.sub(lambda m: f"a link on {(m.group(1) or m.group(2)).removeprefix('www.')}", text)
    text = _PATH_RE.sub(_speak_path, text)
    for rx, rep in _ABBREV:
        text = rx.sub(rep, text)
    text = _UNIT_RE.sub(lambda m: f"{m.group(1)} {_UNITS[m.group(2)]}", text)
    text = text.replace("&", " and ").replace("!", ".")
    text = re.sub(r"\s+([.,?;:])", r"\1", text)  # citation fragments leave "rivals ." behind
    return re.sub(r"\s{2,}", " ", text).strip()


def pause_markup(text: str, ellipsis_ms: int = 400, dash: str = "comma") -> str:
    """Turn the persona's pause markers into TTS pause controls.

    "…" becomes an explicit break. "—" becomes a comma by default: Cartesia warns that many breaks
    in quick succession split generation and can cause artefacts, and a comma already gives a
    short natural pause.
    """
    text = re.sub(r"\s*(?:…|\.\.\.)\s*", f'<break time="{ellipsis_ms}ms"/>', text)
    text = re.sub(r"\s*—\s*", ", " if dash == "comma" else '<break time="150ms"/>', text)
    # Never two breaks in a row, and never a break at the very start of an utterance.
    text = re.sub(r'(<break time="\d+ms"/>)+', lambda m: m.group(0)[: m.group(0).index("/>") + 2], text)
    return re.sub(r'^<break time="\d+ms"/>', "", text)
