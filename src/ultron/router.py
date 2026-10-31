"""Picks the model tier for a turn, locally and in microseconds.

Practical requests (timers, apps, volume, time, system state, short acknowledgements) go to the
fast model, where latency is the feature. Everything else is conversation and goes to the strong
model, where depth is. A misroute costs either a less clever reply or ~500 ms, never correctness,
so a few regexes beat a classifier call that would itself add a network round trip.
"""

from __future__ import annotations

import re

_PRACTICAL = re.compile(
    r"""
    \btimers?\b | \balarm\b | \bremind\ me\b
    | \b(open|launch|start|quit|close)\s+(the\s+)?[A-Z\w]
    | \bvolume\b | \b(louder|quieter|mute|unmute)\b
    | \bwhat\s+time\b | \bwhat'?s\s+the\s+(time|date)\b | \bwhat\s+day\b
    | \b(battery|cpu|disk\s+space)\b | \bsystem\s+stats\b
    | \b(memory|ram|storage)\s+(usage|use|left|free)\b | \bhow\s+much\s+(memory|ram|storage|space)\b
    | \b(make|take|save|add)\s+a\s+note\b | \bnote\s+(this|that)\b
    | \b(play|pause|resume|skip|next\s+track|previous\s+track|what'?s\s+playing)\b
    | \b(weather|forecast|temperature|raining|rain\s+(today|tomorrow)|umbrella)\b
    | \b(calendar|meetings?|agenda|appointments?|schedule\s+(today|tomorrow))\b
    | \b(find|search\s+for|look\s+for|where\s+is|open)\b.*\b(file|folder|document|pdf|notes?)\b
    | \b(read|show\s+me)\s+(the\s+|my\s+)?(file|notes?)\b
    | \bwho\s+won\b | \b(score|result)\s+of\b
    """,
    re.I | re.X,
)

# Whole-utterance acknowledgements and confirmations: nothing to think about.
_SHORT = re.compile(
    r"^(yes|yeah|yep|no|nope|ok(ay)?|sure|proceed|do\ it|cancel|stop|thanks?(\ you)?|cheers|"
    r"hi|hello|hey|good\ (morning|evening|night)|never\ ?mind)[\s.,!?]*(ultron)?[\s.,!?]*$",
    re.I | re.X,
)


def route(text: str, previous: str | None = None) -> str:
    """'fast' or 'strong'. previous: the tier of the turn just before, if it was recent. A
    follow-up in a conversation ("yes", "a to-do app", "why?") stays on the strong model; only a
    clearly practical request leaves it."""
    t = text.strip()
    if _PRACTICAL.search(t):
        return "fast"
    if previous == "strong":
        return "strong"
    return "fast" if _SHORT.match(t) else "strong"
