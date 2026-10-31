import pytest

from ultron.router import route

FAST = [
    "Set a timer for ten minutes.",
    "How long left on the timer?",
    "Open VS Code",
    "open spotify",
    "Turn the volume down",
    "What time is it?",
    "what's the date",
    "How's the battery?",
    "Make a note: buy coffee",
    "pause",
    "Yes.",
    "proceed",
    "Thanks, Ultron.",
    "Good morning",
    "never mind",
    "How much memory is free?",
    "What's the weather tomorrow?",
    "Is it going to rain tomorrow?",
    "What's on my calendar today?",
    "Do I have any meetings tomorrow?",
    "Find the build prompt file in my projects.",
    "Who won the most recent Formula 1 race?",
    "What's playing?",
]
STRONG = [
    "What do you think of humans?",
    "Do you think you're conscious?",
    "I just finished Crime and Punishment.",
    "Why do you even help me?",
    "I had a pretty bad day.",
    "What's the most beautiful thing humans have made?",
    "Do you think time is real?",
    "Is it worth starting over at thirty?",
    "Tell me about Bach.",
    "What do you think about memory, and why humans romanticise it?",
    "Why do people find rain so melancholic?",
    "Good morning. I couldn't sleep, I kept thinking about whether any of this matters.",
]


@pytest.mark.parametrize("text", FAST)
def test_practical_goes_fast(text):
    assert route(text) == "fast"


@pytest.mark.parametrize("text", STRONG)
def test_conversation_goes_strong(text):
    assert route(text) == "strong"


def test_follow_ups_stay_with_the_strong_model():
    assert route("yes", previous="strong") == "strong"
    assert route("a to-do app", previous="strong") == "strong"
    assert route("set a timer for five minutes", previous="strong") == "fast"
    assert route("yes", previous="fast") == "fast"
    assert route("yes") == "fast"
