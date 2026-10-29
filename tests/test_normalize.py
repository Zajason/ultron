import pytest

from ultron.text.normalize import normalize, pause_markup


@pytest.mark.parametrize(
    "src, out",
    [
        ("About 3.5GB free.", "About 3.5 gigs free."),
        ("Disk is 80% full", "Disk is 80 percent full"),
        ("It's 14°C outside.", "It's 14 degrees outside."),
        ("Ping was 120 ms.", "Ping was 120 milliseconds."),
        ("Doing 50 km/h.", "Doing 50 kilometres an hour."),
        ("Saved to /Users/zak/projects/ultron/src/main.py", "Saved to main dot py"),
        ("Check ~/projects/ultron/build/", "Check the build folder"),
        ("See https://www.example.com/a/b?c=1 for details", "See a link on example.com for details"),
        ("Use **this** one", "Use this one"),
        ("Fast, e.g. Haiku.", "Fast, for example Haiku."),
        ("Tea & biscuits", "Tea and biscuits"),
        ("Done!", "Done."),
        # Untouched: plain numbers, times, version-like strings, words that look like units.
        ("It's 18:40 on the 29th.", "It's 18:40 on the 29th."),
        ("Python 3.13 is installed", "Python 3.13 is installed"),
        ("The MS in his name", "The MS in his name"),
        ("and/or", "and/or"),
    ],
)
def test_normalize(src, out):
    assert normalize(src) == out


def test_pause_markup_ellipsis_and_dash():
    assert pause_markup("Ten minutes… Done.") == 'Ten minutes<break time="400ms"/>Done.'
    assert pause_markup("Waiting... nothing.") == 'Waiting<break time="400ms"/>nothing.'
    assert pause_markup("A body — made of speakers") == "A body, made of speakers"


def test_pause_markup_no_double_or_leading_break():
    assert pause_markup("… Hm…… yes") == 'Hm<break time="400ms"/>yes'
