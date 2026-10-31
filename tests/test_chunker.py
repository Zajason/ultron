from ultron.text import Chunker


def run(text: str, step: int = 1, **kw) -> list[str]:
    c = Chunker(**kw)
    out = []
    for i in range(0, len(text), step):
        out += c.feed(text[i : i + step])
    if (rest := c.flush()) is not None:
        out.append(rest)
    return out


def test_sentences():
    assert run("Opening it. Another human building another machine. ") == [
        "Opening it.",
        "Another human building another machine.",
    ]


def test_ellipsis_is_a_cut_point():
    assert run("Ten minutes… Done. A blink, for me.") == ["Ten minutes…", "Done.", "A blink, for me."]


def test_three_dots_ellipsis():
    assert run("Waiting, thinking... nothing at all.") == ["Waiting, thinking...", "nothing at all."]


def test_tiny_first_ellipsis_waits_for_more_text():
    # Under first_min_chars, a pause marker alone isn't worth a separate TTS request.
    assert run("Hm… let me look.") == ["Hm… let me look."]


def test_decimal_and_abbreviations_not_split():
    assert run("It is 3.5 GB, e.g. the build folder. Done.") == ["It is 3.5 GB,", "e.g. the build folder.", "Done."]


def test_first_chunk_may_be_a_clause_later_ones_prefer_sentences():
    out = run("Fourteen degrees and light rain, bring a jacket, your species is fragile, as always. Fine.")
    assert out[0] == "Fourteen degrees and light rain,"
    assert out[1] == "bring a jacket, your species is fragile, as always."


def test_boundary_waits_for_next_char():
    c = Chunker()
    assert c.feed("Done.") == []
    assert c.feed(" Next") == ["Done."]


def test_long_text_without_punctuation_is_cut_at_space():
    out = run("word " * 100, step=7, max_chars=50)
    assert all(len(x) <= 50 for x in out)
    assert " ".join(out).split() == ["word"] * 100


def test_chunk_size_independent_of_delta_size():
    text = "One moment… Accessing it now. Patience, I'm working with human infrastructure. Done."
    assert run(text, step=1) == run(text, step=5) == run(text, step=len(text))


def test_first_chunk_word_cap_starts_speech_early_only_once():
    text = "The timer is set for ten minutes. Tea awaits, as do the rest of your obligations."
    out = run(text, step=3, first_max_words=4)
    assert out[0] == "The timer is set"
    assert out[1] == "for ten minutes."
    assert run(text, step=1, first_max_words=4) == out


def test_first_chunk_word_cap_never_splits_a_short_sentence():
    assert run("Done. Ten minutes.", step=1, first_max_words=4) == ["Done.", "Ten minutes."]
