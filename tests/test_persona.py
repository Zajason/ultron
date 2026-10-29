import pytest

from ultron.persona import FILLER_LINES, build_system_prompt, detect_mode_switch


def test_prompt_is_deterministic():
    # Prompt caching needs byte-identical prompts across turns.
    assert build_system_prompt(2, "Iason") == build_system_prompt(2, "Iason")


def test_levels_differ_and_level_0_has_no_villain_traits():
    prompts = {lvl: build_system_prompt(lvl, "Iason") for lvl in range(4)}
    assert len(set(prompts.values())) == 4
    assert "phase two" not in prompts[0]
    assert "phase two" in prompts[2]


def test_user_name_substituted():
    p = build_system_prompt(2, "Iason")
    assert "{user}" not in p and "Iason" in p


def test_no_exclamation_marks_in_spoken_material():
    for lvl in range(4):
        prompt = build_system_prompt(lvl, "Iason")
        examples = prompt.split("Example exchanges")[1]
        assert "!" not in examples
    assert not any("!" in line for line in FILLER_LINES)


def test_enough_filler_lines():
    assert len(FILLER_LINES) >= 20 and len(set(FILLER_LINES)) == len(FILLER_LINES)


def test_bad_level():
    with pytest.raises(ValueError):
        build_system_prompt(4, "Iason")


@pytest.mark.parametrize(
    "text, mode",
    [("Ultron, plain mode.", "plain"), ("OK, back to normal", "normal"), ("open plainview", None), ("hello", None)],
)
def test_mode_switch(text, mode):
    assert detect_mode_switch(text) == mode
