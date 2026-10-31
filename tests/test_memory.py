from test_brain import make_brain, reply, text

from ultron.memory import Memory


def test_add_forget_and_persist(tmp_path):
    f = tmp_path / "memory.md"
    m = Memory(f)
    assert m.add("Iason prefers his coffee black") == "remembered"
    assert m.add("iason prefers his coffee black") == "already known"
    m.add("Iason's sister is called Eleni")
    assert Memory(f).facts == m.facts  # survives a restart, plain markdown
    assert m.forget("coffee") == ["Iason prefers his coffee black"]
    assert m.forget("nonexistent thing") == []
    assert Memory(f).facts == ["Iason's sister is called Eleni"]


async def test_memory_and_summary_follow_the_cached_persona(tmp_path):
    brain, client = make_brain(tmp_path, [reply(text("Noted.")), reply(text("Yes."))])
    brain.memory = Memory(tmp_path / "m.md")
    await brain.respond("hello")
    persona_block = client.calls[0]["system"][0]
    assert len(client.calls[0]["system"]) == 1  # nothing remembered yet

    brain.memory.add("Iason's sister is called Eleni")
    brain.summary = "They discussed Kant."
    await brain.respond("who is my sister?")
    system = client.calls[1]["system"]
    assert system[0] == persona_block and "cache_control" in system[0]  # persona cache untouched
    assert "Eleni" in system[1]["text"] and "Kant" in system[1]["text"]
