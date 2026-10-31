"""Practical tools against fake HTTP and fake commands (no network, no AppleScript)."""

import httpx
import pytest

from ultron.tools import Registry
from ultron.tools import practical


def make(handler):
    reg = Registry()
    practical.register_practical(reg, "Athens", http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return reg


def meteo(request):
    if "geocoding" in request.url.host:
        return httpx.Response(200, json={"results": [{"name": "Athens", "country": "Greece", "latitude": 38.0, "longitude": 23.7}]})
    return httpx.Response(200, json={
        "current": {"temperature_2m": 19.2, "apparent_temperature": 18.1, "weather_code": 3, "wind_speed_10m": 7.0, "precipitation": 0},
        "daily": {"time": ["2026-10-31", "2026-11-01"], "weather_code": [3, 95], "temperature_2m_max": [20, 18],
                  "temperature_2m_min": [15, 14], "precipitation_probability_max": [10, 88]},
    })


async def test_weather_now_and_tomorrow():
    reg = make(meteo)
    assert (await reg.get("weather").handler()) == "Athens, Greece now: overcast, 19 °C (feels like 18), wind 7 km/h"
    out = await reg.get("weather").handler(when="tomorrow")
    assert "Sunday: thunderstorms, 14 to 18 °C, 88% chance of rain" in out


async def test_unknown_place():
    reg = make(lambda r: httpx.Response(200, json={}))
    with pytest.raises(ValueError):
        await reg.get("weather").handler(location="Atlantis")


async def test_read_file_guards(tmp_path, monkeypatch):
    monkeypatch.setattr(practical.Path, "home", lambda: tmp_path)
    (tmp_path / ".ssh").mkdir()
    (tmp_path / ".ssh" / "config").write_text("secret")
    (tmp_path / "notes.txt").write_text("hello")
    (tmp_path / "blob.bin").write_bytes(b"\xff\xfe\x00")
    read = make(meteo).get("read_file").handler
    assert await read(str(tmp_path / "notes.txt")) == "hello"
    assert "credentials" in await read(str(tmp_path / ".ssh" / "config"))
    assert "not a text file" in await read(str(tmp_path / "blob.bin"))
    assert "home folder" in await read("/etc/hosts")


async def test_search_files_matches_every_word(monkeypatch):
    seen = []

    async def fake_run(*argv, timeout=5.0):
        seen.append(argv)
        return f"{practical.Path.home()}/projects/ultron/ULTRON_BUILD_PROMPT.md"

    monkeypatch.setattr(practical, "run", fake_run)
    out = await make(meteo).get("search_files").handler("build prompt")
    assert out == "~/projects/ultron/ULTRON_BUILD_PROMPT.md"
    assert "*build*" in seen[0][-1] and "*prompt*" in seen[0][-1] and "&&" in seen[0][-1]


async def test_hand_off_copies_prompt_and_opens_app(monkeypatch):
    from ultron.tools import handoff

    copied, opened = [], []

    async def fake_copy(text):
        copied.append(text)

    async def fake_run(*argv, timeout=5.0):
        opened.append(argv)
        return ""

    monkeypatch.setattr(handoff, "_copy", fake_copy)
    monkeypatch.setattr(handoff, "run", fake_run)
    reg = Registry()
    handoff.register_handoff(reg)
    out = await reg.get("hand_off").handler("Plan a 10-day trip to Japan in April.", app="chatgpt")
    assert copied == ["Plan a 10-day trip to Japan in April."]
    assert opened == [("open", "-a", "ChatGPT")]
    assert "clipboard" in out
