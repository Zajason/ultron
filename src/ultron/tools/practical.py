"""Everyday tools: weather, media, finding and reading files, calendar.

All tier 0 (read) or tier 1 (low-risk). Web search is Anthropic's server-side tool and lives in
the brain's tool list, not here.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import re
from pathlib import Path

import httpx

from ..platform.base import CommandError, run
from .registry import Registry, Tier, Tool

# WMO weather codes (Open-Meteo), worded for speech.
_WMO = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains", 80: "light showers", 81: "showers",
    82: "violent showers", 85: "snow showers", 86: "heavy snow showers", 95: "thunderstorms",
    96: "thunderstorms with hail", 99: "thunderstorms with heavy hail",
}

# Never read these, whatever the model asks: credentials and keys.
_SECRET_PARTS = {".ssh", ".gnupg", ".aws", "Keychains", ".netrc", ".env", ".git-credentials", "id_rsa", "id_ed25519",
                 ".config/gh", ".docker/config.json", "Cookies", "Login Data"}


def _is_secret(p: Path) -> bool:
    s = str(p)
    return any(part in s.split("/") or s.endswith(part) or f"/{part}/" in s for part in _SECRET_PARTS)


def register_practical(reg: Registry, default_location: str, http: httpx.AsyncClient | None = None,
                       read_limit: int = 12000) -> None:
    client = http or httpx.AsyncClient(timeout=8.0)
    places: dict[str, dict] = {}

    async def _place(name: str) -> dict:
        key = name.strip().lower()
        if key not in places:
            r = await client.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": name, "count": 1})
            r.raise_for_status()
            hits = r.json().get("results") or []
            if not hits:
                raise ValueError(f"no place called {name!r}")
            places[key] = hits[0]
        return places[key]

    async def weather(location: str = "", when: str = "now") -> str:
        place = await _place(location or default_location)
        r = await client.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": place["latitude"], "longitude": place["longitude"], "timezone": "auto",
            "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m,precipitation",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "forecast_days": 7,
        })
        r.raise_for_status()
        d = r.json()
        where = f"{place['name']}, {place.get('country', '')}".strip(", ")
        if when == "now":
            c = d["current"]
            return (f"{where} now: {_WMO.get(c['weather_code'], 'unknown')}, {c['temperature_2m']:.0f} °C "
                    f"(feels like {c['apparent_temperature']:.0f}), wind {c['wind_speed_10m']:.0f} km/h")
        days = d["daily"]
        idx = {"today": [0], "tomorrow": [1], "week": range(7)}.get(when, [0])
        out = []
        for i in idx:
            day = dt.date.fromisoformat(days["time"][i]).strftime("%A")
            out.append(f"{day}: {_WMO.get(days['weather_code'][i], 'unknown')}, {days['temperature_2m_min'][i]:.0f} to "
                       f"{days['temperature_2m_max'][i]:.0f} °C, {days['precipitation_probability_max'][i]}% chance of rain")
        return f"{where}. " + "; ".join(out)

    async def media(action: str) -> str:
        app = await _media_app()
        if app is None:
            return "neither Spotify nor Music is running"
        if action == "now_playing":
            script = (f'tell application "{app}" to if player state is playing then '
                      f'return (name of current track) & " by " & (artist of current track)')
            playing = await run("osascript", "-e", script)
            return f"{app}: {playing}" if playing else f"{app} is paused"
        verbs = {"play": "play", "pause": "pause", "toggle": "playpause", "next": "next track", "previous": "previous track"}
        await run("osascript", "-e", f'tell application "{app}" to {verbs[action]}')
        return f"{app}: {action} done"

    async def search_files(query: str, limit: int = 8) -> str:
        # Every word must appear in the file name, any case: "build prompt" finds BUILD_PROMPT.md.
        # kMDItemFSName is ~3x faster than kMDItemDisplayName (0.97 vs 2.6 s on this Mac).
        home = str(Path.home())
        words = [w for w in re.split(r"[\s_\-.]+", query) if w]
        expr = " && ".join(f"kMDItemFSName == '*{_quote(w)}*'c" for w in words) or "kMDItemFSName == '*'"
        out = await run("mdfind", "-onlyin", home, expr, timeout=8)
        paths = [p for p in out.splitlines() if p and "/Library/" not in p and "/." not in p][:limit]
        if not paths:
            return f"no files named like {query!r}"
        return "\n".join(p.replace(home, "~", 1) for p in paths)

    async def read_file(path: str) -> str:
        p = Path(path).expanduser().resolve()
        if not p.is_relative_to(Path.home()):
            return "only files inside your home folder can be read"
        if _is_secret(p):
            return "that file holds credentials; refusing to read it"
        if not p.is_file():
            return f"{path} is not a file"
        data = await asyncio.to_thread(p.read_bytes)
        try:
            text = data[: read_limit * 4].decode("utf-8")
        except UnicodeDecodeError:
            return f"{p.name} is not a text file ({len(data)} bytes)"
        more = f"\n… [{len(text) - read_limit} more characters]" if len(text) > read_limit else ""
        return text[:read_limit] + more

    async def calendar(day: str = "today") -> str:
        start = dt.date.today() + dt.timedelta(days={"today": 0, "tomorrow": 1}.get(day, 0))
        if day not in ("today", "tomorrow"):
            start = dt.date.fromisoformat(day)
        script = f'''
        set d0 to current date
        set year of d0 to {start.year}
        set month of d0 to {start.month}
        set day of d0 to {start.day}
        set time of d0 to 0
        set d1 to d0 + (1 * days)
        set out to ""
        tell application "Calendar"
            repeat with c in calendars
                repeat with e in (every event of c whose start date ≥ d0 and start date < d1)
                    set out to out & (time string of (start date of e)) & " | " & (summary of e) & linefeed
                end repeat
            end repeat
        end tell
        return out'''
        try:
            out = await run("osascript", "-e", script, timeout=15)
        except CommandError as e:
            if "Not authorized" in str(e) or "-1743" in str(e):
                return "no permission to read Calendar: allow it under Privacy & Security > Automation"
            raise
        events = sorted(line for line in out.splitlines() if line.strip())
        when = start.strftime("%A %d %B")
        return f"{when}: " + ("; ".join(events) if events else "nothing scheduled")

    for tool in [
        Tool("weather", "Current weather or forecast. Leave location empty for the user's home city.",
             {"location": {"type": "string", "description": "City, empty for home"},
              "when": {"type": "string", "enum": ["now", "today", "tomorrow", "week"]}},
             (), Tier.READ, weather),
        Tool("media", "Control music playback in Spotify or Apple Music, or say what's playing.",
             {"action": {"type": "string", "enum": ["play", "pause", "toggle", "next", "previous", "now_playing"]}},
             ("action",), Tier.LOW_RISK, media),
        Tool("search_files", "Find files and folders in the user's home folder by name (Spotlight). Every word "
             "must appear in the name, in any case and order; search once with the key words.",
             {"query": {"type": "string", "description": "Key words of the name, e.g. 'build prompt'"}},
             ("query",), Tier.READ, search_files),
        Tool("read_file", "Read a text file inside the user's home folder.",
             {"path": {"type": "string", "description": "Path, e.g. ~/Documents/plan.md"}},
             ("path",), Tier.READ, read_file),
        Tool("calendar", "Events on the user's calendars for one day.",
             {"day": {"type": "string", "description": "today, tomorrow or YYYY-MM-DD"}},
             (), Tier.READ, calendar),
    ]:
        reg.add(tool)


def _quote(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")


async def _media_app() -> str | None:
    for app in ("Spotify", "Music"):
        running = await run("osascript", "-e", f'application "{app}" is running')
        if running == "true":
            return app
    return None
