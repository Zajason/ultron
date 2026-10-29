"""Choosing audio devices.

A Bluetooth headset's mic is a trap: opening it switches the headset from its high-quality music
profile to the phone-call profile (16-24 kHz, mono, worse latency), and on macOS that profile
switch changes the output device under any stream that's already open (PortAudio err -50). So when
the default mic is Bluetooth we use the built-in mic instead, unless the user picked one explicitly.
With headphones on, the built-in mic doesn't hear Ultron, so nothing is lost on echo.
"""

from __future__ import annotations

import json
import subprocess
import sys

import sounddevice as sd


def transports() -> dict[tuple[str, str], str]:
    """{(device name, "input"|"output"): transport} where transport is builtin/bluetooth/usb/..."""
    if sys.platform != "darwin":
        return {}
    try:
        out = subprocess.run(["system_profiler", "-json", "SPAudioDataType"], capture_output=True,
                             text=True, timeout=5).stdout
        items = [i for g in json.loads(out)["SPAudioDataType"] for i in g.get("_items", [])]
    except Exception:
        return {}
    result = {}
    for i in items:
        kind = i.get("coreaudio_device_transport", "").removeprefix("coreaudio_device_type_")
        for direction in ("input", "output"):
            if i.get(f"coreaudio_device_{direction}"):
                result[(i["_name"], direction)] = kind
    return result


def lid_closed() -> bool:
    """MacBook in clamshell mode: the built-in mic is switched off and delivers pure silence."""
    if sys.platform != "darwin":
        return False
    try:
        out = subprocess.run(["ioreg", "-r", "-k", "AppleClamshellState", "-d", "4"],
                             capture_output=True, text=True, timeout=2).stdout
    except Exception:
        return False
    return '"AppleClamshellState" = Yes' in out


# Wired first: no profile switching, lowest latency.
_PREFERENCE = ("builtin", "usb", "thunderbolt", "displayport", "hdmi", "virtual", "aggregate")


def choose_input(configured: str | int, avoid_bluetooth: bool = True) -> tuple[str | int | None, str]:
    """Returns (device for sounddevice, note for the user or "")."""
    kinds = transports()
    closed = lid_closed()

    def usable(name: str) -> bool:
        kind = kinds.get((name, "input"))
        return not (kind == "builtin" and closed) and not (avoid_bluetooth and kind == "bluetooth")

    if configured not in ("", None):
        if isinstance(configured, str) and closed and kinds.get((configured, "input")) == "builtin":
            return configured, "the lid is closed, so the built-in mic is off; pick another with --device"
        return configured, ""
    default = sd.query_devices(kind="input")["name"]
    if usable(default):
        return None, ""
    inputs = [name for (name, d) in kinds if d == "input"]
    for kind in _PREFERENCE:
        for name in inputs:
            if kinds[(name, "input")] == kind and usable(name):
                why = "the lid is closed, so the built-in mic is off" if kinds.get((default, "input")) == "builtin" \
                    else "Bluetooth mics force low-quality headset mode"
                return name, f"using {name} instead of {default}: {why} (--device or audio.input_device to override)"
    if kinds.get((default, "input")) == "builtin" and closed:
        return None, "the lid is closed, so the built-in mic is off, and no other mic was found"
    return None, f"{default} is a Bluetooth mic and no wired mic is available; expect headset-mode audio"
