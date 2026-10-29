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


def choose_input(configured: str | int, avoid_bluetooth: bool = True) -> tuple[str | int | None, str]:
    """Returns (device for sounddevice, note for the user or "")."""
    if configured not in ("", None):
        return configured, ""
    default = sd.query_devices(kind="input")["name"]
    if not avoid_bluetooth:
        return None, ""
    kinds = transports()
    if kinds.get((default, "input")) != "bluetooth":
        return None, ""
    builtin = [name for (name, d), k in kinds.items() if d == "input" and k == "builtin"]
    if not builtin:
        return None, f"{default} is a Bluetooth mic and no built-in mic was found; expect headset-mode audio"
    return builtin[0], (f"using {builtin[0]} instead of {default}: Bluetooth mics force low-quality "
                        f"headset mode (set audio.input_device to override)")
