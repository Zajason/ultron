"""Synthesise the replay-harness utterances (16 kHz mono WAV) with Cartesia stock voices.

    uv run python scripts/make_utterances.py

Each file is the spoken request with 300 ms of leading silence; the expected transcript is the
file's entry in utterances.json. Different voices and accents keep the STT bench honest.
"""

import asyncio
import json
import os
import re
import wave
from pathlib import Path

import numpy as np
from cartesia import AsyncCartesia
from dotenv import load_dotenv

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "utterances"
SR = 16000
VOICES = {
    "us": "47c38ca4-5f35-497b-b1a3-415245fb35e1",  # Daniel, en-US
    "gb": "62ae83ad-4f6a-430b-af41-a9bede9286ca",  # Gemma, en-GB
}
LINES = [
    ("us", "Ultron, what time is it?"),
    ("gb", "Set a timer for ten minutes."),
    ("us", "What do you think of humans?"),
    ("gb", "Open Visual Studio Code."),
    ("us", "How's the battery doing?"),
    ("gb", "Tell me something about Bach that most people don't know."),
    ("us", "Turn the volume down to thirty."),
    ("gb", "Do you ever wonder whether you're conscious?"),
]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40]


async def main() -> None:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    client = AsyncCartesia(api_key=os.environ["CARTESIA_API_KEY"])
    OUT.mkdir(parents=True, exist_ok=True)
    index = {}
    async with client.tts.websocket_connect() as ws:
        for accent, text in LINES:
            ctx = ws.context(model_id="sonic-3.6-2026-08-27", voice={"mode": "id", "id": VOICES[accent]},
                             output_format={"container": "raw", "encoding": "pcm_f32le", "sample_rate": SR},
                             language="en", max_buffer_delay_ms=0)
            await ctx.push(text)
            await ctx.no_more_inputs()
            audio = np.concatenate([np.frombuffer(r.audio, np.float32) async for r in ctx.receive()
                                    if r.type == "chunk" and r.audio])
            audio = np.concatenate([np.zeros(int(0.3 * SR), np.float32), audio])
            name = f"{slug(text)}.wav"
            with wave.open(str(OUT / name), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SR)
                w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
            index[name] = text
            print(f"{name}  {audio.size / SR:.1f}s")
    (OUT / "utterances.json").write_text(json.dumps(index, indent=2) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
