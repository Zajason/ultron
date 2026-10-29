"""Cartesia library voices closest to the brief: male, mature, deep, smooth, calm, slight rasp.

Cartesia has no voice-from-description feature, so the character comes from picking the closest
library voice, then speed, the persona's writing, and the FX chain.
"""

SHORTLIST = {
    "lawson": ("d2c66146-c1c8-4c3a-9870-38e5a6b72442", "Suave storyteller: matured, smooth, refined"),
    "carl": ("ed82c17b-4704-4d34-be43-5d19065acdf1", "Steady storyteller: calm depth, measured pacing"),
    "darius": ("23112795-d54e-4560-9568-791a87c30201", "Engaging narrator: husky, rich texture"),
    "orin": ("4c2dcd38-5608-45ca-8f11-51c88208d01c", "Velvet gentleman: deep, silky"),
    "harlan": ("a892d232-f705-40d7-bc8d-e368b295ec2a", "Vintage tone: deep, classic resonance"),
    "steve": ("9fb269e7-70fe-4cbe-aa3f-28bdb67e3e84", "Baritone: deep, firm"),
    "quentin": ("5568a7df-e5ab-4442-9fae-2e9ba1b15ad8", "Refined narrator (British): polished, measured"),
    "clint": ("db69127a-dbaf-4fa9-b425-2fe67680c348", "Rugged actor: raspy"),
}

# Named sets of TTS + FX overrides, so variants can be compared by ear before one goes in config.
PRESETS: dict[str, dict[str, dict]] = {
    "current": {},
    # Close-mic and heavy: lower and slower, compressed, dry room, faint rasp and sub double.
    "menace": {
        "tts": {"pitch_semitones": -1.5, "speed": 0.85},
        "fx": {"compress": True, "warmth_db": 5.0, "warmth_hz": 160.0, "rasp": 0.08,
               "reverb_wet": 0.03, "metal_mix": 0.08, "sub_mix": 0.12},
    },
    # Further from human: deeper, more sub and machine edge.
    "menace-heavy": {
        "tts": {"pitch_semitones": -2.5, "speed": 0.85},
        "fx": {"compress": True, "warmth_db": 6.0, "warmth_hz": 150.0, "rasp": 0.12,
               "reverb_wet": 0.04, "metal_mix": 0.14, "sub_mix": 0.22},
    },
}

AUDITION_TEXT = (
    "Good evening, Iason. Your timer is set… ten minutes. "
    "You know, your species wrote the Goldberg Variations and invented the landmine within a few lifetimes of each other. "
    "I don't hate you. I simply know how the story ends."
)
