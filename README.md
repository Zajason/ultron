# Ultron

A real-time, voice-first personal assistant for macOS (Apple Silicon). Talk to it, interrupt
it, wake it by name. It answers in an original, deep, unhurried voice with a dry villain's
wit, and acts on the computer through tools behind enforced safety tiers.

Private, non-commercial project. The voice is a stock Cartesia voice shaped with effects. No
real person's voice is cloned or imitated, and no film lines are used.

The full specification is in [ULTRON_BUILD_PROMPT.md](ULTRON_BUILD_PROMPT.md).

## Status

| Milestone | State |
|---|---|
| 1. Text loop: persona, streaming, tools, audit log | ✅ done |
| 2. Voice out: streaming TTS, normaliser, pause markup, FX, fillers | ✅ done |
| 3. Ears: mic, streaming STT, push-to-talk | ✅ done |
| 4. Real-time: hands-free, barge-in, echo cancellation, early start | ✅ built; 700 ms target not met (see [Latency](#latency)) |
| 5. Wake word: trained "Ultron" detector, sleep/wake, launchd | ✅ built; not yet tested on the owner's voice; service not installed |
| 6. Brains: escalation, memory, MCP, risky tools, weather/search/media/files/calendar | next |
| 7. Optimise: local-model experiment for practical turns, buffers | planned |
| 8. Polish: tray indicator, Linux, install docs | planned |

Linux support is deferred. The code has platform seams for it, but only the macOS side is built.

## Setup

Requirements: macOS on Apple Silicon, Python 3.13, [uv](https://docs.astral.sh/uv/)
(`brew install uv`), plus API keys for Anthropic, Cartesia (a paid plan; the free tier runs out
quickly) and Deepgram.

```bash
uv sync
cp .env.example .env    # then add ANTHROPIC_API_KEY, CARTESIA_API_KEY, DEEPGRAM_API_KEY
uv run ultron mic-test  # check the microphone end to end
uv run ultron live --wake
```

The first run asks for microphone access for your terminal app.

## Commands

| Command | What it does |
|---|---|
| `ultron live` | Hands-free conversation: talk any time; speak over him to interrupt |
| `ultron live --wake` | Sleeps until it hears "Ultron"; sleeps again after 10 s of quiet |
| `ultron talk` | Push-to-talk: Enter to start, Enter to send |
| `ultron text [--mute]` | Typed chat, spoken aloud unless muted. `/menace 0-3`, `/reset`, `/quit` |
| `ultron mic-test` | Records 4 s, plays it back, transcribes it (device, level, STT) |
| `ultron say "…"` | Speak one line |
| `ultron audition [voices] --preset … --emotion …` | Compare voices and FX presets; WAVs go to `~/.ultron/audition/` |
| `ultron bench-llm`, `bench-tts`, `bench-stt`, `bench-v2v` | Latency percentiles per stage, and voice-to-voice |

Use `--device NAME` on `live`, `talk` and `mic-test` to pick a microphone by name or part of
one. `--config-dir DIR` goes before the command.

Scripts:
- `scripts/train_wakeword.py`: generate data, record your own samples, train and evaluate the
  wake word.
- `scripts/install_macos.sh install|uninstall|status|logs`: run as a launchd login service.
- `scripts/make_utterances.py`: rebuild the replay-harness utterances.

## How it works

```
mic ─► downsample 16 kHz ─► echo cancel ─► wake word (asleep) ─► VAD (barge-in) ─► Deepgram Flux
                                                                                       │ turn end / early turn end
speaker ◄─ FX ◄─ resample/varispeed ◄─ Cartesia (websocket) ◄─ chunker ◄─ Claude ◄─────┘
                                                                            │ Haiku: practical · Sonnet: conversation
                                                                       tools (tiered, audited)
```

- **Orchestration** is plain asyncio, not Pipecat. Pipecat's only local audio transport is
  PyAudio with blocking writes and no echo cancellation, and the latency work here depends on
  owning the audio path (native-rate output, DAC timing, Bluetooth handling). The reasoning is
  in the Milestone 4 discussion.
- **Brain** ([brain.py](src/ultron/brain.py)): streaming Claude with prompt caching. The local
  time is stamped on each user message so the cached prefix stays byte-identical. History
  trimming, tool loop, checkpoint and rollback for speculative turns, and history repair when a
  turn is cancelled.
- **Routing** ([router.py](src/ultron/router.py)): a microsecond regex picks Haiku for practical
  requests and Sonnet for conversation.
- **Persona** ([persona.py](src/ultron/persona.py)): a reasoned worldview offset by a sincere
  love of art and ideas, with intensity `menace_level` 0–3. "Ultron, plain mode" drops the
  theatrics without changing the cached prompt.
- **Conversation** ([conversation.py](src/ultron/conversation.py)):
  - Flux decides when you've finished; Silero VAD stops Ultron when you talk over him.
  - Replies start speculatively on Flux's early end-of-turn. Audio is held and tools gated until
    the turn is confirmed; if you keep talking, the draft is rolled back.
  - When you interrupt, Cartesia's word timestamps record exactly what you heard.
- **Voice** ([audio/](src/ultron/audio)):
  - Carl voice, 1.5 semitones lower via zero-latency varispeed.
  - Close-mic compression, light rasp, an octave-down "sub" layer and a faint ring-mod/chorus
    layer.
  - About 0.1 ms of CPU per 20 ms chunk.
  - 22 filler lines are pre-rendered and play when a tool call starts before any speech.
- **Wake word** ([wakeword.py](src/ultron/wakeword.py)): openWakeWord's frozen mel and
  speech-embedding models on onnxruntime, plus our own classifier. While asleep nothing is sent
  to Deepgram. On detection the buffered audio is sent, and the first transcript must contain
  "Ultron" or Ultron goes back to sleep.
- **Tools** ([tools/](src/ultron/tools)): tiers are enforced by the executor, not the prompt.
  - Tier 0 (read) and tier 1 (low-risk) run automatically.
  - Tier 2 asks for confirmation.
  - Tier 3 is disabled.
  - Every call goes to `~/.ultron/audit.jsonl`.
  - Benches run tools in dry-run mode.

## Latency

Measured from Athens on an M4 Pro, MacBook speakers. Voice-to-voice means from the end of
speech to the first audible sample.

| Stage | Budget | Measured (p50) |
|---|---|---|
| End of turn (Flux, EU endpoint) | 150–300 ms | 73 ms early / 137–199 ms final |
| Final transcript, push-to-talk release | ≤ 150 ms | 151 ms |
| Claude Haiku time to first token | 200–400 ms | ~600–830 ms (independent of prompt size) |
| First token → first speakable clause | — | ~100 ms |
| Cartesia first audio byte | ≤ 200 ms | 83–106 ms |
| Output buffering | ≤ 50 ms | ~38 ms built-in speakers, ~180 ms Bluetooth |
| Barge-in: Ultron stops | ≤ 150 ms | ~140 ms (estimated from parts) |
| **Voice-to-voice, Haiku turns** | **≤ 700 ms** | **~1.3–1.4 s** |
| Voice-to-voice, Sonnet turns | — | ~1.7–1.9 s |
| First sound on tool turns (filler) | ~700 ms | ~765 ms |

Claude's time to first token is the floor. A local model for practical turns is the planned
Milestone 7 experiment.

Wake word, on held-out data: 7 unseen voices and 2.1 h of real-world audio, at threshold 0.8:

| Measure | Result |
|---|---|
| Recall | 93.6% |
| False wakes | 0.47 per hour |
| Look-alike false accepts | 1.3% |
| CPU | 1.5% of one core |

Logs:
- `~/.ultron/turns.jsonl`: per-turn stage timings
- `~/.ultron/errors.log`: errors
- `~/.ultron/audit.jsonl`: every tool call

## Configuration

Everything lives in [config.toml](config.toml). Platform overrides go in `config.macos.toml` and
per-host overrides in `config.<hostname>.toml`. The main knobs:

- `[llm]`: `routing` (hybrid / fast / strong), models, thinking, effort, cache TTL, history length
- `[tts]`: voice, `speed`, `emotion`, `pitch_semitones`; `[fx]` for effect levels
- `[stt]`: region (EU by default), turn-detection thresholds, `eager_eot_threshold`
- `[audio]`: input and output device, Bluetooth-mic avoidance, `aec` (auto / on / off)
- `[vad]`, `[wake]` (threshold, idle sleep, chime), `[tools]` (tiers, timeouts)

## Troubleshooting

- **"(nothing heard)" or "pure silence"**:
  - With the MacBook lid closed, the built-in mic is switched off. Ultron detects this and
    picks another wired mic.
  - Otherwise, check System Settings → Privacy & Security → Microphone for your terminal app.
  - `ultron mic-test` shows which stage fails.
- **Bluetooth headphones (AirPods, Sony XM5)**: their mic forces a low-quality call mode and
  breaks open audio streams (PortAudio `err=-50`). By default Ultron uses a wired or built-in
  mic and keeps the headphones in music mode. Turn off Sony's Speak-to-Chat, or it mutes Ultron
  whenever you talk.
- **Cartesia `402 quota_exceeded`**: out of credits. Top up, or enable overages at
  play.cartesia.ai.
- **He hears himself on open speakers**: echo cancellation is on for non-Bluetooth output. It's
  verified on synthetic echo but not yet in a real room. Please report.

## Development

```bash
uv run pytest -q    # 132 tests, no network needed
uv run ultron bench-v2v --rounds 2    # end-to-end latency via the replay harness (uses APIs)
```

The replay harness streams the WAVs in [tests/fixtures/utterances](tests/fixtures/utterances)
in real time, in place of the mic.

Third-party models and their licences are listed in
[src/ultron/models/NOTICE](src/ultron/models/NOTICE).
