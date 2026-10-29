"""Train the "Ultron" wake-word classifier.

    uv run --group train python scripts/train_wakeword.py all        # generate, features, train, eval
    uv run --group train python scripts/train_wakeword.py record     # optional: record yourself (big win)
    uv run --group train python scripts/train_wakeword.py train eval # retrain from cached features

Only a small classifier is trained; the mel and speech-embedding models are openWakeWord's frozen
feature extractor (src/ultron/wakeword.py). Data:
  positives   "Ultron" (+ "hey/okay Ultron", "Ultron?") from macOS voices, Cartesia voices and your
              own recordings, each placed at many positions and heavily augmented
  look-alikes words that sound close ("ultra", "neutron", "Elton", ...), same voices/augmentation
  generic     openWakeWord's validation_set_features (~10.7 h of real-world audio), 80% for training,
              the last 20% held out to measure false wake-ups per hour
Voices are split by speaker: evaluation positives come from voices the classifier never saw.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures as cf
import json
import os
import random
import re
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ultron.audio.resample import Resampler  # noqa: E402
from ultron.wakeword import CHUNK, MODELS, WINDOW, Features, StreamingFeatures, stream_embed  # noqa: E402

WORK = Path("~/.ultron/wakeword").expanduser()
CLIPS = WORK / "clips"
NEG_FEATURES = Path("~/.ultron/data/validation_set_features.npy").expanduser()
SR = 16000
CLIP_S = 4.08  # per example: 2 s lead-in (the mel model's memory, as live) + the 2.08 s it scores

POSITIVE = ["Ultron", "Ultron.", "Ultron?", "Hey Ultron", "Okay Ultron", "Ultron!"]
LOOKALIKE = ["ultra", "Altron", "neutron", "electron", "Elton", "ultimate", "alternate", "patron",
             "Akron", "Oberon", "ultrasound", "all done", "old one", "multi", "Ulster", "until",
             "Austin", "outrun", "ultimatum", "Voltron", "Ultra-Tron", "and then", "hold on", "Saturn",
             "a little", "cauldron", "Ultimately", "Hey Alton", "Okay, Altman", "Elton John"]
MAC_SKIP = {"Bad News", "Bahh", "Bells", "Boing", "Bubbles", "Cellos", "Wobble", "Good News", "Jester",
            "Organ", "Superstar", "Trinoids", "Zarvox"}


# ---------------------------------------------------------------- generation


def write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def mac_voices() -> list[str]:
    out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
    names = []
    for line in out.splitlines():
        m = re.match(r"^(.+?)\s+(en_[A-Z]{2})\s+#", line)
        if m and m.group(1).strip() not in MAC_SKIP:
            names.append(m.group(1).strip())
    return names


def gen_mac() -> None:
    for voice in mac_voices():
        for kind, texts in (("pos", POSITIVE), ("neg", LOOKALIKE)):
            for text in texts:
                for rate in (150, 185, 220):
                    f = CLIPS / kind / f"mac-{slug(voice)}-{slug(text)}-{rate}.wav"
                    if f.exists():
                        continue
                    f.parent.mkdir(parents=True, exist_ok=True)
                    subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", str(f), "--file-format=WAVE",
                                    "--data-format=LEI16@16000", text], check=True)
    print(f"macOS voices done: {len(mac_voices())} voices")


def gen_speech(per_voice: int = 2) -> None:
    """Long stretches of ordinary speech (random dictionary words) as streaming-style negatives."""
    words = [w for w in Path("/usr/share/dict/words").read_text().split() if w.islower() and 3 <= len(w) <= 9]
    rng = random.Random(3)
    for voice in mac_voices():
        for i in range(per_voice):
            f = CLIPS / "speech" / f"mac-{slug(voice)}-{i}.wav"
            if f.exists():
                continue
            text = " ".join(rng.choice(words) for _ in range(45))
            f.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["say", "-v", voice, "-r", str(rng.choice((160, 190, 220))), "-o", str(f),
                            "--file-format=WAVE", "--data-format=LEI16@16000", text], check=True)
    print(f"speech negatives done: {len(list((CLIPS / 'speech').glob('*.wav')))} clips")


async def gen_cartesia(n_voices: int) -> None:
    from cartesia import AsyncCartesia
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    client = AsyncCartesia(api_key=os.environ["CARTESIA_API_KEY"])
    voices = []
    for gender in ("masculine", "feminine", "gender_neutral"):
        page = await client.voices.list(limit=100, gender=gender)
        voices += [v async for v in page if v.language == "en"]
    random.Random(7).shuffle(voices)
    voices = voices[:n_voices]
    async with client.tts.websocket_connect() as ws:
        for v in voices:
            for kind, texts in (("pos", POSITIVE), ("neg", LOOKALIKE)):
                for text in texts:
                    f = CLIPS / kind / f"cartesia-{slug(v.name)}-{v.id[:8]}-{slug(text)}.wav"
                    if f.exists():
                        continue
                    ctx = ws.context(model_id="sonic-3.6-2026-08-27", voice={"mode": "id", "id": v.id},
                                     output_format={"container": "raw", "encoding": "pcm_f32le", "sample_rate": SR},
                                     language="en", max_buffer_delay_ms=0)
                    await ctx.push(text)
                    await ctx.no_more_inputs()
                    parts = [np.frombuffer(r.audio, np.float32) async for r in ctx.receive() if r.type == "chunk" and r.audio]
                    if parts:
                        write_wav(f, np.concatenate(parts))
    print(f"Cartesia voices done: {len(voices)} voices")


def record(n: int) -> None:
    import sounddevice as sd

    from ultron.audio.devices import choose_input
    from ultron.audio.input import Downsampler

    device, note = choose_input("")
    if note:
        print(note)
    rate = int(sd.query_devices(device, kind="input")["default_samplerate"])
    print(f"Recording {n} samples. Say 'Ultron' (vary it: normal, quiet, fast, from across the room,"
          f" 'hey Ultron') after each beep. Ctrl+C to stop early.")
    for i in range(n):
        input(f"[{i + 1}/{n}] Enter, then say it…")
        x = sd.rec(int(1.8 * rate), samplerate=rate, channels=1, dtype="float32", device=device)
        sd.wait()
        audio = Downsampler(rate, SR).process(x[:, 0])
        write_wav(CLIPS / "pos" / f"user-{i:03d}.wav", audio)
    print("saved; they're used for training (never held out, so evaluation stays on unseen voices)")


# ---------------------------------------------------------------- augmentation + features


def trim(x: np.ndarray, thresh: float = 0.02) -> np.ndarray:
    idx = np.flatnonzero(np.abs(x) > thresh * max(np.abs(x).max(), 1e-6))
    return x[max(0, idx[0] - 800) : idx[-1] + 800] if idx.size else x


def colored_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1 / SR)
    f[0] = f[1]
    spec /= f ** rng.uniform(0, 1)  # white .. pink .. towards brown
    x = np.fft.irfft(spec, n)
    return (x / (np.abs(x).max() + 1e-9)).astype(np.float32)


class Augmenter:
    def __init__(self, babble: list[np.ndarray], seed: int) -> None:
        from pedalboard import Compressor, GSMFullRateCompressor, HighpassFilter, LowpassFilter, MP3Compressor, Pedalboard, Reverb

        self.rng = np.random.default_rng(seed)
        self.babble = babble
        self.P = (Pedalboard, Reverb, HighpassFilter, LowpassFilter, Compressor, MP3Compressor, GSMFullRateCompressor)

    def __call__(self, clip: np.ndarray, end_at: float, length_s: float = CLIP_S) -> np.ndarray:
        """Place the clip so it ends `end_at` seconds before the end of a `length_s` background."""
        r = self.rng
        Pedalboard, Reverb, HighpassFilter, LowpassFilter, Compressor, MP3Compressor, GSM = self.P
        semis = r.uniform(-2.0, 2.0)
        x = Resampler(SR, SR, semis).process(clip) if abs(semis) > 0.2 else clip
        x = x * 10 ** (r.uniform(-18, 0) / 20) / (np.abs(x).max() + 1e-6)
        n = int(length_s * SR)
        out = np.zeros(n, np.float32)
        end = n - int(end_at * SR)
        start = max(0, end - x.size)
        out[start:end] = x[-(end - start):]
        fx = []
        if r.random() < 0.6:
            fx.append(Reverb(room_size=r.uniform(0.05, 0.8), damping=r.uniform(0.2, 0.9),
                             wet_level=r.uniform(0.05, 0.4), dry_level=1.0, width=0.0))
        if r.random() < 0.3:
            fx += [HighpassFilter(cutoff_frequency_hz=r.uniform(200, 400)), LowpassFilter(cutoff_frequency_hz=r.uniform(3000, 5000))]
        if r.random() < 0.3:
            fx.append(Compressor(threshold_db=-24, ratio=4))
        if fx:
            out = Pedalboard(fx)(out[None], SR)[0]
        if r.random() < 0.15:
            out = Pedalboard([MP3Compressor(vbr_quality=r.uniform(4, 9))])(out[None], SR)[0][:n]
        elif r.random() < 0.1:
            out = Pedalboard([GSM()])(out[None], SR)[0][:n]
        sig = np.sqrt(np.mean(out**2)) + 1e-9
        snr = r.uniform(3, 35)
        noise = colored_noise(n, r)
        if self.babble and r.random() < 0.5:
            b = self.babble[r.integers(len(self.babble))]
            b = np.resize(b, n) * r.uniform(0.3, 1.0)
            noise = noise * r.uniform(0, 0.5) + b
        noise *= sig / (np.sqrt(np.mean(noise**2)) + 1e-9) / 10 ** (snr / 20)
        out = out + noise
        return np.clip(out, -1, 1).astype(np.float32)


def speaker_of(path: Path) -> str:
    stem = path.stem
    if stem.startswith("user-"):
        return "user"
    return "-".join(stem.split("-")[:3]) if stem.startswith("cartesia-") else "-".join(stem.split("-")[:2])


_W: dict = {}


def _worker_init(babble_files: list[str], seed: int) -> None:
    _W["feats"] = Features()
    _W["aug"] = Augmenter([trim(read_wav(Path(f))) for f in babble_files], seed + os.getpid())


def _clip_job(path: str, reps: int, lo: float, hi: float) -> np.ndarray:
    aug, feats = _W["aug"], _W["feats"]
    clip = trim(read_wav(Path(path)))
    return np.array([stream_embed(feats, aug(clip, aug.rng.uniform(lo, hi)) * 32767) for _ in range(reps)], np.float32)


def _speech_job(path: str, stride: int) -> np.ndarray:
    """Every `stride`-th live window over an augmented stretch of ordinary speech."""
    aug, feats = _W["aug"], _W["feats"]
    clip = read_wav(Path(path))
    lead = np.zeros(int(2.5 * SR), np.float32)  # warm-up for the mel model's memory
    audio = aug(np.concatenate([lead, clip]), 0.0, length_s=(lead.size + clip.size) / SR + 0.2) * 32767
    s = StreamingFeatures(feats)
    out = []
    for i in range(0, (audio.size // CHUNK) * CHUNK, CHUNK):
        if s.push(audio[i : i + CHUNK]) and i // CHUNK >= 45 and (i // CHUNK) % stride == 0:
            out.append(s.emb[-WINDOW:].copy())
    return np.array(out, np.float32).reshape(-1, WINDOW, 96)


def build_features(per_pos: int, per_neg: int) -> None:
    pos_files = sorted((CLIPS / "pos").glob("*.wav"))
    neg_files = sorted((CLIPS / "neg").glob("*.wav"))
    speech_files = sorted((CLIPS / "speech").glob("*.wav"))
    speakers = sorted({speaker_of(f) for f in pos_files} - {"user"})
    rng = random.Random(1)
    held = set(rng.sample(speakers, max(2, len(speakers) // 8)))  # ~12% of voices never trained on
    babble = [str(f) for f in rng.sample(neg_files, min(200, len(neg_files)))]
    workers = max(1, (os.cpu_count() or 2) - 2)

    with cf.ProcessPoolExecutor(workers, initializer=_worker_init, initargs=(babble, 2)) as pool:
        def run(files, reps, end_range):
            jobs = [(f, pool.submit(_clip_job, str(f), reps if speaker_of(f) not in held else 2, *end_range)) for f in files]
            tr, ev = [], []
            for i, (f, job) in enumerate(jobs):
                (ev if speaker_of(f) in held else tr).extend(job.result())
                if i % 300 == 0:
                    print(f"  {i}/{len(jobs)}", flush=True)
            return np.array(tr, np.float32), np.array(ev, np.float32)

        print(f"positives: {len(pos_files)} clips, {len(speakers)} voices ({len(held)} held out), {workers} workers")
        pos_tr, pos_ev = run(pos_files, per_pos, (0.0, 0.25))
        print(f"look-alikes: {len(neg_files)} clips")
        neg_tr, neg_ev = run(neg_files, per_neg, (0.0, 0.6))
        print(f"speech: {len(speech_files)} clips")
        sp = [(f, pool.submit(_speech_job, str(f), 3)) for f in speech_files]
        sp_tr = [j.result() for f, j in sp if speaker_of(f) not in held]
        sp_ev = [j.result() for f, j in sp if speaker_of(f) in held]
    speech_tr = np.concatenate(sp_tr) if sp_tr else np.zeros((0, WINDOW, 96), np.float32)
    speech_ev = np.concatenate(sp_ev) if sp_ev else np.zeros((0, WINDOW, 96), np.float32)
    np.savez_compressed(WORK / "features.npz", pos_tr=pos_tr, pos_ev=pos_ev, neg_tr=neg_tr, neg_ev=neg_ev,
                        speech_tr=speech_tr, speech_ev=speech_ev)
    (WORK / "split.json").write_text(json.dumps({"held_out_voices": sorted(held)}, indent=1))
    print(f"features: pos {len(pos_tr)} train / {len(pos_ev)} eval, look-alike {len(neg_tr)} / {len(neg_ev)}, "
          f"speech {len(speech_tr)} / {len(speech_ev)}")


def generic_windows(part: str, stride: int) -> np.ndarray:
    x = np.load(NEG_FEATURES, mmap_mode="r")
    cut = int(x.shape[0] * 0.8)
    x = np.asarray(x[:cut] if part == "train" else x[cut:])
    idx = np.arange(0, x.shape[0] - WINDOW, stride)
    return np.stack([x[i : i + WINDOW] for i in idx]).astype(np.float32)


# ---------------------------------------------------------------- model


class MLP:
    """1536 -> h -> h -> 1, ReLU, sigmoid. Inputs standardised with training statistics."""

    def __init__(self, d_in: int, h: int, seed: int = 0) -> None:
        r = np.random.default_rng(seed)
        self.p = {
            "W1": r.normal(0, np.sqrt(2 / d_in), (d_in, h)).astype(np.float32), "b1": np.zeros(h, np.float32),
            "W2": r.normal(0, np.sqrt(2 / h), (h, h)).astype(np.float32), "b2": np.zeros(h, np.float32),
            "W3": r.normal(0, np.sqrt(1 / h), (h, 1)).astype(np.float32), "b3": np.zeros(1, np.float32),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0
        self.mu = self.sd = None

    def forward(self, x):
        p = self.p
        z1 = x @ p["W1"] + p["b1"]; a1 = np.maximum(z1, 0)
        z2 = a1 @ p["W2"] + p["b2"]; a2 = np.maximum(z2, 0)
        z3 = (a2 @ p["W3"] + p["b3"])[:, 0]
        return z3, (x, z1, a1, z2, a2)

    def predict(self, x):
        z, _ = self.forward((x.reshape(len(x), -1) - self.mu) / self.sd)
        return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

    def step(self, x, y, w, lr, wd=1e-4):
        z, (x, z1, a1, z2, a2) = self.forward(x)
        prob = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        g = (prob - y) * w / w.sum()
        p = self.p
        grads = {"W3": a2.T @ g[:, None], "b3": np.array([g.sum()], np.float32)}
        d2 = (g[:, None] @ p["W3"].T) * (z2 > 0)
        grads.update(W2=a1.T @ d2, b2=d2.sum(0))
        d1 = (d2 @ p["W2"].T) * (z1 > 0)
        grads.update(W1=x.T @ d1, b1=d1.sum(0))
        self.t += 1
        for k in p:
            gk = grads[k] + (wd * p[k] if k.startswith("W") else 0)
            self.m[k] = 0.9 * self.m[k] + 0.1 * gk
            self.v[k] = 0.999 * self.v[k] + 0.001 * gk**2
            mh, vh = self.m[k] / (1 - 0.9**self.t), self.v[k] / (1 - 0.999**self.t)
            p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
        bce = -(w * (y * np.log(prob + 1e-7) + (1 - y) * np.log(1 - prob + 1e-7))).sum() / w.sum()
        return float(bce)

    def export(self, path: Path) -> None:
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        c = lambda name, a: numpy_helper.from_array(np.asarray(a, np.float32), name)  # noqa: E731
        nodes = [
            helper.make_node("Reshape", ["input", "shape"], ["flat"]),
            helper.make_node("Sub", ["flat", "mu"], ["c"]), helper.make_node("Div", ["c", "sd"], ["x"]),
            helper.make_node("Gemm", ["x", "W1", "b1"], ["z1"]), helper.make_node("Relu", ["z1"], ["a1"]),
            helper.make_node("Gemm", ["a1", "W2", "b2"], ["z2"]), helper.make_node("Relu", ["z2"], ["a2"]),
            helper.make_node("Gemm", ["a2", "W3", "b3"], ["z3"]), helper.make_node("Sigmoid", ["z3"], ["score"]),
        ]
        inits = [numpy_helper.from_array(np.array([-1, WINDOW * 96], np.int64), "shape"), c("mu", self.mu), c("sd", self.sd)]
        inits += [c(k, v) for k, v in self.p.items()]
        graph = helper.make_graph(nodes, "ultron_wakeword",
                                  [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, WINDOW, 96])],
                                  [helper.make_tensor_value_info("score", TensorProto.FLOAT, [None, 1])], inits)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8
        onnx.checker.check_model(model)
        onnx.save(model, str(path))


def train(epochs: int, hidden: int) -> None:
    d = np.load(WORK / "features.npz")
    gen = generic_windows("train", stride=4)
    pos, neg = d["pos_tr"], np.concatenate([d["neg_tr"], d["speech_tr"], gen])
    x = np.concatenate([pos, neg]).reshape(len(pos) + len(neg), -1)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(np.float32)
    # Look-alikes count extra: they're exactly the mistakes we care about.
    w = np.concatenate([np.full(len(pos), len(neg) / len(pos) * 0.5), np.full(len(d["neg_tr"]), 4.0),
                        np.full(len(d["speech_tr"]), 2.0), np.ones(len(gen))]).astype(np.float32)
    model = MLP(x.shape[1], hidden)
    model.mu, model.sd = x.mean(0), x.std(0) + 1e-3
    xs = (x - model.mu) / model.sd
    rng = np.random.default_rng(0)
    print(f"training on {len(pos)} positive, {len(d['neg_tr'])} look-alike, {len(d['speech_tr'])} speech, {len(gen)} generic windows")
    for ep in range(epochs):
        order = rng.permutation(len(xs))
        lr = 1e-3 * (0.5 ** (ep // max(1, epochs // 3)))
        losses = [model.step(xs[b], y[b], w[b], lr) for b in np.array_split(order, max(1, len(order) // 512))]
        print(f"  epoch {ep + 1}/{epochs}  loss {np.mean(losses):.4f}")
    out = MODELS / "ultron_wakeword.onnx"
    model.export(out)
    print(f"saved {out} ({out.stat().st_size / 1e3:.0f} kB)")


def evaluate() -> None:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(MODELS / "ultron_wakeword.onnx"))
    score = lambda x: np.concatenate([sess.run(None, {"input": b})[0][:, 0] for b in np.array_split(x, max(1, len(x) // 2048))])  # noqa: E731
    d = np.load(WORK / "features.npz")
    gen = generic_windows("eval", stride=1)  # every 80 ms step, like live
    hours = (len(gen) + WINDOW) * 0.08 / 3600
    sp, sl, sg = score(d["pos_ev"]), score(d["neg_ev"]), score(gen)
    ss = score(d["speech_ev"]) if len(d["speech_ev"]) else np.zeros(0)
    print(f"held-out: {len(sp)} positives (unseen voices), {len(sl)} look-alikes, {hours:.2f} h of generic audio")
    print(f"{'threshold':>9} {'recall':>8} {'look-alike FA':>14} {'speech FA':>10} {'false wakes/hour':>17}")
    for th in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
        hits = np.flatnonzero(sg >= th)
        events = int(np.sum(np.diff(hits, prepend=-100) > 25)) if hits.size else 0  # 2 s refractory
        sfa = np.mean(ss >= th) if ss.size else float("nan")
        print(f"{th:>9.2f} {np.mean(sp >= th):>8.1%} {np.mean(sl >= th):>14.1%} {sfa:>10.2%} {events / hours:>17.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+", choices=["generate", "record", "features", "train", "eval", "all"])
    ap.add_argument("--cartesia-voices", type=int, default=30)
    ap.add_argument("--record", type=int, default=20)
    ap.add_argument("--per-pos", type=int, default=12, help="augmented copies per positive clip")
    ap.add_argument("--per-neg", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=64)
    a = ap.parse_args()
    stages = ["generate", "features", "train", "eval"] if "all" in a.stages else a.stages
    WORK.mkdir(parents=True, exist_ok=True)
    for s in stages:
        print(f"== {s}")
        if s == "generate":
            gen_mac()
            gen_speech()
            if a.cartesia_voices:
                asyncio.run(gen_cartesia(a.cartesia_voices))
        elif s == "record":
            record(a.record)
        elif s == "features":
            build_features(a.per_pos, a.per_neg)
        elif s == "train":
            train(a.epochs, a.hidden)
        elif s == "eval":
            evaluate()


if __name__ == "__main__":
    main()
