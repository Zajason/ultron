import wave
from pathlib import Path

import numpy as np

from ultron.wakeword import Features, StreamingFeatures

FIX = Path(__file__).parent / "fixtures" / "utterances"


def pcm(name):
    with wave.open(str(FIX / name)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32)


def test_embeddings_shape():
    e = Features().embed_clip(pcm("ultron-what-time-is-it.wav"))
    assert e.ndim == 2 and e.shape[1] == 96 and e.shape[0] > 10


def test_training_features_are_the_live_features():
    from ultron.wakeword import stream_embed

    x = np.concatenate([np.zeros(48000, np.float32), pcm("ultron-what-time-is-it.wav")])
    x = x[: (x.size // 1280) * 1280]
    f = Features()
    s = StreamingFeatures(f)
    for i in range(0, x.size, 333):  # live: odd block sizes
        s.push(x[i : i + 333])
    np.testing.assert_allclose(stream_embed(f, x), s.emb[-16:], atol=1e-4)


def test_streaming_close_to_full_history_features():
    # 2 s of mel context per step keeps the error vs. full-history (batch) features small; this is
    # what openWakeWord's precomputed negatives were made with. Before this: 2.3 with |e| ~ 15.
    x = np.concatenate([np.zeros(48000, np.float32), pcm("tell-me-something-about-bach-that-most-p.wav"),
                        np.zeros(4000, np.float32)])
    x = x[: (x.size // 1280) * 1280]
    f = Features()
    s = StreamingFeatures(f)
    s.push(x)
    full = f.embed_clip(x, last=16)
    assert np.abs(s.emb[-16:] - full).mean() < 0.3
