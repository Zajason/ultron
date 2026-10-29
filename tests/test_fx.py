import numpy as np

from ultron.audio import VoiceFX
from ultron.config import FXConfig

SR = 24000


def noise(n=SR):
    return (np.random.default_rng(0).standard_normal(n) * 0.1).astype(np.float32)


def test_chunked_equals_whole():
    # Streaming must be seamless: processing in 20 ms chunks gives the same signal as one block.
    x = noise()
    whole = VoiceFX(FXConfig(), SR).process(x)
    fx = VoiceFX(FXConfig(), SR)
    chunked = np.concatenate([fx.process(x[i : i + 480]) for i in range(0, x.size, 480)])
    np.testing.assert_allclose(chunked, whole, atol=1e-6)


def test_shape_dtype_and_range():
    y = VoiceFX(FXConfig(), SR).process(noise() * 20)
    assert y.dtype == np.float32 and y.shape == (SR,) and np.abs(y).max() <= 1.0


def test_empty_chunk():
    assert VoiceFX(FXConfig(), SR).process(np.zeros(0, np.float32)).size == 0


def test_metal_layer_is_subtle():
    x = noise()
    dry = VoiceFX(FXConfig(metal_mix=0.0), SR).process(x)
    wet = VoiceFX(FXConfig(), SR).process(x)
    assert np.sqrt(np.mean((wet - dry) ** 2)) < 0.25 * np.sqrt(np.mean(dry**2))
