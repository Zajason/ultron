import numpy as np

from ultron.audio.fx import OctaveDown
from ultron.audio.resample import Resampler, varispeed_ratio


def tone(hz, sr, secs=1.0):
    return np.sin(2 * np.pi * hz * np.arange(int(sr * secs)) / sr).astype(np.float32)


def dominant_hz(x, sr):
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    return np.fft.rfftfreq(x.size, 1 / sr)[np.argmax(spec)]


def test_plain_upsample_keeps_pitch_and_doubles_length():
    y = Resampler(24000, 48000).process(tone(200, 24000))
    assert abs(y.size - 48000) <= 2
    assert abs(dominant_hz(y, 48000) - 200) < 2


def test_varispeed_lowers_pitch_and_lengthens():
    r = varispeed_ratio(-2)
    y = Resampler(24000, 48000, semitones=-2).process(tone(200, 24000))
    assert abs(y.size - 48000 / r) <= 3  # the last input sample waits for the next chunk
    assert abs(dominant_hz(y, 48000) - 200 * r) < 2


def test_chunked_equals_whole():
    x = tone(173, 24000)
    whole = Resampler(24000, 48000, -1.5).process(x)
    rs = Resampler(24000, 48000, -1.5)
    chunked = np.concatenate([rs.process(x[i : i + 457]) for i in range(0, x.size, 457)])
    n = min(whole.size, chunked.size)
    assert abs(whole.size - chunked.size) <= 1
    np.testing.assert_allclose(chunked[:n], whole[:n], atol=1e-5)


def test_octave_down_halves_fundamental():
    sr = 48000
    y = OctaveDown(sr).process(0.5 * tone(160, sr, 2.0))
    assert abs(dominant_hz(y[sr // 2 :], sr) - 80) < 3


def test_octave_down_chunked_equals_whole():
    sr = 48000
    x = 0.5 * tone(150, sr)
    whole = OctaveDown(sr).process(x)
    od = OctaveDown(sr)
    chunked = np.concatenate([od.process(x[i : i + 960]) for i in range(0, x.size, 960)])
    np.testing.assert_allclose(chunked, whole, atol=1e-5)
