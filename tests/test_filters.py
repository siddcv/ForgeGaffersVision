import random

import pytest

from src.config import Config
from src.filters import Kalman1D, OneEuroFilter

CFG = Config()


def rmse(pairs):
    return (sum((a - b) ** 2 for a, b in pairs) / len(pairs)) ** 0.5


def make_kalman():
    return Kalman1D(CFG.kalman_process_noise, CFG.kalman_measurement_noise)


def test_kalman_reduces_noise_on_constant_velocity():
    rng = random.Random(0)
    kalman = make_kalman()
    raw, filtered = [], []
    for i in range(300):
        t = i / 30
        truth = 100 + 200 * t
        noisy = truth + rng.gauss(0, 2)
        output = kalman.update(t, noisy)
        if i > 30:
            raw.append((noisy, truth))
            filtered.append((output, truth))
    assert rmse(filtered) < rmse(raw)


def test_kalman_predicts_through_a_short_gap():
    kalman = make_kalman()
    for i in range(30):
        kalman.update(i / 30, 300 * i / 30)
    assert kalman.predict(33 / 30) == pytest.approx(330, abs=5)


def test_kalman_follows_a_sharp_reversal_quickly():
    kalman = make_kalman()
    for i in range(30):
        kalman.update(i / 30, 300 * i / 30)
    for i in range(30, 33):
        kalman.update(i / 30, 300 - 300 * (i - 29) / 30)
    assert kalman.v < 0


def test_one_euro_reduces_noise_on_a_still_point():
    rng = random.Random(0)
    euro = OneEuroFilter(CFG.one_euro_min_cutoff, CFG.one_euro_beta, CFG.one_euro_d_cutoff)
    raw, filtered = [], []
    for i in range(300):
        noisy = 50 + rng.gauss(0, 2)
        output = euro.update(i / 30, noisy)
        if i > 30:
            raw.append((noisy, 50))
            filtered.append((output, 50))
    assert rmse(filtered) < rmse(raw)
