import math

import numpy as np

import src.run_clip as rc
from src import metrics
from src.config import Config
from src.pose import LEFT_FOOT, RIGHT_FOOT


def _cache_with_person_missing_first(n_frames: int, missing: int, fps: float = 30.0) -> dict:
    """Synthetic cache: no person (NaN landmarks) for the first `missing` frames, then still feet."""
    landmarks = np.full((n_frames, 33, 3), np.nan)
    for index, x in zip(LEFT_FOOT + RIGHT_FOOT, (300.0, 310.0, 500.0, 510.0)):
        landmarks[missing:, index] = (x, 600.0, 0.9)
    no_balls = np.full((n_frames, 3, 5), np.nan)
    return {
        "t_s": np.arange(n_frames) / fps,
        "landmarks": landmarks,
        "crops": np.full((n_frames, 4), np.nan),
        "crop_balls": no_balls,
        "full_balls": no_balls.copy(),
    }


def test_feet_recover_after_clip_starts_with_no_person(monkeypatch):
    seen = []
    real_step = metrics.step

    def recording_step(state, observation, cfg):
        seen.append(observation)
        return real_step(state, observation, cfg)

    monkeypatch.setattr(metrics, "step", recording_step)
    rc.replay(_cache_with_person_missing_first(n_frames=40, missing=10), Config())

    assert all(obs.left_foot == () and obs.right_foot == () for obs in seen[:10])
    for obs in seen[10:]:
        assert len(obs.left_foot) == 2 and len(obs.right_foot) == 2
        assert all(math.isfinite(v) for point in obs.left_foot + obs.right_foot for v in point)
