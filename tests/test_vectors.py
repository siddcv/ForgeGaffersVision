"""Replay the portable test vectors (tests/vectors, written by scripts.export_test_vectors) in Python.

The Swift / Kotlin port runs the same files. If this fails after an intended change to the logic or
config, regenerate the vectors and commit them, so the port is held to the new behaviour.
"""
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from src import ball as ball_module
from src import metrics
from src.config import Config
from src.filters import Kalman1D, OneEuroFilter
from src.pipeline import FramePipeline

VECTORS = Path(__file__).parent / "vectors"
ABS_TOL, REL_TOL = 1e-6, 1e-9  # the README states the same tolerance for ports


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def config(fields: dict) -> Config:
    return Config(**{k: v for k, v in fields.items() if k != "ball_diameter_m"})


def tuples(value):
    if isinstance(value, list):
        return tuple(tuples(v) for v in value)
    return value


def plain(value):
    """Python outputs in the same JSON shape as the vectors."""
    if hasattr(value, "__dataclass_fields__"):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def assert_close(expected, actual, where="") -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and expected.keys() == actual.keys(), where
        for key in expected:
            assert_close(expected[key], actual[key], f"{where}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(expected) == len(actual), f"{where}: length"
        for i, (e, a) in enumerate(zip(expected, actual)):
            assert_close(e, a, f"{where}[{i}]")
    elif isinstance(expected, float) and not isinstance(actual, bool) and isinstance(actual, (int, float)):
        assert math.isclose(expected, actual, rel_tol=REL_TOL, abs_tol=ABS_TOL), f"{where}: {expected} != {actual}"
    else:
        assert expected == actual, f"{where}: {expected!r} != {actual!r}"


def observation(fields: dict) -> metrics.Observation:
    return metrics.Observation(**{k: tuples(v) for k, v in fields.items()})


@pytest.mark.parametrize("path", sorted((VECTORS / "metrics").glob("*.json")), ids=lambda p: p.stem)
def test_metrics_vector(path):
    vector = load(path)
    cfg = config(vector["config"])
    state = metrics.initial_state()
    events = []
    observations = [observation(o) for o in vector["observations"]]
    for frame, obs in enumerate(observations):
        state, new = metrics.step(state, obs, cfg)
        events += [{"frame": frame, **asdict(e)} for e in new]
    actual = {"events": events, "final_touch_rate": metrics.touch_rate(state, observations[-1].t_s, cfg)}
    assert_close(vector["expected"], plain(actual), path.stem)


def test_filters_vector():
    vector = load(VECTORS / "filters.json")
    cfg = config(vector["config"])
    euro = OneEuroFilter(cfg.one_euro_min_cutoff, cfg.one_euro_beta, cfg.one_euro_d_cutoff)
    for i, row in enumerate(vector["one_euro"]):
        assert_close(row["out"], euro.update(row["t_s"], row["x"]), f"one_euro[{i}]")
    kalman = Kalman1D(cfg.kalman_process_noise, cfg.kalman_measurement_noise)
    for i, row in enumerate(vector["kalman"]):
        if row["op"] == "update":
            out = kalman.update(row["t_s"], row["z"])
        else:
            out = getattr(kalman, row["op"])(row["t_s"])
        assert_close(row["out"], out, f"kalman[{i}]")


def test_search_step_vector():
    vector = load(VECTORS / "search_step.json")
    cfg = config(vector["config"])
    for name, rows in vector["sequences"].items():
        state = ball_module.SearchState()
        for i, row in enumerate(rows):
            crop, crop_det, full_det = tuples(row["crop"]), tuples(row["crop_det"]), tuples(row["full_det"])
            needs = ball_module.needs_full_frame(state, row["t_s"], crop, crop_det, cfg)
            state, detection, searched = ball_module.search_step(
                state, row["t_s"], crop, crop_det, full_det if needs else None, cfg,
            )
            actual = {"needs_full_frame": needs, "detection": detection, "searched_full_frame": searched,
                      "state": asdict(state)}
            assert_close(row["expected"], plain(actual), f"{name}[{i}]")


@pytest.mark.parametrize("path", sorted((VECTORS / "pipeline").glob("*.json")), ids=lambda p: p.stem)
def test_pipeline_vector(path):
    vector = load(path)
    pipeline = FramePipeline(config(vector["config"]))
    for i, frame in enumerate(vector["frames"]):
        given = frame["input"]
        landmarks = np.array([[np.nan if v is None else v for v in row] for row in given["landmarks"]])
        record = pipeline.step(
            given["t_s"], landmarks, tuples(given["crop"]),
            tuples(given["crop_candidates"]), lambda: tuples(given["full_candidates"]),
        )
        actual = {key: record[key] for key in frame["expected"]}
        assert_close(frame["expected"], plain(actual), f"{path.stem} frame {i}")
