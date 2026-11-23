"""Write portable test vectors (JSON) for the Swift / Kotlin port of the per-frame logic.

Each vector holds the inputs, the config and the outputs the Python code produced. A port passes
when it reproduces the outputs; tests/test_vectors.py replays them in Python so they stay current.
See tests/vectors/README.md for the format.

Usage: python -m scripts.export_test_vectors        (needs cache/IMG_1710.npz for the pipeline vector)
Re-run after any intended change to metrics, filters, ball tracking or config, and commit the diff.
"""
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from src import ball as ball_module
from src import metrics
from src.config import Config
from src.filters import Kalman1D, OneEuroFilter
from src.pipeline import FramePipeline
from tests.synth import (
    DIAMETER_PX, LEFT_FOOT, RIGHT_FOOT, Y, ball_at, head, head_levels, missing, path, synth_tick_tock,
)

OUT = Path("tests/vectors")
FORMAT_VERSION = 1
CFG = Config()
UP, DOWN, BETWEEN = head_levels(CFG)  # pitches placed relative to the single head-pitch cutoff

PIPELINE_CLIP = "IMG_1710"
PIPELINE_START_FRAME = 60  # 1.0 s in: tick-tocks, full-frame searching, a lost control and reset
PIPELINE_FRAMES = 300
PIPELINE_STRIDE = 2  # the 60 fps clip at 30 fps, as the phone runs it
ROUND = 3  # decimals kept on real-clip inputs; outputs are computed from the rounded inputs


def clean(value):
    """JSON-safe: tuples to lists, NaN to null, dataclasses to dicts."""
    if hasattr(value, "__dataclass_fields__"):
        return clean(asdict(value))
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return None if math.isnan(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def config_dict(cfg) -> dict:
    return {**asdict(cfg), "ball_diameter_m": cfg.ball_diameter_m}


def dumps(value, indent: str = "") -> str:
    """Objects keep one key per line; lists put one compact item per line, so a git diff shows
    which frame changed without the size of fully indented JSON."""
    inner = indent + " "
    if isinstance(value, dict) and value:
        items = [f"{inner}{json.dumps(k)}: {dumps(v, inner)}" for k, v in value.items()]
        return "{\n" + ",\n".join(items) + "\n" + indent + "}"
    if isinstance(value, list) and value and isinstance(value[0], (dict, list)):
        return "[\n" + ",\n".join(inner + json.dumps(v, separators=(",", ":")) for v in value) + "\n" + indent + "]"
    return json.dumps(value, separators=(",", ":"))


def write(name: str, vector: dict) -> None:
    path = OUT / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps({"format_version": FORMAT_VERSION, **clean(vector)}) + "\n")


# metrics.step ---------------------------------------------------------------------------------

def run_metrics(observations, cfg) -> dict:
    state = metrics.initial_state()
    events = []
    for frame, observation in enumerate(observations):
        state, new = metrics.step(state, observation, cfg)
        events += [{"frame": frame, **asdict(e)} for e in new]
    last_t = observations[-1].t_s
    return {"events": events, "final_touch_rate": metrics.touch_rate(state, last_t, cfg)}


def hidden_at_first_touch() -> list:
    observations = synth_tick_tock(4, 60.0)
    first = next(e for e in run_metrics(observations, CFG)["events"] if e["kind"] == "contact")
    index = next(i for i, o in enumerate(observations) if o.t_s > first["t_s"])
    hidden = [replace(o, ball_xy=None, ball_diameter_px=None) for o in observations[index:index + 8]]
    shifted = [replace(o, ball_xy=(o.ball_xy[0] + 110, Y)) for o in observations[index + 8:]]
    return observations[:index] + hidden + shifted


def brief_occlusion() -> list:
    observations = synth_tick_tock(4)
    first = next(e for e in run_metrics(observations, CFG)["events"] if e["kind"] == "contact")
    index = next(i for i, o in enumerate(observations) if o.t_s >= first["t_s"])
    return [
        replace(o, ball_xy=None, ball_diameter_px=None) if abs(i - index) <= 1 else o
        for i, o in enumerate(observations)
    ]


def repeat_same_foot() -> list:
    observations = synth_tick_tock(1, 30.0)
    t, x = observations[-1].t_s, observations[-1].ball_xy[0]
    extra = []
    for direction in [-1] * 6 + [1] * 6:
        t += 1 / 30
        x += direction * 9
        extra.append(replace(observations[-1], t_s=t, ball_xy=(x, Y)))
    return observations + extra


def metrics_scenarios() -> dict:
    """name -> (description, observations, config)."""
    far_feet = [replace(o, left_foot=((-5000.0, Y),), right_foot=((5000.0, Y),)) for o in synth_tick_tock(4)]
    still = [
        metrics.Observation(t_s=i / 30, ball_xy=(300.0 + (i % 2) * 0.2, Y), ball_diameter_px=DIAMETER_PX,
                            left_foot=LEFT_FOOT, right_foot=RIGHT_FOOT)
        for i in range(60)
    ]
    fast = path([300, 240, 360])
    slow = path([360, 330], speed_px_s=40.0, start_t=fast[-1].t_s)
    late = path([300, 240, 330])
    return {
        "tick_tock_8_at_30fps": ("8 alternating touches at 1.5 m/s, 30 fps", synth_tick_tock(8, 30.0), CFG),
        "tick_tock_8_at_60fps": ("8 alternating touches at 1.5 m/s, 60 fps", synth_tick_tock(8, 60.0), CFG),
        "tick_tock_40_at_60fps": ("40 touches, for the touches-per-minute readout", synth_tick_tock(40, 60.0), CFG),
        "ball_size_3": ("same drill with a size 3 ball: speeds scale by 0.190 / 0.220",
                        synth_tick_tock(6), replace(CFG, ball_size=3)),
        "repeat_same_foot": ("two left contacts in a row: neither is a tick-tock", repeat_same_foot(), CFG),
        "hidden_at_touch_direction_rule": ("ball hidden 8 frames at contact, reappears nearer the other foot",
                                           hidden_at_first_touch(), CFG),
        "hidden_at_touch_nearest_rule": ("as above with touch_foot_rule = nearest",
                                         hidden_at_first_touch(), replace(CFG, touch_foot_rule="nearest")),
        "reversal_far_from_feet": ("direction changes far from both feet are not contacts", far_feet, CFG),
        "still_ball": ("ball jittering 0.2 px in place: no contacts", still, CFG),
        "brief_occlusion": ("ball missing for 3 frames around a contact", brief_occlusion(), CFG),
        "clean_run": ("left, right, left, right: the last has no answer yet",
                      path([300, 240, 360, 240, 360, 300]), CFG),
        "same_foot_twice": ("left touch drifts back without crossing, left again",
                            path([300, 240, 290, 240, 360, 240, 300]), CFG),
        "nudge_not_crossing": ("setup tap short of the midpoint", path([300, 240, 275, 235, 360, 240, 300]), CFG),
        "creeping_jitter": ("slow ball whose detection jitters next to a foot: no contacts",
                            path([300, 352, 355, 352, 355, 351], speed_px_s=45.0), CFG),
        "slow_answer_inside_limit": ("answer inside ticktock_max_arrival_s, confirmed after it",
                                     fast + slow, replace(CFG, ticktock_max_arrival_s=0.5)),
        "crosses_never_answered": ("ball crosses and leaves: a contact, no tick-tock", path([300, 240, 1000]), CFG),
        "answer_after_limit": ("the other foot answers after ticktock_max_arrival_s",
                               late + path([330, 360, 240, 300], start_t=late[-1].t_s + CFG.ticktock_max_arrival_s + 0.3),
                               CFG),
        "lost_control_cancels_pending": ("a full-frame search miss cancels the waiting touch",
                                         late + missing(40, searched=True, start=0.6) + path([330, 360, 240, 300], start_t=2.0),
                                         replace(CFG, ticktock_max_arrival_s=10.0)),
        "crop_miss_not_searched": ("120 frames missing from the crop only: never lost control",
                                   missing(120, searched=False), CFG),
        "full_frame_miss_under_limit": ("full-frame miss for 10 frames: not lost control",
                                        missing(10, searched=True), CFG),
        "full_frame_miss_at_limit": ("full-frame miss for 60 frames: one lost control",
                                     missing(60, searched=True), CFG),
        "far_ball_then_reset": ("ball far from the feet for 1 s, then back: lost control and reset",
                                ball_at(1000.0, 30) + ball_at(240.0, 3, start=1.0), CFG),
        "head_down_under_1s": ("head down for 30 frames at 30 fps: not reported", head([DOWN] * 30), CFG),
        "head_down_over_1s": ("head up 1 s then down: head_down stamped at 1.0 s",
                              head([UP] * 30 + [DOWN] * 40), CFG),
        "head_hysteresis": ("single cutoff: pitch at the boundary ends the head-down state",
                            head([DOWN] * 40 + [BETWEEN]), CFG),
    }


# filters and search switch ---------------------------------------------------------------------

def filter_vectors() -> dict:
    rng = np.random.default_rng(0)
    times = [i / 30 for i in range(90)]
    times = times[:40] + times[46:]  # a gap of 6 frames
    xs = [float(round(100 + 200 * t + 30 * math.sin(6 * t) + rng.normal(0, 2), 3)) for t in times]

    euro = OneEuroFilter(CFG.one_euro_min_cutoff, CFG.one_euro_beta, CFG.one_euro_d_cutoff)
    one_euro = [{"t_s": t, "x": x, "out": euro.update(t, x)} for t, x in zip(times, xs)]

    # Kalman: updates with measurements, plus predict() and peek() calls inside the gap.
    kalman = Kalman1D(CFG.kalman_process_noise, CFG.kalman_measurement_noise)
    steps = []
    for i, (t, x) in enumerate(zip(times, xs)):
        if i == 40:
            for gap_t in (times[39] + k / 30 for k in range(1, 7)):
                steps.append({"op": "peek", "t_s": gap_t, "out": kalman.peek(gap_t)})
                steps.append({"op": "predict", "t_s": gap_t, "out": kalman.predict(gap_t)})
        steps.append({"op": "update", "t_s": t, "z": x, "out": kalman.update(t, x)})
    return {
        "description": "One Euro and constant-velocity Kalman on a noisy, curving track with a 6-frame gap",
        "config": config_dict(CFG),
        "one_euro": one_euro,
        "kalman": steps,
    }


def search_vectors() -> dict:
    crop = (100, 100, 300, 300)
    in_crop = (190, 190, 210, 210, 0.9)
    far = (800, 100, 820, 120, 0.9)
    after = CFG.ball_search_after_s
    sequences = {
        "found_in_crop": [(0.0, crop, in_crop, far)],
        "brief_crop_miss": [(0.0, crop, in_crop, far), (0.03, crop, None, far), (0.06, crop, in_crop, far)],
        "search_after_misses_finds_far_ball": [(0.0, crop, None, far), (after, crop, None, far), (after + 0.03, crop, None, far)],
        "search_finds_nothing": [(0.0, crop, None, None), (after, crop, None, None), (1.0, crop, None, None)],
        "search_returns_when_ball_back_in_crop": [(0.0, crop, None, far), (after, crop, None, far),
                                                  (1.0, crop, None, in_crop), (1.03, crop, in_crop, None)],
        "no_crop_searches_at_once": [(0.0, None, None, far)],
    }
    out = {}
    for name, frames in sequences.items():
        state = ball_module.SearchState()
        rows = []
        for t_s, c, crop_det, full_det in frames:
            needs = ball_module.needs_full_frame(state, t_s, c, crop_det, CFG)
            state, detection, searched = ball_module.search_step(state, t_s, c, crop_det, full_det if needs else None, CFG)
            rows.append({
                "t_s": t_s, "crop": c, "crop_det": crop_det, "full_det": full_det,
                "expected": {"needs_full_frame": needs, "detection": detection, "searched_full_frame": searched,
                             "state": asdict(state)},
            })
        out[name] = rows
    return {
        "description": "ball.search_step with ball.needs_full_frame; full_det is only read when needed",
        "config": config_dict(CFG),
        "sequences": out,
    }


# whole per-frame pipeline on a real clip --------------------------------------------------------

def pipeline_vector() -> dict:
    cache = dict(np.load(Path("cache") / f"{PIPELINE_CLIP}.npz"))
    if "head_pitch_deg" not in cache:
        raise SystemExit(f"cache/{PIPELINE_CLIP}.npz has no Face Landmarker pitch: rebuild it with "
                         "python -m src.run_clip <clip> --refresh --no-overlay")
    end = PIPELINE_START_FRAME + PIPELINE_FRAMES * PIPELINE_STRIDE
    pipeline = FramePipeline(CFG)
    frames = []
    for i in range(PIPELINE_START_FRAME, end, PIPELINE_STRIDE):
        landmarks = np.round(cache["landmarks"][i], ROUND)
        crop_row = cache["crops"][i]
        crop = None if np.isnan(crop_row[0]) else tuple(int(v) for v in crop_row)
        crop_candidates = [tuple(round(float(v), ROUND) for v in row) for row in cache["crop_balls"][i] if not np.isnan(row[0])]
        full_candidates = [tuple(round(float(v), ROUND) for v in row) for row in cache["full_balls"][i] if not np.isnan(row[0])]
        t_s = round(float(cache["t_s"][i]), 6)
        pitch = cache["head_pitch_deg"][i]
        head_pitch = round(float(pitch), ROUND) if np.isfinite(pitch) else None
        record = pipeline.step(t_s, landmarks, crop, crop_candidates, full_candidates, head_pitch)
        frames.append({
            "input": {"t_s": t_s, "landmarks": landmarks, "crop": crop,
                      "crop_candidates": crop_candidates, "full_candidates": full_candidates,
                      "head_pitch_deg": head_pitch},
            "expected": {k: record[k] for k in ("ball", "ball_xy", "predicted", "searched_full_frame",
                                                 "ran_full_frame", "events", "touch_rate")},
        })
    return {
        "description": (f"FramePipeline.step on {PIPELINE_CLIP}, cached model outputs from frame "
                        f"{PIPELINE_START_FRAME}, every {PIPELINE_STRIDE}nd frame (30 fps), fresh state. "
                        "landmarks are 33 x (pixel x, pixel y, visibility), null when no person was found. "
                        "Candidates are (x1, y1, x2, y2, confidence) in full-frame pixels; full_candidates "
                        "must only be used on frames where needs_full_frame is true. head_pitch_deg is "
                        "the Face Landmarker pitch (sampled at 10 Hz and held), null when no face was found."),
        "config": config_dict(CFG),
        "frames": frames,
    }


def main() -> None:
    names = []
    for name, (description, observations, cfg) in metrics_scenarios().items():
        write(f"metrics/{name}", {
            "description": description,
            "config": config_dict(cfg),
            "observations": [asdict(o) for o in observations],
            "expected": run_metrics(observations, cfg),
        })
        names.append(f"metrics/{name}")
    write("filters", filter_vectors())
    write("search_step", search_vectors())
    names += ["filters", "search_step"]
    if (Path("cache") / f"{PIPELINE_CLIP}.npz").exists():
        write(f"pipeline/{PIPELINE_CLIP}", pipeline_vector())
        names.append(f"pipeline/{PIPELINE_CLIP}")
    else:
        print(f"Skipped the pipeline vector: cache/{PIPELINE_CLIP}.npz not found")
    print(f"Wrote {len(names)} vectors to {OUT}/")


if __name__ == "__main__":
    main()
