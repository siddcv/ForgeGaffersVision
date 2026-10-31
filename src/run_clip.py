"""Clip -> cache -> streaming replay -> report and overlay.

Usage: python -m src.run_clip path/to/clip.mp4 [--stride 2] [--refresh] [--no-overlay]
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from . import ball as ball_module
from . import metrics
from .config import Config
from .filters import Kalman1D, OneEuroFilter
from .pose import LEFT_FOOT, RIGHT_FOOT, PoseEstimator, coarse_head_pitch_deg, visible_points
from .video_io import iter_frames, read_timestamps


def _to_row(detection) -> list:
    return list(detection) if detection is not None else [np.nan] * 5


def _from_row(row):
    return None if np.isnan(row[0]) else tuple(float(v) for v in row)


def build_cache(clip_path: str, cache_path: Path, cfg: Config) -> None:
    """Run the models once, saving crop and full-frame detections for every frame."""
    pose = PoseEstimator()
    detector = ball_module.YoloBallDetector(cfg)
    times, landmarks, crops, crop_balls, full_balls = [], [], [], [], []
    last_ms = -1
    try:
        for frame in iter_frames(clip_path):
            height, width = frame.bgr.shape[:2]
            t_ms = max(int(frame.t_s * 1000), last_ms + 1)  # MediaPipe needs strictly increasing
            last_ms = t_ms
            points = pose.detect(cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2RGB), t_ms)
            feet = visible_points(points, LEFT_FOOT + RIGHT_FOOT, cfg.pose_min_visibility)
            crop = ball_module.foot_crop(feet, width, height, cfg)
            crop_det = None
            if crop is not None:
                x1, y1, x2, y2 = crop
                found = detector.detect(frame.bgr[y1:y2, x1:x2])
                crop_det = ball_module.to_full_frame(found, crop) if found else None
            times.append(frame.t_s)
            landmarks.append(points)
            crops.append(list(crop) if crop else [np.nan] * 4)
            crop_balls.append(_to_row(crop_det))
            full_balls.append(_to_row(detector.detect(frame.bgr)))
            if frame.index % 100 == 0:
                print(f"  processed {frame.index} frames")
    finally:
        pose.close()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path, t_s=np.array(times), landmarks=np.array(landmarks), crops=np.array(crops),
        crop_balls=np.array(crop_balls), full_balls=np.array(full_balls),
    )


def replay(cache, cfg: Config, stride: int = 1) -> list:
    """Feed cached model outputs through the same per-frame logic the phone will run."""
    search = ball_module.SearchState()
    state = metrics.initial_state()
    make_kalman = lambda: Kalman1D(cfg.kalman_process_noise, cfg.kalman_measurement_noise)
    kalman_x, kalman_y = make_kalman(), make_kalman()
    euro = {}
    records = []

    def smooth_points(indices, points_at, t_s):
        smoothed = []
        for index in indices:
            if points_at[index, 2] < cfg.pose_min_visibility:
                continue
            xy = []
            for axis in (0, 1):
                key = (index, axis)
                if key not in euro:
                    euro[key] = OneEuroFilter(
                        cfg.one_euro_min_cutoff, cfg.one_euro_beta, cfg.one_euro_d_cutoff,
                    )
                xy.append(euro[key].update(t_s, float(points_at[index, axis])))
            smoothed.append(tuple(xy))
        return tuple(smoothed)

    for i in range(0, len(cache["t_s"]), stride):
        t_s = float(cache["t_s"][i])
        crop_row = cache["crops"][i]
        crop = None if np.isnan(crop_row[0]) else tuple(crop_row)
        search, detection, searched = ball_module.search_step(
            search, t_s, crop, _from_row(cache["crop_balls"][i]), _from_row(cache["full_balls"][i]), cfg,
        )
        ball_xy = diameter = None
        if detection is not None:
            x, y = ball_module.box_center(detection)
            ball_xy = kalman_x.update(t_s, x), kalman_y.update(t_s, y)
            diameter = ball_module.box_diameter_px(detection)
        observation = metrics.Observation(
            t_s=t_s,
            ball_xy=ball_xy,
            ball_diameter_px=diameter,
            searched_full_frame=searched,
            left_foot=smooth_points(LEFT_FOOT, cache["landmarks"][i], t_s),
            right_foot=smooth_points(RIGHT_FOOT, cache["landmarks"][i], t_s),
            head_pitch_deg=coarse_head_pitch_deg(cache["landmarks"][i], cfg.pose_min_visibility),
        )
        state, events = metrics.step(state, observation, cfg)
        records.append({
            "index": i, "t_s": t_s, "crop": crop, "ball": detection,
            "searched_full_frame": searched, "events": events,
        })
    return records


def summarize(records: list, cache, cfg: Config) -> dict:
    events = [event for record in records for event in record["events"]]
    touches = [e for e in events if e.kind == "touch"]
    speeds = {
        foot: [e.value for e in events if e.kind == "speed" and e.foot == foot]
        for foot in ("left", "right")
    }
    indices = [record["index"] for record in records]
    visibility = cache["landmarks"][indices][:, LEFT_FOOT + RIGHT_FOOT, 2]
    gaps = np.diff([record["t_s"] for record in records])
    latencies = [e.reported_t_s - e.t_s for e in touches]
    return {
        "frames": len(records),
        "duration_s": records[-1]["t_s"] if records else 0.0,
        "frame_gap_ms": {
            "median": float(np.median(gaps) * 1000),
            "max": float(gaps.max() * 1000),
        } if len(gaps) else None,
        "timestamps_strictly_increasing": bool((gaps > 0).all()),
        "low_foot_visibility_pct": float(100 * np.mean(~(visibility >= cfg.pose_min_visibility))),
        "ball_found_pct": 100 * float(np.mean([r["ball"] is not None for r in records])),
        "searching_pct": 100 * float(np.mean([r["searched_full_frame"] for r in records])),
        "touches": {
            "left": sum(e.foot == "left" for e in touches),
            "right": sum(e.foot == "right" for e in touches),
        },
        "mean_speed_mps": {
            foot: float(np.mean(values)) if values else None for foot, values in speeds.items()
        },
        "lost_control": sum(e.kind == "lost_control" for e in events),
        "resets": sum(e.kind == "reset" for e in events),
        "head_down": sum(e.kind == "head_down" for e in events),
        "mean_touch_latency_ms": float(np.mean(latencies) * 1000) if latencies else None,
    }


def save_foot_plot(records: list, cache, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    indices = [record["index"] for record in records]
    times = [record["t_s"] for record in records]
    points = cache["landmarks"][indices]
    figure, axis = plt.subplots(figsize=(12, 4))
    axis.plot(times, points[:, LEFT_FOOT[1], 0], label="left toe x")
    axis.plot(times, points[:, RIGHT_FOOT[1], 0], label="right toe x")
    ball_x = [(r["ball"][0] + r["ball"][2]) / 2 if r["ball"] else np.nan for r in records]
    axis.plot(times, ball_x, ".", markersize=3, label="ball x")
    axis.set(xlabel="time (s)", ylabel="x (pixels)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip")
    parser.add_argument("--stride", type=int, default=1, help="2 turns a 60 fps clip into 30 fps")
    parser.add_argument("--refresh", action="store_true", help="re-run the models")
    parser.add_argument("--no-overlay", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    stem = Path(args.clip).stem
    cache_path = Path("cache") / f"{stem}.npz"
    if args.refresh or not cache_path.exists():
        print(f"Running models on {args.clip}")
        build_cache(args.clip, cache_path, cfg)
    cache = dict(np.load(cache_path))

    expected = len(read_timestamps(args.clip))
    if len(cache["t_s"]) != expected:
        print(f"WARNING: decoded {len(cache['t_s'])} frames but ffprobe reports {expected}")

    records = replay(cache, cfg, args.stride)
    summary = summarize(records, cache, cfg)

    out_dir = Path("output") / f"{stem}_stride{args.stride}"
    out_dir.mkdir(parents=True, exist_ok=True)
    events = [asdict(event) for record in records for event in record["events"]]
    (out_dir / "report.json").write_text(json.dumps({"summary": summary, "events": events}, indent=2))
    save_foot_plot(records, cache, out_dir / "feet_and_ball.png")
    if not args.no_overlay:
        from .overlay import render

        render(args.clip, str(out_dir / "overlay.mp4"), records, cache["landmarks"], cfg)

    print(json.dumps(summary, indent=2))
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
