"""Compare cached Face Landmarker pitch against manually labelled head-down intervals.

Usage:
  python -m src.diagnose_head_pose 20261102_134245
  python -m src.diagnose_head_pose cache/20261102_134245.npz --labels data/labels/20261102_134245.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np


def load_intervals(path: Path) -> list[tuple[float, float]]:
    intervals = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("kind") != "head_down":
                continue
            start = (row.get("t_s") or "").strip()
            end = (row.get("end_s") or "").strip()
            if start and end:
                intervals.append((float(start), float(end)))
    return intervals


def describe(name: str, values: np.ndarray) -> None:
    values = values[np.isfinite(values)]
    if not len(values):
        print(f"{name}: no valid face-pose samples")
        return
    q10, median, q90 = np.percentile(values, [10, 50, 90])
    print(
        f"{name}: n={len(values)} min={values.min():.1f} "
        f"p10={q10:.1f} median={median:.1f} p90={q90:.1f} max={values.max():.1f}"
    )
    print(
        f"  fraction <= -20 deg: {np.mean(values <= -20):.1%}; "
        f"fraction <= -15 deg: {np.mean(values <= -15):.1%}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip_or_cache", help="clip stem or path to cache/<stem>.npz")
    parser.add_argument("--labels", help="defaults to data/labels/<stem>.csv")
    args = parser.parse_args()

    supplied = Path(args.clip_or_cache)
    if supplied.suffix == ".npz":
        cache_path = supplied
        stem = supplied.stem
    else:
        stem = supplied.stem
        cache_path = Path("cache") / f"{stem}.npz"
    labels_path = Path(args.labels) if args.labels else Path("data/labels") / f"{stem}.csv"

    with np.load(cache_path) as cache:
        if "head_pitch_deg" not in cache.files:
            raise SystemExit("This cache has no Face Landmarker data. Re-run src.run_clip with --refresh.")
        times = cache["t_s"]
        pitch = cache["head_pitch_deg"]

    intervals = load_intervals(labels_path)
    down_mask = np.zeros(len(times), dtype=bool)
    for start, end in intervals:
        down_mask |= (times >= start) & (times <= end)

    print(f"Clip: {stem}")
    print(f"Manual head-down intervals: {len(intervals)}")
    print(f"Valid pitch samples: {np.isfinite(pitch).sum()}/{len(pitch)} ({np.isfinite(pitch).mean():.1%})")
    describe("MANUALLY HEAD-DOWN", pitch[down_mask])
    describe("OUTSIDE LABELLED INTERVALS", pitch[~down_mask])

    valid_down = pitch[down_mask & np.isfinite(pitch)]
    valid_up = pitch[(~down_mask) & np.isfinite(pitch)]
    if len(valid_down) and len(valid_up):
        if len(valid_down) and len(valid_up):
        print("\nThreshold sweep (lower pitch = more head-down):")
        for threshold in (5, 8, 10, 12, 15, 18, 20, 22, 25, 28, 30):
            recall = np.mean(valid_down <= threshold)
            false_positive_rate = np.mean(valid_up <= threshold)
            print(f"  pitch <= {threshold:>2} deg: labelled-down coverage={recall:.1%}, outside coverage={false_positive_rate:.1%}")
        print("\nThreshold sweep (higher pitch = more head-down; alternate direction):")
        for threshold in (5, 8, 10, 12, 15, 18, 20, 22, 25, 28, 30):
            recall = np.mean(valid_down >= threshold)
            false_positive_rate = np.mean(valid_up >= threshold)
            print(f"  pitch >= {threshold:>2} deg: labelled-down coverage={recall:.1%}, outside coverage={false_positive_rate:.1%}")



if __name__ == "__main__":
    main()
