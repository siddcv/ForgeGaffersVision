"""Score a live drill against hand labels, next to a full-rate replay of the same video.

Usage:
    python -m src.evaluate_live output/live/drill_<date>_<time>.json
    python -m src.evaluate_live output/live/drill_<date>_<time>.json --labels data/labels/drill_<date>_<time>.csv
    python -m src.evaluate_live output/live/drill_<date>_<time>.json --no-replay

Two runs of the same drill are scored by the same rules (evaluate.py):
  live     the events the live screen reported (output/live/<drill>.json), with frames skipped
           whenever the laptop fell behind
  replay   the drill's recording replayed through the same per-frame pipeline, every frame
           (from cache/<clip>.npz; the models are run once to build it if it does not exist)

Comparing them says where errors come from:
  wrong live only     skipped frames: the laptop is too slow for that moment
  wrong in both       the rules, the detector or the pose model: watch that moment in the
                      overlay video (output/live/<drill>_overlay.mp4) to see which
  wrong replay only   rare; live happened to skip a frame that would have misled it

Labels: data/labels/<clip>.csv, where <clip> is the recording's name (drill_<date>_<time>), or the
clip's name when a clip was played live (python -m src.live --source <clip>). Make them with
python -m src.label_clip <recording>. Writes output/live/<drill>.evaluation.json; nothing in
output/<clip>_stride<N>/ is touched.

This is offline scoring, so it may look at the whole drill at once.
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .config import Config
from .evaluate import SCORED_KINDS, Mark, drill_window, evaluate, in_window, load_labels

VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".MOV", ".MP4", ".AVI")


def clip_name(summary: dict) -> str:
    """The name labels and caches are filed under: the recording's for a camera drill, the
    clip's for a clip played live."""
    if summary.get("recording"):
        return Path(summary["recording"]).stem
    return Path(summary["source"]).stem


def find_video(summary: dict, clip: str, clips_dir: Path = Path("data/clips")) -> Path | None:
    """The drill's video: where live saved it, or wherever it was filed since (tune/, test/)."""
    for key in ("recording", "source"):
        value = summary.get(key)
        if value and not str(value).isdigit() and Path(value).exists():
            return Path(value)
    matches = [p for p in clips_dir.rglob(f"{clip}.*") if p.suffix in VIDEO_SUFFIXES]
    return matches[0] if matches else None


def marks_from_events(events: list) -> list:
    """Live summary or replay events -> the Marks evaluate() scores."""
    return [
        Mark(e["kind"], float(e["t_s"]), e.get("foot"), e.get("reported_t_s"))
        for e in events
        if e["kind"] in SCORED_KINDS
    ]


def replay_events(clip: str, video: Path | None, cfg: Config, cache_dir: Path = Path("cache")) -> list:
    """Every-frame replay of the drill through the shared pipeline, in memory."""
    import numpy as np

    from .run_clip import build_cache, replay

    cache_path = cache_dir / f"{clip}.npz"
    rebuild = not cache_path.exists()
    if not rebuild:
        with np.load(cache_path) as existing:
            rebuild = "head_pitch_deg" not in existing.files  # live uses Face Landmarker pitch
    if rebuild:
        if video is None:
            raise SystemExit(f"No cache/{clip}.npz and no video found for {clip}; pass --no-replay, "
                             "or put the recording back under data/clips/.")
        print(f"Running the models on {video} once to build {cache_path} (a few minutes)")
        build_cache(str(video), cache_path, cfg)
    cache = dict(np.load(cache_path))
    return [asdict(event) for record in replay(cache, cfg) for event in record["events"]]


def _key_misses(results: dict, kind: str) -> set:
    return set(results.get(kind, {}).get("missed_at_s", []))


def _key_wrong_foot(results: dict) -> set:
    return set(results.get("touch", {}).get("wrong_foot_at_s", []))


def split(live: set, replay: set) -> dict:
    """Label times wrong in both runs, in live only, in replay only."""
    return {"both": sorted(live & replay), "live_only": sorted(live - replay),
            "replay_only": sorted(replay - live)}


def split_false_alarms(live: list, replay: list, tolerance_s: float) -> dict:
    """False alarms (detection times) shared by both runs within tolerance_s, or in one only."""
    replay_left = sorted(replay)
    both, live_only = [], []
    for t in sorted(live):
        nearest = min(replay_left, key=lambda r: abs(r - t), default=None)
        if nearest is not None and abs(nearest - t) <= tolerance_s:
            both.append(t)
            replay_left.remove(nearest)
        else:
            live_only.append(t)
    return {"both": both, "live_only": live_only, "replay_only": replay_left}


def compare(live: dict, replay: dict) -> dict:
    """Per kind: which misses, false alarms and wrong feet come from live alone."""
    out = {}
    for kind in sorted(set(live) | set(replay)):
        tolerance = (live.get(kind) or replay.get(kind))["tolerance_s"]
        out[kind] = {
            "misses": split(_key_misses(live, kind), _key_misses(replay, kind)),
            "false_alarms": split_false_alarms(live.get(kind, {}).get("false_alarm_at_s", []),
                                               replay.get(kind, {}).get("false_alarm_at_s", []), tolerance),
        }
        if kind == "touch":
            out[kind]["wrong_foot"] = split(_key_wrong_foot(live), _key_wrong_foot(replay))
    return out


def _cell(value) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def print_side_by_side(live: dict, replay: dict | None) -> None:
    rows = ("labelled", "detected", "hits", "misses", "false_alarms", "precision", "recall", "foot_accuracy")
    for kind in SCORED_KINDS:
        if kind not in live and not (replay and kind in replay):
            continue
        print(f"\n{kind}" + ("" if replay is None else f"{'live':>12}{'replay':>12}"))
        for row in rows:
            a = live.get(kind, {}).get(row)
            b = None if replay is None else replay.get(kind, {}).get(row)
            if a is None and b is None:
                continue
            print(f"  {row:<16}{_cell(a):>10}" + ("" if replay is None else f"{_cell(b):>12}"))


def print_comparison(comparison: dict) -> None:
    print("\nWhere the errors come from (times in s):")
    for kind, parts in comparison.items():
        for what, groups in parts.items():
            if not any(groups.values()):
                continue
            print(f"  {kind} {what}: " + "; ".join(
                f"{name.replace('_', ' ')} {times}" for name, times in groups.items() if times))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summary", help="output/live/<drill>.json written by src.live")
    parser.add_argument("--labels", help="defaults to data/labels/<clip>.csv")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--no-replay", action="store_true", help="score the live run only")
    args = parser.parse_args()

    cfg = Config()
    summary_path = Path(args.summary)
    report = json.loads(summary_path.read_text())
    summary = report["summary"]
    clip = clip_name(summary)
    labels_path = Path(args.labels) if args.labels else Path("data/labels") / f"{clip}.csv"
    if not labels_path.exists():
        raise SystemExit(f"No labels at {labels_path}. Label the recording first: "
                         f"python -m src.label_clip <path to {clip}.mp4>")

    start_s, end_s = drill_window(Path(args.manifest), clip)
    labels = in_window(load_labels(labels_path), start_s, end_s)
    live = evaluate(labels, in_window(marks_from_events(report["events"]), start_s, end_s), cfg)
    replay = None
    if not args.no_replay:
        events = replay_events(clip, find_video(summary, clip), cfg)
        replay = evaluate(labels, in_window(marks_from_events(events), start_s, end_s), cfg)

    drill_s = summary.get("drill_s") or 0
    processed = summary.get("frames_processed") or 0
    output = {
        "drill": summary_path.stem,
        "clip": clip,
        "labels": str(labels_path),
        "drill_window_s": [start_s, end_s],
        "live_processing": {
            "fps": round(processed / drill_s, 1) if drill_s else None,
            "feedback_lag_ms": summary.get("feedback_lag_ms"),
        },
        "live": live,
        "replay": replay,
        "comparison": compare(live, replay) if replay is not None else None,
    }
    out_path = summary_path.with_name(f"{summary_path.stem}.evaluation.json")
    out_path.write_text(json.dumps(output, indent=2))

    print(f"Drill {summary_path.stem}, labels {labels_path}")
    if output["live_processing"]["fps"] is not None:
        lag = (summary.get("feedback_lag_ms") or {}).get("median")
        print(f"Live processed {output['live_processing']['fps']} fps, median lag {lag} ms")
    print_side_by_side(live, replay)
    if output["comparison"] is not None:
        print_comparison(output["comparison"])
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
