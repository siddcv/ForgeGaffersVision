"""Score a run_clip report against hand labels.
 
Usage: python -m src.evaluate output/<clip>_stride1/report.json [--labels data/labels/<clip>.csv]
                              [--manifest data/manifest.csv]
 
Reads the events the pipeline detected (report.json) and the hand-labelled ground truth
(data/labels/<clip>.csv), pairs each label with the closest detection of the same kind
within a time tolerance, and reports precision, recall, foot accuracy and timing error.
Writes evaluation.json next to the report.
 
This is offline scoring, so unlike metrics.py it may look at the whole clip at once.
"""
import argparse
import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
 
from .config import Config
 
# Label kinds and the pipeline event kinds they are scored against.
SCORED_KINDS = ("touch", "lost_control", "reset", "head_down")
 
 
@dataclass(frozen=True)
class Mark:
    """One event in time, from either the labels or the pipeline."""
    kind: str
    t_s: float
    foot: str | None = None
    reported_t_s: float | None = None  # pipeline events only: when the system knew
 
 
def load_labels(path: Path) -> list:
    marks = []
    with open(path, newline="") as file:
        for row in csv.DictReader(file):
            kind = (row.get("kind") or "").strip()
            if kind not in SCORED_KINDS or not (row.get("t_s") or "").strip():
                continue
            foot = (row.get("foot") or "").strip() or None
            marks.append(Mark(kind, float(row["t_s"]), foot))
    return marks
 
 
def load_detections(report_path: Path) -> list:
    report = json.loads(report_path.read_text())
    return [
        Mark(e["kind"], float(e["t_s"]), e.get("foot"), e.get("reported_t_s"))
        for e in report["events"]
        if e["kind"] in SCORED_KINDS
    ]
 
 
def drill_window(manifest_path: Path, clip: str) -> tuple:
    """(start_s, end_s) from manifest.csv; None for either end that is blank or missing."""
    if not manifest_path.exists():
        return None, None
    with open(manifest_path, newline="") as file:
        for row in csv.DictReader(file):
            if row.get("clip") == clip:
                start = (row.get("drill_start_s") or "").strip()
                end = (row.get("drill_end_s") or "").strip()
                return (float(start) if start else None), (float(end) if end else None)
    return None, None
 
 
def in_window(marks: list, start_s: float | None, end_s: float | None) -> list:
    return [
        m for m in marks
        if (start_s is None or m.t_s >= start_s) and (end_s is None or m.t_s <= end_s)
    ]
 
 
def match(labels: list, detections: list, tolerance_s: float) -> tuple:
    """Pair labels with detections, closest in time first, each used at most once.
 
    Returns (pairs, missed_labels, false_detections).
    """
    candidates = sorted(
        (abs(d.t_s - l.t_s), li, di)
        for li, l in enumerate(labels)
        for di, d in enumerate(detections)
        if abs(d.t_s - l.t_s) <= tolerance_s
    )
    used_labels, used_detections, pairs = set(), set(), []
    for _, li, di in candidates:
        if li in used_labels or di in used_detections:
            continue
        used_labels.add(li)
        used_detections.add(di)
        pairs.append((labels[li], detections[di]))
    pairs.sort(key=lambda pair: pair[0].t_s)
    missed = [l for i, l in enumerate(labels) if i not in used_labels]
    false = [d for i, d in enumerate(detections) if i not in used_detections]
    return pairs, missed, false
 
 
def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None
 
 
def score_kind(labels: list, detections: list, tolerance_s: float) -> dict:
    """Precision, recall and timing for one event kind."""
    pairs, missed, false = match(labels, detections, tolerance_s)
    hits = len(pairs)
    precision = _ratio(hits, hits + len(false))
    recall = _ratio(hits, hits + len(missed))
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    timing_ms = [abs(d.t_s - l.t_s) * 1000 for l, d in pairs]
    latency_ms = [
        (d.reported_t_s - d.t_s) * 1000 for _, d in pairs if d.reported_t_s is not None
    ]
    result = {
        "labelled": len(labels),
        "detected": len(detections),
        "hits": hits,
        "misses": len(missed),
        "false_alarms": len(false),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "median_timing_error_ms": statistics.median(timing_ms) if timing_ms else None,
        "mean_latency_ms": statistics.mean(latency_ms) if latency_ms else None,
        "missed_at_s": [round(m.t_s, 3) for m in missed],
        "false_alarm_at_s": [round(d.t_s, 3) for d in false],
    }
    foot_pairs = [(l, d) for l, d in pairs if l.foot and d.foot]
    if foot_pairs:
        result["foot_accuracy"] = sum(l.foot == d.foot for l, d in foot_pairs) / len(foot_pairs)
        result["wrong_foot_at_s"] = [round(l.t_s, 3) for l, d in foot_pairs if l.foot != d.foot]
    return result
 
 
def evaluate(labels: list, detections: list, cfg: Config) -> dict:
    results = {}
    for kind in SCORED_KINDS:
        tolerance = cfg.eval_touch_tolerance_s if kind == "touch" else cfg.eval_event_tolerance_s
        kind_labels = [m for m in labels if m.kind == kind]
        kind_detections = [m for m in detections if m.kind == kind]
        if kind_labels or kind_detections:
            results[kind] = {"tolerance_s": tolerance} | score_kind(
                kind_labels, kind_detections, tolerance,
            )
    return results
 
 
def _format(value) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}" if isinstance(value, float) else str(value)
 
 
def print_table(results: dict) -> None:
    columns = ("labelled", "detected", "hits", "misses", "false_alarms", "precision", "recall")
    print(f"{'kind':<14}" + "".join(f"{c:>14}" for c in columns))
    for kind, scores in results.items():
        print(f"{kind:<14}" + "".join(f"{_format(scores[c]):>14}" for c in columns))
    if "touch" in results and "foot_accuracy" in results["touch"]:
        print(f"\nTouch foot accuracy: {results['touch']['foot_accuracy']:.0%}")
 
 
def clip_name(report_path: Path) -> str:
    """output/20261101_183403_stride1/report.json -> 20261101_183403"""
    return report_path.parent.name.rsplit("_stride", 1)[0]
 
 
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="report.json written by run_clip")
    parser.add_argument("--labels", help="defaults to data/labels/<clip>.csv")
    parser.add_argument("--manifest", default="data/manifest.csv")
    args = parser.parse_args()
 
    report_path = Path(args.report)
    clip = clip_name(report_path)
    labels_path = Path(args.labels) if args.labels else Path("data/labels") / f"{clip}.csv"
 
    start_s, end_s = drill_window(Path(args.manifest), clip)
    labels = in_window(load_labels(labels_path), start_s, end_s)
    detections = in_window(load_detections(report_path), start_s, end_s)
 
    results = evaluate(labels, detections, Config())
    output = {"clip": clip, "drill_window_s": [start_s, end_s], "results": results}
    out_path = report_path.parent / "evaluation.json"
    out_path.write_text(json.dumps(output, indent=2))
 
    print_table(results)
    print(f"Saved {out_path}")
 
 
if __name__ == "__main__":
    main()
 
