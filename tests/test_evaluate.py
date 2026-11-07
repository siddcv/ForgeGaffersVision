import json
 
from src.config import Config
from src.evaluate import (
    Mark, clip_name, drill_window, evaluate, in_window, load_detections, load_labels, match,
    score_kind,
)
 
CFG = Config()
 
 
def touch(t, foot="left", reported=None):
    return Mark("touch", t, foot, reported)
 
 
def test_perfect_detection_scores_one():
    labels = [touch(1.0), touch(1.5, "right")]
    detections = [touch(1.02), touch(1.48, "right")]
    scores = score_kind(labels, detections, 0.1)
    assert scores["precision"] == scores["recall"] == scores["foot_accuracy"] == 1.0
 
 
def test_miss_and_false_alarm_are_counted():
    labels = [touch(1.0), touch(2.0)]
    detections = [touch(1.03), touch(3.0)]
    scores = score_kind(labels, detections, 0.1)
    assert (scores["hits"], scores["misses"], scores["false_alarms"]) == (1, 1, 1)
    assert scores["missed_at_s"] == [2.0]
    assert scores["false_alarm_at_s"] == [3.0]
 
 
def test_detection_outside_tolerance_does_not_match():
    pairs, missed, false = match([touch(1.0)], [touch(1.2)], 0.1)
    assert pairs == [] and len(missed) == 1 and len(false) == 1
 
 
def test_each_detection_is_used_once_and_closest_wins():
    # One detection sits between two labels; it should pair with the closer label only.
    pairs, missed, false = match([touch(1.0), touch(1.12)], [touch(1.1)], 0.1)
    assert [(l.t_s, d.t_s) for l, d in pairs] == [(1.12, 1.1)]
    assert [m.t_s for m in missed] == [1.0]
 
 
def test_wrong_foot_is_a_hit_but_lowers_foot_accuracy():
    scores = score_kind([touch(1.0, "left")], [touch(1.0, "right")], 0.1)
    assert scores["hits"] == 1
    assert scores["foot_accuracy"] == 0.0
    assert scores["wrong_foot_at_s"] == [1.0]
 
 
def test_latency_uses_reported_time():
    scores = score_kind([touch(1.0)], [touch(1.0, reported=1.05)], 0.1)
    assert abs(scores["mean_latency_ms"] - 50) < 1e-6
 
 
def test_no_labels_or_detections_gives_no_ratio():
    scores = score_kind([], [], 0.1)
    assert scores["precision"] is None and scores["recall"] is None and scores["f1"] is None
 
 
def test_kinds_use_their_own_tolerance():
    labels = [Mark("lost_control", 8.2), touch(1.0)]
    detections = [Mark("lost_control", 8.5), touch(1.3)]
    results = evaluate(labels, detections, CFG)
    assert results["lost_control"]["hits"] == 1  # 0.3 s is within the 0.5 s event tolerance
    assert results["touch"]["hits"] == 0  # 0.3 s is outside the 0.1 s touch tolerance
 
 
def test_drill_window_filters_marks():
    marks = [touch(0.5), touch(2.0), touch(9.0)]
    assert [m.t_s for m in in_window(marks, 1.0, 8.0)] == [2.0]
    assert len(in_window(marks, None, None)) == 3
 
 
def test_loaders_read_template_format(tmp_path):
    labels_csv = tmp_path / "clip.csv"
    labels_csv.write_text(
        "kind,t_s,end_s,foot,note\n"
        "touch,1.33,,left,\n"
        "lost_control,8.20,10.60,,ball rolled away\n"
        "head_down,12.00,14.50,,\n"
    )
    labels = load_labels(labels_csv)
    assert [(m.kind, m.t_s, m.foot) for m in labels] == [
        ("touch", 1.33, "left"), ("lost_control", 8.2, None), ("head_down", 12.0, None),
    ]
 
    report = tmp_path / "clip_stride1" / "report.json"
    report.parent.mkdir()
    report.write_text(json.dumps({"summary": {}, "events": [
        {"kind": "touch", "t_s": 1.3, "reported_t_s": 1.34, "foot": "left", "value": None},
        {"kind": "speed", "t_s": 1.4, "reported_t_s": 1.5, "foot": "left", "value": 1.2},
        {"kind": "head_up", "t_s": 14.5, "reported_t_s": 14.5, "foot": None, "value": 2.5},
    ]}))
    assert [m.kind for m in load_detections(report)] == ["touch"]  # speed and head_up are not scored
    assert clip_name(report) == "clip"
 
 
def test_drill_window_reads_manifest(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("clip,drill_start_s,drill_end_s\nabc,1.5,20\nxyz,,\n")
    assert drill_window(manifest, "abc") == (1.5, 20.0)
    assert drill_window(manifest, "xyz") == (None, None)
    assert drill_window(manifest, "missing") == (None, None)
    assert drill_window(tmp_path / "nope.csv", "abc") == (None, None)
 
Claude finished the response
