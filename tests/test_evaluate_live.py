import json
import sys

import pytest

from src import evaluate_live
from src.evaluate_live import clip_name, compare, find_video, split, split_false_alarms


def test_clip_name_comes_from_the_recording_or_the_played_clip():
    assert clip_name({"source": "0", "recording": "data/clips/live/2026-11-19/drill_101500.mp4"}) == "drill_101500"
    assert clip_name({"source": "data/clips/tune/s/20261102_134206.mp4", "recording": None}) == "20261102_134206"


def test_find_video_follows_a_recording_filed_elsewhere(tmp_path):
    filed = tmp_path / "tune" / "2026-11-19_gym" / "drill_101500.mp4"
    filed.parent.mkdir(parents=True)
    filed.write_bytes(b"x")
    (filed.parent / "drill_101500.timestamps.csv").write_text("t_s\n0\n")  # not a video
    summary = {"source": "0", "recording": str(tmp_path / "live" / "drill_101500.mp4")}  # moved away
    assert find_video(summary, "drill_101500", clips_dir=tmp_path) == filed
    assert find_video(summary, "drill_999999", clips_dir=tmp_path) is None


def test_split_sorts_errors_by_run():
    assert split({1.0, 2.0, 3.0}, {2.0, 4.0}) == {"both": [2.0], "live_only": [1.0, 3.0], "replay_only": [4.0]}


def test_false_alarms_in_both_runs_match_within_tolerance():
    groups = split_false_alarms([1.00, 5.00, 9.0], [1.08, 7.0], tolerance_s=0.2)
    assert groups == {"both": [1.0], "live_only": [5.0, 9.0], "replay_only": [7.0]}


def test_compare_reports_touch_misses_false_alarms_and_feet():
    live = {"touch": {"tolerance_s": 0.2, "missed_at_s": [1.4, 8.0], "false_alarm_at_s": [29.1],
                      "wrong_foot_at_s": [12.0]}}
    replay = {"touch": {"tolerance_s": 0.2, "missed_at_s": [1.4], "false_alarm_at_s": [29.13],
                        "wrong_foot_at_s": []}}
    result = compare(live, replay)["touch"]
    assert result["misses"] == {"both": [1.4], "live_only": [8.0], "replay_only": []}
    assert result["false_alarms"]["both"] == [29.1]
    assert result["wrong_foot"] == {"both": [], "live_only": [12.0], "replay_only": []}


def write_drill(tmp_path, events, labels_rows):
    summary = {"source": "0", "recording": "data/clips/live/2026-11-19/drill_101500.mp4",
               "drill_s": 10.0, "frames_processed": 120, "feedback_lag_ms": {"median": 80.0, "p90": 95.0}}
    summary_path = tmp_path / "drill_101500.json"
    summary_path.write_text(json.dumps({"summary": summary, "events": events}))
    labels = tmp_path / "drill_101500.csv"
    labels.write_text("kind,t_s,end_s,foot\n" + "".join(f"touch,{t},,{foot}\n" for t, foot in labels_rows))
    return summary_path, labels


def event(t, foot, kind="touch"):
    return {"frame": 0, "kind": kind, "t_s": t, "reported_t_s": t + 0.4, "foot": foot, "value": None}


def run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["evaluate_live", *argv])
    evaluate_live.main()


def test_live_only_scoring_writes_evaluation_next_to_the_summary(tmp_path, monkeypatch, capsys):
    summary_path, labels = write_drill(
        tmp_path,
        [event(1.0, "left"), event(1.4, "right"), event(1.8, "left"), event(5.0, "right", "contact")],
        [(1.05, "left"), (1.45, "right"), (2.6, "left")],
    )
    run_main(monkeypatch, str(summary_path), "--labels", str(labels), "--no-replay",
             "--manifest", str(tmp_path / "none.csv"))
    result = json.loads((tmp_path / "drill_101500.evaluation.json").read_text())
    touch = result["live"]["touch"]
    assert (touch["hits"], touch["misses"], touch["false_alarms"]) == (2, 1, 1)
    assert result["replay"] is None and result["live_processing"]["fps"] == 12.0
    assert "live" in capsys.readouterr().out


def test_live_and_replay_are_compared(tmp_path, monkeypatch):
    summary_path, labels = write_drill(
        tmp_path, [event(1.0, "left"), event(1.8, "left")], [(1.05, "left"), (1.45, "right"), (1.85, "left")],
    )
    # The replay finds the touch live skipped.
    monkeypatch.setattr(evaluate_live, "replay_events",
                        lambda clip, video, cfg: [event(1.0, "left"), event(1.42, "right"), event(1.8, "left")])
    run_main(monkeypatch, str(summary_path), "--labels", str(labels), "--manifest", str(tmp_path / "none.csv"))
    result = json.loads((tmp_path / "drill_101500.evaluation.json").read_text())
    assert result["live"]["touch"]["recall"] == pytest.approx(2 / 3)
    assert result["replay"]["touch"]["recall"] == 1.0
    assert result["comparison"]["touch"]["misses"] == {"both": [], "live_only": [1.45], "replay_only": []}


def test_missing_labels_say_how_to_make_them(tmp_path, monkeypatch):
    summary_path, _ = write_drill(tmp_path, [], [])
    with pytest.raises(SystemExit, match="label_clip"):
        run_main(monkeypatch, str(summary_path), "--labels", str(tmp_path / "missing.csv"), "--no-replay")
