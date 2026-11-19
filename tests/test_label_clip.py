from src.evaluate import load_labels
from src.label_clip import PASSES, Label, LabelSession, load, save, summarize


def test_lost_control_start_and_end_make_one_interval_and_a_reset():
    session = LabelSession()
    session.start("lost_control", 8.2)
    assert session.open_interval("lost_control").t_s == 8.2
    session.end("lost_control", 10.6)
    assert session.sorted_labels() == [Label("lost_control", 8.2, 10.6), Label("reset", 10.6)]
    assert session.open_interval("lost_control") is None


def test_head_down_and_up_make_an_interval_without_a_reset():
    session = LabelSession()
    session.start("head_down", 12.0)
    session.end("head_down", 14.5)
    assert session.labels == [Label("head_down", 12.0, 14.5)]


def test_cannot_end_without_a_start_or_before_it():
    session = LabelSession()
    assert "no open" in session.end("lost_control", 5.0)
    session.start("lost_control", 5.0)
    assert "after the start" in session.end("lost_control", 4.0)
    assert session.open_interval("lost_control") is not None


def test_cannot_open_twice_or_inside_a_closed_interval():
    session = LabelSession()
    session.start("head_down", 1.0)
    assert "already open" in session.start("head_down", 2.0)
    session.end("head_down", 3.0)
    assert "inside" in session.start("head_down", 2.5)
    assert len(session.labels) == 1


def test_lost_control_and_head_down_can_overlap():
    session = LabelSession()
    session.start("lost_control", 1.0)
    session.start("head_down", 1.5)
    assert session.open_interval("lost_control") and session.open_interval("head_down")


def test_touches_and_duplicates():
    session = LabelSession()
    session.touch(1.0, "left")
    assert "already" in session.touch(1.0, "left")
    session.touch(1.0, "right")  # a different foot on the same frame is kept
    assert len(session.labels) == 2


def test_undo_reverts_one_edit_at_a_time():
    session = LabelSession()
    session.touch(1.0, "left")
    session.start("lost_control", 2.0)
    session.end("lost_control", 3.0)
    session.undo()  # removes the end and its reset together
    assert session.labels == [Label("touch", 1.0, foot="left"), Label("lost_control", 2.0)]
    session.undo()
    session.undo()
    assert session.labels == [] and session.undo() == "nothing to undo"


def test_erase_nearest_takes_its_reset_with_it():
    session = LabelSession([Label("touch", 1.0, foot="left")])
    session.start("lost_control", 2.0)
    session.end("lost_control", 3.0)
    session.erase_nearest(2.95)  # near the end of the interval
    assert session.labels == [Label("touch", 1.0, foot="left")]
    assert "no label within" in session.erase_nearest(5.0)


def test_erase_in_a_pass_only_touches_that_pass():
    session = LabelSession([Label("touch", 2.95, foot="left"), Label("head_down", 1.0, 3.0)])
    session.erase_nearest(2.95, kinds=PASSES["head"][0])  # the touch is nearer, but not in this pass
    assert session.labels == [Label("touch", 2.95, foot="left")]
    assert session.erase_nearest(2.95, kinds=PASSES["lost"][0]) == "nothing to erase"


def test_passes_cover_every_kind_and_key_once():
    passes = [PASSES[p] for p in ("touches", "lost", "head")]
    keys = "".join(k for _, k, _ in passes)
    assert len(keys) == len(set(keys))
    assert PASSES["all"][0] == {"touch", "lost_control", "reset", "head_down"}


def test_passes_add_to_the_same_file(tmp_path):
    path = tmp_path / "clip.csv"
    touches = LabelSession(load(path))
    touches.touch(1.0, "left")
    save(path, touches.labels)
    lost = LabelSession(load(path))
    lost.start("lost_control", 5.0)
    lost.end("lost_control", 6.0)
    save(path, lost.labels)
    head = LabelSession(load(path))
    head.start("head_down", 2.0)
    head.end("head_down", 3.5)
    save(path, head.labels)
    assert [l.kind for l in load(path)] == ["touch", "head_down", "lost_control", "reset"]


def test_save_and_load_round_trip_and_evaluate_reads_it(tmp_path):
    path = tmp_path / "clip.csv"
    labels = [
        Label("touch", 1.3333, foot="left"),
        Label("lost_control", 8.2, 10.6),
        Label("reset", 10.6),
        Label("head_down", 12.0),  # still open
    ]
    save(path, labels)
    assert path.read_text().splitlines()[0] == "kind,t_s,end_s,foot,note"
    loaded = load(path)
    assert loaded[0] == Label("touch", 1.333, foot="left")
    assert loaded[-1] == Label("head_down", 12.0)
    resumed = LabelSession(loaded)
    assert resumed.open_interval("head_down").t_s == 12.0
    assert [(m.kind, m.t_s) for m in load_labels(path)] == [
        ("touch", 1.333), ("lost_control", 8.2), ("reset", 10.6), ("head_down", 12.0),
    ]


def test_save_retries_while_the_file_is_locked(tmp_path, monkeypatch):
    # Windows refuses the rename while OneDrive or Excel holds the file; it clears quickly.
    import src.label_clip as label_clip

    real_replace, calls = label_clip.os.replace, []

    def locked_twice(src, dst):
        calls.append(dst)
        if len(calls) <= 2:
            raise PermissionError(5, "Access is denied")
        real_replace(src, dst)

    monkeypatch.setattr(label_clip.os, "replace", locked_twice)
    path = tmp_path / "clip.csv"
    save(path, [Label("touch", 1.0, foot="left")], wait_s=0)
    assert len(calls) == 3 and load(path) == [Label("touch", 1.0, foot="left")]


def test_try_save_reports_a_lock_that_does_not_clear(tmp_path, monkeypatch):
    import src.label_clip as label_clip

    def always_locked(src, dst):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(label_clip.os, "replace", always_locked)
    monkeypatch.setattr(label_clip.time, "sleep", lambda s: None)
    message = label_clip.try_save(tmp_path / "clip.csv", [Label("touch", 1.0, foot="left")])
    assert message and message.startswith("NOT SAVED")


def test_load_missing_file_is_empty(tmp_path):
    assert load(tmp_path / "none.csv") == []


def test_summary_gives_threshold_hints():
    labels = [
        Label("touch", 1.0, foot="left"), Label("touch", 1.3, foot="right"),
        Label("touch", 1.5, foot="right"), Label("touch", 1.9, foot="left"),
        Label("lost_control", 5.0, 6.5), Label("head_down", 7.0, 8.25), Label("head_down", 9.0),
    ]
    summary = summarize(labels)
    assert (summary["touches_left"], summary["touches_right"], summary["same_foot_repeats"]) == (2, 2, 1)
    assert summary["touch_gap_min_s"] == 0.2 and summary["touch_gap_median_s"] == 0.3
    assert summary["lost_control_durations_s"] == [1.5]
    assert summary["head_down_durations_s"] == [1.25]
    assert summary["open_intervals"] == ["head_down"]
