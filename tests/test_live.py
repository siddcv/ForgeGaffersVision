import pytest
import cv2
import numpy as np

from src.live import FrameSource


def test_player_report_stays_visible_until_continue_is_clicked(monkeypatch):
    from src.live import WINDOW, DrillButton, show_player_report

    button = DrillButton()
    shown = []
    rendered_text = []
    monkeypatch.setattr(cv2, "imshow", lambda window, image: shown.append(window))
    monkeypatch.setattr(cv2, "putText", lambda image, text, *args: rendered_text.append(text))
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *args: -1)

    waits = 0

    def wait_key(delay):
        nonlocal waits
        waits += 1
        if waits == 2:
            x, y, _, _ = button.rect
            button.on_mouse(cv2.EVENT_LBUTTONDOWN, x + 1, y + 1, None, None)
        return -1 if waits <= 2 else ord("q")

    monkeypatch.setattr(cv2, "waitKey", wait_key)
    lines = ["Drill report", "Head-down time  20.0%", "Ball control lost for  5.0 seconds",
             "Average pace  10.0 tick-tocks per minute", "Tick-tocks performed  5"]

    assert show_player_report(lines, button, "CONTINUE")
    assert shown == [WINDOW]
    assert rendered_text == [
        "Drill report", "Tick-tocks performed", "5", "Head-down time", "20.0%",
        "Ball control lost for", "5.0", "seconds", "Average pace", "10.0", "tick-tocks per minute",
        "CONTINUE",
    ]
    assert waits == 2
    assert not button.clicked


@pytest.mark.parametrize("frames, duration", [(20, 30.0), (1, 0.0), (0, 0.0)])
def test_save_drill_writes_player_report(tmp_path, monkeypatch, frames, duration):
    import json
    from types import SimpleNamespace

    from src.config import Config
    from src.live import Drill, save_drill

    monkeypatch.chdir(tmp_path)
    cfg = Config()
    drill = Drill(cfg, "test_drill", None)
    drill.proc_ms = [1.0] * frames
    if duration:
        drill.hud.counts["touch", "left"] = 3
        drill.hud.counts["touch", "right"] = 2
        drill.observe_head_pitch(0.0, cfg.head_down_pitch_deg - 1)
        drill.observe_head_pitch(5.0, None)
        drill.observe_head_pitch(10.0, cfg.head_down_pitch_deg)
        drill.observe_head_pitch(duration, None)
        drill.events = [
            {"kind": "lost_control", "source": "rhythm", "t_s": 1.0},
            {"kind": "lost_control", "source": "spatial", "t_s": 5.0},
            {"kind": "reset", "t_s": 7.0},
            {"kind": "lost_control", "source": "spatial", "t_s": 27.0},
        ]
    source = SimpleNamespace(frame_size=(1280, 720), last_drill_frames=frames)
    args = SimpleNamespace(source="0", manifest="")

    lines = save_drill(drill, None, source, args, is_camera=True)

    output = tmp_path / "output" / "live"
    assert (output / "test_drill.txt").read_text().splitlines() == lines
    data = json.loads((output / "test_drill.json").read_text())
    assert data["events"] == drill.events
    assert data["summary"]["frames_processed"] == frames
    if duration:
        assert lines == [
            "Drill report",
            "Head-down time  20.0% of tracked head time",
            "Ball control lost for  5.0 seconds",
            "Average pace  10.0 tick-tocks per minute",
            "Tick-tocks performed  5",
        ]
        assert data["summary"]["head_tracked_s"] == 25.0
    else:
        assert "Head-down time  Unavailable" in lines
        assert "Ball control lost for  Unavailable" in lines
        assert "Average pace  Unavailable" in lines
        assert lines[-1] == f"Tick-tocks performed  {'0' if frames else 'Unavailable'}"


def write_clip(path, n_frames, fps=30.0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for value in range(n_frames):
        writer.write(np.full((48, 64, 3), value * 8, dtype=np.uint8))
    writer.release()


def test_clip_source_plays_in_order_and_ends(tmp_path):
    path = tmp_path / "clip.mp4"
    write_clip(path, 10)
    source = FrameSource(str(path))
    seen, last = [], -1
    while (item := source.next_after(last)) is not None:
        index, t_s, bgr, arrival = item
        assert index > last and bgr.shape == (48, 64, 3)
        seen.append(t_s)
        last = index
    source.stop()
    assert source.grabbed == 10
    assert seen == sorted(seen) and seen[-1] > 0.25  # played at its own pace, newest frames reached


def test_rotate_and_flip_are_applied_before_processing(tmp_path):
    path = tmp_path / "clip.mp4"
    write_clip(path, 2)
    source = FrameSource(str(path), rotate=90, flip=True)
    item = source.next_after(-1)
    source.stop()
    assert item[2].shape == (64, 48, 3)  # portrait after a 90 degree turn


def test_video_writer_falls_back_past_a_codec_that_does_not_open(tmp_path):
    from src.live import open_video_writer

    writer, path = open_video_writer(tmp_path / "s.mp4", 30.0, (64, 48), codecs=(("mp4v", ".notavideo"), ("mp4v", ".mp4")))
    for _ in range(3):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()
    assert path == tmp_path / "s.mp4" and cv2.VideoCapture(str(path)).read()[0]


def test_video_writer_fails_loudly_when_no_codec_opens(tmp_path):
    import pytest

    from src.live import open_video_writer

    with pytest.raises(RuntimeError, match="--no-record"):
        open_video_writer(tmp_path / "s.mp4", 30.0, (64, 48), codecs=(("mp4v", ".notavideo"),))
    assert not list(tmp_path.iterdir())  # nothing half-written left behind


class FakeCamera:
    """Stands in for cv2.VideoCapture(index): endless 64 x 48 frames at about 100 fps."""

    def __init__(self, index):
        self.n = 0

    def isOpened(self):
        return True

    def set(self, prop, value):
        return True

    def get(self, prop):
        return 30.0

    def read(self):
        import time

        time.sleep(0.01)
        self.n += 1
        return True, np.full((48, 64, 3), self.n % 250, np.uint8)

    def release(self):
        pass


def wait_frames(source, n, last=-1):
    """Consume frames until n more have arrived; returns the last index seen."""
    target = last + n
    while last < target:
        last = source.next_after(last)[0]
    return last


def frames_in(path):
    capture = cv2.VideoCapture(str(path))
    frames = 0
    while capture.read()[0]:
        frames += 1
    return frames


def test_each_drill_is_recorded_from_start_to_stop(tmp_path, monkeypatch):
    import src.live as live
    from src.video_io import read_timestamps

    monkeypatch.setattr(live, "open_camera", FakeCamera)
    source = live.FrameSource("0")
    last = wait_frames(source, 5)  # READY: shown but not recorded
    assert source.recording_path is None and source.drill_t0 is None

    source.start_drill(tmp_path / "drill_1.mp4")
    last = wait_frames(source, 20, last)
    first = source.stop_drill()
    last = wait_frames(source, 5, last)  # READY again between drills
    source.start_drill(tmp_path / "drill_2.mp4")
    last = wait_frames(source, 10, last)
    second = source.stop_drill()
    source.stop()

    assert (first, second) == (tmp_path / "drill_1.mp4", tmp_path / "drill_2.mp4")
    for path, at_least in ((first, 20), (second, 10)):
        times = read_timestamps(str(path))  # from the sidecar, starting at 0 on the drill's first frame
        assert frames_in(path) == len(times) >= at_least
        assert times[0] == 0 and (times[1:] > times[:-1]).all()
    assert source.recording_path is None and source.drill_t0 is None


def test_overlay_video_plays_at_real_speed(tmp_path):
    from src.live import OverlayRecorder

    recorder = OverlayRecorder(tmp_path / "drill_overlay.mp4")
    # Views arrive unevenly, as the live loop shows them (about 12 per second), over a 2 s drill.
    for i, t_s in enumerate([0.0, 0.07, 0.2, 0.31, 0.5, 0.9, 1.3, 1.5]):
        recorder.add(np.full((48, 64, 3), 30 * i, np.uint8), t_s)
    path = recorder.close(end_t_s=2.0)

    capture = cv2.VideoCapture(str(path))
    assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(30.0)
    frames = []
    while (frame := capture.read())[0]:
        frames.append(frame[1])
    assert len(frames) == 60  # 2.0 s at 30 fps
    # The view shown at 0.5 s (index 4) stays on screen until the next one at 0.9 s.
    # Compression shifts flat gray by a few levels; neighbouring views differ by 30.
    assert abs(int(frames[20][0, 0, 0]) - 120) <= 8 and abs(int(frames[26][0, 0, 0]) - 120) <= 8


def test_no_overlay_video_when_nothing_was_shown(tmp_path):
    from src.live import OverlayRecorder

    assert OverlayRecorder(tmp_path / "x.mp4").close(5.0) is None
    assert not list(tmp_path.iterdir())


def test_camera_report_warns_about_a_landscape_frame():
    from src.live import camera_report, orientation

    assert orientation((720, 1280)) == "portrait" and orientation((1280, 720)) == "landscape"
    assert "WARNING" not in camera_report((720, 1280), 30.0)
    report = camera_report((1280, 720), 30.0)
    assert "landscape" in report and "--rotate 90" in report


def test_recorded_drill_gets_a_manifest_row(tmp_path):
    import csv
    from datetime import datetime

    from src.live import append_manifest_row

    manifest = tmp_path / "manifest.csv"
    manifest.write_text("clip,session,date,take,fps,resolution,orientation,drill_start_s,drill_end_s,split,notes\r\n"
                        "old,s,d,tick-tock,60,1080x1920,portrait,,,tune,\r\n")
    summary = {"drill_s": 21.37, "camera_fps": 29.97, "frame_size": [720, 1280], "orientation": "portrait"}
    append_manifest_row(manifest, tmp_path / "drill_20261119_101500.mp4", summary, datetime(2026, 11, 19, 10, 15))

    rows = list(csv.DictReader(open(manifest, newline="")))
    assert rows[0]["clip"] == "old"  # existing rows untouched
    new = rows[1]
    assert (new["clip"], new["session"], new["date"]) == ("drill_20261119_101500", "2026-11-19_live", "2026-11-19")
    assert (new["fps"], new["resolution"], new["orientation"]) == ("30", "720x1280", "portrait")
    assert (new["drill_start_s"], new["drill_end_s"], new["split"]) == ("0", "21.37", "")


def standing_player(width=720, height=1280):
    """Landmarks of a player standing in the middle of a portrait frame, every point visible."""
    from src.pose import FACE, LEFT_FOOT, RIGHT_FOOT
    landmarks = np.full((33, 3), [width / 2, height / 2, 0.9])
    for k, i in enumerate(FACE):  # face around (360, 200), 60 px wide
        landmarks[i, :2] = width / 2 - 30 + 6 * k, 200 + (k % 3) * 5
    for i, x in zip(LEFT_FOOT + RIGHT_FOOT, (420, 440, 300, 280)):
        landmarks[i, :2] = x, 1150
    return landmarks


BALL = (340, 1100, 380, 1140, 0.8)


def test_framing_ok_when_head_feet_and_ball_are_in_frame():
    from src.config import Config
    from src.live import framing_problems
    assert framing_problems(standing_player(), BALL, 720, 1280, Config()) == []


def test_framing_reports_each_missing_part():
    from src.config import Config
    from src.live import framing_problems
    from src.pose import LEFT_TOE, RIGHT_HEEL
    cfg = Config()
    no_person = np.full((33, 3), np.nan)
    assert framing_problems(no_person, BALL, 720, 1280, cfg) == ["no player found"]

    cut_feet = standing_player()
    cut_feet[LEFT_TOE, 1] = 1290  # past the bottom edge
    cut_feet[RIGHT_HEEL, 2] = 0.1  # pose unsure: off frame or hidden
    assert framing_problems(cut_feet, BALL, 720, 1280, cfg) == ["left foot not in frame", "right foot not in frame"]

    head_cut = standing_player()
    head_cut[:11, 1] -= 160  # eyes at ~40 px: the top of the head is above the frame
    assert framing_problems(head_cut, BALL, 720, 1280, cfg) == ["head not in frame"]
    no_face = standing_player()
    no_face[:11, 2] = 0.0
    assert "head not in frame" in framing_problems(no_face, BALL, 720, 1280, cfg)

    assert framing_problems(standing_player(), None, 720, 1280, cfg) == ["ball not found"]
    edge_ball = (690, 1100, 730, 1140, 0.8)
    assert framing_problems(standing_player(), edge_ball, 720, 1280, cfg) == ["ball not in frame"]


def test_framing_gate_waits_before_enabling_and_ignores_brief_misses():
    from src.config import Config
    from src.live import FramingGate
    cfg = Config(framing_hold_s=0.5, framing_grace_s=0.5)
    gate = FramingGate(cfg)
    t = [i / 30 for i in range(120)]
    assert not gate.update(t[0], [])
    assert not gate.update(t[10], [])  # 0.33 s in frame
    assert gate.update(t[16], [])  # 0.53 s
    assert gate.update(t[17], ["ball not found"])  # one missed frame: still enabled
    assert gate.update(t[18], [])
    assert gate.update(t[20], ["right foot not in frame"])
    assert gate.update(t[30], ["right foot not in frame"])  # 0.33 s out
    assert not gate.update(t[36], ["right foot not in frame"])  # 0.53 s out: disabled
    assert gate.problems == ["right foot not in frame"]


def test_recenter_warning_is_about_the_body_and_waits_before_showing():
    from src.config import Config
    from src.live import FramingGate, body_problems
    assert body_problems(["left foot not in frame", "ball not found", "ball not in frame"]) == ["left foot not in frame"]
    assert body_problems(["ball not found"]) == []  # a lost ball is lost control, not a framing problem
    recenter = FramingGate(Config(framing_hold_s=0.5, framing_grace_s=0.5), ready=True)
    assert recenter.ready and recenter.problems == []  # the drill starts in frame (START needed it)
    assert recenter.update(0.0, ["head not in frame"])
    assert recenter.update(0.3, ["head not in frame"])  # brief: no warning yet
    assert not recenter.update(0.6, ["head not in frame"])  # out 0.6 s: RECENTER YOURSELF
    assert not recenter.update(0.8, [])
    assert not recenter.update(1.2, [])  # back 0.4 s: still shown
    assert recenter.update(1.4, [])  # back 0.6 s: warning cleared
