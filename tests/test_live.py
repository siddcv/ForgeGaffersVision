import cv2
import numpy as np

from src.live import FrameSource


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
