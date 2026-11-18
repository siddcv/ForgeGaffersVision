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
