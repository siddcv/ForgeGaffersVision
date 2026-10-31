import cv2
import numpy as np

from src.video_io import iter_frames, read_timestamps


def test_frames_have_increasing_timestamps(tmp_path):
    path = str(tmp_path / "clip.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for value in range(30):
        writer.write(np.full((48, 64, 3), value * 8, dtype=np.uint8))
    writer.release()

    timestamps = read_timestamps(path)
    frames = list(iter_frames(path))
    assert len(frames) == len(timestamps) == 30
    assert (np.diff(timestamps) > 0).all()
    assert frames[1].t_s == timestamps[1]
    assert frames[0].bgr.shape == (48, 64, 3)
