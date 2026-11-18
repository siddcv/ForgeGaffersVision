"""Frame loader with real per-frame timestamps."""
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Frame:
    index: int
    t_s: float
    bgr: np.ndarray


def timestamps_path(video_path: str) -> Path:
    """Sidecar written by live.py: one real capture time per frame, because OpenCV's video writer
    stores a constant frame rate and would lose the camera's actual timing."""
    return Path(video_path).with_suffix(".timestamps.csv")


def read_timestamps(path: str) -> np.ndarray:
    """Return each frame's presentation time in seconds, starting at 0.

    Uses the live recording's timestamps sidecar when there is one, else the file's own timestamps.
    """
    sidecar = timestamps_path(path)
    if sidecar.exists():
        times = np.loadtxt(sidecar, delimiter=",", skiprows=1, ndmin=1)
        return times - times[0]
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "frame=best_effort_timestamp_time",
            "-of", "csv=p=0", path,
        ],
        capture_output=True, text=True, check=True,
    )
    times = []
    for line in result.stdout.splitlines():
        field = line.split(",")[0].strip()
        if field and field != "N/A":
            times.append(float(field))
    if not times:
        raise ValueError(f"No frame timestamps found in {path}")
    return np.array(times) - times[0]


def iter_frames(path: str) -> Iterator[Frame]:
    """Yield upright BGR frames. OpenCV applies the rotation stored in the file."""
    timestamps = read_timestamps(path)
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise ValueError(f"Could not open {path}")
    try:
        for index, t_s in enumerate(timestamps):
            ok, bgr = capture.read()
            if not ok:
                return
            yield Frame(index, float(t_s), bgr)
    finally:
        capture.release()
