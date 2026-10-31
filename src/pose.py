"""MediaPipe Pose wrapper and the geometry that reads its landmarks."""
import math
import urllib.request
from pathlib import Path

import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
MODEL_PATH = Path("models/pose_landmarker_lite.task")

NOSE, LEFT_EAR, RIGHT_EAR = 0, 7, 8
LEFT_HEEL, RIGHT_HEEL, LEFT_TOE, RIGHT_TOE = 29, 30, 31, 32
LEFT_FOOT = (LEFT_HEEL, LEFT_TOE)
RIGHT_FOOT = (RIGHT_HEEL, RIGHT_TOE)
FACE = tuple(range(11))
SKELETON = (
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    # Arms and hands
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # Torso and legs
    (11, 12), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (29, 31), (27, 31), (28, 30), (30, 32), (28, 32),
)


class PoseEstimator:
    """Returns a (33, 3) array of pixel x, pixel y, visibility, or NaNs if no person."""

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            PoseLandmarker, PoseLandmarkerOptions, RunningMode,
        )

        if not MODEL_PATH.exists():
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        self._mp = mp
        self._landmarker = PoseLandmarker.create_from_options(
            PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
                running_mode=RunningMode.VIDEO,
            )
        )

    def detect(self, rgb: np.ndarray, t_ms: int) -> np.ndarray:
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, t_ms)
        landmarks = np.full((33, 3), np.nan)
        if result.pose_landmarks:
            height, width = rgb.shape[:2]
            for i, point in enumerate(result.pose_landmarks[0]):
                landmarks[i] = point.x * width, point.y * height, point.visibility
        return landmarks

    def close(self) -> None:
        self._landmarker.close()


def visible_points(landmarks: np.ndarray, indices, min_visibility: float) -> tuple:
    """Return (x, y) for the listed landmarks that are visible enough."""
    return tuple(
        (float(landmarks[i, 0]), float(landmarks[i, 1]))
        for i in indices
        if landmarks[i, 2] >= min_visibility
    )


def head_box(landmarks: np.ndarray, min_visibility: float, padding: float) -> tuple | None:
    """Box around the visible face landmarks, padded by a fraction of its larger side."""
    points = visible_points(landmarks, FACE, min_visibility)
    if len(points) < 2:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    margin = padding * max(max(xs) - min(xs), max(ys) - min(ys))
    return min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin


def coarse_head_pitch_deg(landmarks: np.ndarray, min_visibility: float) -> float | None:
    """Rough pitch from nose height relative to the ears; negative is looking down.

    This is the pose-only fallback and is uncalibrated, so the zero point is not
    exactly level. Face Landmarker replaces it in build step 7.
    """
    # Written so that NaN visibility (no person detected) also returns None.
    if not all(landmarks[i, 2] >= min_visibility for i in (NOSE, LEFT_EAR, RIGHT_EAR)):
        return None
    ear_span = math.dist(landmarks[LEFT_EAR, :2], landmarks[RIGHT_EAR, :2])
    if ear_span == 0:
        return None
    ear_y = (landmarks[LEFT_EAR, 1] + landmarks[RIGHT_EAR, 1]) / 2
    return -math.degrees(math.atan2(landmarks[NOSE, 1] - ear_y, ear_span))
