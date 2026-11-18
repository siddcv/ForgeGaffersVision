"""MediaPipe Face Landmarker wrapper for causal head-pitch estimation.

Face Landmarker is sampled at a configurable rate (normally 10 Hz), rather than
running on every video frame. The latest sample is held until the next sample;
no future frames are used.
"""
from pathlib import Path
import urllib.request

import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_PATH = Path("models/face_landmarker.task")


def pitch_from_transformation_matrix(matrix) -> float | None:
    """Return face pitch in degrees from MediaPipe's 4x4 face transform.

    The face's horizontal axis is the rotation axis for nodding. This is a
    camera-relative angle, not a neutral-calibrated anatomical angle: camera
    orientation and the player's natural stance can shift its baseline. Use
    labelled upright/downward samples to choose direction and thresholds.
    """
    try:
        transform = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        return None

    rotation = transform[:3, :3]
    try:
        u, _, vh = np.linalg.svd(rotation)
    except np.linalg.LinAlgError:
        return None
    rotation = u @ vh
    # Keep a proper rotation (determinant +1), not a reflection.
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh

    # Extract rotation about the face's horizontal (x) axis.
    angle_rad = np.arctan2(rotation[2, 1], rotation[1, 1])
    pitch = -float(np.degrees(angle_rad))
    if not np.isfinite(pitch):
        return None
    # Keep values in the principal range for stable thresholding.
    return float((pitch + 180.0) % 360.0 - 180.0)


class FacePoseEstimator:
    """MediaPipe Face Landmarker in VIDEO mode; returns pitch or None."""

    def __init__(self, model_path: Path = MODEL_PATH):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceLandmarker, FaceLandmarkerOptions, RunningMode,
        )

        model_path = Path(model_path)
        if not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(MODEL_URL, model_path)
        self._mp = mp
        self._landmarker = FaceLandmarker.create_from_options(
            FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=RunningMode.VIDEO,
                num_faces=1,
                output_facial_transformation_matrixes=True,
            )
        )

    def detect(self, rgb: np.ndarray, t_ms: int) -> float | None:
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, int(t_ms))
        matrices = result.facial_transformation_matrixes
        if not matrices:
            return None
        return pitch_from_transformation_matrix(matrices[0])

    def close(self) -> None:
        self._landmarker.close()
