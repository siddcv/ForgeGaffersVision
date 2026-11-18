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


class HeadPitchSampler:
    """Face Landmarker pitch, sampled causally every cfg.head_pose_sample_interval_s (10 Hz).

    Pose already localizes the player's head reliably. Face Landmarker misses the small face in
    the full portrait frame, so each sample crops around the pose head box, makes it square with
    1.5x padding, and enlarges it to 256 x 256 before inference. The latest sample is held until
    the next one; None means no face was found at the last sample. Used by build_cache (replay)
    and live mode, so both measure pitch the same way.
    """

    CROP_SCALE = 1.5
    INPUT_SIZE = 256

    def __init__(self, estimator: "FacePoseEstimator", cfg):
        self._estimator = estimator
        self._cfg = cfg
        self._next_sample_s = 0.0
        self.latest = None

    def update(self, bgr: np.ndarray, landmarks: np.ndarray, t_s: float, t_ms: int) -> float | None:
        """Pitch for this frame. t_ms must increase from call to call, as for MediaPipe video."""
        import cv2

        from .pose import head_box

        cfg = self._cfg
        if t_s + 1e-9 < self._next_sample_s:
            return self.latest
        self.latest = None
        box = head_box(landmarks, cfg.pose_min_visibility, cfg.head_box_padding)
        if box is not None:
            height, width = bgr.shape[:2]
            x1, y1, x2, y2 = box
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            side = max(x2 - x1, y2 - y1) * self.CROP_SCALE
            crop_x1 = max(0, int(center_x - side / 2))
            crop_y1 = max(0, int(center_y - side / 2))
            crop_x2 = min(width, int(center_x + side / 2))
            crop_y2 = min(height, int(center_y + side / 2))
            head_crop = bgr[crop_y1:crop_y2, crop_x1:crop_x2]
            if head_crop.size:
                enlarged = cv2.resize(head_crop, (self.INPUT_SIZE, self.INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
                self.latest = self._estimator.detect(cv2.cvtColor(enlarged, cv2.COLOR_BGR2RGB), t_ms)
        self._next_sample_s = t_s + cfg.head_pose_sample_interval_s
        return self.latest


def open_head_pitch_sampler(cfg) -> tuple:
    """(sampler, estimator), or (None, None) with a warning if Face Landmarker cannot start,
    in which case callers fall back to the pose-only estimate. Close the estimator when done."""
    try:
        estimator = FacePoseEstimator()
    except Exception as exc:
        print(f"  WARNING: Face Landmarker unavailable; using pose-only fallback: {exc}")
        return None, None
    print("  Face Landmarker enabled (10 Hz head-pose sampling)")
    return HeadPitchSampler(estimator, cfg), estimator
