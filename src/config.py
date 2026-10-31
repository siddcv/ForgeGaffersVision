"""All thresholds. Values are placeholders until tuned on real clips.

Durations are in seconds, not frames, so they mean the same at 30 and 60 fps.
"""
from dataclasses import dataclass

BALL_DIAMETER_M = {1: 0.153, 2: 0.170, 3: 0.190, 4: 0.205, 5: 0.220}


@dataclass(frozen=True)
class Config:
    # Ball size
    ball_size: int = 5
    ball_diameter_override_m: float | None = None
    diameter_smoothing: float = 0.2  # weight of the newest measurement

    # Touches
    touch_min_speed_mps: float = 0.2
    touch_foot_distance_diam: float = 1.5  # allows for the ball being hidden at the touch
    touch_min_interval_s: float = 0.08
    touch_max_gap_s: float = 0.25
    speed_window_s: float = 0.10

    # Lost control
    lost_distance_diam: float = 1.5
    lost_control_min_s: float = 0.5

    # Head position
    head_down_pitch_deg: float = -20.0
    head_up_pitch_deg: float = -15.0
    head_down_min_s: float = 1.0

    # Ball detection
    ball_search_after_s: float = 0.15
    ball_confidence: float = 0.15
    ball_class_id: int = 32  # COCO "sports ball"
    detector_weights: str = "yolov8n.pt"
    crop_span_scale: float = 2.0
    crop_min_frac: float = 0.25  # minimum crop side, as a fraction of frame height

    # Pose
    pose_min_visibility: float = 0.5
    head_box_padding: float = 0.5

    # Filters
    kalman_process_noise: float = 4.0e6
    kalman_measurement_noise: float = 4.0
    one_euro_min_cutoff: float = 1.5
    one_euro_beta: float = 0.02
    one_euro_d_cutoff: float = 1.0

    @property
    def ball_diameter_m(self) -> float:
        if self.ball_diameter_override_m is not None:
            return self.ball_diameter_override_m
        return BALL_DIAMETER_M[self.ball_size]
