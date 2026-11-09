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
    ball_candidates: int = 3  # boxes kept per frame, so the tracker can pick the one that fits
    detector_weights: str = "yolov8n.pt"
    crop_imgsz: int = 320  # detector input size for the foot crop
    full_imgsz: int = 640  # detector input size for the full frame
    # The untuned detector finds the ball best when it is about 40-80 px in the detector input,
    # so the crop is sized from the last seen ball: diameter / crop_ball_frac.
    crop_ball_frac: float = 0.18
    crop_span_min_scale: float = 1.2  # the crop always covers the feet with this margin
    crop_span_scale: float = 2.0  # used before the ball has been seen
    crop_min_frac: float = 0.25  # minimum crop side before the ball has been seen, fraction of frame height
    crop_pad_value: int = 114  # gray fill where the square crop extends past the frame edge

    # Ball tracking on top of detection
    track_gate_diam: float = 2.5  # accept a box only within this many ball diameters of the prediction
    track_gate_max_s: float = 0.3  # after this long unseen, reacquire the ball nearest the feet
    track_max_predict_s: float = 0.1  # bridge detection gaps up to this long; never while searching

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
