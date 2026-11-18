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
    # Ball speed needed before and after a reversal. Tried 0.3-0.5 on 134129: it removed a false
    # contact from a creeping ball (0.95 s) but missed the first real touch, which starts from that
    # slow ball, and let a stale remembered direction turn a stop-and-carry into a contact (9.65 s).
    touch_min_speed_mps: float = 0.2
    # A direction change counts only if the ball travelled this many ball diameters into the turn
    # and away from it, so detection jitter on a slowly creeping ball is not a contact (134129, 0.95 s).
    # Contacts are therefore reported once the ball has moved this far after the turn.
    # Kept small: at contact the foot covers the ball and its box centre wobbles, and each wobble
    # restarts the count. On 134129 + park (126 labelled): 0 -> 121 hits / 5 false, 0.05 -> 120 / 3,
    # 0.1 -> 117 / 2, 0.5 -> 112 / 1.
    touch_min_travel_diam: float = 0.1
    touch_foot_distance_diam: float = 1.5  # allows for the ball being hidden at the touch
    # Which foot made a touch: "direction" = the foot the ball was travelling toward (100% on the
    # first labelled clip, vs 90% for "nearest"), "nearest" = the foot closest to the ball.
    touch_foot_rule: str = "direction"
    touch_min_interval_s: float = 0.08
    # A contact counts as a tick-tock only if the ball crosses the midpoint between the feet and
    # the other foot makes the next contact within this time. Labelled touch gaps were 0.30-0.6 s.
    ticktock_max_arrival_s: float = 1.0
    touch_max_gap_s: float = 0.25
    speed_window_s: float = 0.10

    # Touches per minute
    touches_per_minute_goal: int = 150  # the target players work up to
    touch_minute_s: float = 60.0  # the window the goal is counted over
    touch_pace_window_s: float = 10.0  # live pace: touches in this window, scaled to a minute
    touch_pace_min_s: float = 3.0  # no pace until this long after the first touch
 
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

    # Evaluation (evaluate.py): how far apart in time a detection and a label can be and still match
    # Labels mark push-off, once the ball is clearly sent back; the pipeline marks first contact,
    # 0.13-0.15 s earlier on real clips. Real touches are >= 0.30 s apart, so 0.2 s cannot pair
    # a detection with a neighbouring touch.
    eval_touch_tolerance_s: float = 0.20
    eval_event_tolerance_s: float = 0.50  # lost control, reset, head down
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
 


