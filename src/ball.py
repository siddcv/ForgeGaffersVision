"""Ball detector wrapper, foot crop, coordinate mapping, search-mode switch and ball tracker."""
import math
from dataclasses import dataclass, replace

from .filters import Kalman1D

Box = tuple  # x1, y1, x2, y2
Detection = tuple  # x1, y1, x2, y2, confidence


class YoloBallDetector:
    """Untuned baseline. Ultralytics is AGPL, so this is not a shipping choice yet."""

    def __init__(self, cfg):
        from ultralytics import YOLO

        self._model = YOLO(cfg.detector_weights)
        self._cfg = cfg

    def detect(self, bgr, imgsz: int) -> list:
        """Up to cfg.ball_candidates detections, highest confidence first."""
        result = self._model.predict(
            bgr,
            classes=[self._cfg.ball_class_id],
            conf=self._cfg.ball_confidence,
            imgsz=imgsz,
            device="cpu",
            verbose=False,
        )[0]
        order = result.boxes.conf.argsort(descending=True)[: self._cfg.ball_candidates]
        return [
            (*map(float, result.boxes.xyxy[i]), float(result.boxes.conf[i])) for i in order
        ]


def foot_crop(foot_points, frame_width: int, frame_height: int, cfg, ball_diameter_px=None) -> Box | None:
    """Square integer crop centred on the feet. It may extend past the frame edge (see crop_image).

    Once the ball has been seen, the side is set so the ball fills about cfg.crop_ball_frac of it,
    because the detector misses balls that look too large or too small in its input.
    """
    if not foot_points:
        return None
    xs = [p[0] for p in foot_points]
    ys = [p[1] for p in foot_points]
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    if ball_diameter_px:
        side = max(ball_diameter_px / cfg.crop_ball_frac, span * cfg.crop_span_min_scale)
    else:
        side = max(span * cfg.crop_span_scale, cfg.crop_min_frac * frame_height)
    side = int(side)
    center_x, center_y = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    x1, y1 = int(center_x - side / 2), int(center_y - side / 2)
    x2, y2 = x1 + side, y1 + side
    # Drop crops that lie entirely outside the frame.
    if x2 <= 0 or y2 <= 0 or x1 >= frame_width or y1 >= frame_height:
        return None
    return x1, y1, x2, y2


def crop_image(image, crop: Box, pad_value: int):
    """Cut the crop out of an HxWxC array, filling any part outside the frame with pad_value.

    Keeping the crop square avoids letterboxing, so the ball keeps the size foot_crop chose.
    """
    import numpy as np

    height, width = image.shape[:2]
    x1, y1, x2, y2 = crop
    out = np.full((y2 - y1, x2 - x1) + image.shape[2:], pad_value, dtype=image.dtype)
    sx1, sy1, sx2, sy2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
    if sx2 > sx1 and sy2 > sy1:
        out[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1] = image[sy1:sy2, sx1:sx2]
    return out


def to_full_frame(detection: Detection, crop: Box) -> Detection:
    x1, y1, x2, y2, confidence = detection
    return x1 + crop[0], y1 + crop[1], x2 + crop[0], y2 + crop[1], confidence


def box_center(detection: Detection) -> tuple:
    return (detection[0] + detection[2]) / 2, (detection[1] + detection[3]) / 2


def box_diameter_px(detection: Detection) -> float:
    """The shorter side, since motion blur stretches the longer one."""
    return min(detection[2] - detection[0], detection[3] - detection[1])


def _inside(detection: Detection, crop: Box | None) -> bool:
    if crop is None:
        return False
    x, y = box_center(detection)
    return crop[0] <= x <= crop[2] and crop[1] <= y <= crop[3]


@dataclass(frozen=True)
class SearchState:
    searching: bool = False
    crop_miss_since_s: float | None = None


def search_step(state: SearchState, t_s: float, crop: Box | None, crop_det, full_det, cfg):
    """Choose this frame's detection. Returns (state, detection, searched_full_frame).

    Both detections are supplied because replay reads them from the cache; on a
    phone, full_det would only be computed when state.searching is true.
    """
    if state.searching:
        if full_det is None:
            return state, None, True
        if _inside(full_det, crop):
            return SearchState(), full_det, True
        return state, full_det, True

    if crop_det is not None:
        return SearchState(), crop_det, False

    miss_since = state.crop_miss_since_s if state.crop_miss_since_s is not None else t_s
    state = replace(state, crop_miss_since_s=miss_since)
    if crop is None or t_s - miss_since >= cfg.ball_search_after_s:
        return replace(state, searching=True), full_det, True
    return state, None, False


class BallTracker:
    """Causal track on top of per-frame detections.

    choose() picks the candidate box that fits the track: the one nearest the predicted position,
    within track_gate_diam ball diameters. With no recent track it takes the candidate nearest the
    feet. update() smooths the chosen box with a Kalman filter and, when there is no box, predicts
    through gaps up to track_max_predict_s, but never once a full-frame search has started, since
    by then a prediction is no evidence of where the ball is.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._x = Kalman1D(cfg.kalman_process_noise, cfg.kalman_measurement_noise)
        self._y = Kalman1D(cfg.kalman_process_noise, cfg.kalman_measurement_noise)
        self.diameter_px = None
        self.last_seen_s = None

    def _tracking(self, t_s: float, limit_s: float) -> bool:
        return self.last_seen_s is not None and t_s - self.last_seen_s <= limit_s

    def choose(self, t_s: float, candidates, feet=()) -> Detection | None:
        if not candidates:
            return None
        if self._tracking(t_s, self.cfg.track_gate_max_s):
            expected = (self._x.peek(t_s), self._y.peek(t_s))
            best = min(candidates, key=lambda c: math.dist(box_center(c), expected))
            gate = self.cfg.track_gate_diam * self.diameter_px
            return best if math.dist(box_center(best), expected) <= gate else None
        if feet:
            center = (sum(p[0] for p in feet) / len(feet), sum(p[1] for p in feet) / len(feet))
            return min(candidates, key=lambda c: math.dist(box_center(c), center))
        return max(candidates, key=lambda c: c[4])

    def update(self, t_s: float, detection: Detection | None, searched_full_frame: bool):
        """Returns (ball_xy or None, predicted)."""
        if detection is not None:
            x, y = box_center(detection)
            self.diameter_px = box_diameter_px(detection)
            self.last_seen_s = t_s
            return (self._x.update(t_s, x), self._y.update(t_s, y)), False
        if searched_full_frame or not self._tracking(t_s, self.cfg.track_max_predict_s):
            return None, False
        return (self._x.predict(t_s), self._y.predict(t_s)), True
