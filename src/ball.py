"""Ball detector wrapper, foot crop, coordinate mapping and search-mode switch."""
from dataclasses import dataclass, replace

Box = tuple  # x1, y1, x2, y2
Detection = tuple  # x1, y1, x2, y2, confidence


class YoloBallDetector:
    """Untuned baseline. Ultralytics is AGPL, so this is not a shipping choice yet."""

    def __init__(self, cfg):
        from ultralytics import YOLO

        self._model = YOLO(cfg.detector_weights)
        self._cfg = cfg

    def detect(self, bgr) -> Detection | None:
        result = self._model.predict(
            bgr,
            classes=[self._cfg.ball_class_id],
            conf=self._cfg.ball_confidence,
            device="cpu",
            verbose=False,
        )[0]
        if len(result.boxes) == 0:
            return None
        best = int(result.boxes.conf.argmax())
        return (*map(float, result.boxes.xyxy[best]), float(result.boxes.conf[best]))


def foot_crop(foot_points, frame_width: int, frame_height: int, cfg) -> Box | None:
    """Square integer crop around the feet, clipped to the frame."""
    if not foot_points:
        return None
    xs = [p[0] for p in foot_points]
    ys = [p[1] for p in foot_points]
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    side = max(span * cfg.crop_span_scale, cfg.crop_min_frac * frame_height)
    center_x, center_y = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    x1 = max(0, int(center_x - side / 2))
    y1 = max(0, int(center_y - side / 2))
    x2 = min(frame_width, int(center_x + side / 2))
    y2 = min(frame_height, int(center_y + side / 2))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


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
