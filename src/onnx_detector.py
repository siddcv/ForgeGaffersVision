"""Ball detector on an exported YOLO model (ONNX Runtime), without PyTorch or Ultralytics.

The preprocessing and decoding here are written out by hand because the phone has to repeat them
exactly (see docs/PREPROCESSING.md). They match what Ultralytics does for a fixed-size export:
  1. letterbox: scale the longer side to imgsz (bilinear), pad the rest with gray 114, centred
  2. BGR -> RGB, uint8 -> float32 in [0, 1], HWC -> 1xCxHxW
  3. model output (1, 4 + classes, anchors): box centre x, y, width, height in input pixels, then
     one score per class (already sigmoid)
  4. keep the ball class above the confidence threshold, non-maximum suppression, map back
A square foot crop is resized without padding, so the ball keeps the size foot_crop chose.
"""
from pathlib import Path

import numpy as np

PAD_VALUE = 114


def export_onnx(weights: str, imgsz: int) -> Path:
    """models/<weights>_<imgsz>.onnx, exporting it with Ultralytics the first time (dev only)."""
    target = Path("models") / f"{Path(weights).stem}_{imgsz}.onnx"
    if not target.exists():
        import shutil

        from ultralytics import YOLO

        exported = YOLO(weights).export(
            format="onnx", imgsz=imgsz, dynamic=False, simplify=True, nms=False, verbose=False,
        )
        target.parent.mkdir(exist_ok=True)
        shutil.move(exported, target)
    return target


def letterbox(bgr: np.ndarray, imgsz: int) -> tuple:
    """Return (imgsz x imgsz BGR image, scale, pad_x, pad_y)."""
    import cv2

    height, width = bgr.shape[:2]
    scale = min(imgsz / height, imgsz / width)
    new_w, new_h = round(width * scale), round(height * scale)
    if (new_w, new_h) != (width, height):
        bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = (imgsz - new_w) / 2, (imgsz - new_h) / 2
    top, bottom = round(pad_h - 0.1), round(pad_h + 0.1)
    left, right = round(pad_w - 0.1), round(pad_w + 0.1)
    if top or bottom or left or right:
        bgr = cv2.copyMakeBorder(
            bgr, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3,
        )
    return bgr, scale, left, top


def to_tensor(bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 HxWx3 -> RGB float32 1x3xHxW in [0, 1]."""
    rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def decode(output: np.ndarray, class_id: int, confidence: float, iou_threshold: float, keep: int) -> list:
    """Raw (1, 4 + classes, anchors) output -> up to `keep` (x1, y1, x2, y2, conf) in input pixels,
    highest confidence first."""
    rows = output[0]
    scores = rows[4 + class_id]
    order = np.argsort(-scores)
    order = order[scores[order] >= confidence]
    kept = []
    for index in order:
        cx, cy, w, h = (float(v) for v in rows[:4, index])
        box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, float(scores[index]))
        if all(iou(box, other) <= iou_threshold for other in kept):
            kept.append(box)
            if len(kept) == keep:
                break
    return kept


def to_source(box, scale: float, pad_x: float, pad_y: float) -> tuple:
    """Map a box from model input pixels back to the image passed to letterbox()."""
    x1, y1, x2, y2, conf = box
    return (x1 - pad_x) / scale, (y1 - pad_y) / scale, (x2 - pad_x) / scale, (y2 - pad_y) / scale, conf


class OnnxBallDetector:
    """Same interface as ball.YoloBallDetector: detect(bgr, imgsz) -> list of detections.

    One fixed-size model per input size, e.g. {320: "models/yolov8n_320.onnx"}.
    """

    def __init__(self, cfg, model_paths: dict, threads: int = 0):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads  # 0 lets ONNX Runtime choose
        options.inter_op_num_threads = 1
        self._cfg = cfg
        self._sessions = {
            size: ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
            for size, path in model_paths.items()
        }

    def run(self, tensor: np.ndarray, imgsz: int) -> np.ndarray:
        session = self._sessions[imgsz]
        return session.run(None, {session.get_inputs()[0].name: tensor})[0]

    def detect(self, bgr: np.ndarray, imgsz: int) -> list:
        image, scale, pad_x, pad_y = letterbox(bgr, imgsz)
        output = self.run(to_tensor(image), imgsz)
        boxes = decode(
            output, self._cfg.ball_class_id, self._cfg.ball_confidence,
            self._cfg.ball_nms_iou, self._cfg.ball_candidates,
        )
        return [to_source(box, scale, pad_x, pad_y) for box in boxes]
