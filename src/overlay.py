"""Render a debug video: skeleton, ball box, search mode and live counts."""
import cv2
import numpy as np

from .pose import SKELETON, coarse_head_pitch_deg, head_box
from .video_io import iter_frames

RED = (0, 0, 255)
WHITE = (255, 255, 255)


def update_head_down(head_down: bool, events) -> bool:
    """Follow the head_down and head_up events reported on this frame."""
    for event in events:
        if event.kind == "head_down":
            head_down = True
        elif event.kind == "head_up":
            head_down = False
    return head_down


def _draw_head(image, landmarks, head_down, cfg):
    box = head_box(landmarks, cfg.pose_min_visibility, cfg.head_box_padding)
    color = RED if head_down else WHITE
    if box is not None:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    pitch = coarse_head_pitch_deg(landmarks, cfg.pose_min_visibility)
    text = "pitch: n/a" if pitch is None else f"pitch: {pitch:.0f} deg"
    if head_down:
        text = f"HEAD DOWN  {text}"
    cv2.putText(image, text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)


def _draw_skeleton(image, landmarks, min_visibility):
    for a, b in SKELETON:
        if landmarks[a, 2] >= min_visibility and landmarks[b, 2] >= min_visibility:
            cv2.line(
                image, tuple(map(int, landmarks[a, :2])), tuple(map(int, landmarks[b, :2])),
                (0, 255, 0), 2,
            )


def render(clip_path: str, out_path: str, records: list, landmarks, cfg) -> None:
    by_index = {record["index"]: record for record in records}
    times = np.array([record["t_s"] for record in records])
    fps = 1.0 / float(np.median(np.diff(times))) if len(times) > 1 else 30.0
    writer = None
    left = right = 0
    head_down = False
    try:
        for frame in iter_frames(clip_path):
            record = by_index.get(frame.index)
            if record is None:
                continue
            image = frame.bgr.copy()
            if writer is None:
                height, width = image.shape[:2]
                writer = cv2.VideoWriter(
                    out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height),
                )
            _draw_skeleton(image, landmarks[frame.index], cfg.pose_min_visibility)
            if record["crop"] is not None:
                x1, y1, x2, y2 = map(int, record["crop"])
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 1)
            if record["ball"] is not None:
                x1, y1, x2, y2 = map(int, record["ball"][:4])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 165, 255), 2)
            for event in record["events"]:
                if event.kind == "touch":
                    left += event.foot == "left"
                    right += event.foot == "right"
            head_down = update_head_down(head_down, record["events"])
            _draw_head(image, landmarks[frame.index], head_down, cfg)
            mode = "SEARCH" if record["searched_full_frame"] else "TRACK"
            cv2.putText(
                image, f"{mode}  L:{left} R:{right}  t={frame.t_s:.2f}s", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
            )
            writer.write(image)
    finally:
        if writer is not None:
            writer.release()
