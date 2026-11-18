"""Render a debug video: skeleton, ball box, search mode and live counts."""
import cv2
import numpy as np

from .pose import SKELETON, coarse_head_pitch_deg, head_box
from .video_io import iter_frames

RED = (0, 0, 255)
WHITE = (255, 255, 255)
GRAY = (180, 180, 180)
GREEN = (0, 255, 0)


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


def touch_rate_text(rate, goal: int) -> str:
    """'pace 138/min  last min 71  goal 150', or the first-minute form before a minute is up."""
    if rate is None:
        return f"goal {goal}/min"
    pace = "pace --" if rate.pace_per_min is None else f"pace {rate.pace_per_min:.0f}/min"
    minute = f"last min {rate.last_minute}" if rate.minute_complete else f"{rate.last_minute} so far"
    return f"{pace}  {minute}  goal {goal}"


def _draw_touch_rate(image, rate, goal: int) -> None:
    met = rate is not None and rate.minute_complete and rate.last_minute >= goal
    cv2.putText(image, touch_rate_text(rate, goal), (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 255, 0) if met else WHITE, 2)


FLASH_S = 0.3  # how long a contact / tick-tock label stays on screen


def _draw_flashes(image, last_reported: dict, t_s: float) -> None:
    """Gray 'contact' when a contact is reported; green 'TICK-TOCK' when one is confirmed, which
    happens at the next contact, by the other foot."""
    if t_s - last_reported["touch"] <= FLASH_S:
        cv2.putText(image, "TICK-TOCK", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, GREEN, 3)
    if t_s - last_reported["contact"] <= FLASH_S:
        cv2.putText(image, "contact", (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, GRAY, 3)


def render(clip_path: str, out_path: str, records: list, landmarks, cfg) -> None:
    by_index = {record["index"]: record for record in records}
    times = np.array([record["t_s"] for record in records])
    fps = 1.0 / float(np.median(np.diff(times))) if len(times) > 1 else 30.0
    writer = None
    counts = {("touch", "left"): 0, ("touch", "right"): 0, ("contact", "left"): 0, ("contact", "right"): 0}
    last_reported = {"touch": float("-inf"), "contact": float("-inf")}
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
            elif record.get("predicted"):
                # Tracker prediction through a short detection gap.
                center = tuple(map(int, record["ball_xy"]))
                cv2.circle(image, center, 25, (0, 165, 255), 2)
                cv2.putText(image, "pred", (center[0] + 30, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            for event in record["events"]:
                if (event.kind, event.foot) in counts:
                    counts[(event.kind, event.foot)] += 1
                    last_reported[event.kind] = frame.t_s
            head_down = update_head_down(head_down, record["events"])
            _draw_head(image, landmarks[frame.index], head_down, cfg)
            _draw_flashes(image, last_reported, frame.t_s)
            mode = "SEARCH" if record["searched_full_frame"] else "TRACK"
            cv2.putText(
                image, f"{mode}  tick-tock L:{counts['touch', 'left']} R:{counts['touch', 'right']}"
                f"  t={frame.t_s:.2f}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2,
            )
            cv2.putText(
                image, f"contacts L:{counts['contact', 'left']} R:{counts['contact', 'right']}", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, GRAY, 2,
            )
            _draw_touch_rate(image, record.get("touch_rate"), cfg.touches_per_minute_goal)
            writer.write(image)
    finally:
        if writer is not None:
            writer.release()
