"""Live tick-tock feedback from a camera, on the laptop: the phone app's loop, in Python.

    camera -> pose -> foot crop -> ball detector (ONNX) -> FramePipeline.step -> overlay window

Frames are read on a background thread that keeps only the newest one, so feedback never falls
behind the camera: frames that arrive while one is being processed are skipped (and still
recorded). The metric thresholds are in seconds, so skipped frames do not change their meaning.
Laptop speed says nothing about a phone; use this to feel the feedback and try camera placement.

Drills: the window opens in READY (overlay running, nothing counted). START stays grey, with the
reason shown above it, until the whole player (head to toe) and the ball have been in frame for
framing_hold_s (--no-framing-check skips this). START (click the button or press space) counts
down, then resets the counts and starts the drill; STOP ends it. During the countdown and the
drill, RECENTER YOURSELF shows while the player's head or feet are out of frame (nothing pauses;
the ball leaving the frame is lost control, not a framing problem). Each drill
from a camera is recorded to data/clips/live/<date>/drill_<date>_<time>.mp4 with a timestamps sidecar
holding each frame's real capture time, and its summary goes to output/live/drill_<date>_<time>.json,
with times from the drill's first frame, so it can be labelled (src.label_clip), replayed
(src.run_clip) and scored (src.evaluate) like any clip. The window's view (overlay, counts, REC)
is also saved, as output/live/drill_<date>_<time>_overlay.mp4, playing at real speed; it is not a clip
to label or replay. A clip source starts one drill at once.
Completed drills also produce a player report on screen and beside the JSON as a text file.

Usage:
    python -m src.live                       # camera 0 (phone-as-webcam apps show up as a camera)
    python -m src.live --source 1            # another camera
    python -m src.live --rotate 90 --flip    # sideways feed / undo a mirrored feed
    python -m src.live --source data/clips/tune/<session>/<clip>.mp4   # a clip, paced as if live
Keys: space starts / stops a drill (same as the button), q or Esc quits (saving a running drill).
"""
import argparse
import csv
import json
import statistics
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import ball as ball_module
from .config import Config
from .onnx_detector import OnnxBallDetector, export_onnx
from .overlay import Hud, draw, update_head_down
from .face_pose import open_head_pitch_sampler
from .pipeline import POSE_PITCH, FramePipeline
from .pose import FACE, LEFT_FOOT, RIGHT_FOOT, PoseEstimator
from .video_io import iter_frames, timestamps_path

WINDOW = "Gaffer's Vision live"
DISPLAY_MAX_PX = 900  # longest side of the window
ROTATIONS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
MEASURE_FPS_AFTER_S = 3.0  # report the camera's real frame rate once this much has been read


def orientation(frame_size) -> str | None:
    if frame_size is None:
        return None
    width, height = frame_size
    return "portrait" if height > width else "landscape"


def camera_report(frame_size, reported_fps: float) -> str:
    """The camera's frame size and frame rate, with a warning for a landscape frame: the
    recorded clips and the phone app are portrait, and a landscape frame makes the phone stand
    further back to fit the player, so the ball is smaller and harder to detect."""
    width, height = frame_size
    text = f"Camera: {width} x {height} ({orientation(frame_size)}), reported {reported_fps:.0f} fps"
    if width > height:
        text += ("  WARNING: landscape. Set the phone-as-webcam app to portrait, or if the picture is "
                 "on its side add --rotate 90 (or 270).")
    return text


def framing_problems(landmarks, ball, width: int, height: int, cfg) -> list[str]:
    """What keeps the player and ball from being fully in frame; empty when they are.

    Checks the head (face landmarks, plus an estimate of the top of the head), both feet (heels
    and toes) and the detected ball box, each inside the frame with a margin at every edge.
    Off-frame landmarks come back from pose with low visibility, or with coordinates past the
    edge, so both are treated as not in frame.
    """
    margin_x, margin_y = cfg.framing_margin_frac * width, cfg.framing_margin_frac * height

    def inside(x, y) -> bool:
        return margin_x <= x <= width - margin_x and margin_y <= y <= height - margin_y

    def visible(i) -> bool:
        return bool(landmarks[i, 2] >= cfg.pose_min_visibility)  # False for NaN (no person)

    problems = []
    if not any(visible(i) for i in range(len(landmarks))):
        return ["no player found"]
    face = [i for i in FACE if visible(i)]
    if not face:
        problems.append("head not in frame")
    else:
        xs, ys = [landmarks[i, 0] for i in face], [landmarks[i, 1] for i in face]
        face_width = max(xs) - min(xs)
        head_top = min(ys) - cfg.framing_head_top_frac * face_width
        if not all(inside(x, y) for x, y in zip(xs, ys)) or head_top < margin_y:
            problems.append("head not in frame")
    for name, foot in (("left foot", LEFT_FOOT), ("right foot", RIGHT_FOOT)):
        if not all(visible(i) and inside(landmarks[i, 0], landmarks[i, 1]) for i in foot):
            problems.append(f"{name} not in frame")
    if ball is None:
        problems.append("ball not found")
    elif not (inside(ball[0], ball[1]) and inside(ball[2], ball[3])):
        problems.append("ball not in frame")
    return problems


class FramingGate:
    """Enables START once the player and ball have been fully in frame for framing_hold_s, and
    disables it after they have been out for framing_grace_s, so a single missed detection (the
    ball hidden by a foot for a frame) does not make the button flicker.

    During a drill the same gate, started as ready (START needed the player in frame), drives the
    RECENTER YOURSELF warning: shown while not ready."""

    def __init__(self, cfg, ready: bool = False):
        self.cfg = cfg
        self.ready = ready
        self.problems = [] if ready else ["no player found"]
        self._since = None  # when the current in-frame or out-of-frame run began

    def update(self, t_s: float, problems: list[str]) -> bool:
        in_frame = not problems
        if problems:
            self.problems = problems
        if in_frame != self.ready:
            if self._since is None:
                self._since = t_s
            wait = self.cfg.framing_hold_s if in_frame else self.cfg.framing_grace_s
            if t_s - self._since >= wait:
                self.ready, self._since = in_frame, None
        else:
            self._since = None
        return self.ready


# Tried in order: the first the local OpenCV build can open is used. mp4v is the usual choice;
# some macOS builds only open avc1 (H.264); MJPG in .avi is the last resort, larger but widely
# supported. The timestamps sidecar follows whichever file is written.
RECORD_CODECS = (("mp4v", ".mp4"), ("avc1", ".mp4"), ("MJPG", ".avi"))


def open_video_writer(path: Path, fps: float, size: tuple, codecs=RECORD_CODECS) -> tuple:
    """(writer, actual path) for the first codec that opens, or RuntimeError if none does.

    cv2.VideoWriter never raises: with an unsupported codec it silently writes nothing, so each
    writer is checked with isOpened() before it is used.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tried = []
    for fourcc, suffix in codecs:
        target = path.with_suffix(suffix)
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if writer.isOpened():
            return writer, target
        writer.release()
        target.unlink(missing_ok=True)
        tried.append(fourcc)
    raise RuntimeError(f"Could not open a video writer for {path} (tried {', '.join(tried)}). "
                       "Run with --no-record to go on without recording.")


def open_camera(index: int):
    """The camera, as an OpenCV capture. A separate function so tests can swap in a fake camera."""
    return cv2.VideoCapture(index)


class FrameSource:
    """Background reader of a camera (index) or a clip (path), keeping only the newest frame.

    Each frame comes with its arrival time on time.perf_counter(), so feedback lag = now - arrival.
    Frame times use time.perf_counter(): time.monotonic() ticks only every ~15.6 ms on Windows,
    which gave frames repeated timestamps and jittered every time by up to 15 ms.
    A clip plays at its own timing; if decoding cannot keep up (60 fps HEVC on a busy laptop),
    behind_s records how far behind the clip's clock it fell. A camera never falls behind.

    Drills: start_drill() marks the next frame as the drill's first (drill_t0) and, for a camera,
    starts recording from that frame; stop_drill() ends it and finishes the file. Requests are
    applied on the reader thread between frames, so a recording holds every frame of its drill.
    """

    def __init__(self, source: str, rotate: int = 0, flip: bool = False, width: int = 1280, height: int = 720):
        self._is_camera = source.isdigit()
        self._source = source
        self._rotate = ROTATIONS[rotate]
        self._flip = flip
        self._size = (width, height)
        self._cond = threading.Condition()
        self._latest = None  # (index, t_s, bgr, arrival)
        self._done = False
        self._stop = False
        self.error = None
        self.grabbed = 0
        self.behind_s = 0.0
        self._start_request = None  # (path or None,) once start_drill() is called
        self._stop_request = False
        self.drill_t0 = None  # source time of the current drill's first frame
        self.recording_path = None  # file being written for the current drill (suffix set by the codec)
        self.last_recording = None  # the most recently finished recording
        self.frame_size = None  # (width, height) of the frames as processed, after --rotate
        self.drill_frames = 0  # camera frames in the current drill (all recorded, processed or not)
        self.last_drill_frames = 0  # the same, for the most recently finished drill
        self._writer = self._times = None
        self._fps = 30.0
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def now_s(self) -> float:
        return time.perf_counter() - self._t0

    def start_drill(self, record_path: Path | None) -> None:
        with self._cond:
            self._start_request = (record_path if self._is_camera else None,)

    def stop_drill(self, timeout_s: float = 3.0) -> Path | None:
        """End the drill; returns the finished recording (or None) once its file is closed."""
        with self._cond:
            self._stop_request = True
            self._cond.wait_for(lambda: not self._stop_request or self._done, timeout_s)
            if self._stop_request:  # reader already finished: close here
                self._close_drill()
            return self.last_recording

    def _transform(self, bgr):
        if self._rotate is not None:
            bgr = cv2.rotate(bgr, self._rotate)
        return cv2.flip(bgr, 1) if self._flip else bgr

    def _close_drill(self) -> None:
        """Call with self._cond held."""
        if self._writer is not None:
            self._writer.release()
            self._times.close()
            self._writer = self._times = None
            self.last_recording = self.recording_path
        if self.drill_t0 is not None:
            self.last_drill_frames = self.drill_frames
        self.drill_frames = 0
        self.recording_path = None
        self.drill_t0 = None
        self._stop_request = False
        self._cond.notify_all()

    def _handle_drill(self, t_s: float, bgr) -> None:
        """Apply start / stop requests, then record this frame if a recording is open."""
        with self._cond:
            if self._stop_request:
                self._close_drill()
            if self._start_request is not None:
                (path,) = self._start_request
                self._start_request = None
                self._close_drill()
                self.last_recording = None
                if path is not None:
                    height, width = bgr.shape[:2]
                    self._writer, self.recording_path = open_video_writer(path, self._fps, (width, height))
                    self._times = open(timestamps_path(str(self.recording_path)), "w")
                    self._times.write("t_s\n")
                    print(f"Recording to {self.recording_path.resolve()}")
                self.drill_t0 = t_s
            if self.drill_t0 is not None:
                self.drill_frames += 1
        if self._writer is not None:
            self._writer.write(bgr)
            self._times.write(f"{t_s:.6f}\n")
            if self.grabbed % 30 == 0:
                self._times.flush()  # keep the timestamps if the session is killed

    def _publish(self, t_s: float, bgr) -> None:
        with self._cond:
            self._latest = (self.grabbed, t_s, bgr, time.perf_counter())
            self.grabbed += 1
            self._cond.notify_all()

    def _run(self) -> None:
        try:
            self._run_camera() if self._is_camera else self._run_clip()
        except Exception as error:  # reported by the main loop
            self.error = error
        finally:
            with self._cond:
                self._close_drill()
                self._done = True
                self._cond.notify_all()

    def _run_camera(self) -> None:
        capture = open_camera(int(self._source))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open camera {self._source}. Try --source 1 or 2, or check "
                               "that the phone-as-webcam app is connected.")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._size[0])
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._size[1])
        self._fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        first_s = None
        measured = False
        try:
            while not self._stop:
                ok, bgr = capture.read()
                if not ok:
                    raise RuntimeError("The camera stopped sending frames.")
                now = time.perf_counter()
                if first_s is None:
                    first_s = now
                    self._t0 = now  # frame times start at 0 on the first frame
                t_s = now - first_s
                bgr = self._transform(bgr)
                if self.frame_size is None:
                    self.frame_size = (bgr.shape[1], bgr.shape[0])
                    print(camera_report(self.frame_size, self._fps))
                if not measured and t_s >= MEASURE_FPS_AFTER_S:
                    measured = True
                    print(f"Camera is delivering {self.grabbed / t_s:.1f} fps")
                self._handle_drill(t_s, bgr)
                self._publish(t_s, bgr)
        finally:
            capture.release()

    def _run_clip(self) -> None:
        """A recorded clip played at its own timing, as a stand-in for a camera."""
        for frame in iter_frames(self._source):
            if self._stop:
                return
            if frame.index == 0:
                self._t0 = time.perf_counter()  # after ffprobe has read the timestamps
            wait = frame.t_s - self.now_s()
            if wait > 0:
                time.sleep(wait)
            self.behind_s = max(self.behind_s, -wait)
            bgr = self._transform(frame.bgr)
            if self.frame_size is None:
                self.frame_size = (bgr.shape[1], bgr.shape[0])
            self._handle_drill(frame.t_s, bgr)
            self._publish(frame.t_s, bgr)

    def next_after(self, index: int):
        """The newest frame with a higher index than `index`, waiting for one; None at the end."""
        with self._cond:
            # Short waits in a loop so Ctrl+C still works while waiting (it would not inside one wait).
            while not (self._done or (self._latest is not None and self._latest[0] > index)):
                self._cond.wait(0.2)
            if self._latest is not None and self._latest[0] > index:
                return self._latest
            if self.error is not None:
                raise self.error
            return None

    def stop(self) -> None:
        self._stop = True
        self._thread.join(timeout=5)


def summary_stats(values: list) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)
    return {"median": round(statistics.median(ordered), 1), "p90": round(ordered[int(0.9 * (len(ordered) - 1))], 1)}


def draw_status(image, proc_ms: float, lag_ms: float, processed: int, grabbed: int) -> None:
    skipped = 100 * (1 - processed / grabbed) if grabbed else 0
    text = f"{1000 / proc_ms:.0f} fps  proc {proc_ms:.0f} ms  lag {lag_ms:.0f} ms  skipped {skipped:.0f}%"
    height = image.shape[0]
    cv2.putText(image, text, (20, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
    cv2.putText(image, text, (20, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


class DrillButton:
    """START / countdown / STOP button drawn on the displayed image; clicks are read through an
    OpenCV mouse callback, which works with every OpenCV build (its own buttons need Qt)."""

    def __init__(self):
        self.rect = None  # (x1, y1, x2, y2) in displayed-image pixels
        self.clicked = False

    def on_mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and self.rect is not None:
            x1, y1, x2, y2 = self.rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.clicked = True

    def draw(self, image, label: str, color: tuple) -> None:
        height, width = image.shape[:2]
        x2, y2 = width - 20, height - 70
        x1, y1 = x2 - 170, y2 - 60
        self.rect = (x1, y1, x2, y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(image, label, (x1 + (170 - text_w) // 2, y1 + (60 + text_h) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)


def draw_banner(image, text: str, color: tuple) -> None:
    """Short state line in the top-right corner: READY, REC 0:12, DRILL 0:12."""
    width = image.shape[1]
    (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    x = width - text_w - 20
    if text.startswith("REC"):
        cv2.circle(image, (x - 18, 33), 10, (0, 0, 255), -1)
    cv2.putText(image, text, (x, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
    cv2.putText(image, text, (x, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def body_problems(problems: list[str]) -> list[str]:
    """The framing problems about the player (head, feet), leaving out the ball's."""
    return [p for p in problems if not p.startswith("ball")]


def draw_recenter(image, problems: list[str]) -> None:
    """Mid-drill warning a third of the way down (below the counts, above the countdown digit):
    RECENTER YOURSELF, and what is out of frame."""
    height, width = image.shape[:2]
    y = height // 3
    for text, scale, thickness, y in (("RECENTER YOURSELF", 1.3, 3, y), (", ".join(problems), 0.8, 2, y + 40)):
        (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        origin = (max(10, (width - text_w) // 2), y)
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 4)
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 200, 255), thickness)


def draw_framing_hint(image, problems: list[str]) -> None:
    """Why START is disabled, above the button: 'Fit head to toe and the ball: right foot not in frame'."""
    height, width = image.shape[:2]
    lines = ["Fit head to toe and the ball in frame", ", ".join(problems)]
    for i, text in enumerate(lines):
        (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        origin = (max(10, width - text_w - 20), height - 175 + 30 * i)
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5)
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)


def draw_countdown(image, seconds_left: float) -> None:
    height, width = image.shape[:2]
    text = str(max(1, int(seconds_left + 0.999)))
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 5.0, 12)
    origin = ((width - text_w) // 2, (height + text_h) // 2)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 5.0, (0, 0, 0), 20)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 5.0, (0, 200, 255), 12)


def fit(image, longest: int):
    height, width = image.shape[:2]
    scale = longest / max(height, width)
    return image if scale >= 1 else cv2.resize(image, (int(width * scale), int(height * scale)))


def minutes(seconds: float) -> str:
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


class OverlayRecorder:
    """The live window's view (overlay, counts, REC, STOP), saved as a video that plays at real
    speed. The window only updates when a frame has been processed (about 10-15 times a second on
    a laptop), so each view is repeated until the next one, at a steady FPS. It shows exactly what
    the player saw; for a smooth overlay on every frame, replay the raw recording with run_clip."""

    FPS = 30.0

    def __init__(self, path: Path):
        self.path = path
        self._writer = None
        self._size = None
        self._last = None
        self.written = 0

    def add(self, view, t_s: float) -> None:
        """view: the displayed image; t_s: seconds since the drill started."""
        if self._writer is None:
            self._size = (view.shape[1], view.shape[0])
            self._writer, self.path = open_video_writer(self.path, self.FPS, self._size)
        if (view.shape[1], view.shape[0]) != self._size:
            view = cv2.resize(view, self._size)
        self._fill_to(t_s)
        self._writer.write(view)
        self.written += 1
        self._last = view

    def _fill_to(self, t_s: float) -> None:
        target = int(round(t_s * self.FPS))
        while self._last is not None and self.written < target:
            self._writer.write(self._last)
            self.written += 1

    def close(self, end_t_s: float) -> Path | None:
        """Finish the file, holding the last view until the drill's end; the saved path or None."""
        if self._writer is None:
            return None
        self._fill_to(end_t_s)
        self._writer.release()
        self._writer = None
        return self.path


class Drill:
    """One drill: from START (after the countdown) to STOP. Counts, events and timing restart for
    each drill, and its times are seconds from the drill's first frame, the same time base as its
    recording, so the summary can be scored against labels made on that recording."""

    def __init__(self, cfg, name: str, record_path: Path | None, save_overlay: bool = False):
        self.pipeline, self.hud = FramePipeline(cfg), Hud()
        self.name = name
        self.record_path = record_path
        self.overlay = OverlayRecorder(Path("output") / "live" / f"{name}_overlay.mp4") if save_overlay else None
        self.proc_ms, self.lag_ms, self.events = [], [], []
        self.full_frame_runs = 0
        self.recenter_warnings = 0  # times the head or feet went out of frame mid-drill
        self.last_t = 0.0
        self.last_head_pitch = None
        self.head_tracked_s = 0.0
        self.looking_down_s = 0.0
        self.head_down_pitch_deg = cfg.head_down_pitch_deg
        self.started = datetime.now()

    def observe_head_pitch(self, t_s: float, pitch: float | None) -> None:
        # Each known sample represents the interval until the next processed frame.
        if self.last_head_pitch is not None:
            elapsed = t_s - self.last_t
            self.head_tracked_s += elapsed
            if self.last_head_pitch < self.head_down_pitch_deg:
                self.looking_down_s += elapsed
        self.last_head_pitch = pitch
        self.last_t = t_s


def append_manifest_row(manifest: Path, recording: Path, summary: dict, when: datetime) -> None:
    """One row in data/manifest.csv for a recorded drill, as data/README.md asks for every clip.

    What live mode knows is filled in: the drill window (0 to its length: the recording starts at
    START and ends at STOP), frame rate, resolution and orientation. Player, camera height and
    distance, lighting and split are left blank for the person filing the clip to fill in.
    """
    with open(manifest, newline="") as file:
        columns = next(csv.reader(file))
    size = summary.get("frame_size")
    known = {
        "clip": recording.stem,
        "session": f"{when:%Y-%m-%d}_live",
        "date": f"{when:%Y-%m-%d}",
        "take": "tick-tock",
        "fps": round(summary["camera_fps"]) if summary.get("camera_fps") else "",
        "resolution": f"{size[0]}x{size[1]}" if size else "",
        "orientation": summary.get("orientation") or "",
        "drill_start_s": 0,
        "drill_end_s": summary["drill_s"],
        "notes": "live drill (src.live); fill in player, camera height/distance, lighting, split",
    }
    with open(manifest, "a", newline="") as file:
        csv.DictWriter(file, fieldnames=columns).writerow({c: known.get(c, "") for c in columns})


def ball_out_of_control_seconds(events: list[dict], end_s: float) -> float:
    total = 0.0
    lost_since = None
    for event in events:
        if event["kind"] == "lost_control" and event.get("source") == "spatial":
            lost_since = event["t_s"]
        elif event["kind"] == "reset" and lost_since is not None:
            total += event["t_s"] - lost_since
            lost_since = None
    if lost_since is not None:
        total += end_s - lost_since
    return total


def player_report(summary: dict) -> list[str]:
    looking_down = summary["looking_down_pct"]
    lost_seconds = summary["ball_out_of_control_s"]
    pace = summary["average_pace_tpm"]
    down_text = f"{looking_down:.1f}% of tracked head time" if looking_down is not None else "Unavailable"
    lost_text = f"{lost_seconds:.1f} seconds" if lost_seconds is not None else "Unavailable"
    pace_text = f"{pace:.1f} tick-tocks per minute" if pace is not None else "Unavailable"
    tick_tocks = summary["tick_tocks"]
    return [
        "Drill report",
        f"Head-down time  {down_text}",
        f"Ball control lost for  {lost_text}",
        f"Average pace  {pace_text}",
        f"Tick-tocks performed  {tick_tocks}" if tick_tocks is not None else "Tick-tocks performed  Unavailable",
    ]


def show_player_report(lines: list[str], button: DrillButton, label: str) -> bool:
    """Show results until dismissed. Return False if the player quits."""
    view = np.full((492, 900, 3), (35, 24, 19), dtype=np.uint8)
    primary, secondary = (250, 246, 239), (193, 180, 167)
    mint, ink = (173, 236, 165), (44, 52, 22)

    def write(text: str, x: int, y: int, scale: float, color: tuple, weight: int = 1) -> None:
        cv2.putText(view, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, weight, cv2.LINE_AA)

    def panel(x: int, y: int, width: int, height: int, color: tuple, radius: int) -> None:
        x2, y2 = x + width, y + height
        cv2.rectangle(view, (x + radius, y), (x2 - radius, y2), color, -1)
        cv2.rectangle(view, (x, y + radius), (x2, y2 - radius), color, -1)
        for cx in (x + radius, x2 - radius):
            for cy in (y + radius, y2 - radius):
                cv2.circle(view, (cx, cy), radius, color, -1, cv2.LINE_AA)

    write(lines[0], 36, 64, 1.35, primary, 2)
    headline, total = lines[-1].split("  ", 1)
    panel(36, 93, 828, 128, mint, 18)
    write(headline, 64, 165, 0.92, ink, 2)
    total_scale = 1.5 if total == "Unavailable" else 2.65
    (total_width, total_height), _ = cv2.getTextSize(total, cv2.FONT_HERSHEY_SIMPLEX, total_scale, 3)
    write(total, 831 - total_width, 156 + total_height // 2, total_scale, ink, 3)

    for index, (line, color) in enumerate(zip(lines[1:-1], [(224, 197, 174), (170, 203, 255), mint])):
        x, y = 36 + index * 282, 239
        panel(x, y, 264, 145, (55, 42, 35), 14)
        metric, reading = line.split("  ", 1)
        value, _, units = reading.partition(" ")
        write(metric, x + 20, y + 37, 0.58, primary)
        write(value, x + 20, y + 93, 0.88 if value == "Unavailable" else 1.62, color, 2)
        if units:
            write(units, x + 20, y + 122, 0.55, secondary)

    button.rect = (664, 405, 864, 459)
    panel(664, 405, 200, 54, mint, 15)
    (label_width, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.64, 2)
    write(label, 749 - label_width // 2, 439, 0.64, ink, 2)
    cv2.arrowedLine(view, (816, 432), (839, 432), ink, 2, cv2.LINE_AA, tipLength=0.32)
    cv2.imshow(WINDOW, view)
    visibility_supported = False
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (ord("q"), 27):
            return False
        if key in (ord(" "), 13) or button.clicked:
            button.clicked = False
            return True
        # HighGUI can report -1 when the property is unsupported or the window is gone.
        try:
            visible = cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            visible = -1
        if visible == 0 or (visibility_supported and visible < 0):
            return False
        visibility_supported = visibility_supported or visible > 0


def save_drill(drill: Drill, recording: Path | None, source: "FrameSource", args, is_camera: bool) -> list[str]:
    overlay = drill.overlay.close(drill.last_t) if drill.overlay is not None else None
    counts = {f"{kind}_{foot}": n for (kind, foot), n in drill.hud.counts.items()}
    tick_tocks = counts["touch_left"] + counts["touch_right"]
    summary = {
        "source": args.source,
        "recording": str(recording) if recording else None,
        "overlay_video": str(overlay) if overlay else None,
        "drill_s": round(drill.last_t, 2),
        "frame_size": list(source.frame_size) if source.frame_size else None,
        "orientation": orientation(source.frame_size),
        "camera_fps": round(source.last_drill_frames / drill.last_t, 1) if drill.last_t and is_camera else None,
        "processed_fps": round(len(drill.proc_ms) / drill.last_t, 1) if drill.last_t else None,
        "frames_processed": len(drill.proc_ms),
        "full_frame_runs": drill.full_frame_runs,
        "recenter_warnings": drill.recenter_warnings,
        "clip_reader_behind_s": None if is_camera else round(source.behind_s, 2),
        "processing_ms": summary_stats(drill.proc_ms),
        "feedback_lag_ms": summary_stats(drill.lag_ms),
        **counts,
        "tick_tocks": tick_tocks if drill.proc_ms else None,
        "average_pace_tpm": 60 * tick_tocks / drill.last_t if drill.last_t else None,
        "looking_down_pct": 100 * drill.looking_down_s / drill.head_tracked_s if drill.head_tracked_s else None,
        "head_tracked_s": drill.head_tracked_s,
        "ball_out_of_control_s": ball_out_of_control_seconds(drill.events, drill.last_t) if drill.last_t else None,
        "lost_control": sum(e["kind"] == "lost_control" for e in drill.events),
        "head_down": sum(e["kind"] == "head_down" for e in drill.events),
    }
    if recording is not None and args.manifest and Path(args.manifest).exists():
        append_manifest_row(Path(args.manifest), recording, summary, drill.started)
        print(f"Added {recording.stem} to {args.manifest} (fill in player, camera height/distance, split)")
    out = Path("output") / "live" / f"{drill.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "events": drill.events}, indent=2))
    lines = player_report(summary)
    report_path = out.with_suffix(".txt")
    report = "\n".join(lines) + "\n"
    report_path.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"Saved player report to {report_path}")
    if recording is not None and recording.exists() and recording.stat().st_size > 0:
        print(f"Saved {out} and the recording {recording.resolve()} with its timestamps")
    elif drill.record_path is not None:
        print(f"Saved {out}. WARNING: no recording was written (expected near {drill.record_path.resolve()}).")
    else:
        print(f"Saved {out}")
    if overlay is not None:
        print(f"Saved the overlay video {overlay.resolve()}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="0", help="camera index, or a clip path to replay at its own pace")
    parser.add_argument("--rotate", type=int, choices=sorted(ROTATIONS), default=0, help="clockwise degrees")
    parser.add_argument("--flip", action="store_true", help="mirror the input, to undo a mirrored webcam feed")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    # 2 measured best on a 12-core laptop: with every core, the detector starved pose and decoding
    # (proc 123 ms vs 71 ms median on 134129).
    parser.add_argument("--threads", type=int, default=2, help="detector CPU threads (0: ONNX Runtime decides)")
    parser.add_argument("--countdown", type=float, default=3.0, help="seconds between START and the drill")
    parser.add_argument("--autostart", action="store_true",
                        help="start a drill at once, without the button (always on for clips and --no-window)")
    parser.add_argument("--no-record", action="store_true", help="count drills without saving video")
    parser.add_argument("--manifest", default="data/manifest.csv",
                        help="add a row per recorded drill here ('' to skip)")
    parser.add_argument("--no-overlay-video", action="store_true",
                        help="do not save the window's view (output/live/drill_<date>_<time>_overlay.mp4)")
    parser.add_argument("--no-framing-check", action="store_true",
                        help="enable START without waiting for the player and ball to be fully in frame")
    parser.add_argument("--no-window", action="store_true", help="no display, e.g. for a timing run")
    parser.add_argument("--max-seconds", type=float, help="stop after this long")
    args = parser.parse_args()

    cfg = Config()
    sizes = (cfg.crop_imgsz, cfg.full_imgsz)
    detector = OnnxBallDetector(cfg, {size: export_onnx(cfg.detector_weights, size) for size in sizes}, args.threads)
    pose = PoseEstimator()
    head_sampler, face_pose = open_head_pitch_sampler(cfg)  # head pitch at 10 Hz, as in replay
    is_camera = args.source.isdigit()
    autostart = args.autostart or args.no_window or not is_camera

    def new_drill() -> Drill:
        now = datetime.now()
        record_path = None
        if is_camera and not args.no_record:
            # The date is in the name too: names must be unique across sessions (data/README.md),
            # since labels, caches and outputs are filed by name.
            record_path = Path("data/clips/live") / f"{now:%Y-%m-%d}" / f"drill_{now:%Y%m%d_%H%M%S}.mp4"
        name = record_path.stem if record_path else f"{Path(args.source).stem if not is_camera else 'drill'}_{now:%H%M%S}"
        return Drill(cfg, name, record_path, save_overlay=not (args.no_overlay_video or args.no_window))

    source = FrameSource(args.source, args.rotate, args.flip, args.width, args.height)
    preview = FramePipeline(cfg)  # keeps the overlay (feet, ball) running between drills; never counts
    preview_hud = Hud()
    button = DrillButton()
    framing = FramingGate(cfg)  # READY: enables START
    recenter = FramingGate(cfg, ready=True)  # countdown and drill: RECENTER YOURSELF while not ready
    if not args.no_window:
        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, button.on_mouse)
        print("Press START (or space) to begin a drill, STOP (or space) to end it, q to quit.")

    state, drill, countdown_end = "ready", None, None
    if autostart:
        drill = new_drill()
        source.start_drill(drill.record_path)
        state = "drill"
    last_index, last_ms, processed, recent_proc = -1, -1, 0, []
    quitting = False
    loop_finished = False
    try:
        while True:
            try:
                item = source.next_after(last_index)
            except KeyboardInterrupt:  # Ctrl+C: stop like q, so the drill is finished and saved
                quitting = True
                break
            if item is None:
                break
            index, t_s, bgr, arrival = item
            last_index = index
            if args.max_seconds and t_s > args.max_seconds:
                break
            if state == "countdown" and time.perf_counter() >= countdown_end:
                drill = new_drill()
                source.start_drill(drill.record_path)
                state = "drill"

            start = time.perf_counter()
            height, width = bgr.shape[:2]
            t_ms = max(int(t_s * 1000), last_ms + 1)  # MediaPipe needs strictly increasing times
            last_ms = t_ms
            # Frames from the drill's first frame on go to the drill, timed from that frame.
            drill_t0 = source.drill_t0
            counting = state == "drill" and drill_t0 is not None and t_s >= drill_t0
            pipeline = drill.pipeline if counting else preview
            t_used = t_s - drill_t0 if counting else t_s
            landmarks = pose.detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), t_ms)
            crop = pipeline.foot_crop(landmarks, width, height)
            crop_candidates = []
            if crop is not None:
                image = ball_module.crop_image(bgr, crop, cfg.crop_pad_value)
                crop_candidates = [ball_module.to_full_frame(d, crop) for d in detector.detect(image, cfg.crop_imgsz)]
            head_pitch = head_sampler.update(bgr, landmarks, t_s, t_ms) if head_sampler else POSE_PITCH
            record = pipeline.step(t_used, landmarks, crop, crop_candidates,
                                   lambda: detector.detect(bgr, cfg.full_imgsz), head_pitch)
            problems = framing_problems(landmarks, record["ball"], width, height, cfg)
            if state == "ready":
                framing.update(t_s, problems)
            else:  # countdown or drill: warn about the body only; a lost ball is lost control
                was_in_frame = recenter.ready
                recenter.update(t_s, body_problems(problems))
                if counting and was_in_frame and not recenter.ready:
                    drill.recenter_warnings += 1
            start_enabled = args.no_framing_check or framing.ready
            proc = (time.perf_counter() - start) * 1000
            lag = (time.perf_counter() - arrival) * 1000
            processed += 1
            recent_proc = (recent_proc + [proc])[-30:]
            if counting:
                drill.proc_ms.append(proc)
                drill.lag_ms.append(lag)
                drill.full_frame_runs += record["ran_full_frame"]
                drill.hud.update(record["events"], t_used)
                drill.events += [{"frame": index, **asdict(event)} for event in record["events"]]
                drill.observe_head_pitch(t_used, record["head_pitch_deg"])
            else:
                # Keep head feedback active in preview and countdown without counting drill events.
                preview_hud.head_down = update_head_down(preview_hud.head_down, record["events"])

            if args.no_window:
                continue
            view = bgr.copy()
            draw(view, record, landmarks, drill.hud if counting else preview_hud, cfg, t_used)
            draw_status(view, statistics.median(recent_proc), lag, processed, source.grabbed)
            view = fit(view, DISPLAY_MAX_PX)
            if state == "ready" and start_enabled:
                draw_banner(view, "READY", (255, 255, 255))
                button.draw(view, "START", (0, 150, 0))
            elif state == "ready":
                draw_banner(view, "STEP INTO FRAME", (0, 200, 255))
                draw_framing_hint(view, framing.problems)
                button.draw(view, "START", (110, 110, 110))
            elif state == "countdown":
                draw_countdown(view, countdown_end - time.perf_counter())
                button.draw(view, "CANCEL", (0, 140, 255))
                if not recenter.ready:
                    draw_recenter(view, recenter.problems)
            else:
                recording = source.recording_path is not None
                draw_banner(view, f"{'REC' if recording else 'DRILL'} {minutes(drill.last_t)}",
                            (0, 0, 255) if recording else (255, 255, 255))
                button.draw(view, "STOP", (0, 0, 200))
                if not recenter.ready:
                    draw_recenter(view, recenter.problems)
                if counting and drill.overlay is not None:
                    drill.overlay.add(view, t_used)
            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                quitting = True
                break
            if key == ord(" ") or button.clicked:
                button.clicked = False
                if state == "ready" and not start_enabled:
                    pass  # START is disabled until the player and ball are fully in frame
                elif state == "ready":
                    state, countdown_end = "countdown", time.perf_counter() + args.countdown
                    recenter = FramingGate(cfg, ready=True)
                elif state == "countdown":
                    state = "ready"
                else:
                    lines = save_drill(drill, source.stop_drill(), source, args, is_camera)
                    state, drill = "ready", None
                    if not show_player_report(lines, button, "CONTINUE" if is_camera else "DONE") or not is_camera:
                        break
                    preview, preview_hud = FramePipeline(cfg), Hud()
                    framing = FramingGate(cfg)
        loop_finished = True
    finally:
        recording = source.stop_drill() if state == "drill" else None
        source.stop()
        pose.close()
        if face_pose is not None:
            face_pose.close()
        try:
            if loop_finished and state == "drill" and drill is not None:
                lines = save_drill(drill, recording, source, args, is_camera)
                if not args.no_window and not quitting:
                    show_player_report(lines, button, "DONE")
        finally:
            if not args.no_window:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
