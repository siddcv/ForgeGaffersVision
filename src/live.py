"""Live tick-tock feedback from a camera, on the laptop: the phone app's loop, in Python.

    camera -> pose -> foot crop -> ball detector (ONNX) -> FramePipeline.step -> overlay window

Frames are read on a background thread that keeps only the newest one, so feedback never falls
behind the camera: frames that arrive while one is being processed are skipped (and still
recorded). The metric thresholds are in seconds, so skipped frames do not change their meaning.
Laptop speed says nothing about a phone; use this to feel the feedback and try camera placement.

Each camera session is recorded to data/clips/live/<date>/live_<time>.mp4 with a timestamps
sidecar holding each frame's real capture time, so it can be labelled (src.label_clip), replayed
(src.run_clip) and scored (src.evaluate) like any clip.

Usage:
    python -m src.live                       # camera 0 (phone-as-webcam apps show up as a camera)
    python -m src.live --source 1            # another camera
    python -m src.live --rotate 90 --flip    # sideways feed / undo a mirrored feed
    python -m src.live --source data/clips/tune/<session>/<clip>.mp4   # a clip, paced as if live
Keys: q or Esc quits, r resets the counts.
"""
import argparse
import json
import statistics
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2

from . import ball as ball_module
from .config import Config
from .onnx_detector import OnnxBallDetector, export_onnx
from .overlay import Hud, draw
from .face_pose import open_head_pitch_sampler
from .pipeline import POSE_PITCH, FramePipeline
from .pose import PoseEstimator
from .video_io import iter_frames, timestamps_path

WINDOW = "Gaffer's Vision live"
DISPLAY_MAX_PX = 900  # longest side of the window
ROTATIONS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


class FrameSource:
    """Background reader of a camera (index) or a clip (path), keeping only the newest frame.

    Each frame comes with its arrival time on time.monotonic(), so feedback lag = now - arrival.
    A clip plays at its own timing; if decoding cannot keep up (60 fps HEVC on a busy laptop),
    behind_s records how far behind the clip's clock it fell. A camera never falls behind.
    """

    def __init__(self, source: str, rotate: int = 0, flip: bool = False, record_path: Path | None = None,
                 width: int = 1280, height: int = 720):
        self._is_camera = source.isdigit()
        self._source = source
        self._rotate = ROTATIONS[rotate]
        self._flip = flip
        self._record_path = record_path
        self._size = (width, height)
        self._cond = threading.Condition()
        self._latest = None  # (index, t_s, bgr, arrival)
        self._done = False
        self._stop = False
        self.error = None
        self.grabbed = 0
        self.behind_s = 0.0
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def now_s(self) -> float:
        return time.monotonic() - self._t0

    def _transform(self, bgr):
        if self._rotate is not None:
            bgr = cv2.rotate(bgr, self._rotate)
        return cv2.flip(bgr, 1) if self._flip else bgr

    def _publish(self, t_s: float, bgr) -> None:
        with self._cond:
            self._latest = (self.grabbed, t_s, bgr, time.monotonic())
            self.grabbed += 1
            self._cond.notify_all()

    def _run(self) -> None:
        try:
            self._run_camera() if self._is_camera else self._run_clip()
        except Exception as error:  # reported by the main loop
            self.error = error
        finally:
            with self._cond:
                self._done = True
                self._cond.notify_all()

    def _run_camera(self) -> None:
        capture = cv2.VideoCapture(int(self._source))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open camera {self._source}. Try --source 1 or 2, or check "
                               "that the phone-as-webcam app is connected.")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._size[0])
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._size[1])
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        writer = times = None
        first_s = None
        try:
            while not self._stop:
                ok, bgr = capture.read()
                if not ok:
                    raise RuntimeError("The camera stopped sending frames.")
                now = time.monotonic()
                if first_s is None:
                    first_s = now
                    self._t0 = now  # frame times start at 0 on the first frame
                t_s = now - first_s
                bgr = self._transform(bgr)
                if self._record_path is not None:
                    if writer is None:
                        self._record_path.parent.mkdir(parents=True, exist_ok=True)
                        height, width = bgr.shape[:2]
                        writer = cv2.VideoWriter(str(self._record_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                                 fps, (width, height))
                        times = open(timestamps_path(str(self._record_path)), "w")
                        times.write("t_s\n")
                    writer.write(bgr)
                    times.write(f"{t_s:.6f}\n")
                self._publish(t_s, bgr)
        finally:
            capture.release()
            if writer is not None:
                writer.release()
                times.close()

    def _run_clip(self) -> None:
        """A recorded clip played at its own timing, as a stand-in for a camera."""
        for frame in iter_frames(self._source):
            if self._stop:
                return
            if frame.index == 0:
                self._t0 = time.monotonic()  # after ffprobe has read the timestamps
            wait = frame.t_s - self.now_s()
            if wait > 0:
                time.sleep(wait)
            self.behind_s = max(self.behind_s, -wait)
            self._publish(frame.t_s, self._transform(frame.bgr))

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


def fit(image, longest: int):
    height, width = image.shape[:2]
    scale = longest / max(height, width)
    return image if scale >= 1 else cv2.resize(image, (int(width * scale), int(height * scale)))


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
    parser.add_argument("--no-record", action="store_true", help="do not save the camera session")
    parser.add_argument("--no-window", action="store_true", help="no display, e.g. for a timing run")
    parser.add_argument("--max-seconds", type=float, help="stop after this long")
    args = parser.parse_args()

    cfg = Config()
    sizes = (cfg.crop_imgsz, cfg.full_imgsz)
    detector = OnnxBallDetector(cfg, {size: export_onnx(cfg.detector_weights, size) for size in sizes}, args.threads)
    pose = PoseEstimator()
    head_sampler, face_pose = open_head_pitch_sampler(cfg)  # head pitch at 10 Hz, as in replay

    is_camera = args.source.isdigit()
    started = datetime.now()
    record_path = None
    if is_camera and not args.no_record:
        record_path = Path("data/clips/live") / f"{started:%Y-%m-%d}" / f"live_{started:%H%M%S}.mp4"
    name = record_path.stem if record_path else f"{Path(args.source).stem if not is_camera else 'camera'}_{started:%H%M%S}"

    pipeline, hud = FramePipeline(cfg), Hud()
    source = FrameSource(args.source, args.rotate, args.flip, record_path, args.width, args.height)
    proc_ms, lag_ms, events, full_frame_runs = [], [], [], 0
    last_index, last_ms = -1, -1
    try:
        while True:
            item = source.next_after(last_index)
            if item is None:
                break
            index, t_s, bgr, arrival = item
            last_index = index
            if args.max_seconds and t_s > args.max_seconds:
                break

            start = time.perf_counter()
            height, width = bgr.shape[:2]
            t_ms = max(int(t_s * 1000), last_ms + 1)  # MediaPipe needs strictly increasing times
            last_ms = t_ms
            landmarks = pose.detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), t_ms)
            crop = pipeline.foot_crop(landmarks, width, height)
            crop_candidates = []
            if crop is not None:
                image = ball_module.crop_image(bgr, crop, cfg.crop_pad_value)
                crop_candidates = [ball_module.to_full_frame(d, crop) for d in detector.detect(image, cfg.crop_imgsz)]
            head_pitch = head_sampler.update(bgr, landmarks, t_s, t_ms) if head_sampler else POSE_PITCH
            record = pipeline.step(t_s, landmarks, crop, crop_candidates,
                                   lambda: detector.detect(bgr, cfg.full_imgsz), head_pitch)
            proc_ms.append((time.perf_counter() - start) * 1000)
            lag_ms.append((time.monotonic() - arrival) * 1000)
            full_frame_runs += record["ran_full_frame"]
            hud.update(record["events"], t_s)
            events += [{"frame": index, **asdict(event)} for event in record["events"]]

            if not args.no_window:
                view = bgr.copy()
                draw(view, record, landmarks, hud, cfg, t_s)
                draw_status(view, statistics.median(proc_ms[-30:]), lag_ms[-1], len(proc_ms), source.grabbed)
                cv2.imshow(WINDOW, fit(view, DISPLAY_MAX_PX))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    pipeline, hud = FramePipeline(cfg), Hud()
                    events.append({"frame": index, "kind": "counts_reset", "t_s": t_s, "reported_t_s": t_s,
                                   "foot": None, "value": None})
    finally:
        source.stop()
        pose.close()
        if face_pose is not None:
            face_pose.close()
        if not args.no_window:
            cv2.destroyAllWindows()

    counts = {f"{kind}_{foot}": n for (kind, foot), n in hud.counts.items()}
    summary = {
        "source": args.source,
        "recording": str(record_path) if record_path else None,
        "frames_grabbed": source.grabbed,
        "frames_processed": len(proc_ms),
        "full_frame_runs": full_frame_runs,
        "clip_reader_behind_s": None if is_camera else round(source.behind_s, 2),
        "processing_ms": summary_stats(proc_ms),
        "feedback_lag_ms": summary_stats(lag_ms),
        **counts,
        "lost_control": sum(e["kind"] == "lost_control" for e in events),
        "head_down": sum(e["kind"] == "head_down" for e in events),
    }
    out = Path("output") / "live" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "events": events}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved {out}" + (f" and the recording {record_path}" if record_path else ""))


if __name__ == "__main__":
    main()
