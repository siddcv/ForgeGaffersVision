"""Hand-label a clip by pressing keys while it plays: touches, lost control, head down.

Usage: python -m src.label_clip data/clips/<tune|test>/<session>/<clip>.mp4 [--focus touches|lost|head|all]
                                [--out data/labels/<clip>.csv]

Each key press is stamped with the current frame's real timestamp, read the same way as the
pipeline (video_io.read_timestamps), so labels and detections share one time base.
Writes data/labels/<clip>.csv in the format evaluate.py reads, saving after every change.
If the file exists, labelling resumes from it.

Label in passes, one kind of event at a time: --focus touches, then --focus lost, then
--focus head. Every pass adds to the same file. A pass only accepts its own keys and only
erases its own labels, so it never disturbs the other passes. Run one pass at a time per clip.

Keys
  space        play / pause                  , .    one frame back / forward
  j k          one second back / forward     [ ]    slower / faster
  l r          touch, player's LEFT / RIGHT foot (the player's own foot, not the screen side)
  x y          lost control starts / control regained (also writes a reset)
  h u          head down starts / head back up
  e            erase the nearest label of this pass
  z            undo                          q Esc  save and quit
"""
import argparse
import csv
import os
import statistics
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path

INTERVAL_KINDS = ("lost_control", "head_down")
FIELDS = ("kind", "t_s", "end_s", "foot", "note")
SPEEDS = (0.1, 0.25, 0.5, 1.0)
ERASE_WITHIN_S = 0.25

# Each pass: the label kinds it owns, its keys, and its starting playback speed.
# Touches need slow playback; lost control and head movements are slower events.
PASSES = {
    "touches": ({"touch"}, "lr", 0.25),
    "lost": ({"lost_control", "reset"}, "xy", 0.5),
    "head": ({"head_down"}, "hu", 0.5),
}
PASSES["all"] = (
    set().union(*(kinds for kinds, _, _ in PASSES.values())),
    "".join(keys for _, keys, _ in PASSES.values()),
    0.25,
)


@dataclass(frozen=True)
class Label:
    kind: str
    t_s: float
    end_s: float | None = None
    foot: str | None = None
    note: str = ""


class LabelSession:
    """The labels for one clip and the edits made to them. No video or UI code.

    An interval (lost_control, head_down) is open while its end_s is None. Every method
    returns a short message for the status line.
    """

    def __init__(self, labels=()):
        self.labels = list(labels)
        self._history = []

    def _remember(self) -> None:
        self._history.append(list(self.labels))

    def open_interval(self, kind: str) -> Label | None:
        return next((l for l in self.labels if l.kind == kind and l.end_s is None), None)

    def touch(self, t_s: float, foot: str) -> str:
        if any(l.kind == "touch" and l.foot == foot and abs(l.t_s - t_s) < 1e-6 for l in self.labels):
            return f"{foot} touch already marked at {t_s:.3f} s"
        self._remember()
        self.labels.append(Label("touch", t_s, foot=foot))
        return f"touch {foot} at {t_s:.3f} s"

    def start(self, kind: str, t_s: float) -> str:
        name = kind.replace("_", " ")
        open_label = self.open_interval(kind)
        if open_label is not None:
            return f"{name} already open since {open_label.t_s:.3f} s; close it first"
        if any(l.kind == kind and l.end_s is not None and l.t_s <= t_s <= l.end_s for l in self.labels):
            return f"{t_s:.3f} s is inside an existing {name} interval"
        self._remember()
        self.labels.append(Label(kind, t_s))
        return f"{name} starts at {t_s:.3f} s"

    def end(self, kind: str, t_s: float) -> str:
        name = kind.replace("_", " ")
        open_label = self.open_interval(kind)
        if open_label is None:
            return f"no open {name} to close"
        if t_s <= open_label.t_s:
            return f"end must be after the start ({open_label.t_s:.3f} s)"
        self._remember()
        self.labels[self.labels.index(open_label)] = replace(open_label, end_s=t_s)
        if kind == "lost_control":
            self.labels.append(Label("reset", t_s))
        return f"{name} ends at {t_s:.3f} s ({t_s - open_label.t_s:.2f} s long)"

    def erase_nearest(self, t_s: float, within_s: float = ERASE_WITHIN_S, kinds=None) -> str:
        """Erase the label nearest t_s, considering only `kinds` when given."""
        def distance(label):
            ends = [label.t_s] + ([label.end_s] if label.end_s is not None else [])
            return min(abs(e - t_s) for e in ends)

        candidates = [l for l in self.labels if kinds is None or l.kind in kinds]
        if not candidates:
            return "nothing to erase"
        nearest = min(candidates, key=distance)
        if distance(nearest) > within_s:
            return f"no label within {within_s} s"
        self._remember()
        self.labels.remove(nearest)
        if nearest.kind == "lost_control" and nearest.end_s is not None:
            # Its reset goes with it.
            self.labels = [
                l for l in self.labels
                if not (l.kind == "reset" and abs(l.t_s - nearest.end_s) < 1e-6)
            ]
        return f"erased {nearest.kind} at {nearest.t_s:.3f} s"

    def undo(self) -> str:
        if not self._history:
            return "nothing to undo"
        self.labels = self._history.pop()
        return "undone"

    def sorted_labels(self) -> list:
        return sorted(self.labels, key=lambda l: (l.t_s, l.kind))


def load(path: Path) -> list:
    if not path.exists():
        return []
    labels = []
    with open(path, newline="") as file:
        for row in csv.DictReader(file):
            kind = (row.get("kind") or "").strip()
            t_s = (row.get("t_s") or "").strip()
            if not kind or not t_s:
                continue
            end_s = (row.get("end_s") or "").strip()
            labels.append(Label(
                kind, float(t_s), float(end_s) if end_s else None,
                (row.get("foot") or "").strip() or None, (row.get("note") or "").strip(),
            ))
    return labels


def save(path: Path, labels: list, attempts: int = 20, wait_s: float = 0.05) -> None:
    """Write atomically, so a crash mid-save never leaves a half-written file.

    On Windows the final rename fails while another program holds the file open, which OneDrive
    does briefly while syncing it and Excel does while it is open. Retry for about a second
    before giving up; the caller keeps the labels in memory and tries again later.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(FIELDS)
        for l in sorted(labels, key=lambda l: (l.t_s, l.kind)):
            writer.writerow([
                l.kind, f"{l.t_s:.3f}", "" if l.end_s is None else f"{l.end_s:.3f}", l.foot or "", l.note,
            ])
    for attempt in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(wait_s * (attempt + 1) / 4)


def try_save(path: Path, labels: list) -> str | None:
    """save(), returning an error message instead of raising when the file stays locked."""
    try:
        save(path, labels)
        return None
    except OSError as error:
        return f"NOT SAVED ({error.__class__.__name__}): close Excel / wait for OneDrive; retries on next key"


def summarize(labels: list) -> dict:
    """What the labels say about the drill, as a starting point for config.py thresholds."""
    touches = sorted((l for l in labels if l.kind == "touch"), key=lambda l: l.t_s)
    gaps = [b.t_s - a.t_s for a, b in zip(touches, touches[1:])]
    repeats = sum(a.foot == b.foot for a, b in zip(touches, touches[1:]))

    def durations(kind):
        return [round(l.end_s - l.t_s, 3) for l in labels if l.kind == kind and l.end_s is not None]

    return {
        "touches_left": sum(l.foot == "left" for l in touches),
        "touches_right": sum(l.foot == "right" for l in touches),
        "same_foot_repeats": repeats,
        "touch_gap_min_s": round(min(gaps), 3) if gaps else None,
        "touch_gap_median_s": round(statistics.median(gaps), 3) if gaps else None,
        "lost_control_durations_s": durations("lost_control"),
        "head_down_durations_s": durations("head_down"),
        "open_intervals": [l.kind for l in labels if l.kind in INTERVAL_KINDS and l.end_s is None],
    }


class FrameSource:
    """Random access to display-sized frames, in the same order as video_io.iter_frames.

    Recent frames are kept in memory so stepping back is instant. Going back further reopens
    the video and decodes forward, which is slow but always lands on the right frame.
    """

    def __init__(self, path: str, times, display_height: int, buffer_frames: int):
        self.path = path
        self.times = times
        self.display_height = display_height
        self.buffer_frames = buffer_frames
        self._capture = None
        self._next = 0
        self._cache = OrderedDict()

    def __len__(self) -> int:
        return len(self.times)

    def _open(self) -> None:
        import cv2

        if self._capture is not None:
            self._capture.release()
        self._capture = cv2.VideoCapture(self.path)
        if not self._capture.isOpened():
            raise ValueError(f"Could not open {self.path}")
        self._next = 0

    def get(self, index: int):
        import cv2

        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        if self._capture is None or index < self._next:
            self._open()
        while self._next < index:
            self._capture.grab()
            self._next += 1
        ok, bgr = self._capture.read()
        self._next += 1
        if not ok:
            return None
        scale = self.display_height / bgr.shape[0]
        image = cv2.resize(bgr, (int(bgr.shape[1] * scale), self.display_height), interpolation=cv2.INTER_AREA)
        self._cache[index] = image
        while len(self._cache) > self.buffer_frames:
            self._cache.popitem(last=False)
        return image

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()


COLORS = {  # BGR
    "left": (255, 160, 0),
    "right": (0, 200, 255),
    "lost_control": (0, 0, 230),
    "head_down": (230, 0, 230),
    "reset": (0, 220, 0),
}


def _text(image, text, origin, color=(255, 255, 255), scale=0.6):
    import cv2

    x, y = origin
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(image, (x - 4, y - h - 6), (x + w + 4, y + 6), (0, 0, 0), -1)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw(frame, session: LabelSession, index: int, times, speed: float, playing: bool, message: str,
         focus: str = "all"):
    """Frame with a status panel on top and a label timeline underneath.

    Labels from other passes stay on the timeline, dimmed, for context.
    """
    import cv2
    import numpy as np

    kinds, keys, _ = PASSES[focus]
    image = frame.copy()
    width = image.shape[1]
    t_s, duration = float(times[index]), float(times[-1]) or 1.0
    state = "PLAY" if playing else "PAUSED"
    _text(image, f"{t_s:7.3f} s   frame {index}/{len(times) - 1}   {state} x{speed}", (10, 24))
    _text(image, f"PASS: {focus}   keys: {' '.join(keys)}", (10, 52), (120, 255, 255))
    y = 80
    if "touch" in kinds:
        touches = [l for l in session.labels if l.kind == "touch"]
        _text(image, f"touches L {sum(l.foot == 'left' for l in touches)}  R {sum(l.foot == 'right' for l in touches)}",
              (10, y))
        y += 28
    for kind in INTERVAL_KINDS:
        open_label = session.open_interval(kind)
        if open_label is not None and kind in kinds:
            _text(image, f"{kind.replace('_', ' ').upper()} since {open_label.t_s:.2f} s", (10, y), COLORS[kind])
            y += 28
    for label in session.labels:  # flash marks on the frame they were made
        if label.kind in kinds and abs(label.t_s - t_s) < 1e-6:
            name = f"TOUCH {label.foot.upper()}" if label.kind == "touch" else label.kind.upper()
            _text(image, name, (10, y), COLORS.get(label.foot or label.kind, (255, 255, 255)), 0.9)
            y += 34
    if message:
        _text(image, message, (10, image.shape[0] - 16), (200, 255, 200))

    bar = np.full((44, width, 3), 40, np.uint8)
    to_x = lambda t: int(round(t / duration * (width - 1)))
    for label in session.labels:
        dim = 1.0 if label.kind in kinds else 0.35
        if label.kind in INTERVAL_KINDS:
            row = 2 if label.kind == "lost_control" else 12
            end = label.end_s if label.end_s is not None else t_s
            color = tuple(int(c * dim) for c in COLORS[label.kind])
            cv2.rectangle(bar, (to_x(label.t_s), row), (max(to_x(end), to_x(label.t_s) + 1), row + 8), color, -1)
        elif label.kind == "touch":
            row = 22 if label.foot == "left" else 32
            color = tuple(int(c * dim) for c in COLORS[label.foot])
            cv2.line(bar, (to_x(label.t_s), row), (to_x(label.t_s), row + 9), color, 1)
    cv2.line(bar, (to_x(t_s), 0), (to_x(t_s), 43), (255, 255, 255), 1)
    return np.vstack([image, bar])


def main() -> None:
    import cv2

    from .video_io import read_timestamps

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clip")
    parser.add_argument("--focus", choices=list(PASSES), default="all",
                        help="label one kind of event per pass; all passes add to the same file")
    parser.add_argument("--out", help="defaults to data/labels/<clip>.csv")
    parser.add_argument("--display-height", type=int, default=900)
    parser.add_argument("--buffer-frames", type=int, default=240, help="frames kept for instant stepping back")
    args = parser.parse_args()

    out = Path(args.out) if args.out else Path("data/labels") / f"{Path(args.clip).stem}.csv"
    session = LabelSession(load(out))
    times = read_timestamps(args.clip)
    source = FrameSource(args.clip, times, args.display_height, args.buffer_frames)
    window = f"label {Path(args.clip).name}"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    focus_kinds, focus_keys, start_speed = PASSES[args.focus]
    index, speed_i, playing, unsaved = 0, SPEEDS.index(start_speed), False, False
    message = f"resumed {len(session.labels)} labels from {out}" if session.labels else "space to play"
    last = len(times) - 1
    all_edits = {
        "l": lambda t: session.touch(t, "left"),
        "r": lambda t: session.touch(t, "right"),
        "x": lambda t: session.start("lost_control", t),
        "y": lambda t: session.end("lost_control", t),
        "h": lambda t: session.start("head_down", t),
        "u": lambda t: session.end("head_down", t),
    }
    edits = {ord(k): f for k, f in all_edits.items() if k in focus_keys}
    edits[ord("e")] = lambda t: session.erase_nearest(t, kinds=focus_kinds)
    edits[ord("z")] = lambda t: session.undo()
    other_keys = {ord(k) for k in all_edits if k not in focus_keys}
    print(__doc__.split("Keys", 1)[1])
    try:
        while True:
            frame = source.get(index)
            if frame is None:  # decoder ran out before the timestamp list did
                last = index = max(0, index - 1)
                playing = False
                continue
            cv2.imshow(window, draw(frame, session, index, times, SPEEDS[speed_i], playing, message, args.focus))
            delay = 0
            if playing:
                step_s = float(times[min(index + 1, last)] - times[index]) or 1 / 30
                delay = max(1, int(step_s / SPEEDS[speed_i] * 1000))
            key = cv2.waitKey(delay) & 0xFF
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            t_s = float(times[index])
            if key == 255:  # no key: keep playing
                if playing:
                    index = min(index + 1, last)
                    playing = index < last
                continue
            if key in (ord("q"), 27):
                break
            if key in edits:
                before = list(session.labels)
                message = edits[key](t_s)
                unsaved = unsaved or session.labels != before
            elif key in other_keys:
                message = f"'{chr(key)}' is not used in the {args.focus} pass (keys: {' '.join(focus_keys)})"
            elif key == ord(" "):
                playing = not playing and index < last
            elif key == ord(","):
                index, playing = max(0, index - 1), False
            elif key == ord("."):
                index, playing = min(last, index + 1), False
            elif key in (ord("j"), ord("k")):
                target = t_s + (1.0 if key == ord("k") else -1.0)
                index = int(min(range(len(times)), key=lambda i: abs(times[i] - target)))
                index = min(index, last)
            elif key == ord("["):
                speed_i = max(0, speed_i - 1)
            elif key == ord("]"):
                speed_i = min(len(SPEEDS) - 1, speed_i + 1)
            if unsaved:  # after every key, so a failed save is retried on the next one
                error = try_save(out, session.labels)
                if error:
                    message = error
                else:
                    unsaved = False
    finally:
        source.close()
        cv2.destroyAllWindows()
        if unsaved and try_save(out, session.labels):
            # The labels file is still locked: keep the work in a separate file instead of losing it.
            backup = out.with_name(f"{out.stem}.unsaved-{time.strftime('%Y%m%d-%H%M%S')}.csv")
            save(backup, session.labels)
            print(f"Could not write {out} (open in Excel or locked by OneDrive).")
            print(f"Your labels are in {backup}; rename it to {out.name} once the file is free.")
            raise SystemExit(1)

    summary = summarize(session.labels)
    print(f"Saved {len(session.labels)} labels to {out}")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    if summary["open_intervals"]:
        print("  WARNING: some intervals have no end; reopen the clip and close them with y / u")


if __name__ == "__main__":
    main()
