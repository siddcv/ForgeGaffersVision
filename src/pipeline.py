"""The per-frame logic shared by replay (cached clips), live (camera) and, later, the phone app.

Everything after the neural networks happens here, once per frame and in this order:
  1. One Euro smoothing of the heel and toe keypoints
  2. tracker picks the best crop candidate
  3. full-frame candidates are requested only if the search switch needs them
  4. search switch chooses this frame's detection
  5. Kalman tracker update
  6. metrics.step

The Swift port mirrors FramePipeline.step line by line. The models stay outside: the caller passes
landmarks and crop candidates in, plus the full-frame candidates as a list (replay, from the cache)
or a zero-argument function (live), which is called only when the search switch needs it.
"""
from typing import Callable, Sequence

from . import ball as ball_module
from . import metrics
from .filters import OneEuroFilter
from .pose import LEFT_FOOT, RIGHT_FOOT, coarse_head_pitch_deg, visible_points

Candidates = Sequence[tuple]  # (x1, y1, x2, y2, confidence) in full-frame pixels
POSE_PITCH = object()  # step(head_pitch_deg=...) default: the rough pose-only estimate


class FramePipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.search = ball_module.SearchState()
        self.tracker = ball_module.BallTracker(cfg)
        self.metrics_state = metrics.initial_state()
        self._euro = {}

    def foot_crop(self, landmarks, frame_width: int, frame_height: int):
        """Crop for this frame's ball detector, sized from the tracked ball's diameter.

        Live code calls this between pose and ball detection. Replay uses the crop stored in the
        cache instead, which build_cache sized from the previous frame's raw detection.
        """
        feet = visible_points(landmarks, LEFT_FOOT + RIGHT_FOOT, self.cfg.pose_min_visibility)
        return ball_module.foot_crop(feet, frame_width, frame_height, self.cfg, self.tracker.diameter_px)

    def _smooth_points(self, indices, landmarks, t_s: float) -> tuple:
        smoothed = []
        for index in indices:
            # Written as "not >=" so a frame with no pose (NaN) is skipped too: one NaN fed to a
            # One Euro filter stays in its state and turns every later foot position into NaN.
            if not landmarks[index, 2] >= self.cfg.pose_min_visibility:
                continue
            xy = []
            for axis in (0, 1):
                key = (index, axis)
                if key not in self._euro:
                    self._euro[key] = OneEuroFilter(
                        self.cfg.one_euro_min_cutoff, self.cfg.one_euro_beta, self.cfg.one_euro_d_cutoff,
                    )
                xy.append(self._euro[key].update(t_s, float(landmarks[index, axis])))
            smoothed.append(tuple(xy))
        return tuple(smoothed)

    def step(
        self,
        t_s: float,
        landmarks,
        crop,
        crop_candidates: Candidates,
        full_candidates: Candidates | Callable[[], Candidates],
        head_pitch_deg=POSE_PITCH,
    ) -> dict:
        """Process one frame. landmarks is a (33, 3) array of pixel x, pixel y, visibility.

        head_pitch_deg is the Face Landmarker pitch (None when no face was found this frame).
        Leave it out to use the rough pose-only estimate instead, which is on a different scale:
        the head thresholds in config are calibrated for Face Landmarker pitch.
        """
        cfg = self.cfg
        if head_pitch_deg is POSE_PITCH:
            head_pitch_deg = coarse_head_pitch_deg(landmarks, cfg.pose_min_visibility)
        left_foot = self._smooth_points(LEFT_FOOT, landmarks, t_s)
        right_foot = self._smooth_points(RIGHT_FOOT, landmarks, t_s)
        feet = left_foot + right_foot

        crop_det = self.tracker.choose(t_s, crop_candidates, feet)
        full_det = None
        ran_full_frame = ball_module.needs_full_frame(self.search, t_s, crop, crop_det, cfg)
        if ran_full_frame:
            candidates = full_candidates() if callable(full_candidates) else full_candidates
            full_det = self.tracker.choose(t_s, candidates, feet)
        self.search, detection, searched = ball_module.search_step(
            self.search, t_s, crop, crop_det, full_det, cfg,
        )
        ball_xy, predicted = self.tracker.update(t_s, detection, searched)

        observation = metrics.Observation(
            t_s=t_s,
            # Metrics see detections only: on a real clip, snapping from a prediction back to the
            # next detection looks like a direction change and adds false touches.
            ball_xy=None if predicted else ball_xy,
            ball_diameter_px=ball_module.box_diameter_px(detection) if detection else None,
            searched_full_frame=searched,
            left_foot=left_foot,
            right_foot=right_foot,
            head_pitch_deg=head_pitch_deg,
        )
        self.metrics_state, events = metrics.step(self.metrics_state, observation, cfg)
        return {
            "t_s": t_s, "crop": crop, "ball": detection, "ball_xy": ball_xy,
            "predicted": predicted, "searched_full_frame": searched, "ran_full_frame": ran_full_frame,
            "events": events, "touch_rate": metrics.touch_rate(self.metrics_state, t_s, cfg),
            "head_pitch_deg": head_pitch_deg,
        }
