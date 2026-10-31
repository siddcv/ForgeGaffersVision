"""Streaming tick-tock metrics. Standard library only, and never reads future frames.

Call step(state, observation, cfg) once per frame. cfg is any object with the
threshold fields of config.Config, passed in so this module imports nothing else.
"""
import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Observation:
    t_s: float
    ball_xy: tuple | None = None
    ball_diameter_px: float | None = None
    searched_full_frame: bool = False
    left_foot: tuple = ()  # (x, y) points
    right_foot: tuple = ()
    head_pitch_deg: float | None = None


@dataclass(frozen=True)
class Event:
    kind: str  # touch, speed, lost_control, reset, head_down, head_up
    t_s: float  # when it happened
    reported_t_s: float  # when the system knew; the difference is detection latency
    foot: str | None = None
    value: float | None = None


@dataclass(frozen=True)
class State:
    diameter_px: float | None = None
    last_ball_t_s: float | None = None
    last_ball_xy: tuple | None = None
    last_vx_mps: float | None = None
    last_foot_distances: tuple | None = None
    last_touch_t_s: float | None = None
    speed_foot: str | None = None
    speed_start_t_s: float | None = None
    speed_end_t_s: float | None = None
    speed_distance_m: float = 0.0
    lost: bool = False
    far_since_s: float | None = None
    not_found_since_s: float | None = None
    head_low_since_s: float | None = None
    head_down_since_s: float | None = None


def initial_state() -> State:
    return State()


def smooth_diameter(previous: float | None, measured: float | None, weight: float) -> float | None:
    if measured is None or measured <= 0:
        return previous
    if previous is None:
        return measured
    return weight * measured + (1 - weight) * previous


def foot_distance(ball_xy: tuple, foot: tuple) -> float | None:
    if not foot:
        return None
    return min(math.dist(ball_xy, point) for point in foot)


def nearer_foot(distances: tuple) -> tuple:
    """Return (foot name, distance) for the closer foot, or (None, None)."""
    left, right = distances
    if left is None and right is None:
        return None, None
    if right is None or (left is not None and left <= right):
        return "left", left
    return "right", right


def is_reversal(previous_vx: float | None, vx: float, min_speed: float) -> bool:
    return (
        previous_vx is not None
        and previous_vx * vx < 0
        and abs(previous_vx) >= min_speed
        and abs(vx) >= min_speed
    )


def _speed_event(state: State, reported_t_s: float) -> Event | None:
    if state.speed_start_t_s is None or state.speed_end_t_s is None:
        return None
    elapsed = state.speed_end_t_s - state.speed_start_t_s
    if elapsed <= 0:
        return None
    return Event(
        "speed", state.speed_start_t_s, reported_t_s,
        foot=state.speed_foot, value=state.speed_distance_m / elapsed,
    )


def _step_ball(state: State, obs: Observation, distances: tuple, cfg) -> tuple:
    """Update velocity, touches and speed from a frame where the ball was detected."""
    events = []
    diameter = state.diameter_px
    meters_per_px = cfg.ball_diameter_m / diameter
    updates = {
        "last_ball_t_s": obs.t_s,
        "last_ball_xy": obs.ball_xy,
        "last_foot_distances": distances,
    }
    dt = None if state.last_ball_t_s is None else obs.t_s - state.last_ball_t_s
    if dt is None or dt <= 0 or dt > cfg.touch_max_gap_s:
        return replace(state, last_vx_mps=None, **updates), events

    vx = (obs.ball_xy[0] - state.last_ball_xy[0]) * meters_per_px / dt
    # Keep the last clear direction: across an occluded touch, net motion is near zero.
    if abs(vx) >= cfg.touch_min_speed_mps:
        updates["last_vx_mps"] = vx
    moved_m =math.dist(obs.ball_xy, state.last_ball_xy) * meters_per_px

    touched = False
    if is_reversal(state.last_vx_mps, vx, cfg.touch_min_speed_mps):
        recent = state.last_touch_t_s is not None and (
            state.last_ball_t_s - state.last_touch_t_s < cfg.touch_min_interval_s
        )
        # Check both frames, since a foot may hide the ball right at the touch.
        before, before_distance = nearer_foot(state.last_foot_distances or (None, None))
        after, after_distance = nearer_foot(distances)
        candidates = [c for c in ((before_distance, before), (after_distance, after)) if c[1]]
        limit = cfg.touch_foot_distance_diam * diameter
        if candidates and not recent and min(c[0] for c in candidates) <= limit:
            touched = True
            foot = min(candidates, key=lambda c: c[0])[1]
            touch_t_s = state.last_ball_t_s
            pending = _speed_event(state, obs.t_s)
            if pending:
                events.append(pending)
            events.append(Event("touch", touch_t_s, obs.t_s, foot=foot))
            updates.update(
                last_touch_t_s=touch_t_s,
                speed_foot=foot,
                # Start after the turnaround frame, whose motion partly cancels out.
                speed_start_t_s=obs.t_s,
                speed_end_t_s=obs.t_s,
                speed_distance_m=0.0,
            )

    if not touched and state.speed_start_t_s is not None:
        updates["speed_end_t_s"] = obs.t_s
        updates["speed_distance_m"] = state.speed_distance_m + moved_m
        if obs.t_s - state.speed_start_t_s >= cfg.speed_window_s:
            finished = replace(state, **updates)
            events.append(_speed_event(finished, obs.t_s))
            updates.update(speed_start_t_s=None, speed_end_t_s=None, speed_distance_m=0.0)

    return replace(state, **updates), events


def _step_lost_control(state: State, obs: Observation, distances: tuple, cfg) -> tuple:
    if obs.ball_xy is None:
        # A miss in the foot crop proves nothing; only a full-frame search counts.
        if not obs.searched_full_frame:
            return state, []
        since = state.not_found_since_s if state.not_found_since_s is not None else obs.t_s
        state = replace(state, not_found_since_s=since, far_since_s=None)
        if not state.lost and obs.t_s - since >= cfg.lost_control_min_s:
            return replace(state, lost=True), [Event("lost_control", since, obs.t_s)]
        return state, []

    state = replace(state, not_found_since_s=None)
    _, nearest = nearer_foot(distances)
    if nearest is None:
        return replace(state, far_since_s=None), []

    if state.lost:
        if nearest <= cfg.touch_foot_distance_diam * state.diameter_px:
            return replace(state, lost=False, far_since_s=None), [Event("reset", obs.t_s, obs.t_s)]
        return state, []

    if nearest <= cfg.lost_distance_diam * state.diameter_px:
        return replace(state, far_since_s=None), []
    since = state.far_since_s if state.far_since_s is not None else obs.t_s
    state = replace(state, far_since_s=since)
    if obs.t_s - since >= cfg.lost_control_min_s:
        return replace(state, lost=True), [Event("lost_control", since, obs.t_s)]
    return state, []


def _step_head(state: State, obs: Observation, cfg) -> tuple:
    pitch = obs.head_pitch_deg
    if pitch is None:
        return state, []

    if state.head_down_since_s is not None:
        if pitch > cfg.head_up_pitch_deg:
            duration = obs.t_s - state.head_down_since_s
            event = Event("head_up", obs.t_s, obs.t_s, value=duration)
            return replace(state, head_down_since_s=None, head_low_since_s=None), [event]
        return state, []

    if pitch > cfg.head_down_pitch_deg:
        return replace(state, head_low_since_s=None), []
    since = state.head_low_since_s if state.head_low_since_s is not None else obs.t_s
    state = replace(state, head_low_since_s=since)
    if obs.t_s - since >= cfg.head_down_min_s:
        return replace(state, head_down_since_s=since), [Event("head_down", since, obs.t_s)]
    return state, []


def step(state: State, obs: Observation, cfg) -> tuple:
    """Advance one frame. Returns (new_state, events)."""
    events = []
    state = replace(
        state,
        diameter_px=smooth_diameter(state.diameter_px, obs.ball_diameter_px, cfg.diameter_smoothing),
    )
    distances = (None, None)
    if obs.ball_xy is not None and state.diameter_px is not None:
        distances = (foot_distance(obs.ball_xy, obs.left_foot), foot_distance(obs.ball_xy, obs.right_foot))
        state, new_events = _step_ball(state, obs, distances, cfg)
        events += new_events
    state, new_events = _step_lost_control(state, obs, distances, cfg)
    events += new_events
    state, new_events = _step_head(state, obs, cfg)
    events += new_events
    return state, events
