"""Streaming tick-tock metrics. Standard library only, and never reads future frames.

Call step(state, observation, cfg) once per frame. cfg is any object with the
threshold fields of config.Config, passed in so this module imports nothing else.

Two touch counts:
  contact  any touch on the ball: a direction change next to a foot. Setup taps, recovery
           taps and repeats on one foot all count. Reported straight away.
  touch    a tick-tock: a contact by one foot that sends the ball across the midpoint between
           the feet and whose next contact is by the other foot, within ticktock_max_arrival_s.
           Stamped at the first foot's contact, reported when the other foot touches the ball,
           so it lags about one touch gap. Touches per minute counts these.
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
    kind: str  # contact, touch (tick-tock), speed, lost_control, reset, head_down, head_up
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
    last_feet_x: tuple | None = None  # (left, right) mean foot x on the last ball frame
    last_touch_t_s: float | None = None  # last contact, for touch_min_interval_s
    # The latest contact, waiting to be confirmed as a tick-tock by the other foot.
    pending_foot: str | None = None
    pending_t_s: float | None = None
    pending_crossed: bool = False  # the ball has since crossed the midpoint toward the other foot
    # Minimum travel: where the current run of motion started, and a turn near a foot that becomes a
    # contact once the ball has moved touch_min_travel_diam away from it in the new direction.
    run_start_x: float | None = None
    turn_t_s: float | None = None
    turn_x: float | None = None
    turn_foot: str | None = None
    turn_direction: int = 0  # +1 / -1, the new direction of travel in image x
    first_touch_t_s: float | None = None
    recent_touch_t_s: tuple = ()  # touch times within the last touch_minute_s, oldest first
    speed_foot: str | None = None
    speed_start_t_s: float | None = None
    speed_end_t_s: float | None = None
    speed_distance_m: float = 0.0
    lost: bool = False
    far_since_s: float | None = None
    not_found_since_s: float | None = None
    head_low_since_s: float | None = None
    head_down_since_s: float | None = None


@dataclass(frozen=True)
class TouchRate:
    last_minute: int  # touches in the last touch_minute_s, the number the goal is set on
    minute_complete: bool  # a full minute has passed since the first touch
    pace_per_min: float | None  # touches in the last touch_pace_window_s, scaled to a minute


def initial_state() -> State:
    return State()


def record_touch(state: State, t_s: float, cfg) -> State:
    """Add a touch to the rolling window, dropping touches older than a minute."""
    recent = tuple(t for t in state.recent_touch_t_s if t_s - t < cfg.touch_minute_s) + (t_s,)
    first = state.first_touch_t_s if state.first_touch_t_s is not None else t_s
    return replace(state, recent_touch_t_s=recent, first_touch_t_s=first)


def touch_rate(state: State, t_s: float, cfg) -> TouchRate:
    """Touches per minute as of t_s, for a live readout. Call on any frame.

    The pace uses a shorter window so it responds within seconds and falls when the player
    stops. Before touch_pace_window_s has passed since the first touch it uses the time so far,
    and it stays None until touch_pace_min_s, when a couple of touches would give a wild number.
    """
    if state.first_touch_t_s is None:
        return TouchRate(0, False, None)
    elapsed = t_s - state.first_touch_t_s
    last_minute = sum(0 <= t_s - t < cfg.touch_minute_s for t in state.recent_touch_t_s)
    window = min(cfg.touch_pace_window_s, elapsed)
    pace = None
    if window >= cfg.touch_pace_min_s:
        pace = sum(0 <= t_s - t <= window for t in state.recent_touch_t_s) / window * 60
    return TouchRate(last_minute, elapsed >= cfg.touch_minute_s, pace)


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


def mean_x(points: tuple) -> float | None:
    return sum(p[0] for p in points) / len(points) if points else None


def foot_by_direction(incoming_vx: float | None, feet_x: tuple | None) -> str | None:
    """The foot the ball was travelling toward before it reversed, or None if unclear.

    In a tick-tock the ball crosses to a foot and comes back, so the foot ahead of it made the
    touch. Unlike the nearest foot, this needs no ball detection at the moment of contact,
    when the foot often hides the ball. Works for mirrored video too, since it compares the
    feet's own image positions.
    """
    if incoming_vx is None or incoming_vx == 0 or feet_x is None:
        return None
    left_x, right_x = feet_x
    if left_x is None or right_x is None or left_x == right_x:
        return None
    ahead_is_larger_x = incoming_vx > 0
    return "left" if (left_x > right_x) == ahead_is_larger_x else "right"


def crossed_midpoint(ball_x: float, left_foot: tuple, right_foot: tuple, from_foot: str) -> bool:
    """Whether the ball is past the midpoint between the feet, on the side away from from_foot.

    The midpoint is halfway between the two feet, each foot being the mean x of its heel and
    toe on this frame, so it moves with the player. Compares the feet's own image positions,
    so it works for mirrored video too. False when either foot is not visible.
    """
    left_x, right_x = mean_x(left_foot), mean_x(right_foot)
    if left_x is None or right_x is None or left_x == right_x:
        return False
    midpoint = (left_x + right_x) / 2
    target_x = right_x if from_foot == "left" else left_x
    return (ball_x - midpoint) * (target_x - midpoint) > 0


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
        "last_feet_x": (mean_x(obs.left_foot), mean_x(obs.right_foot)),
    }
    if (state.pending_foot is not None and not state.pending_crossed
            and crossed_midpoint(obs.ball_xy[0], obs.left_foot, obs.right_foot, state.pending_foot)):
        updates["pending_crossed"] = True
        state = replace(state, pending_crossed=True)
    no_turn = {"turn_t_s": None, "turn_x": None, "turn_foot": None, "turn_direction": 0}
    dt = None if state.last_ball_t_s is None else obs.t_s - state.last_ball_t_s
    if dt is None or dt <= 0 or dt > cfg.touch_max_gap_s:
        # New track: no direction yet, and the run of motion starts here.
        return replace(state, last_vx_mps=None, run_start_x=obs.ball_xy[0], **no_turn, **updates), events

    vx = (obs.ball_xy[0] - state.last_ball_xy[0]) * meters_per_px / dt
    # Keep the last clear direction: across an occluded touch, net motion is near zero.
    if abs(vx) >= cfg.touch_min_speed_mps:
        updates["last_vx_mps"] = vx
    moved_m =math.dist(obs.ball_xy, state.last_ball_xy) * meters_per_px
    min_travel_px = cfg.touch_min_travel_diam * diameter
    run_start_x = state.run_start_x if state.run_start_x is not None else state.last_ball_xy[0]

    if is_reversal(state.last_vx_mps, vx, cfg.touch_min_speed_mps):
        turn_x = state.last_ball_xy[0]
        travelled_in = abs(turn_x - run_start_x)
        # A new run starts at the turn. Any earlier turn that has not yet travelled far enough was
        # jitter, so it is dropped.
        updates.update(run_start_x=turn_x, **no_turn)
        recent = state.last_touch_t_s is not None and (
            state.last_ball_t_s - state.last_touch_t_s < cfg.touch_min_interval_s
        )
        # Check both frames, since a foot may hide the ball right at the touch.
        before, before_distance = nearer_foot(state.last_foot_distances or (None, None))
        after, after_distance = nearer_foot(distances)
        candidates = [c for c in ((before_distance, before), (after_distance, after)) if c[1]]
        limit = cfg.touch_foot_distance_diam * diameter
        if (candidates and not recent and min(c[0] for c in candidates) <= limit
                and travelled_in >= min_travel_px):
            foot = None
            if cfg.touch_foot_rule == "direction":
                foot = foot_by_direction(state.last_vx_mps, state.last_feet_x)
            if foot is None:
                # Nearest foot. The frame after can come well after contact when the ball was
                # hidden, by which time it is heading for the other foot, so prefer the frame before.
                foot = before or min(candidates, key=lambda c: c[0])[1]
            updates.update(
                turn_t_s=state.last_ball_t_s, turn_x=turn_x, turn_foot=foot,
                turn_direction=1 if vx > 0 else -1,
            )

    # A turn becomes a contact once the ball has travelled far enough in the new direction, so a
    # few pixels of detection jitter on a slow ball never count.
    turn_t_s = updates.get("turn_t_s", state.turn_t_s)
    touched = False
    if turn_t_s is not None:
        turn_x = updates.get("turn_x", state.turn_x)
        direction = updates.get("turn_direction", state.turn_direction)
        if (obs.ball_xy[0] - turn_x) * direction >= min_travel_px:
            touched = True
            foot = updates.get("turn_foot", state.turn_foot)
            touch_t_s = turn_t_s
            updates.update(no_turn)
            pending = _speed_event(state, obs.t_s)
            if pending:
                events.append(pending)
            events.append(Event("contact", touch_t_s, obs.t_s, foot=foot))
            # The pending contact is a tick-tock if the ball crossed over and this foot is the other one.
            if (state.pending_foot is not None and foot != state.pending_foot and state.pending_crossed
                    and touch_t_s - state.pending_t_s <= cfg.ticktock_max_arrival_s):
                events.append(Event("touch", state.pending_t_s, obs.t_s, foot=state.pending_foot))
                counted = record_touch(state, state.pending_t_s, cfg)
                updates.update(
                    recent_touch_t_s=counted.recent_touch_t_s,
                    first_touch_t_s=counted.first_touch_t_s,
                )
            updates.update(
                pending_foot=foot,
                pending_t_s=touch_t_s,
                pending_crossed=False,
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


def _expire_pending(state: State, t_s: float, cfg) -> State:
    """Drop the pending contact once the other foot can no longer confirm it: too much time has
    passed, or control was lost."""
    if state.pending_foot is None:
        return state
    # A turn in time that is still travelling far enough to become a contact may yet answer it.
    answer_coming = state.turn_t_s is not None and state.turn_t_s - state.pending_t_s <= cfg.ticktock_max_arrival_s
    if state.lost or (t_s - state.pending_t_s > cfg.ticktock_max_arrival_s and not answer_coming):
        return replace(state, pending_foot=None, pending_t_s=None, pending_crossed=False)
    return state


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
    state = _expire_pending(state, obs.t_s, cfg)
    state, new_events = _step_head(state, obs, cfg)
    events += new_events
    return state, events
