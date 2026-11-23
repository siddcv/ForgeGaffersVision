"""Synthetic tick-tock generator, so metric tests have known answers."""
from dataclasses import replace

from src.metrics import Observation

LEFT_X, RIGHT_X = 200.0, 400.0  # image positions of the feet
BALL_MIN_X, BALL_MAX_X = 240.0, 360.0  # where the ball reverses
DIAMETER_PX = 40.0
Y = 500.0
LEFT_FOOT = ((LEFT_X, Y), (LEFT_X - 10, Y))
RIGHT_FOOT = ((RIGHT_X, Y), (RIGHT_X + 10, Y))
METERS_PER_PX = 0.22 / DIAMETER_PX


def synth_tick_tock(n_touches: int, fps: float = 30.0, speed_mps: float = 1.5) -> list:
    """Ball starts mid-way moving toward the player's left foot in the image.

    Reversals alternate left, right, left, ... Returns one Observation per frame.
    """
    speed_px = speed_mps / METERS_PER_PX
    span = BALL_MAX_X - BALL_MIN_X
    crossing_s = span / speed_px
    total_s = crossing_s * n_touches
    observations = []
    for i in range(int(total_s * fps) + 1):
        t = i / fps
        # Distance travelled, measured from the first (left) reversal
        travelled = speed_px * t - span / 2
        phase = travelled % (2 * span)
        offset = phase if phase <= span else 2 * span - phase
        x = BALL_MIN_X + offset
        observations.append(Observation(
            t_s=t, ball_xy=(x, Y), ball_diameter_px=DIAMETER_PX,
            left_foot=LEFT_FOOT, right_foot=RIGHT_FOOT, head_pitch_deg=0.0,
        ))
    return observations


def insert_frames(observations: list, at_s: float, inserted_xy: list, fps: float = 30.0) -> list:
    """Splice extra frames in at time at_s and shift everything after them later.

    inserted_xy holds one ball position per inserted frame (None for a frame with no ball).
    """
    i = next(k for k, o in enumerate(observations) if o.t_s >= at_s)
    base = observations[i - 1]
    shift = len(inserted_xy) / fps
    inserted = [
        replace(base, t_s=base.t_s + (k + 1) / fps, ball_xy=xy,
                ball_diameter_px=DIAMETER_PX if xy is not None else None)
        for k, xy in enumerate(inserted_xy)
    ]
    later = [replace(o, t_s=o.t_s + shift) for o in observations[i:]]
    return observations[:i] + inserted + later


def with_hold(observations: list, at_s: float, hold_s: float, fps: float = 30.0) -> list:
    """At the first turnaround from at_s, the ball sits at the foot for hold_s.

    Gives one late touch, then normal pace, with the ball never far from the feet.
    """
    def at_turnaround(o):
        return min(abs(o.ball_xy[0] - BALL_MIN_X), abs(o.ball_xy[0] - BALL_MAX_X)) <= 10.0

    i = next(k for k, o in enumerate(observations) if o.t_s >= at_s and at_turnaround(o))
    xy = observations[i].ball_xy
    return insert_frames(observations, observations[i + 1].t_s, [xy] * round(hold_s * fps), fps)


def with_ball_away(observations: list, at_s: float, away_s: float, fps: float = 30.0) -> list:
    """The ball escapes far from both feet for away_s, then play carries on."""
    far_xy = ((LEFT_X + RIGHT_X) / 2, Y + 400.0)
    return insert_frames(observations, at_s, [far_xy] * round(away_s * fps), fps)


# Short scripted scenarios. Feet at x 200 (left) and 400 (right), midpoint 300.

def path(waypoints, fps=60.0, speed_px_s=300.0, start_t=0.0, feet=(LEFT_FOOT, RIGHT_FOOT)):
    """Ball moving at constant speed through the given x positions, turning at each one."""
    observations, t = [], start_t
    x = waypoints[0]
    for target in waypoints[1:]:
        while abs(target - x) > 1e-9:
            t += 1 / fps
            step_px = min(speed_px_s / fps, abs(target - x))
            x += step_px if target > x else -step_px
            observations.append(Observation(
                t_s=t, ball_xy=(x, Y), ball_diameter_px=DIAMETER_PX,
                left_foot=feet[0], right_foot=feet[1],
            ))
    return observations


def missing(n_frames, searched, start=0.0):
    return [
        Observation(
            t_s=start + i / 30, searched_full_frame=searched,
            left_foot=LEFT_FOOT, right_foot=RIGHT_FOOT,
        )
        for i in range(n_frames)
    ]


def ball_at(x, n_frames, start=0.0):
    return [
        Observation(
            t_s=start + i / 30, ball_xy=(x, Y), ball_diameter_px=DIAMETER_PX,
            left_foot=LEFT_FOOT, right_foot=RIGHT_FOOT,
        )
        for i in range(n_frames)
    ]


def head_levels(cfg) -> tuple:
    """(up, down, boundary) pitches for tests around the single head-pitch cutoff."""
    up = cfg.head_down_pitch_deg + 10.0
    down = cfg.head_down_pitch_deg - 10.0
    between = cfg.head_down_pitch_deg
    return up, down, between


def head(pitches, fps=30):
    return [Observation(t_s=i / fps, head_pitch_deg=p) for i, p in enumerate(pitches)]
