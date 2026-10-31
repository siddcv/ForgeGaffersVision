"""Synthetic tick-tock generator, so metric tests have known answers."""
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
