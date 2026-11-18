"""Synthetic tests for the frequency-disruption ball-loss detector."""
import pytest

from src.ball_loss import BallLossConfig, initial_state, step

# Steady tick-tock at 2 touches/s.
STEADY_INTERVAL_S = 0.5
CFG = BallLossConfig(
    baseline_s=4.0,
    drop_frac=0.3,
    dip_min_s=0.4,
    dip_max_s=2.5,
    min_baseline_intervals=4,
)


def run(touch_schedule, fps=30.0, cfg=CFG):
    """Replay frames; touch_schedule is times that count as touches."""
    state = initial_state()
    events = []
    if not touch_schedule:
        return events
    end_s = max(touch_schedule) + 1.0
    t = 0.0
    dt = 1.0 / fps
    remaining = sorted(touch_schedule)
    while t <= end_s + 1e-9:
        touched = False
        if remaining and abs(t - remaining[0]) < dt * 0.51:
            touched = True
            remaining.pop(0)
        state, new_events = step(state, t, touched, cfg)
        events += new_events
        t += dt
    return events


def steady_touches(duration_s, interval_s=STEADY_INTERVAL_S):
    times = []
    t = interval_s
    while t <= duration_s + 1e-9:
        times.append(round(t, 6))
        t += interval_s
    return times


def test_steady_play_has_no_ball_loss():
    assert run(steady_touches(12.0)) == []


def test_brief_gap_then_recover_is_ball_loss():
    # Baseline, ~1.2s with no touches, then two steady touches to confirm recovery.
    before = steady_touches(5.0)
    after = [6.2, 6.7, 7.2, 7.7]
    events = run(before + after)
    losses = [e for e in events if e.kind == "ball-loss"]
    assert len(losses) == 1
    assert 0.4 <= losses[0].value <= 2.5


def test_sustained_slowdown_is_not_ball_loss():
    # Steady, then permanently half speed (long gaps keep coming).
    before = steady_touches(5.0)
    after = [round(5.5 + i * 1.0, 6) for i in range(10)]
    assert run(before + after) == []


def test_speed_up_is_not_ball_loss():
    before = steady_touches(5.0)
    after = [round(5.2 + i * 0.25, 6) for i in range(16)]
    assert run(before + after) == []


def test_gap_before_baseline_is_not_ball_loss():
    early = steady_touches(2.0)
    late = [round(3.5 + i * STEADY_INTERVAL_S, 6) for i in range(10)]
    assert run(early + late) == []


@pytest.mark.parametrize("fps", [30.0, 60.0])
def test_brief_gap_works_at_common_frame_rates(fps):
    before = steady_touches(5.0)
    after = [6.2, 6.7, 7.2, 7.7]
    losses = [e for e in run(before + after, fps=fps) if e.kind == "ball-loss"]
    assert len(losses) == 1
