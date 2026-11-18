"""Synthetic tests for the frequency-disruption ball-loss detector."""
import pytest

from src.ball_loss import initial_state, step
from src.config import Config

# Steady tick-tock at 2 touches/s.
STEADY_INTERVAL_S = 0.5
# Pinned here so retuning config.py defaults doesn't silently change what these tests check.
CFG = Config(
    ball_loss_baseline_s=4.0,
    ball_loss_drop_frac=0.3,
    ball_loss_dip_min_s=0.4,
    ball_loss_dip_max_s=2.5,
    ball_loss_min_baseline_intervals=4,
)


def run(touch_schedule, fps=30.0, cfg=CFG, delays_s=(0.0,)):
    """Feed touch events as metrics would emit them.

    Touch times snap to the frame grid. Each touch is reported delays_s[i % len]
    seconds after it happened (the touch detector's latency, up to touch_max_gap_s).
    """
    state = initial_state()
    events = []
    for i, t in enumerate(sorted(touch_schedule)):
        touch_t_s = round(t * fps) / fps
        reported_t_s = touch_t_s + delays_s[i % len(delays_s)]
        state, new_events = step(state, touch_t_s, reported_t_s, cfg)
        events += new_events
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


def test_uneven_touch_report_delay_is_not_ball_loss():
    # Steady play, but the detector reports touches alternately on time and 0.25 s late
    # (foot hides the ball). Report-time gaps go 0.75, 0.25, 0.75, ... which looks like
    # late-then-recover; gaps must be measured on touch times instead.
    assert run(steady_touches(12.0), delays_s=(0.0, 0.25)) == []


def test_ball_loss_times_gap_start_and_confirming_report():
    before = steady_touches(5.0)  # last steady touch at 5.0
    after = [6.2, 6.7, 7.2, 7.7]  # 6.7 confirms recovery
    losses = [e for e in run(before + after, delays_s=(0.1,)) if e.kind == "ball-loss"]
    assert len(losses) == 1
    assert losses[0].t_s == pytest.approx(5.0)
    assert losses[0].value == pytest.approx(1.2)
    assert losses[0].reported_t_s == pytest.approx(6.7 + 0.1)
