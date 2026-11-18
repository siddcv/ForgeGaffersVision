"""Synthetic tests for the frequency-disruption ball-loss detector."""
import pytest

from src.ball_loss import initial_state, interrupt, step
from src.config import Config

# Steady tick-tock at 2 touches/s.
STEADY_INTERVAL_S = 0.5
# Pinned here so retuning config.py defaults doesn't silently change what these tests check.
CFG = Config(
    ball_loss_baseline_s=4.0,
    ball_loss_drop_frac=0.3,
    ball_loss_dip_min_s=0.4,
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
    assert losses[0].value == pytest.approx(1.2)


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


def feed(state, times):
    events = []
    for t in times:
        state, new_events = step(state, t, t, CFG)
        events += new_events
    return state, events


def test_interrupt_drops_the_gap_across_it_but_keeps_the_baseline():
    state, _ = feed(initial_state(), steady_touches(5.0))
    baseline = state.baseline_interval_s
    state = interrupt(state)  # e.g. the spatial detector saw the ball escape at ~5.3 s
    state, events = feed(state, [6.2, 6.7, 7.2, 7.7])
    assert events == []
    assert state.baseline_interval_s == baseline


def test_interrupt_clears_a_pending_loss():
    state, _ = feed(initial_state(), steady_touches(5.0) + [6.2])  # late gap, awaiting confirmation
    state, events = feed(interrupt(state), [6.7, 7.2])
    assert events == []


# Bug 2 (clip 20261102_135033): long gaps used to switch the detector off, because a gap over
# 2.5 s was never a loss and replaced the baseline, which then never came back down.


def test_long_gap_then_recover_is_ball_loss():
    before = steady_touches(5.0)
    after = [8.5, 9.0, 9.5, 10.0]  # 3.5 s with no touches, then back to pace
    losses = [e for e in run(before + after) if e.kind == "ball-loss"]
    assert len(losses) == 1
    assert losses[0].value == pytest.approx(3.5)


def test_one_long_gap_does_not_raise_the_baseline():
    # Classroom pattern: steady ~0.4 s, a 3.4 s gap, steady again, then a 1.0 s fumble.
    before = steady_touches(5.0, interval_s=0.4)
    middle = [round(8.2 + i * 0.4, 6) for i in range(8)]  # last steady touch 4.8; ends at 11.0
    after = [12.0, 12.4, 12.8, 13.2]
    losses = [e for e in run(before + middle + after) if e.kind == "ball-loss"]
    assert [round(e.value, 1) for e in losses] == [3.4, 1.0]


def test_long_gap_during_warmup_does_not_inflate_the_baseline():
    early = steady_touches(3.0, interval_s=0.4)  # last touch 2.8
    middle = [round(5.5 + i * 0.4, 6) for i in range(10)]  # 2.7 s gap, steady to 9.1
    after = [10.1, 10.5, 10.9, 11.3]  # 1.0 s fumble
    losses = [e for e in run(early + middle + after) if e.kind == "ball-loss"]
    assert any(e.value == pytest.approx(1.0) for e in losses)


def test_baseline_comes_back_down_when_play_speeds_up():
    slow = steady_touches(5.0, interval_s=1.0)  # baseline learned at 1 touch/s
    fast = [round(5.5 + i * 0.5, 6) for i in range(16)]  # sustained 2 touches/s, ends at 13.0
    after = [14.0, 14.5, 15.0, 15.5]  # 1.0 s gap: late at the new pace
    losses = [e for e in run(slow + fast + after) if e.kind == "ball-loss"]
    assert len(losses) == 1
    assert losses[0].value == pytest.approx(1.0)
