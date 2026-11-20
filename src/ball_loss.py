"""Frequency-disruption ball-loss detector.

Standalone and causal: call step(state, touch_t_s, reported_t_s, cfg) once per touch.
Standard library only. cfg is any object with the ball_loss_* fields of
config.Config, passed in so this module imports nothing else.

A ball loss is a single unusually long gap between touches, of any length, that
is then followed by a return to the baseline pace. When late gaps come in a row,
the newest one is judged; ball_loss_pace_change_gaps of them in a row is a
sustained slowdown (pace change), not a loss. Faster play is never a loss.

The baseline is the median of recent intervals, so one long gap cannot move it,
a sustained slowdown raises it, and faster play brings it back down. Late gaps
join that history only once their run is resolved (so the baseline never shifts
while a loss is pending), and a gap confirmed as a loss never joins it.
"""
from dataclasses import dataclass, replace
from statistics import median


@dataclass(frozen=True)
class BallLossEvent:
    kind: str  # "ball-loss"
    t_s: float  # when the long gap started (previous touch time)
    reported_t_s: float  # when recovery confirmed the loss
    value: float | None = None  # long-gap duration in seconds


@dataclass(frozen=True)
class BallLossState:
    last_touch_t_s: float | None = None
    start_t_s: float | None = None
    intervals: tuple = ()
    baseline_interval_s: float | None = None
    # After a late gap, wait for the next interval to confirm recovery.
    pending_loss_t_s: float | None = None
    pending_gap_s: float | None = None
    late_gaps: tuple = ()  # current run of consecutive late gaps; the last one is pending


def initial_state() -> BallLossState:
    return BallLossState()


def interrupt(state: BallLossState) -> BallLossState:
    """Forget the last touch and any pending loss; keep the learned baseline.

    Called when another detector has already reported this incident, so the gap
    spanning it is never judged and the incident is not reported twice.
    """
    return replace(
        state, last_touch_t_s=None, pending_loss_t_s=None, pending_gap_s=None, late_gaps=()
    )


def _late_threshold_s(baseline_interval_s: float, cfg) -> float:
    return max(cfg.ball_loss_dip_min_s, baseline_interval_s / (1.0 - cfg.ball_loss_drop_frac))


def step(
    state: BallLossState,
    touch_t_s: float,
    reported_t_s: float,
    cfg,
) -> tuple:
    """Advance one touch. Returns (new_state, events).

    touch_t_s is when the touch happened (a metrics touch Event's t_s) and drives
    all gap timing. reported_t_s is when the touch was detected (its reported_t_s);
    it only stamps when a loss is confirmed, so detection latency stays measurable.
    """
    events = []
    start_t_s = state.start_t_s if state.start_t_s is not None else touch_t_s
    last_touch_t_s = state.last_touch_t_s
    intervals = state.intervals
    baseline_interval_s = state.baseline_interval_s
    pending_loss_t_s = state.pending_loss_t_s
    pending_gap_s = state.pending_gap_s
    late_gaps = state.late_gaps

    if last_touch_t_s is not None:
        gap = touch_t_s - last_touch_t_s
        if gap > 0:
            if baseline_interval_s is None:
                intervals = (intervals + (gap,))[-cfg.ball_loss_history_intervals :]
                # Lock baseline after enough intervals in the warmup window.
                if (
                    touch_t_s - start_t_s >= cfg.ball_loss_baseline_s
                    and len(intervals) >= cfg.ball_loss_min_baseline_intervals
                ):
                    baseline_interval_s = median(intervals)
            else:
                late_s = _late_threshold_s(baseline_interval_s, cfg)
                if gap > late_s:
                    late_gaps = late_gaps + (gap,)
                    if len(late_gaps) >= cfg.ball_loss_pace_change_gaps:
                        # Sustained slowdown: a pace change, not a loss.
                        intervals = intervals + late_gaps
                        late_gaps = ()
                        pending_loss_t_s = None
                        pending_gap_s = None
                    else:
                        # Judge the newest late gap; held out of the history until resolved.
                        pending_loss_t_s = last_touch_t_s
                        pending_gap_s = gap
                else:
                    if pending_loss_t_s is not None:
                        # Back near baseline pace: the pending late gap was a loss.
                        events.append(
                            BallLossEvent(
                                "ball-loss",
                                pending_loss_t_s,
                                reported_t_s,
                                value=pending_gap_s,
                            )
                        )
                        # Earlier late gaps of the run were not losses; they count as play.
                        intervals = intervals + late_gaps[:-1]
                    intervals = intervals + (gap,)
                    late_gaps = ()
                    pending_loss_t_s = None
                    pending_gap_s = None
                intervals = intervals[-cfg.ball_loss_history_intervals :]
                baseline_interval_s = median(intervals)

    new_state = replace(
        state,
        last_touch_t_s=touch_t_s,
        start_t_s=start_t_s,
        intervals=intervals,
        baseline_interval_s=baseline_interval_s,
        pending_loss_t_s=pending_loss_t_s,
        pending_gap_s=pending_gap_s,
        late_gaps=late_gaps,
    )
    return new_state, events
