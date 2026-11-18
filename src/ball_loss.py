"""Frequency-disruption ball-loss detector.

Standalone and causal: call step(state, t_s, touched, cfg) once per frame.
Standard library only. cfg is any object with the ball_loss_* fields of
config.Config, passed in so this module imports nothing else.

A ball loss is a single unusually long gap between touches that is then
followed by a return to the baseline pace. A sustained slowdown (another
long gap right after) is a pace change, not a loss. Faster play is never
a loss.
"""
from dataclasses import dataclass, replace


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


def initial_state() -> BallLossState:
    return BallLossState()


def _mean(values: tuple) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _late_threshold_s(baseline_interval_s: float, cfg) -> float:
    return max(cfg.ball_loss_dip_min_s, baseline_interval_s / (1.0 - cfg.ball_loss_drop_frac))


def step(
    state: BallLossState,
    t_s: float,
    touched: bool,
    cfg,
) -> tuple:
    """Advance one frame. Returns (new_state, events)."""
    if not touched:
        return state, []

    events = []
    start_t_s = state.start_t_s if state.start_t_s is not None else t_s
    last_touch_t_s = state.last_touch_t_s
    intervals = state.intervals
    baseline_interval_s = state.baseline_interval_s
    pending_loss_t_s = state.pending_loss_t_s
    pending_gap_s = state.pending_gap_s

    if last_touch_t_s is not None:
        gap = t_s - last_touch_t_s
        if gap > 0:
            intervals = (intervals + (gap,))[-cfg.ball_loss_history_intervals :]

            # Lock baseline after enough steady intervals in the warmup window.
            if (
                baseline_interval_s is None
                and t_s - start_t_s >= cfg.ball_loss_baseline_s
                and len(intervals) >= cfg.ball_loss_min_baseline_intervals
            ):
                baseline_interval_s = _mean(intervals)

            if baseline_interval_s is not None:
                late_s = _late_threshold_s(baseline_interval_s, cfg)
                if pending_loss_t_s is not None:
                    # Confirm: next interval is back near baseline → loss.
                    # Another late interval → sustained slowdown / pace change.
                    if gap <= late_s:
                        events.append(
                            BallLossEvent(
                                "ball-loss",
                                pending_loss_t_s,
                                t_s,
                                value=pending_gap_s,
                            )
                        )
                    else:
                        baseline_interval_s = gap
                    pending_loss_t_s = None
                    pending_gap_s = None
                elif late_s <= gap <= cfg.ball_loss_dip_max_s:
                    pending_loss_t_s = last_touch_t_s
                    pending_gap_s = gap
                elif gap > cfg.ball_loss_dip_max_s:
                    baseline_interval_s = gap

    new_state = replace(
        state,
        last_touch_t_s=t_s,
        start_t_s=start_t_s,
        intervals=intervals,
        baseline_interval_s=baseline_interval_s,
        pending_loss_t_s=pending_loss_t_s,
        pending_gap_s=pending_gap_s,
    )
    return new_state, events
