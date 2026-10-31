from dataclasses import replace

import pytest

from src.config import Config
from src.metrics import Observation, initial_state, step
from tests.synth import DIAMETER_PX, LEFT_FOOT, RIGHT_FOOT, Y, synth_tick_tock

CFG = Config()


def run(observations, cfg=CFG):
    state = initial_state()
    events = []
    for observation in observations:
        state, new_events = step(state, observation, cfg)
        events += new_events
    return events


def kinds(events, kind):
    return [event for event in events if event.kind == kind]


@pytest.mark.parametrize("fps", [30.0, 60.0])
def test_counts_alternating_touches(fps):
    touches = kinds(run(synth_tick_tock(8, fps)), "touch")
    assert [t.foot for t in touches] == ["left", "right"] * 4


def test_repeat_touch_on_one_foot_is_not_forced_to_alternate():
    observations = synth_tick_tock(1, 30.0)
    first_touch_s = kinds(run(observations), "touch")[0].t_s
    # Ball comes back to the left foot and is sent away again, with no right touch.
    t = observations[-1].t_s
    x = observations[-1].ball_xy[0]
    extra = []
    for direction in [-1] * 6 + [1] * 6:
        t += 1 / 30
        x += direction * 9
        extra.append(replace(observations[-1], t_s=t, ball_xy=(x, Y)))
    touches = kinds(run(observations + extra), "touch")
    assert touches[0].t_s == first_touch_s
    assert [touch.foot for touch in touches] == ["left", "left"]


def test_reversal_far_from_feet_is_not_a_touch():
    observations = [
        replace(o, left_foot=((-5000.0, Y),), right_foot=((5000.0, Y),))
        for o in synth_tick_tock(4)
    ]
    assert kinds(run(observations), "touch") == []


def test_still_ball_has_no_touches():
    observations = [
        Observation(
            t_s=i / 30, ball_xy=(300.0 + (i % 2) * 0.2, Y), ball_diameter_px=DIAMETER_PX,
            left_foot=LEFT_FOOT, right_foot=RIGHT_FOOT,
        )
        for i in range(60)
    ]
    assert kinds(run(observations), "touch") == []


def test_touch_survives_a_brief_occlusion_at_the_foot():
    observations = synth_tick_tock(4)
    first = kinds(run(observations), "touch")[0]
    index = next(i for i, o in enumerate(observations) if o.t_s >= first.t_s)
    hidden = [
        replace(o, ball_xy=None, ball_diameter_px=None) if abs(i - index) <= 1 else o
        for i, o in enumerate(observations)
    ]
    assert len(kinds(run(hidden), "touch")) == 4


@pytest.mark.parametrize("fps", [30.0, 60.0])
def test_ball_speed_matches_known_speed(fps):
    speeds = kinds(run(synth_tick_tock(8, fps, speed_mps=1.5)), "speed")
    assert speeds
    for speed in speeds:
        assert speed.value == pytest.approx(1.5, rel=0.1)


def test_ball_size_changes_reported_speed():
    size5 = kinds(run(synth_tick_tock(6)), "speed")[0].value
    size3 = kinds(run(synth_tick_tock(6), Config(ball_size=3)), "speed")[0].value
    assert size3 == pytest.approx(size5 * 0.190 / 0.220)


def missing(n_frames, searched, start=0.0):
    return [
        Observation(
            t_s=start + i / 30, searched_full_frame=searched,
            left_foot=LEFT_FOOT, right_foot=RIGHT_FOOT,
        )
        for i in range(n_frames)
    ]


def test_crop_miss_without_search_is_never_lost_control():
    assert kinds(run(missing(120, searched=False)), "lost_control") == []


def test_full_frame_miss_under_limit_is_not_lost_control():
    assert kinds(run(missing(10, searched=True)), "lost_control") == []


def test_full_frame_miss_at_limit_is_lost_control_once():
    assert len(kinds(run(missing(60, searched=True)), "lost_control")) == 1


def ball_at(x, n_frames, start=0.0):
    return [
        Observation(
            t_s=start + i / 30, ball_xy=(x, Y), ball_diameter_px=DIAMETER_PX,
            left_foot=LEFT_FOOT, right_foot=RIGHT_FOOT,
        )
        for i in range(n_frames)
    ]


def test_far_ball_is_lost_control_then_reset_when_it_returns():
    events = run(ball_at(1000.0, 30) + ball_at(240.0, 3, start=1.0))
    assert len(kinds(events, "lost_control")) == 1
    assert len(kinds(events, "reset")) == 1


def test_normal_play_has_no_lost_control():
    assert kinds(run(synth_tick_tock(20)), "lost_control") == []


def head(pitches, fps=30):
    return [Observation(t_s=i / fps, head_pitch_deg=p) for i, p in enumerate(pitches)]


def test_head_down_just_under_one_second_is_not_reported():
    assert kinds(run(head([-30.0] * 30)), "head_down") == []


def test_head_down_over_one_second_is_reported_with_start_time():
    events = kinds(run(head([0.0] * 30 + [-30.0] * 40)), "head_down")
    assert len(events) == 1
    assert events[0].t_s == pytest.approx(1.0)
    assert events[0].reported_t_s - events[0].t_s == pytest.approx(1.0)


def test_head_hysteresis_does_not_flicker():
    pitches = [-30.0] * 40 + [-17.0, -30.0] * 10 + [0.0]
    events = run(head(pitches))
    assert len(kinds(events, "head_down")) == 1
    assert len(kinds(events, "head_up")) == 1


def test_step_does_not_mutate_previous_state():
    state = initial_state()
    new_state, _ = step(state, synth_tick_tock(1)[0], CFG)
    assert state == initial_state()
    assert new_state != state
