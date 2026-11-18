from dataclasses import replace

import pytest

from src.config import Config
from src.metrics import Observation, foot_by_direction, initial_state, step
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
def test_counts_alternating_contacts(fps):
    contacts = kinds(run(synth_tick_tock(8, fps)), "contact")
    assert [c.foot for c in contacts] == ["left", "right"] * 4


@pytest.mark.parametrize("fps", [30.0, 60.0])
def test_alternating_contacts_are_tick_tocks_except_the_last(fps):
    events = run(synth_tick_tock(8, fps))
    contacts, touches = kinds(events, "contact"), kinds(events, "touch")
    # The last contact has no answer from the other foot yet, so it is not a tick-tock.
    assert [t.foot for t in touches] == (["left", "right"] * 4)[:7]
    for touch, contact, answer in zip(touches, contacts, contacts[1:]):
        assert touch.t_s == contact.t_s  # stamped at the first foot's contact
        assert touch.reported_t_s == answer.reported_t_s  # reported when the other foot answers


def test_repeat_contact_on_one_foot_is_not_forced_to_alternate():
    observations = synth_tick_tock(1, 30.0)
    first_touch_s = kinds(run(observations), "contact")[0].t_s
    # Ball comes back to the left foot and is sent away again, with no right touch.
    t = observations[-1].t_s
    x = observations[-1].ball_xy[0]
    extra = []
    for direction in [-1] * 6 + [1] * 6:
        t += 1 / 30
        x += direction * 9
        extra.append(replace(observations[-1], t_s=t, ball_xy=(x, Y)))
    events = run(observations + extra)
    contacts = kinds(events, "contact")
    assert contacts[0].t_s == first_touch_s
    assert [c.foot for c in contacts] == ["left", "left"]
    assert kinds(events, "touch") == []  # a same-foot repeat is never a tick-tock


def test_foot_by_direction_picks_the_foot_ahead_of_the_ball():
    feet_x = (200.0, 400.0)  # player's left foot on the image left
    assert foot_by_direction(-1.0, feet_x) == "left"
    assert foot_by_direction(+1.0, feet_x) == "right"
    # Mirrored video or rear camera: the left foot is on the image right; the rule follows the feet.
    assert foot_by_direction(+1.0, (400.0, 200.0)) == "left"
    assert foot_by_direction(None, feet_x) is None
    assert foot_by_direction(1.0, (300.0, 300.0)) is None
    assert foot_by_direction(1.0, (None, 300.0)) is None


@pytest.mark.parametrize("rule", ["direction", "nearest"])
def test_ball_hidden_at_the_touch_keeps_the_right_foot(rule):
    # As on the first real clip: the foot hides the ball at contact, and it is next detected
    # about 130 ms later, already closer to the other foot.
    observations = synth_tick_tock(4, 60.0)
    first = kinds(run(observations), "contact")[0]
    assert first.foot == "left"
    index = next(i for i, o in enumerate(observations) if o.t_s > first.t_s)
    hidden = [replace(o, ball_xy=None, ball_diameter_px=None) for o in observations[index:index + 8]]
    shifted = [
        replace(o, ball_xy=(o.ball_xy[0] + 110, Y)) for o in observations[index + 8:]
    ]  # reappears nearer the right foot than the ball last was to the left foot
    contacts = kinds(run(observations[:index] + hidden + shifted, replace(CFG, touch_foot_rule=rule)), "contact")
    assert contacts[0].foot == "left"


def test_reversal_far_from_feet_is_not_a_touch():
    observations = [
        replace(o, left_foot=((-5000.0, Y),), right_foot=((5000.0, Y),))
        for o in synth_tick_tock(4)
    ]
    events = run(observations)
    assert kinds(events, "contact") == [] and kinds(events, "touch") == []


def test_still_ball_has_no_touches():
    observations = [
        Observation(
            t_s=i / 30, ball_xy=(300.0 + (i % 2) * 0.2, Y), ball_diameter_px=DIAMETER_PX,
            left_foot=LEFT_FOOT, right_foot=RIGHT_FOOT,
        )
        for i in range(60)
    ]
    assert kinds(run(observations), "contact") == []


def test_touch_survives_a_brief_occlusion_at_the_foot():
    observations = synth_tick_tock(4)
    first = kinds(run(observations), "contact")[0]
    index = next(i for i, o in enumerate(observations) if o.t_s >= first.t_s)
    hidden = [
        replace(o, ball_xy=None, ball_diameter_px=None) if abs(i - index) <= 1 else o
        for i, o in enumerate(observations)
    ]
    events = run(hidden)
    assert len(kinds(events, "contact")) == 4
    assert len(kinds(events, "touch")) == 3


# Tick-tocks: the ball must cross the midpoint and the other foot must make the next contact.
# Feet at x 200 (left) and 400 (right), midpoint 300; the ball turns at 240 / 360 in a normal touch.

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


def contacts_and_touches(observations, cfg=CFG):
    events = run(observations, cfg)
    return [c.foot for c in kinds(events, "contact")], [t.foot for t in kinds(events, "touch")]


def test_crossed_midpoint_follows_the_feet():
    from src.metrics import crossed_midpoint
    assert crossed_midpoint(320.0, LEFT_FOOT, RIGHT_FOOT, "left")
    assert not crossed_midpoint(280.0, LEFT_FOOT, RIGHT_FOOT, "left")
    assert crossed_midpoint(280.0, LEFT_FOOT, RIGHT_FOOT, "right")
    # Mirrored: the left foot on the image right; the midpoint test follows the feet.
    assert crossed_midpoint(280.0, RIGHT_FOOT, LEFT_FOOT, "left")
    # The midpoint moves with the feet: shift both feet right by 100, midpoint 400.
    shifted_left = tuple((x + 100, y) for x, y in LEFT_FOOT)
    shifted_right = tuple((x + 100, y) for x, y in RIGHT_FOOT)
    assert not crossed_midpoint(320.0, shifted_left, shifted_right, "left")
    assert not crossed_midpoint(320.0, (), RIGHT_FOOT, "left")  # a foot not visible


def test_clean_run_counts_every_answered_touch():
    contacts, touches = contacts_and_touches(path([300, 240, 360, 240, 360, 300]))
    assert contacts == ["left", "right", "left", "right"]
    assert touches == ["left", "right", "left"]  # the last right has no answer from the left


def test_same_foot_twice_counts_only_the_touch_that_crosses():
    # Left touch, ball drifts back to the left foot without crossing, left again, then a normal run.
    contacts, touches = contacts_and_touches(path([300, 240, 290, 240, 360, 240, 300]))
    assert contacts == ["left", "left", "right", "left"]
    assert touches == ["left", "right"]  # the first left is a contact only


def test_nudge_that_does_not_cross_is_not_a_tick_tock():
    # A setup tap: the left foot nudges the ball part way (to 275, short of the midpoint at 300),
    # it rolls back, and the left foot plays it properly into a normal run.
    events = run(path([300, 240, 275, 235, 360, 240, 300]))
    contacts = kinds(events, "contact")
    assert [c.foot for c in contacts] == ["left", "left", "right", "left"]
    assert contacts[0].t_s not in [t.t_s for t in kinds(events, "touch")]  # the nudge never crossed
    assert [t.foot for t in kinds(events, "touch")] == ["left", "right"]


def test_jitter_on_a_creeping_ball_is_not_a_contact():
    # As at the start of 134129: the ball creeps toward the right foot at ~0.2 m/s and detection
    # jitter of a few pixels flips its direction, next to the foot.
    observations = path([300, 352, 355, 352, 355, 351], speed_px_s=45.0)  # 3 px = 0.075 ball widths
    assert kinds(run(observations), "contact") == []
    # The same turns with real travel (a ball-width each way) are contacts.
    assert len(kinds(run(path([300, 360, 320, 362, 300])), "contact")) == 2


def test_slow_answer_inside_the_arrival_limit_still_confirms_the_tick_tock():
    # As at 8.3-8.8 s on 134129: the right foot answers within the limit, but the ball leaves it so
    # slowly that the answer is only confirmed (half a ball-width later) after the limit has passed.
    cfg = replace(CFG, ticktock_max_arrival_s=0.5)
    fast = path([300, 240, 360])  # left turn at 0.2 s, right turn at 0.6 s: inside the limit
    slow = path([360, 330], speed_px_s=40.0, start_t=fast[-1].t_s)  # 20 px of travel takes 0.5 s
    contacts, touches = contacts_and_touches(fast + slow, cfg)
    assert contacts == ["left", "right"]
    assert touches == ["left"]


def test_contact_is_reported_once_the_ball_has_travelled_after_the_turn():
    events = run(path([300, 240, 360]))
    contact = kinds(events, "contact")[0]
    travel_s = CFG.touch_min_travel_diam * DIAMETER_PX / 300.0
    assert contact.reported_t_s - contact.t_s == pytest.approx(travel_s, abs=2 / 60)


def test_ball_that_crosses_but_never_reaches_the_other_foot_is_not_a_tick_tock():
    # Left touch sends the ball across and away: no contact by the right foot.
    contacts, touches = contacts_and_touches(path([300, 240, 1000]))
    assert contacts == ["left"] and touches == []


def test_answer_after_the_arrival_limit_is_not_a_tick_tock():
    slow = path([300, 240, 330])
    gap_s = CFG.ticktock_max_arrival_s + 0.3
    # Ball sits unseen past the midpoint, then reaches the right foot and is sent back.
    rest = path([330, 360, 240, 300], start_t=slow[-1].t_s + gap_s)
    contacts, touches = contacts_and_touches(slow + rest)
    assert contacts == ["left", "right", "left"]
    assert touches == ["right"]  # the first left timed out; right was answered by the left


def test_lost_control_cancels_the_pending_touch():
    cfg = replace(CFG, ticktock_max_arrival_s=10.0)  # so only lost control can cancel it
    observations = path([300, 240, 330]) + missing(40, searched=True, start=0.6) + path([330, 360, 240, 300], start_t=2.0)
    events = run(observations, cfg)
    assert len(kinds(events, "lost_control")) == 1
    assert [t.foot for t in kinds(events, "touch")] == ["right"]


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


# Touches per minute

def touched_at(times, cfg=CFG):
    from src.metrics import record_touch
    state = initial_state()
    for t in times:
        state = record_touch(state, t, cfg)
    return state


def test_no_touches_gives_no_rate():
    from src.metrics import touch_rate
    rate = touch_rate(initial_state(), 5.0, CFG)
    assert (rate.last_minute, rate.minute_complete, rate.pace_per_min) == (0, False, None)


def test_steady_150_per_minute_meets_the_goal_after_a_full_minute():
    from src.metrics import touch_rate
    times = [i * 0.4 for i in range(200)]  # one touch every 0.4 s = 150 per minute
    state = touched_at([t for t in times if t <= 70.0])
    rate = touch_rate(state, 70.0, CFG)
    assert rate.minute_complete and rate.last_minute == 150
    assert rate.pace_per_min == pytest.approx(150, abs=6)


def test_minute_is_not_complete_before_sixty_seconds():
    from src.metrics import touch_rate
    state = touched_at([i * 0.4 for i in range(100)])  # last touch at 39.6 s
    rate = touch_rate(state, 40.0, CFG)
    assert not rate.minute_complete and rate.last_minute == 100


def test_pace_is_withheld_for_the_first_seconds_then_uses_the_time_so_far():
    from src.metrics import touch_rate
    state = touched_at([0.0, 0.5, 1.0, 1.5, 2.0])
    assert touch_rate(state, 2.0, CFG).pace_per_min is None  # under touch_pace_min_s
    state = touched_at([i * 0.5 for i in range(9)])  # 0 .. 4 s, 120 per minute
    assert touch_rate(state, 4.0, CFG).pace_per_min == pytest.approx(9 / 4 * 60)


def test_pace_falls_when_the_player_stops():
    from src.metrics import touch_rate
    state = touched_at([i * 0.4 for i in range(50)])  # stops at 19.6 s
    assert touch_rate(state, 19.6, CFG).pace_per_min == pytest.approx(150, abs=6)
    assert touch_rate(state, 25.0, CFG).pace_per_min < 80
    assert touch_rate(state, 30.0, CFG).pace_per_min == 0


def test_old_touches_drop_out_of_the_window():
    state = touched_at([i * 0.4 for i in range(400)])  # 160 s of touches
    assert len(state.recent_touch_t_s) == 150  # only the last minute is kept


def test_step_counts_touches_into_the_rate():
    from src.metrics import touch_rate
    observations = synth_tick_tock(40, 60.0)
    state = initial_state()
    touches = 0
    for o in observations:
        state, events = step(state, o, CFG)
        touches += len(kinds(events, "touch"))
    rate = touch_rate(state, observations[-1].t_s, CFG)
    assert rate.last_minute == touches == len(state.recent_touch_t_s)
    assert rate.pace_per_min is not None
