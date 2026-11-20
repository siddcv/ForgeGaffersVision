import numpy as np

from src.metrics import Event
from src.overlay import LOST_BANNER_S, latest_lost_control, lost_banner, update_head_down
from src.pose import SKELETON, coarse_head_pitch_deg, head_box


def landmarks_with_face(visibility=1.0):
    landmarks = np.zeros((33, 3))
    landmarks[:11] = [(100 + 10 * i, 200 + 2 * i, visibility) for i in range(11)]
    return landmarks


def test_skeleton_covers_all_33_landmarks():
    assert {index for pair in SKELETON for index in pair} == set(range(33))


def test_head_box_surrounds_the_face_with_padding():
    assert head_box(landmarks_with_face(), 0.5, 0.5) == (50.0, 150.0, 250.0, 270.0)


def test_head_box_needs_visible_face():
    assert head_box(landmarks_with_face(visibility=0.1), 0.5, 0.5) is None
    assert head_box(np.full((33, 3), np.nan), 0.5, 0.5) is None


def test_no_person_gives_no_head_pitch():
    assert coarse_head_pitch_deg(np.full((33, 3), np.nan), 0.5) is None


def test_head_down_label_follows_events():
    down = Event("head_down", 0.0, 1.0)
    up = Event("head_up", 2.0, 2.0)
    assert update_head_down(False, [down]) is True
    assert update_head_down(True, []) is True
    assert update_head_down(True, [up]) is False


def test_latest_lost_control_keeps_the_newest_report():
    first = Event("lost_control", 1.0, 1.5, source="spatial")
    second = Event("lost_control", 3.0, 4.0, source="rhythm")
    assert latest_lost_control(None, [Event("touch", 0.5, 0.6)]) is None
    assert latest_lost_control(None, [first]) == first
    assert latest_lost_control(first, []) == first
    assert latest_lost_control(first, [second]) == second


def test_lost_banner_shows_from_report_for_a_while():
    event = Event("lost_control", 3.0, 4.0, source="rhythm")
    assert lost_banner(None, 4.0) is None
    assert lost_banner(event, 3.9) is None  # not known yet: show it when it was reported
    assert lost_banner(event, 4.0) == "LOST CONTROL (rhythm)"
    assert lost_banner(event, 4.0 + LOST_BANNER_S - 0.01) == "LOST CONTROL (rhythm)"
    assert lost_banner(event, 4.0 + LOST_BANNER_S) is None


def test_touch_rate_text():
    from src.metrics import TouchRate
    from src.overlay import touch_rate_text
    assert touch_rate_text(None, 150) == "goal 150/min"
    assert touch_rate_text(TouchRate(12, False, None), 150) == "pace --  12 so far  goal 150"
    assert touch_rate_text(TouchRate(151, True, 148.4), 150) == "pace 148/min  last min 151  goal 150"
