import numpy as np

from src.metrics import Event
from src.overlay import update_head_down
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
