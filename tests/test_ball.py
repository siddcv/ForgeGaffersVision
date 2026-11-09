import numpy as np
import pytest

from src.ball import (
    BallTracker, SearchState, box_center, box_diameter_px, crop_image, foot_crop, search_step,
    to_full_frame,
)
from src.config import Config

CFG = Config()
CROP = (100, 100, 300, 300)
IN_CROP = (190, 190, 210, 210, 0.9)
FAR = (800, 100, 820, 120, 0.9)


def test_crop_mapping_returns_full_frame_coordinates():
    assert to_full_frame((10, 20, 30, 40, 0.5), (100, 200, 300, 400)) == (110, 220, 130, 240, 0.5)


def test_foot_crop_is_square_and_contains_feet():
    crop = foot_crop(((50, 700), (120, 710)), 1280, 720, CFG)
    assert crop[0] <= 50 and crop[2] >= 120 and crop[1] <= 700 and crop[3] >= 710
    assert crop[2] - crop[0] == crop[3] - crop[1]


def test_foot_crop_is_sized_so_the_ball_fills_the_target_fraction():
    crop = foot_crop(((500, 1700), (700, 1720)), 1080, 1920, CFG, ball_diameter_px=180)
    assert crop[2] - crop[0] == int(180 / CFG.crop_ball_frac)


def test_foot_crop_still_covers_wide_feet_when_the_ball_is_small():
    crop = foot_crop(((100, 500), (900, 500)), 1080, 1920, CFG, ball_diameter_px=20)
    assert crop[0] <= 100 and crop[2] >= 900


def test_foot_crop_needs_feet():
    assert foot_crop((), 1280, 720, CFG) is None


def test_crop_image_pads_outside_the_frame():
    image = np.zeros((100, 100, 3), np.uint8)
    out = crop_image(image, (50, 50, 150, 150), 114)
    assert out.shape == (100, 100, 3)
    assert out[0, 0, 0] == 0 and out[99, 99, 0] == 114


def test_diameter_uses_shorter_side():
    assert box_diameter_px((0, 0, 50, 30, 0.9)) == 30
    assert box_center((0, 0, 50, 30, 0.9)) == (25, 15)


def test_found_in_crop_stays_tracking():
    state, detection, searched = search_step(SearchState(), 0.0, CROP, IN_CROP, IN_CROP, CFG)
    assert detection == IN_CROP and not searched and not state.searching


def test_brief_miss_does_not_search_yet():
    state, _, _ = search_step(SearchState(), 0.0, CROP, IN_CROP, IN_CROP, CFG)
    state, detection, searched = search_step(state, 0.03, CROP, None, IN_CROP, CFG)
    assert detection is None and not searched and not state.searching
    state, detection, searched = search_step(state, 0.06, CROP, IN_CROP, IN_CROP, CFG)
    assert detection == IN_CROP and state == SearchState()


def enter_search():
    state = SearchState()
    state, _, _ = search_step(state, 0.0, CROP, None, FAR, CFG)
    return search_step(state, CFG.ball_search_after_s, CROP, None, FAR, CFG)


def test_long_miss_starts_search_and_finds_far_ball():
    state, detection, searched = enter_search()
    assert state.searching and searched and detection == FAR


def test_search_returning_nothing_is_flagged_as_searched():
    state, _, _ = enter_search()
    state, detection, searched = search_step(state, 1.0, CROP, None, None, CFG)
    assert detection is None and searched and state.searching


def test_ball_back_in_crop_returns_to_tracking():
    state, _, _ = enter_search()
    state, detection, _ = search_step(state, 1.0, CROP, IN_CROP, IN_CROP, CFG)
    assert detection == IN_CROP and state == SearchState()


def test_no_crop_searches_immediately():
    state, detection, searched = search_step(SearchState(), 0.0, None, None, FAR, CFG)
    assert state.searching and searched and detection == FAR


def ball(x, y=500.0, d=40.0, conf=0.5):
    return (x - d / 2, y - d / 2, x + d / 2, y + d / 2, conf)


def tracked(tracker, xs, fps=60):
    """Feed a ball moving through xs, one detection per frame. Returns the next frame time."""
    for i, x in enumerate(xs):
        t = i / fps
        tracker.update(t, tracker.choose(t, [ball(x)]), False)
    return len(xs) / fps


def test_tracker_picks_the_box_that_fits_the_track_not_the_most_confident():
    tracker = BallTracker(CFG)
    t = tracked(tracker, [300 + 5 * i for i in range(20)])
    fixture = ball(150, y=160, conf=0.9)  # a static lookalike far from the ball, as in the first clip
    real = ball(400, conf=0.3)
    assert tracker.choose(t, [fixture, real]) == real


def test_tracker_rejects_a_jump_far_from_the_prediction():
    tracker = BallTracker(CFG)
    t = tracked(tracker, [300 + 5 * i for i in range(20)])
    assert tracker.choose(t, [ball(900)]) is None


def test_tracker_without_a_track_takes_the_candidate_nearest_the_feet():
    tracker = BallTracker(CFG)
    near, far = ball(520, y=1700, conf=0.3), ball(150, y=160, conf=0.9)
    assert tracker.choose(0.0, [far, near], feet=((500, 1720), (560, 1720))) == near


def test_tracker_predicts_through_a_short_gap_along_the_motion():
    tracker = BallTracker(CFG)
    t = tracked(tracker, [300 + 5 * i for i in range(20)])
    xy, predicted = tracker.update(t, None, False)
    assert predicted and xy[0] == pytest.approx(400, abs=3)


def test_tracker_stops_predicting_after_the_limit():
    tracker = BallTracker(CFG)
    t = tracked(tracker, [300 + 5 * i for i in range(20)])
    xy, predicted = tracker.update(t + CFG.track_max_predict_s + 0.02, None, False)
    assert xy is None and not predicted


def test_tracker_never_predicts_once_a_full_frame_search_has_started():
    tracker = BallTracker(CFG)
    t = tracked(tracker, [300 + 5 * i for i in range(20)])
    assert tracker.update(t, None, True) == (None, False)


def test_tracker_reacquires_after_a_long_absence():
    tracker = BallTracker(CFG)
    t = tracked(tracker, [300 + 5 * i for i in range(20)])
    later = t + CFG.track_gate_max_s + 0.1
    assert tracker.choose(later, [ball(900)]) is not None
