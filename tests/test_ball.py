from src.ball import (
    SearchState, box_center, box_diameter_px, foot_crop, search_step, to_full_frame,
)
from src.config import Config

CFG = Config()
CROP = (100, 100, 300, 300)
IN_CROP = (190, 190, 210, 210, 0.9)
FAR = (800, 100, 820, 120, 0.9)


def test_crop_mapping_returns_full_frame_coordinates():
    assert to_full_frame((10, 20, 30, 40, 0.5), (100, 200, 300, 400)) == (110, 220, 130, 240, 0.5)


def test_foot_crop_contains_feet_and_stays_in_frame():
    crop = foot_crop(((50, 700), (120, 710)), 1280, 720, CFG)
    assert crop[0] <= 50 and crop[2] >= 120 and crop[1] <= 700 and crop[3] == 720
    assert crop[0] >= 0


def test_foot_crop_needs_feet():
    assert foot_crop((), 1280, 720, CFG) is None


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
