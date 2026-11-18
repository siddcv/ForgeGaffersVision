import numpy as np
import pytest

from src.face_pose import pitch_from_transformation_matrix


def rotation_x(degrees):
    angle = np.radians(degrees)
    c, s = np.cos(angle), np.sin(angle)
    matrix = np.eye(4)
    matrix[:3, :3] = [[1, 0, 0], [0, c, -s], [0, s, c]]
    return matrix


def test_level_head_has_zero_pitch():
    assert pitch_from_transformation_matrix(np.eye(4)) == pytest.approx(0.0)


def test_downward_pitch_is_negative():
    assert pitch_from_transformation_matrix(rotation_x(25)) == pytest.approx(-25.0)


def test_upward_pitch_is_positive():
    assert pitch_from_transformation_matrix(rotation_x(-15)) == pytest.approx(15.0)


def test_invalid_matrix_returns_none():
    assert pitch_from_transformation_matrix(None) is None
    assert pitch_from_transformation_matrix(np.full((4, 4), np.nan)) is None
    assert pitch_from_transformation_matrix(np.eye(3)) is None


def test_non_rotation_scale_is_removed():
    matrix = rotation_x(20)
    matrix[:3, :3] *= 1.2
    assert pitch_from_transformation_matrix(matrix) == pytest.approx(-20.0)
