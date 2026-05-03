import math
import pytest
from energym_ai.utils.geometry import calculate_angle


def test_angle_90_degrees():
    angle = calculate_angle((0, 0), (1, 0), (1, 1))
    assert math.isclose(angle, 90.0, abs_tol=0.5)


def test_angle_180_degrees_lurus():
    angle = calculate_angle((0, 0), (1, 0), (2, 0))
    assert math.isclose(angle, 180.0, abs_tol=0.5)


def test_angle_0_degrees_terlipat_penuh():
    angle = calculate_angle((0, 0), (1, 0), (0, 0.001))
    assert angle < 5.0


def test_angle_bicep_curl_down_position():
    shoulder = (0.5, 0.3)
    elbow = (0.5, 0.5)
    wrist = (0.5, 0.7)  # lurus vertikal
    angle = calculate_angle(shoulder, elbow, wrist)
    assert 175 <= angle <= 180


def test_angle_bicep_curl_up_position():
    shoulder = (0.5, 0.3)
    elbow = (0.5, 0.5)
    wrist = (0.55, 0.32) 
    angle = calculate_angle(shoulder, elbow, wrist)
    assert angle < 50, f"Expected < 50°, got {angle:.1f}°"


def test_angle_3d_coordinates():
    angle = calculate_angle((0, 0, 0), (1, 0, 0), (1, 1, 0))
    assert math.isclose(angle, 90.0, abs_tol=0.5)
