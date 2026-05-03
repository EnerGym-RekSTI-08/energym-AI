import pytest
from unittest.mock import MagicMock
from energym_ai.exercises.bicep_curl import BicepCurlAnalyzer, CurlState
from energym_ai.core.pose_detector import PoseResult, Landmark


@pytest.fixture
def config():
    return {
        "display_name": "Bicep Curl Test",
        "primary_joint": "elbow",
        "landmarks": {"shoulder": 0, "elbow": 1, "wrist": 2, "hip": 3},
        "angle_thresholds": {"down_position": 160, "up_position": 50},
        "form_rules": {"max_body_sway": 15, "max_elbow_drift": 25, "min_rom_angle": 90},
        "tempo": {"min_concentric": 0.0, "min_eccentric": 0.0},  # disable tempo cek
    }


def make_pose(shoulder, elbow, wrist, hip) -> PoseResult:
    landmarks = [
        Landmark(*shoulder, 0, 1.0),  
        Landmark(*elbow, 0, 1.0),     
        Landmark(*wrist, 0, 1.0),     
        Landmark(*hip, 0, 1.0),       
    ]
    return PoseResult(landmarks=landmarks)


def test_no_pose_detected(config):
    analyzer = BicepCurlAnalyzer(config)
    result = analyzer.analyze(PoseResult(landmarks=None))
    assert result.elbow_angle is None
    assert "pose_not_detected" in result.form_issues


def test_full_rep_counted(config):
    analyzer = BicepCurlAnalyzer(config)

    # Posisi DOWN: lengan lurus (sudut elbow ~180°)
    down_pose = make_pose(
        shoulder=(0.5, 0.3), elbow=(0.5, 0.5), wrist=(0.5, 0.7), hip=(0.5, 0.8),
    )
    analyzer.analyze(down_pose)
    assert analyzer.state == CurlState.DOWN
    assert analyzer.rep_count == 0

    # Posisi UP: lengan tertekuk (sudut ~30°)
    up_pose = make_pose(
        shoulder=(0.5, 0.3), elbow=(0.5, 0.5), wrist=(0.52, 0.32), hip=(0.5, 0.8),
    )
    analyzer.analyze(up_pose)
    assert analyzer.state == CurlState.UP
    assert analyzer.rep_count == 0  # belum balik ke DOWN

    # Kembali ke DOWN → +1 rep
    result = analyzer.analyze(down_pose)
    assert analyzer.state == CurlState.DOWN
    assert analyzer.rep_count == 1
    assert result.rep_count == 1


def test_partial_rep_not_counted(config):
    analyzer = BicepCurlAnalyzer(config)

    down = make_pose((0.5, 0.3), (0.5, 0.5), (0.5, 0.7), (0.5, 0.8))
    # Sudut sekitar 90° (di antara up & down threshold)
    middle = make_pose((0.5, 0.3), (0.5, 0.5), (0.7, 0.5), (0.5, 0.8))

    analyzer.analyze(down)
    analyzer.analyze(middle)
    analyzer.analyze(down)

    assert analyzer.rep_count == 0


def test_session_summary_format(config):
    analyzer = BicepCurlAnalyzer(config)
    summary = analyzer.session_summary()
    assert "exercise" in summary
    assert "duration_seconds" in summary
    assert "valid_reps" in summary
    assert "bad_reps" in summary
    assert "accuracy" in summary
