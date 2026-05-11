from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from time import time

from ..core.pose_detector import PoseResult
from ..utils.geometry import calculate_angle, body_sway


class CurlState(str, Enum):
    DOWN = "down"
    UP = "up"
    UNKNOWN = "unknown"


@dataclass
class FrameAnalysis:
    timestamp: float
    elbow_angle: float | None
    state: CurlState
    rep_count: int
    is_bad_form: bool
    form_issues: list[str] = field(default_factory=list)


class HammerCurlAnalyzer:
    def __init__(self, exercise_config: dict) -> None:
        self.cfg = exercise_config
        self.lm = exercise_config["landmarks"]
        self.thr = exercise_config["angle_thresholds"]
        self.rules = exercise_config["form_rules"]

        self.state: CurlState = CurlState.UNKNOWN
        self.rep_count: int = 0
        self.last_state_change: float = time()

        self.reference_shoulder: tuple[float, float] | None = None
        self.reference_hip: tuple[float, float] | None = None
        self.session_start: float = time()
        self.bad_rep_count: int = 0

    def analyze(self, pose: PoseResult) -> FrameAnalysis:
        now = time()

        if not pose.detected:
            return FrameAnalysis(
                timestamp=now,
                elbow_angle=None,
                state=self.state,
                rep_count=self.rep_count,
                is_bad_form=False,
                form_issues=["pose_not_detected"],
            )

        shoulder = pose.get(self.lm["shoulder"]).as_xy()
        elbow = pose.get(self.lm["elbow"]).as_xy()
        wrist = pose.get(self.lm["wrist"]).as_xy()
        hip = pose.get(self.lm["hip"]).as_xy()

        elbow_angle = calculate_angle(shoulder, elbow, wrist)

        if self.reference_shoulder is None:
            self.reference_shoulder = shoulder
            self.reference_hip = hip

        issues: list[str] = []

        # --- Form checks ---
        sway = body_sway(shoulder, hip, self.reference_shoulder, self.reference_hip)
        if sway > self.rules["max_body_sway"]:
            issues.append(f"body_sway_{sway:.1f}deg")

        elbow_drift_pct = abs(elbow[0] - shoulder[0]) * 100
        if elbow_drift_pct > self.rules["max_elbow_drift"]:
            issues.append(f"elbow_drift_{elbow_drift_pct:.1f}%")

        # Hammer curl grip check — wrist harus sejajar (netral grip).
        # Jika wrist terlalu ke luar (supinasi), horizontal distance
        # antara wrist dan elbow akan besar → peringatan.
        max_wrist_deviation = self.rules.get("max_wrist_deviation", 30)
        wrist_deviation_pct = abs(wrist[0] - elbow[0]) * 100
        if wrist_deviation_pct > max_wrist_deviation:
            issues.append(f"grip_rotation_{wrist_deviation_pct:.1f}%")

        # --- State machine ---
        prev_state = self.state
        if elbow_angle > self.thr["down_position"]:
            self.state = CurlState.DOWN
        elif elbow_angle < self.thr["up_position"]:
            self.state = CurlState.UP

        if prev_state == CurlState.UP and self.state == CurlState.DOWN:
            phase_duration = now - self.last_state_change
            if phase_duration < self.cfg["tempo"]["min_eccentric"]:
                issues.append(f"too_fast_{phase_duration:.2f}s")

            if issues:
                self.bad_rep_count += 1
                self.trigger_iot_alert(issues[0])
            else:
                self.rep_count += 1

            self.last_state_change = now
        elif prev_state != self.state:
            self.last_state_change = now

        return FrameAnalysis(
            timestamp=now,
            elbow_angle=elbow_angle,
            state=self.state,
            rep_count=self.rep_count,
            is_bad_form=len(issues) > 0,
            form_issues=issues,
        )
    
    def trigger_iot_alert(self, issue_code):
        if hasattr(self, 'alerter'):
            self.alerter.send_alert(level="warn", code=issue_code)

    def session_summary(self) -> dict:
        duration = time() - self.session_start
        return {
            "exercise": self.cfg["display_name"],
            "duration_seconds": round(duration, 1),
            "valid_reps": self.rep_count,
            "bad_reps": self.bad_rep_count,
            "accuracy": round(
                self.rep_count / max(1, self.rep_count + self.bad_rep_count), 3
            ),
        }
