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
    active_arm: str  # "left" or "right"
    form_issues: list[str] = field(default_factory=list)


class AlternatingCurlAnalyzer:
    def __init__(self, exercise_config: dict) -> None:
        self.cfg = exercise_config
        self.lm = exercise_config["landmarks"]
        self.thr = exercise_config["angle_thresholds"]
        self.rules = exercise_config["form_rules"]

        # State per sisi
        self.state_left: CurlState = CurlState.UNKNOWN
        self.state_right: CurlState = CurlState.UNKNOWN
        self.last_change_left: float = time()
        self.last_change_right: float = time()

        self.rep_count_left: int = 0
        self.rep_count_right: int = 0
        self.rep_count: int = 0

        self.reference_shoulder: tuple[float, float] | None = None
        self.reference_hip: tuple[float, float] | None = None
        self.session_start: float = time()
        self.bad_rep_count: int = 0

    # ------------------------------------------------------------------ #
    def _check_arm(
        self,
        shoulder: tuple[float, float],
        elbow: tuple[float, float],
        wrist: tuple[float, float],
        hip: tuple[float, float],
        side: str,
        now: float,
    ) -> tuple[float, CurlState, bool, list[str]]:
        angle = calculate_angle(shoulder, elbow, wrist)
        issues: list[str] = []

        # Form checks
        sway = body_sway(shoulder, hip, self.reference_shoulder, self.reference_hip)
        if sway > self.rules["max_body_sway"]:
            issues.append(f"body_sway_{sway:.1f}deg")

        elbow_drift_pct = abs(elbow[0] - shoulder[0]) * 100
        if elbow_drift_pct > self.rules["max_elbow_drift"]:
            issues.append(f"elbow_drift_{elbow_drift_pct:.1f}%")

        # State machine per sisi
        if side == "left":
            prev = self.state_left
            if angle > self.thr["down_position"]:
                self.state_left = CurlState.DOWN
            elif angle < self.thr["up_position"]:
                self.state_left = CurlState.UP
            cur = self.state_left
            last_change = self.last_change_left
        else:
            prev = self.state_right
            if angle > self.thr["down_position"]:
                self.state_right = CurlState.DOWN
            elif angle < self.thr["up_position"]:
                self.state_right = CurlState.UP
            cur = self.state_right
            last_change = self.last_change_right

        counted = False
        if prev == CurlState.UP and cur == CurlState.DOWN:
            phase_dur = now - last_change
            if phase_dur < self.cfg["tempo"]["min_eccentric"]:
                issues.append(f"too_fast_{phase_dur:.2f}s")

            if issues:
                self.bad_rep_count += 1
                self.trigger_iot_alert(issues[0])
            else:
                if side == "left":
                    self.rep_count_left += 1
                else:
                    self.rep_count_right += 1
                self.rep_count += 1
                counted = True

            if side == "left":
                self.last_change_left = now
            else:
                self.last_change_right = now
        elif prev != cur:
            if side == "left":
                self.last_change_left = now
            else:
                self.last_change_right = now

        return angle, cur, counted, issues

    # ------------------------------------------------------------------ #
    def analyze(self, pose: PoseResult) -> FrameAnalysis:
        now = time()

        if not pose.detected:
            return FrameAnalysis(
                timestamp=now,
                elbow_angle=None,
                state=CurlState.UNKNOWN,
                rep_count=self.rep_count,
                is_bad_form=False,
                active_arm="none",
                form_issues=["pose_not_detected"],
            )

        # Kanan (right-side landmarks)
        r_shoulder = pose.get(self.lm["right_shoulder"]).as_xy()
        r_elbow = pose.get(self.lm["right_elbow"]).as_xy()
        r_wrist = pose.get(self.lm["right_wrist"]).as_xy()
        r_hip = pose.get(self.lm["right_hip"]).as_xy()

        # Kiri (left-side landmarks)
        l_shoulder = pose.get(self.lm["left_shoulder"]).as_xy()
        l_elbow = pose.get(self.lm["left_elbow"]).as_xy()
        l_wrist = pose.get(self.lm["left_wrist"]).as_xy()
        l_hip = pose.get(self.lm["left_hip"]).as_xy()

        if self.reference_shoulder is None:
            self.reference_shoulder = r_shoulder
            self.reference_hip = r_hip

        r_angle, r_state, r_counted, r_issues = self._check_arm(
            r_shoulder, r_elbow, r_wrist, r_hip, "right", now
        )
        l_angle, l_state, l_counted, l_issues = self._check_arm(
            l_shoulder, l_elbow, l_wrist, l_hip, "left", now
        )

        # Tentukan arm aktif (sudut lebih kecil = sedang curl)
        if l_angle < r_angle:
            active = "left"
            angle = l_angle
            state = l_state
            issues = l_issues
        else:
            active = "right"
            angle = r_angle
            state = r_state
            issues = r_issues

        return FrameAnalysis(
            timestamp=now,
            elbow_angle=angle,
            state=state,
            rep_count=self.rep_count,
            is_bad_form=len(issues) > 0,
            active_arm=active,
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
            "valid_reps_left": self.rep_count_left,
            "valid_reps_right": self.rep_count_right,
            "bad_reps": self.bad_rep_count,
            "accuracy": round(
                self.rep_count / max(1, self.rep_count + self.bad_rep_count), 3
            ),
        }
