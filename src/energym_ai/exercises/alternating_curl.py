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
    active_arm: str  # "left" | "right"
    form_issues: list[str] = field(default_factory=list)


class AlternatingCurlAnalyzer:
    """
    Alternating dumbbell curl analyzer.

    Each arm is tracked independently. A rep is counted only when arms
    strictly alternate — if the same side is curled consecutively, the
    second attempt is marked as bad form (same_side_consecutive).

    Form checks per rep:
    - body_sway             : trunk leaning / swinging
    - elbow_drift           : upper arm flaring laterally
    - too_fast              : eccentric phase too quick
    - same_side_consecutive : same arm curled twice in a row
    """

    def __init__(self, exercise_config: dict) -> None:
        self.cfg   = exercise_config
        self.lm    = exercise_config["landmarks"]
        self.thr   = exercise_config["angle_thresholds"]
        self.rules = exercise_config["form_rules"]

        self.state_left:  CurlState = CurlState.UNKNOWN
        self.state_right: CurlState = CurlState.UNKNOWN
        self.last_change_left:  float = time()
        self.last_change_right: float = time()

        self.rep_count_left:  int = 0
        self.rep_count_right: int = 0
        self.rep_count:       int = 0
        self.bad_rep_count:   int = 0

        # Track which arm was last successfully counted to enforce alternation
        self.last_counted_arm: str | None = None
        # Track time of last valid rep to prevent bilateral simultaneous counting
        self.last_rep_time: float = 0.0

        self.reference_r_shoulder: tuple | None = None
        self.reference_r_hip:      tuple | None = None
        self.session_start: float = time()

    # ── Internal arm checker ──────────────────────────────────────────────────

    def _check_arm(
        self,
        shoulder: tuple,
        elbow: tuple,
        wrist: tuple,
        hip: tuple,
        side: str,
        now: float,
    ) -> tuple[float, CurlState, bool, list[str]]:
        angle = calculate_angle(shoulder, elbow, wrist)
        issues: list[str] = []

        # Body sway (using right side as reference baseline)
        sway = body_sway(
            shoulder, hip,
            self.reference_r_shoulder, self.reference_r_hip,
        )
        if sway > self.rules["max_body_sway"]:
            issues.append(f"body_sway_{sway:.1f}deg")

        # Upper arm alignment
        drift = abs(elbow[0] - shoulder[0]) * 100
        if drift > self.rules["max_elbow_drift"]:
            issues.append(f"elbow_drift_{drift:.1f}%")

        # State machine per side
        if side == "left":
            prev = self.state_left
            if angle > self.thr["down_position"]:
                self.state_left = CurlState.DOWN
            elif angle < self.thr["up_position"]:
                self.state_left = CurlState.UP
            cur         = self.state_left
            last_change = self.last_change_left
        else:
            prev = self.state_right
            if angle > self.thr["down_position"]:
                self.state_right = CurlState.DOWN
            elif angle < self.thr["up_position"]:
                self.state_right = CurlState.UP
            cur         = self.state_right
            last_change = self.last_change_right

        counted = False

        if prev == CurlState.UP and cur == CurlState.DOWN:
            phase_dur = now - last_change
            min_gap = float(self.rules.get("min_arm_gap_sec", 1.0))

            # Check alternation and bilateral violations BEFORE tempo so the
            # more-specific form issue is reported (not a generic too_fast).
            if self.last_counted_arm == side:
                issues.append("same_side_consecutive")
                self.bad_rep_count += 1
                self._trigger_alert("same_side_consecutive")
            elif (now - self.last_rep_time) < min_gap:
                issues.append("bilateral_move_too_close")
                self.bad_rep_count += 1
                self._trigger_alert("bilateral_move_too_close")
            else:
                if phase_dur < self.cfg["tempo"]["min_eccentric"]:
                    issues.append(f"too_fast_{phase_dur:.2f}s")

                if issues:
                    self.bad_rep_count += 1
                    self._trigger_alert(issues[0])
                else:
                    if side == "left":
                        self.rep_count_left += 1
                    else:
                        self.rep_count_right += 1
                    self.rep_count        += 1
                    self.last_counted_arm  = side
                    self.last_rep_time     = now
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

    # ── Public interface ──────────────────────────────────────────────────────

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

        r_shoulder = pose.get(self.lm["right_shoulder"]).as_xy()
        r_elbow    = pose.get(self.lm["right_elbow"]).as_xy()
        r_wrist    = pose.get(self.lm["right_wrist"]).as_xy()
        r_hip      = pose.get(self.lm["right_hip"]).as_xy()
        l_shoulder = pose.get(self.lm["left_shoulder"]).as_xy()
        l_elbow    = pose.get(self.lm["left_elbow"]).as_xy()
        l_wrist    = pose.get(self.lm["left_wrist"]).as_xy()
        l_hip      = pose.get(self.lm["left_hip"]).as_xy()

        if self.reference_r_shoulder is None:
            self.reference_r_shoulder = r_shoulder
            self.reference_r_hip      = r_hip

        r_angle, r_state, _, r_issues = self._check_arm(
            r_shoulder, r_elbow, r_wrist, r_hip, "right", now
        )
        l_angle, l_state, _, l_issues = self._check_arm(
            l_shoulder, l_elbow, l_wrist, l_hip, "left", now
        )

        # Active arm = whichever has the smaller elbow angle (more curled)
        if l_angle < r_angle:
            active, angle, state, issues = "left",  l_angle, l_state, l_issues
        else:
            active, angle, state, issues = "right", r_angle, r_state, r_issues

        return FrameAnalysis(
            timestamp=now,
            elbow_angle=angle,
            state=state,
            rep_count=self.rep_count,
            is_bad_form=len(issues) > 0,
            active_arm=active,
            form_issues=issues,
        )

    def _trigger_alert(self, issue_code: str) -> None:
        if hasattr(self, "alerter"):
            self.alerter.send_alert(level="warn", code=issue_code)

    def trigger_iot_alert(self, issue_code: str) -> None:
        self._trigger_alert(issue_code)

    def session_summary(self) -> dict:
        duration = time() - self.session_start
        return {
            "exercise":          self.cfg["display_name"],
            "duration_seconds":  round(duration, 1),
            "valid_reps":        self.rep_count,
            "valid_reps_left":   self.rep_count_left,
            "valid_reps_right":  self.rep_count_right,
            "bad_reps":          self.bad_rep_count,
            "accuracy": round(
                self.rep_count / max(1, self.rep_count + self.bad_rep_count), 3
            ),
        }
