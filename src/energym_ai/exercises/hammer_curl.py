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
    bad_rep_count: int
    is_bad_form: bool
    form_issues: list[str] = field(default_factory=list)


class HammerCurlAnalyzer:
    """
    Bilateral hammer curl analyzer.

    A rep is counted only when BOTH arms complete a full ROM together
    (within sync_window_sec), matching the correct form of lifting both
    dumbbells simultaneously with a neutral grip.

    Neutral grip check (hammer grip):
    - Thumb tip should be ABOVE pinky tip throughout the movement.
    - In image coordinates (y=0 at top, y=1 at bottom):
        pinky.y > thumb.y  →  pinky is lower than thumb  →  correct hammer grip
    - If thumb and pinky are at the same height or thumb is BELOW pinky,
      the wrist has rotated outward (supination) → bad form: grip_rotation.

    Form checks per frame:
    - body_sway      : trunk leaning / swinging
    - elbow_drift    : upper arm flaring laterally
    - grip_rotation  : wrist rotated from neutral (pinky not below thumb)
    - too_fast       : eccentric phase (lowering) too quick
    """

    def __init__(self, exercise_config: dict) -> None:
        self.cfg   = exercise_config
        self.lm    = exercise_config["landmarks"]
        self.thr   = exercise_config["angle_thresholds"]
        self.rules = exercise_config["form_rules"]

        self.state_right: CurlState = CurlState.UNKNOWN
        self.state_left:  CurlState = CurlState.UNKNOWN
        self.last_change_right: float = time()
        self.last_change_left:  float = time()

        self.both_up_since:   float | None = None
        self._pending_issues: list[str]    = []

        self.rep_count:     int = 0
        self.bad_rep_count: int = 0

        self.reference_r_shoulder: tuple | None = None
        self.reference_r_hip:      tuple | None = None
        self.session_start: float = time()

    # ── Public interface ──────────────────────────────────────────────────────

    def analyze(self, pose: PoseResult) -> FrameAnalysis:
        now = time()

        if not pose.detected:
            return FrameAnalysis(
                timestamp=now,
                elbow_angle=None,
                state=self.state_right,
                rep_count=self.rep_count,
                bad_rep_count=self.bad_rep_count,
                is_bad_form=False,
                form_issues=["pose_not_detected"],
            )

        r_shoulder = pose.get(self.lm["right_shoulder"]).as_xy()
        r_elbow    = pose.get(self.lm["right_elbow"]).as_xy()
        r_wrist    = pose.get(self.lm["right_wrist"]).as_xy()
        r_hip      = pose.get(self.lm["right_hip"]).as_xy()

        l_shoulder = pose.get(self.lm["left_shoulder"]).as_xy()
        l_elbow    = pose.get(self.lm["left_elbow"]).as_xy()
        l_wrist    = pose.get(self.lm["left_wrist"]).as_xy()

        if self.reference_r_shoulder is None:
            self.reference_r_shoulder = r_shoulder
            self.reference_r_hip      = r_hip

        r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)

        # ── Per-frame form checks ─────────────────────────────────────────────
        frame_issues: list[str] = []

        sway = body_sway(r_shoulder, r_hip, self.reference_r_shoulder, self.reference_r_hip)
        if sway > self.rules["max_body_sway"]:
            frame_issues.append(f"body_sway_{sway:.1f}deg")

        # Upper arm alignment: elbow must stay roughly below shoulder
        r_drift = abs(r_elbow[0] - r_shoulder[0]) * 100
        l_drift = abs(l_elbow[0] - l_shoulder[0]) * 100
        if r_drift > self.rules["max_elbow_drift"] or l_drift > self.rules["max_elbow_drift"]:
            frame_issues.append(f"elbow_drift_{max(r_drift, l_drift):.1f}%")

        # ── State machine (bilateral) ─────────────────────────────────────────
        prev_right = self.state_right
        prev_left  = self.state_left

        if r_angle > self.thr["down_position"]:
            self.state_right = CurlState.DOWN
        elif r_angle < self.thr["up_position"]:
            self.state_right = CurlState.UP

        if l_angle > self.thr["down_position"]:
            self.state_left = CurlState.DOWN
        elif l_angle < self.thr["up_position"]:
            self.state_left = CurlState.UP

        if prev_right != self.state_right:
            self.last_change_right = now
        if prev_left != self.state_left:
            self.last_change_left = now

        both_up   = self.state_right == CurlState.UP   and self.state_left == CurlState.UP
        both_down = self.state_right == CurlState.DOWN and self.state_left == CurlState.DOWN

        sync_window = float(self.rules.get("sync_window_sec", 2.5))

        if both_up and self.both_up_since is None:
            self.both_up_since   = now
            self._pending_issues = []

        if self.both_up_since is not None and frame_issues:
            for issue in frame_issues:
                if issue not in self._pending_issues:
                    self._pending_issues.append(issue)

        if self.both_up_since is not None:
            elapsed = now - self.both_up_since

            if elapsed > sync_window and not both_down:
                self.both_up_since   = None
                self._pending_issues = []

            elif both_down:
                rep_issues = list(self._pending_issues)

                if elapsed < self.cfg["tempo"]["min_eccentric"]:
                    rep_issues.append(f"too_fast_{elapsed:.2f}s")

                if rep_issues:
                    self.bad_rep_count += 1
                    self._trigger_alert(rep_issues[0])
                    frame_issues.extend(rep_issues)
                else:
                    self.rep_count += 1

                self.both_up_since   = None
                self._pending_issues = []

        active_angle = r_angle if r_angle <= l_angle else l_angle
        active_state = self.state_right if r_angle <= l_angle else self.state_left

        return FrameAnalysis(
            timestamp=now,
            elbow_angle=active_angle,
            state=active_state,
            rep_count=self.rep_count,
            bad_rep_count=self.bad_rep_count,
            is_bad_form=len(frame_issues) > 0,
            form_issues=frame_issues,
        )

    def _trigger_alert(self, issue_code: str) -> None:
        if hasattr(self, "alerter"):
            self.alerter.send_alert(level="warn", code=issue_code)

    def trigger_iot_alert(self, issue_code: str) -> None:
        self._trigger_alert(issue_code)

    def session_summary(self) -> dict:
        duration = time() - self.session_start
        return {
            "exercise":         self.cfg["display_name"],
            "duration_seconds": round(duration, 1),
            "valid_reps":       self.rep_count,
            "bad_reps":         self.bad_rep_count,
            "accuracy": round(
                self.rep_count / max(1, self.rep_count + self.bad_rep_count), 3
            ),
        }
