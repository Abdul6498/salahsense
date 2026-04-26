"""Feature-based Salah state machine (threshold-free across person height).

This module classifies coarse posture from relative body geometry and then
applies a guarded Salah-state transition graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any


class BasePosture(str, Enum):
    STAND = "STAND"
    RUKU = "RUKU"
    SUJUD = "SUJUD"
    SIT = "SIT"
    UNKNOWN = "UNKNOWN"


class SalahState(str, Enum):
    QIYAM = "QIYAM"
    RUKU = "RUKU"
    QAUMA = "QAUMA"
    SUJUD_1 = "SUJUD_1"
    JALSA = "JALSA"
    SUJUD_2 = "SUJUD_2"
    QIYAM_NEXT = "QIYAM_NEXT"
    TASHAHHUD = "TASHAHHUD"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class PoseFeatures:
    nose_y: float | None
    shoulder_mid_y: float | None
    hip_mid_y: float | None
    knee_mid_y: float | None
    ankle_mid_y: float | None
    wrist_mid_y: float | None
    torso_from_vertical_deg: float | None
    knee_angle_deg: float | None
    nose_minus_hip_y: float | None
    shoulder_minus_hip_y: float | None
    wrist_minus_knee_y: float | None


@dataclass(frozen=True)
class SalahStateUpdate:
    detected_posture: BasePosture
    state: SalahState
    state_changed: bool
    completed_rakats: int
    current_rakat: int
    reason: str
    features: PoseFeatures


class SalahStateMachine:
    """State machine using relative pose geometry and valid Salah transitions."""

    def __init__(self, min_stable_frames: int = 4) -> None:
        self.min_stable_frames = min_stable_frames

        self._current_state = SalahState.UNKNOWN
        self._completed_rakats = 0
        self._current_rakat = 1

        self._last_candidate = BasePosture.UNKNOWN
        self._candidate_count = 0

    def update(self, observation: Any) -> SalahStateUpdate:
        features = self._extract_features(observation)
        candidate = self._classify_posture(features)

        if candidate == self._last_candidate:
            self._candidate_count += 1
        else:
            self._last_candidate = candidate
            self._candidate_count = 1

        state_changed = False
        reason = "candidate_not_stable"
        if self._candidate_count >= self.min_stable_frames:
            next_state, reason = self._transition(self._current_state, candidate)
            if next_state != self._current_state:
                self._current_state = next_state
                state_changed = True

        return SalahStateUpdate(
            detected_posture=candidate,
            state=self._current_state,
            state_changed=state_changed,
            completed_rakats=self._completed_rakats,
            current_rakat=self._current_rakat,
            reason=reason,
            features=features,
        )

    def _transition(self, state: SalahState, posture: BasePosture) -> tuple[SalahState, str]:
        if posture == BasePosture.UNKNOWN:
            return state, "posture_unknown"

        if state == SalahState.UNKNOWN:
            if posture == BasePosture.STAND:
                return SalahState.QIYAM, "start_qiyam"
            return state, "wait_for_initial_qiyam"

        if state in (SalahState.QIYAM, SalahState.QIYAM_NEXT):
            if posture == BasePosture.RUKU:
                return SalahState.RUKU, "qiyam_to_ruku"
            if posture == BasePosture.SUJUD:
                # Recovery path: user skipped Ruku and Qauma, then went directly to first sajda.
                return SalahState.SUJUD_1, "qiyam_to_sujud_1_skipped_ruku_and_qauma"
            return state, "holding_qiyam"

        if state == SalahState.RUKU:
            if posture == BasePosture.STAND:
                return SalahState.QAUMA, "ruku_to_qauma"
            if posture == BasePosture.SUJUD:
                # Recovery path: user skipped Qauma and moved directly to first sajda.
                return SalahState.SUJUD_1, "ruku_to_sujud_1_skipped_qauma"
            return state, "holding_ruku"

        if state == SalahState.QAUMA:
            if posture == BasePosture.SUJUD:
                return SalahState.SUJUD_1, "qauma_to_sujud_1"
            return state, "holding_qauma"

        if state == SalahState.SUJUD_1:
            if posture == BasePosture.SIT:
                return SalahState.JALSA, "sujud_1_to_jalsa"
            if posture == BasePosture.STAND:
                self._current_rakat += 1
                return SalahState.QIYAM_NEXT, "left_sujud_1_to_qiyam_next_incomplete_rakat"
            return state, "holding_sujud_1"

        if state == SalahState.JALSA:
            if posture == BasePosture.SUJUD:
                return SalahState.SUJUD_2, "jalsa_to_sujud_2"
            if posture == BasePosture.STAND:
                self._current_rakat += 1
                return SalahState.QIYAM_NEXT, "left_jalsa_to_qiyam_next_incomplete_rakat"
            return state, "holding_jalsa"

        if state == SalahState.SUJUD_2:
            if posture == BasePosture.STAND:
                self._completed_rakats += 1
                self._current_rakat = self._completed_rakats + 1
                return SalahState.QIYAM_NEXT, "left_sujud_2_to_qiyam_next_rakat_counted"
            if posture == BasePosture.SIT:
                # Treat as sitting after completed rakat (e.g. tashahhud situations).
                self._completed_rakats += 1
                # Keep current rakat unchanged while still sitting.
                # It advances only when user stands for the next Qiyam.
                self._current_rakat = max(1, self._completed_rakats)
                return SalahState.TASHAHHUD, "left_sujud_2_to_tashahhud_rakat_counted"
            return state, "holding_sujud_2"

        if state == SalahState.TASHAHHUD:
            if posture == BasePosture.STAND:
                self._current_rakat = self._completed_rakats + 1
                return SalahState.QIYAM_NEXT, "tashahhud_to_qiyam_next"
            return state, "holding_tashahhud"

        return state, "no_transition_rule"

    def _extract_features(self, observation: Any) -> PoseFeatures:
        if not observation.pose_detected or observation.landmarks is None:
            return PoseFeatures(
                nose_y=None,
                shoulder_mid_y=None,
                hip_mid_y=None,
                knee_mid_y=None,
                ankle_mid_y=None,
                wrist_mid_y=None,
                torso_from_vertical_deg=None,
                knee_angle_deg=None,
                nose_minus_hip_y=None,
                shoulder_minus_hip_y=None,
                wrist_minus_knee_y=None,
            )

        lm = observation.landmarks

        nose = lm[0]
        l_sh, r_sh = lm[11], lm[12]
        l_hip, r_hip = lm[23], lm[24]
        l_knee, r_knee = lm[25], lm[26]
        l_ankle, r_ankle = lm[27], lm[28]
        l_wrist, r_wrist = lm[15], lm[16]

        sh_mid = ((l_sh.x + r_sh.x) / 2.0, (l_sh.y + r_sh.y) / 2.0)
        hip_mid = ((l_hip.x + r_hip.x) / 2.0, (l_hip.y + r_hip.y) / 2.0)
        knee_mid = ((l_knee.x + r_knee.x) / 2.0, (l_knee.y + r_knee.y) / 2.0)
        ankle_mid = ((l_ankle.x + r_ankle.x) / 2.0, (l_ankle.y + r_ankle.y) / 2.0)
        wrist_mid = ((l_wrist.x + r_wrist.x) / 2.0, (l_wrist.y + r_wrist.y) / 2.0)

        torso_deg = _angle_from_vertical_deg(sh_mid, hip_mid)
        knee_angle = (_joint_angle_deg(l_hip, l_knee, l_ankle) + _joint_angle_deg(r_hip, r_knee, r_ankle)) / 2.0

        return PoseFeatures(
            nose_y=float(nose.y),
            shoulder_mid_y=float(sh_mid[1]),
            hip_mid_y=float(hip_mid[1]),
            knee_mid_y=float(knee_mid[1]),
            ankle_mid_y=float(ankle_mid[1]),
            wrist_mid_y=float(wrist_mid[1]),
            torso_from_vertical_deg=torso_deg,
            knee_angle_deg=knee_angle,
            nose_minus_hip_y=float(nose.y - hip_mid[1]),
            shoulder_minus_hip_y=float(sh_mid[1] - hip_mid[1]),
            wrist_minus_knee_y=float(wrist_mid[1] - knee_mid[1]),
        )

    def _classify_posture(self, features: PoseFeatures) -> BasePosture:
        if features.nose_y is None:
            return BasePosture.UNKNOWN

        torso = features.torso_from_vertical_deg or 0.0
        knee = features.knee_angle_deg or 180.0
        nose_minus_hip = features.nose_minus_hip_y or 0.0
        shoulder_minus_hip = features.shoulder_minus_hip_y or 0.0
        wrist_minus_knee = features.wrist_minus_knee_y
        hip_y = features.hip_mid_y or 0.0
        knee_y = features.knee_mid_y or 1.0

        # Sujud: head significantly lower than hips, hips close to floor region.
        if nose_minus_hip > 0.18 and hip_y > 0.60:
            return BasePosture.SUJUD

        # Jalsa/Tashahhud-like sitting: bent knees and hips low but head above sujud line.
        if knee < 125 and hip_y > 0.64 and nose_minus_hip < 0.16:
            return BasePosture.SIT

        # Ruku: strong forward hinge with straight knees and body geometry
        # consistent with bowing, not with active descent to sujud.
        if (
            torso > 40
            and knee > 155
            and -0.22 < shoulder_minus_hip < -0.03
            and nose_minus_hip < 0.05
            and 0.40 < hip_y < 0.60
            and (knee_y - hip_y) > 0.14
            and (wrist_minus_knee is None or -0.20 < wrist_minus_knee < 0.15)
        ):
            return BasePosture.RUKU

        # Qiyam/Qauma-like standing: torso near vertical with straighter knees.
        if torso < 22 and knee > 145:
            return BasePosture.STAND

        return BasePosture.UNKNOWN


def _angle_from_vertical_deg(a: tuple[float, float], b: tuple[float, float]) -> float:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return abs(math.degrees(math.atan2(dx, dy)))


def _joint_angle_deg(a, b, c) -> float:
    ba_x = a.x - b.x
    ba_y = a.y - b.y
    bc_x = c.x - b.x
    bc_y = c.y - b.y

    dot = ba_x * bc_x + ba_y * bc_y
    mag_ba = math.sqrt(ba_x * ba_x + ba_y * ba_y)
    mag_bc = math.sqrt(bc_x * bc_x + bc_y * bc_y)
    if mag_ba < 1e-6 or mag_bc < 1e-6:
        return 180.0

    cos_theta = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_theta))
