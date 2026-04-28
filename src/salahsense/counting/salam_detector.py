"""Detect end-of-prayer salam (right/left head turns) in final tashahhud."""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from enum import Enum
from typing import Any


class SalamStage(str, Enum):
    IDLE = "IDLE"
    WAIT_FIRST_TURN = "WAIT_FIRST_TURN"
    WAIT_RECENTER = "WAIT_RECENTER"
    WAIT_OPPOSITE_TURN = "WAIT_OPPOSITE_TURN"
    FINISHED = "FINISHED"


@dataclass(frozen=True)
class SalamUpdate:
    prayer_finished: bool
    stage: SalamStage
    yaw_score: float | None
    first_turn_direction: str | None


class SalamDetector:
    """Lightweight yaw-pattern detector for salam.

    The detector is enabled only during final tashahhud and marks prayer as finished
    when it sees one clear head turn followed by a clear opposite turn.
    """

    def __init__(
        self,
        *,
        first_turn_threshold: float = 0.22,
        recenter_threshold: float = 0.08,
        opposite_turn_threshold: float = 0.16,
        turn_hold_frames: int = 3,
        recenter_hold_frames: int = 2,
        smooth_window: int = 7,
        expected_first_turn_direction: str = "LEFT",
    ) -> None:
        self.first_turn_threshold = first_turn_threshold
        self.recenter_threshold = recenter_threshold
        self.opposite_turn_threshold = opposite_turn_threshold
        self.turn_hold_frames = turn_hold_frames
        self.recenter_hold_frames = recenter_hold_frames
        self.expected_first_turn_direction = expected_first_turn_direction.upper()
        self._expected_first_sign = -1 if self.expected_first_turn_direction == "LEFT" else 1
        self._stage = SalamStage.IDLE
        self._baseline_yaw: float | None = None
        self._first_turn_sign: int | None = None
        self._prayer_finished = False
        self._yaw_window: deque[float] = deque(maxlen=smooth_window)
        self._turn_hold_count = 0
        self._recenter_hold_count = 0

    def update(
        self,
        observation: Any | None = None,
        *,
        enabled: bool,
        yaw_score: float | None = None,
    ) -> SalamUpdate:
        if yaw_score is None:
            yaw_score = _compute_yaw_score(observation)
        smooth_yaw = self._smooth_yaw(yaw_score)

        if self._prayer_finished:
            return SalamUpdate(
                prayer_finished=True,
                stage=SalamStage.FINISHED,
                yaw_score=smooth_yaw,
                first_turn_direction=_sign_to_label(self._first_turn_sign),
            )

        if not enabled:
            self._reset_transient()
            return SalamUpdate(
                prayer_finished=False,
                stage=self._stage,
                yaw_score=smooth_yaw,
                first_turn_direction=None,
            )

        if self._stage == SalamStage.IDLE:
            self._stage = SalamStage.WAIT_FIRST_TURN

        if smooth_yaw is None:
            return SalamUpdate(
                prayer_finished=False,
                stage=self._stage,
                yaw_score=None,
                first_turn_direction=_sign_to_label(self._first_turn_sign),
            )

        self._baseline_yaw = _ema(self._baseline_yaw, smooth_yaw, alpha=0.12)
        yaw_delta = smooth_yaw - self._baseline_yaw

        if self._stage == SalamStage.WAIT_FIRST_TURN:
            current_sign = 1 if yaw_delta > 0 else -1
            if (
                abs(yaw_delta) >= self.first_turn_threshold
                and current_sign == self._expected_first_sign
            ):
                self._turn_hold_count += 1
            else:
                self._turn_hold_count = 0

            if self._turn_hold_count >= self.turn_hold_frames:
                self._first_turn_sign = current_sign
                self._stage = SalamStage.WAIT_RECENTER
                self._turn_hold_count = 0
            return SalamUpdate(
                prayer_finished=False,
                stage=self._stage,
                yaw_score=smooth_yaw,
                first_turn_direction=_sign_to_label(self._first_turn_sign),
            )

        if self._stage == SalamStage.WAIT_RECENTER:
            if abs(yaw_delta) <= self.recenter_threshold:
                self._recenter_hold_count += 1
            else:
                self._recenter_hold_count = 0

            if self._recenter_hold_count >= self.recenter_hold_frames:
                self._stage = SalamStage.WAIT_OPPOSITE_TURN
                self._recenter_hold_count = 0

            return SalamUpdate(
                prayer_finished=self._prayer_finished,
                stage=self._stage,
                yaw_score=smooth_yaw,
                first_turn_direction=_sign_to_label(self._first_turn_sign),
            )

        if self._stage == SalamStage.WAIT_OPPOSITE_TURN:
            opposite_hit = (
                abs(yaw_delta) >= self.opposite_turn_threshold
                and self._first_turn_sign is not None
                and (1 if yaw_delta > 0 else -1) == -self._first_turn_sign
            )
            if opposite_hit:
                self._turn_hold_count += 1
            else:
                self._turn_hold_count = 0

            if self._turn_hold_count >= self.turn_hold_frames:
                self._prayer_finished = True
                self._stage = SalamStage.FINISHED

            return SalamUpdate(
                prayer_finished=self._prayer_finished,
                stage=self._stage,
                yaw_score=smooth_yaw,
                first_turn_direction=_sign_to_label(self._first_turn_sign),
            )

        return SalamUpdate(
            prayer_finished=self._prayer_finished,
            stage=self._stage,
            yaw_score=smooth_yaw,
            first_turn_direction=_sign_to_label(self._first_turn_sign),
        )

    def _reset_transient(self) -> None:
        if self._prayer_finished:
            return
        self._stage = SalamStage.IDLE
        self._baseline_yaw = None
        self._first_turn_sign = None
        self._turn_hold_count = 0
        self._recenter_hold_count = 0
        self._yaw_window.clear()

    def _smooth_yaw(self, yaw_score: float | None) -> float | None:
        if yaw_score is None:
            return None
        self._yaw_window.append(yaw_score)
        values = sorted(self._yaw_window)
        mid = len(values) // 2
        if len(values) % 2 == 1:
            return values[mid]
        return (values[mid - 1] + values[mid]) / 2.0


def _compute_yaw_score(observation: Any) -> float | None:
    if observation is None:
        return None
    if not observation.pose_detected or observation.landmarks is None:
        return None

    landmarks = observation.landmarks
    if len(landmarks) <= 8:
        return None

    nose = landmarks[0]
    left_ear = landmarks[7]
    right_ear = landmarks[8]
    left_eye = landmarks[2] if len(landmarks) > 2 else None
    right_eye = landmarks[5] if len(landmarks) > 5 else None

    ear_distance = abs(left_ear.x - right_ear.x)
    eye_distance = abs(left_eye.x - right_eye.x) if left_eye and right_eye else 0.0

    yaw_values: list[float] = []

    if ear_distance >= 1e-3:
        ear_mid_x = (left_ear.x + right_ear.x) / 2.0
        yaw_values.append(float((nose.x - ear_mid_x) / ear_distance))

    if eye_distance >= 1e-3 and left_eye and right_eye:
        eye_mid_x = (left_eye.x + right_eye.x) / 2.0
        yaw_values.append(float((nose.x - eye_mid_x) / eye_distance))

    if not yaw_values:
        return None

    return sum(yaw_values) / len(yaw_values)


def _sign_to_label(sign: int | None) -> str | None:
    if sign is None:
        return None
    return "RIGHT" if sign > 0 else "LEFT"


def _ema(previous: float | None, current: float, *, alpha: float) -> float:
    if previous is None:
        return current
    return (1.0 - alpha) * previous + alpha * current
