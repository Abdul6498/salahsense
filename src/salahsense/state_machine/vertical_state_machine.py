"""Vertical state machine for simple rakat-position reasoning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from salahsense.config.settings import ThresholdSettings


class VerticalLevel(str, Enum):
    """Coarse posture level from head Y position."""

    HIGH = "HIGH"
    MID = "MID"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


class MovementDirection(str, Enum):
    """Direction of head movement between frames."""

    GOING_DOWN = "GOING_DOWN"
    GOING_UP = "GOING_UP"
    STILL = "STILL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class VerticalState:
    """State-machine output for one frame."""

    level: VerticalLevel
    direction: MovementDirection
    level_changed: bool


class VerticalStateMachine:
    """Track direction and level using head Y value."""

    def __init__(self, thresholds: ThresholdSettings) -> None:
        self.thresholds = thresholds
        self._previous_y: float | None = None
        self._previous_level: VerticalLevel = VerticalLevel.UNKNOWN

    def update(self, nose_y: float | None) -> VerticalState:
        """Update machine with current head Y coordinate."""
        level = self._classify_level(nose_y)
        direction = self._classify_direction(nose_y)
        level_changed = level != self._previous_level

        self._previous_y = nose_y
        self._previous_level = level

        return VerticalState(level=level, direction=direction, level_changed=level_changed)

    def _classify_level(self, nose_y: float | None) -> VerticalLevel:
        if nose_y is None:
            return VerticalLevel.UNKNOWN

        if nose_y <= self.thresholds.high_y:
            return VerticalLevel.HIGH

        if nose_y >= self.thresholds.low_y:
            return VerticalLevel.LOW

        if abs(nose_y - self.thresholds.mid_y) <= self.thresholds.mid_tolerance:
            return VerticalLevel.MID

        return VerticalLevel.UNKNOWN

    def _classify_direction(self, nose_y: float | None) -> MovementDirection:
        if nose_y is None or self._previous_y is None:
            return MovementDirection.UNKNOWN

        delta = nose_y - self._previous_y
        if delta > self.thresholds.direction_delta:
            return MovementDirection.GOING_DOWN
        if delta < -self.thresholds.direction_delta:
            return MovementDirection.GOING_UP
        return MovementDirection.STILL
