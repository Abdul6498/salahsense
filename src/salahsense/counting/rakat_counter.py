"""Rakat counting with a tolerant stage-based FSM.

This FSM is designed for real video where intermediate frames may become UNKNOWN.
Target phase-1 flow:
- Start from standing (HIGH)
- First sujud (LOW)
- Leave LOW (usually to MID/UNKNOWN)
- Second sujud (LOW)
- Leave second sujud (not LOW anymore) => 1 rakat
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from salahsense.state_machine import VerticalLevel


class RakatStage(str, Enum):
    WAIT_START_HIGH = "WAIT_START_HIGH"
    WAIT_FIRST_LOW = "WAIT_FIRST_LOW"
    WAIT_LEAVE_FIRST_LOW = "WAIT_LEAVE_FIRST_LOW"
    WAIT_SECOND_LOW = "WAIT_SECOND_LOW"
    WAIT_EXIT_SECOND_LOW = "WAIT_EXIT_SECOND_LOW"


@dataclass(frozen=True)
class CounterUpdate:
    """Counter output after processing one level transition."""

    rakat_count: int
    current_rakat: int
    matched_pattern: list[VerticalLevel]
    completed_rakat: bool
    stage: RakatStage
    reason: str


class RakatCounter:
    """Count rakats with a robust stage machine.

    Why this works better than strict sequence matching:
    - allows UNKNOWN between key poses,
    - requires two distinct LOW visits,
    - increments rakat when leaving second sujud.
    """

    def __init__(self) -> None:
        self._stage = RakatStage.WAIT_START_HIGH
        self._rakat_count = 0
        self._current_rakat = 1
        self._matched_pattern: list[VerticalLevel] = []

    @property
    def rakat_count(self) -> int:
        return self._rakat_count

    @property
    def current_rakat(self) -> int:
        return self._current_rakat

    @property
    def matched_pattern(self) -> list[VerticalLevel]:
        return list(self._matched_pattern)

    @property
    def stage(self) -> RakatStage:
        return self._stage

    def on_level_transition(self, level: VerticalLevel) -> CounterUpdate:
        """Advance FSM using changed level (including UNKNOWN)."""
        completed = False
        reason = "no_change"

        if self._stage == RakatStage.WAIT_START_HIGH:
            if level == VerticalLevel.HIGH:
                self._stage = RakatStage.WAIT_FIRST_LOW
                self._matched_pattern = [VerticalLevel.HIGH]
                # Entering Qiyam after a completed rakat means next current rakat.
                if self._rakat_count >= self._current_rakat:
                    self._current_rakat = self._rakat_count + 1
                    reason = "start_high_detected_current_rakat_advanced"
                else:
                    reason = "start_high_detected"

        elif self._stage == RakatStage.WAIT_FIRST_LOW:
            if level == VerticalLevel.LOW:
                self._stage = RakatStage.WAIT_LEAVE_FIRST_LOW
                self._matched_pattern.append(VerticalLevel.LOW)
                reason = "first_sujud_detected"
            elif level == VerticalLevel.HIGH:
                reason = "still_in_start_standing"
            else:
                reason = "waiting_first_low"

        elif self._stage == RakatStage.WAIT_LEAVE_FIRST_LOW:
            # We only need to confirm user leaves first sujud.
            if level != VerticalLevel.LOW:
                self._stage = RakatStage.WAIT_SECOND_LOW
                # Keep MID explicitly in pattern when available for visibility.
                if level == VerticalLevel.MID:
                    self._matched_pattern.append(VerticalLevel.MID)
                    reason = "left_first_sujud_via_mid"
                else:
                    reason = "left_first_sujud"
            else:
                reason = "still_first_sujud"

        elif self._stage == RakatStage.WAIT_SECOND_LOW:
            if level == VerticalLevel.LOW:
                self._stage = RakatStage.WAIT_EXIT_SECOND_LOW
                self._matched_pattern.append(VerticalLevel.LOW)
                reason = "second_sujud_detected_waiting_exit"
            else:
                reason = "waiting_second_low"

        elif self._stage == RakatStage.WAIT_EXIT_SECOND_LOW:
            if level == VerticalLevel.LOW:
                reason = "still_in_second_sujud"
            elif level == VerticalLevel.HIGH:
                self._rakat_count += 1
                completed = True
                self._stage = RakatStage.WAIT_FIRST_LOW
                self._matched_pattern = [VerticalLevel.HIGH]
                self._current_rakat = self._rakat_count + 1
                reason = "left_second_sujud_rakat_complete_and_new_rakat_start_high"
            else:
                # Usually Jalsa or transitional frames after counting.
                self._rakat_count += 1
                completed = True
                self._stage = RakatStage.WAIT_START_HIGH
                self._matched_pattern = []
                reason = "left_second_sujud_rakat_complete_waiting_start_high"

        return CounterUpdate(
            rakat_count=self._rakat_count,
            current_rakat=self._current_rakat,
            matched_pattern=list(self._matched_pattern),
            completed_rakat=completed,
            stage=self._stage,
            reason=reason,
        )
