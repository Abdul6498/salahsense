"""Tests for vertical state machine behavior."""

from salahsense.config.settings import ThresholdSettings
from salahsense.state_machine import MovementDirection, VerticalLevel, VerticalStateMachine


def test_vertical_level_classification() -> None:
    machine = VerticalStateMachine(
        ThresholdSettings(
            high_y=0.30,
            mid_y=0.55,
            low_y=0.78,
            mid_tolerance=0.07,
            direction_delta=0.004,
        )
    )

    assert machine.update(0.25).level == VerticalLevel.HIGH
    assert machine.update(0.56).level == VerticalLevel.MID
    assert machine.update(0.85).level == VerticalLevel.LOW


def test_direction_detection() -> None:
    machine = VerticalStateMachine(
        ThresholdSettings(
            high_y=0.30,
            mid_y=0.55,
            low_y=0.78,
            mid_tolerance=0.07,
            direction_delta=0.004,
        )
    )

    machine.update(0.30)
    assert machine.update(0.35).direction == MovementDirection.GOING_DOWN
    assert machine.update(0.32).direction == MovementDirection.GOING_UP


def test_level_change_includes_unknown_transitions() -> None:
    machine = VerticalStateMachine(
        ThresholdSettings(
            high_y=0.30,
            mid_y=0.55,
            low_y=0.78,
            mid_tolerance=0.07,
            direction_delta=0.004,
        )
    )

    state1 = machine.update(0.20)  # HIGH
    state2 = machine.update(0.45)  # UNKNOWN
    state3 = machine.update(0.80)  # LOW

    assert state1.level_changed is True
    assert state2.level_changed is True
    assert state3.level_changed is True
