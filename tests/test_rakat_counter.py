"""Tests for phase-1 rakat sequence counting."""

from salahsense.counting import RakatCounter
from salahsense.state_machine import VerticalLevel


def test_rakat_counter_counts_one_full_sequence() -> None:
    counter = RakatCounter()
    sequence = [
        VerticalLevel.HIGH,
        VerticalLevel.LOW,
        VerticalLevel.MID,
        VerticalLevel.LOW,
        VerticalLevel.MID,
    ]

    completed = False
    for level in sequence:
        update = counter.on_level_transition(level)
        completed = completed or update.completed_rakat

    assert completed is True
    assert counter.rakat_count == 1
    # After completing and leaving second sujud via MID, current rakat
    # advances when HIGH is reached again.
    counter.on_level_transition(VerticalLevel.HIGH)
    assert counter.current_rakat == 2


def test_rakat_counter_resets_on_wrong_order() -> None:
    counter = RakatCounter()

    counter.on_level_transition(VerticalLevel.HIGH)
    counter.on_level_transition(VerticalLevel.MID)  # wrong after HIGH
    counter.on_level_transition(VerticalLevel.LOW)
    counter.on_level_transition(VerticalLevel.HIGH)

    assert counter.rakat_count == 0


def test_rakat_counter_tolerates_unknown_between_key_positions() -> None:
    counter = RakatCounter()
    sequence = [
        VerticalLevel.HIGH,
        VerticalLevel.LOW,
        VerticalLevel.UNKNOWN,  # leaving first sujud through transitional frames
        VerticalLevel.LOW,
        VerticalLevel.UNKNOWN,  # transitional rise
        VerticalLevel.MID,
    ]

    completed = False
    for level in sequence:
        update = counter.on_level_transition(level)
        completed = completed or update.completed_rakat

    assert completed is True
    assert counter.rakat_count == 1


def test_rakat_is_counted_after_second_sujud_exit() -> None:
    counter = RakatCounter()
    sequence = [
        VerticalLevel.HIGH,
        VerticalLevel.LOW,
        VerticalLevel.MID,
        VerticalLevel.LOW,
        VerticalLevel.UNKNOWN,
    ]

    last_update = None
    for level in sequence:
        last_update = counter.on_level_transition(level)

    assert last_update is not None
    assert last_update.completed_rakat is True
    assert counter.rakat_count == 1
    assert counter.current_rakat == 1


def test_current_rakat_advances_immediately_on_direct_high_after_second_sujud() -> None:
    counter = RakatCounter()
    sequence = [
        VerticalLevel.HIGH,
        VerticalLevel.LOW,
        VerticalLevel.MID,
        VerticalLevel.LOW,
        VerticalLevel.HIGH,
    ]

    for level in sequence:
        counter.on_level_transition(level)

    assert counter.rakat_count == 1
    assert counter.current_rakat == 2
