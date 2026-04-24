"""Tests for feature-based Salah FSM transitions."""

from salahsense.state_machine import BasePosture, SalahState, SalahStateMachine


def _force(machine: SalahStateMachine, posture: BasePosture) -> None:
    machine._last_candidate = posture  # noqa: SLF001
    machine._candidate_count = machine.min_stable_frames  # noqa: SLF001


def test_transition_path_and_rakat_counting() -> None:
    machine = SalahStateMachine(min_stable_frames=1)

    # Use internal transition method for deterministic FSM checks.
    machine._current_state = SalahState.UNKNOWN  # noqa: SLF001
    assert machine._transition(machine._current_state, BasePosture.STAND)[0] == SalahState.QIYAM  # noqa: SLF001

    machine._current_state = SalahState.QIYAM  # noqa: SLF001
    assert machine._transition(machine._current_state, BasePosture.RUKU)[0] == SalahState.RUKU  # noqa: SLF001

    machine._current_state = SalahState.RUKU  # noqa: SLF001
    assert machine._transition(machine._current_state, BasePosture.STAND)[0] == SalahState.QAUMA  # noqa: SLF001

    machine._current_state = SalahState.QAUMA  # noqa: SLF001
    assert machine._transition(machine._current_state, BasePosture.SUJUD)[0] == SalahState.SUJUD_1  # noqa: SLF001

    machine._current_state = SalahState.SUJUD_1  # noqa: SLF001
    assert machine._transition(machine._current_state, BasePosture.SIT)[0] == SalahState.JALSA  # noqa: SLF001

    machine._current_state = SalahState.JALSA  # noqa: SLF001
    assert machine._transition(machine._current_state, BasePosture.SUJUD)[0] == SalahState.SUJUD_2  # noqa: SLF001

    machine._current_state = SalahState.SUJUD_2  # noqa: SLF001
    next_state, _ = machine._transition(machine._current_state, BasePosture.STAND)  # noqa: SLF001
    assert next_state == SalahState.QIYAM_NEXT
    assert machine._completed_rakats == 1  # noqa: SLF001
    assert machine._current_rakat == 2  # noqa: SLF001


def test_current_rakat_advances_only_when_standing_from_tashahhud() -> None:
    machine = SalahStateMachine(min_stable_frames=1)

    machine._current_state = SalahState.SUJUD_2  # noqa: SLF001
    next_state, _ = machine._transition(machine._current_state, BasePosture.SIT)  # noqa: SLF001
    assert next_state == SalahState.TASHAHHUD
    assert machine._completed_rakats == 1  # noqa: SLF001
    assert machine._current_rakat == 1  # noqa: SLF001

    machine._current_state = SalahState.TASHAHHUD  # noqa: SLF001
    next_state, _ = machine._transition(machine._current_state, BasePosture.STAND)  # noqa: SLF001
    assert next_state == SalahState.QIYAM_NEXT
    assert machine._current_rakat == 2  # noqa: SLF001
