"""Tests for feature-based Salah FSM transitions."""

from salahsense.state_machine import (
    BasePosture,
    PoseFeatures,
    SalahState,
    SalahStateMachine,
    StandingSubtype,
)


def _force(machine: SalahStateMachine, posture: BasePosture) -> None:
    machine._last_candidate = posture  # noqa: SLF001
    machine._candidate_count = machine.min_stable_frames  # noqa: SLF001


def test_transition_path_and_rakat_counting() -> None:
    machine = SalahStateMachine(min_stable_frames=1)

    # Use internal transition method for deterministic FSM checks.
    machine._current_state = SalahState.UNKNOWN  # noqa: SLF001
    assert machine._transition(machine._current_state, BasePosture.STAND, StandingSubtype.UNKNOWN)[0] == SalahState.QIYAM  # noqa: SLF001

    machine._current_state = SalahState.QIYAM  # noqa: SLF001
    assert machine._transition(machine._current_state, BasePosture.RUKU, StandingSubtype.UNKNOWN)[0] == SalahState.RUKU  # noqa: SLF001

    machine._current_state = SalahState.RUKU  # noqa: SLF001
    assert machine._transition(machine._current_state, BasePosture.STAND, StandingSubtype.HANDS_DOWN)[0] == SalahState.QAUMA  # noqa: SLF001

    machine._current_state = SalahState.QAUMA  # noqa: SLF001
    assert machine._transition(machine._current_state, BasePosture.SUJUD, StandingSubtype.UNKNOWN)[0] == SalahState.SUJUD_1  # noqa: SLF001

    machine._current_state = SalahState.SUJUD_1  # noqa: SLF001
    assert machine._transition(machine._current_state, BasePosture.SIT, StandingSubtype.UNKNOWN)[0] == SalahState.JALSA  # noqa: SLF001

    machine._current_state = SalahState.JALSA  # noqa: SLF001
    assert machine._transition(machine._current_state, BasePosture.SUJUD, StandingSubtype.UNKNOWN)[0] == SalahState.SUJUD_2  # noqa: SLF001

    machine._current_state = SalahState.SUJUD_2  # noqa: SLF001
    next_state, _ = machine._transition(machine._current_state, BasePosture.STAND, StandingSubtype.UNKNOWN)  # noqa: SLF001
    assert next_state == SalahState.QIYAM_NEXT
    assert machine._completed_rakats == 1  # noqa: SLF001
    assert machine._current_rakat == 2  # noqa: SLF001


def test_current_rakat_advances_only_when_standing_from_tashahhud() -> None:
    machine = SalahStateMachine(min_stable_frames=1)

    machine._current_state = SalahState.SUJUD_2  # noqa: SLF001
    next_state, _ = machine._transition(machine._current_state, BasePosture.SIT, StandingSubtype.UNKNOWN)  # noqa: SLF001
    assert next_state == SalahState.TASHAHHUD
    assert machine._completed_rakats == 1  # noqa: SLF001
    assert machine._current_rakat == 1  # noqa: SLF001

    machine._current_state = SalahState.TASHAHHUD  # noqa: SLF001
    next_state, _ = machine._transition(machine._current_state, BasePosture.STAND, StandingSubtype.UNKNOWN)  # noqa: SLF001
    assert next_state == SalahState.QIYAM_NEXT
    assert machine._current_rakat == 2  # noqa: SLF001


def test_current_rakat_advances_when_standing_from_any_sajda_phase() -> None:
    machine = SalahStateMachine(min_stable_frames=1)

    machine._current_state = SalahState.SUJUD_1  # noqa: SLF001
    machine._current_rakat = 1  # noqa: SLF001
    next_state, _ = machine._transition(machine._current_state, BasePosture.STAND, StandingSubtype.UNKNOWN)  # noqa: SLF001
    assert next_state == SalahState.QIYAM_NEXT
    assert machine._current_rakat == 2  # noqa: SLF001

    machine._current_state = SalahState.JALSA  # noqa: SLF001
    machine._current_rakat = 2  # noqa: SLF001
    next_state, _ = machine._transition(machine._current_state, BasePosture.STAND, StandingSubtype.UNKNOWN)  # noqa: SLF001
    assert next_state == SalahState.QIYAM_NEXT
    assert machine._current_rakat == 3  # noqa: SLF001


def test_ruku_has_recovery_transitions_to_avoid_stuck_state() -> None:
    machine = SalahStateMachine(min_stable_frames=1)

    machine._current_state = SalahState.RUKU  # noqa: SLF001
    next_state, reason = machine._transition(machine._current_state, BasePosture.SUJUD, StandingSubtype.UNKNOWN)  # noqa: SLF001
    assert next_state == SalahState.SUJUD_1
    assert reason == "ruku_to_sujud_1_skipped_qauma"

    machine._current_state = SalahState.RUKU  # noqa: SLF001
    next_state, reason = machine._transition(machine._current_state, BasePosture.SIT, StandingSubtype.UNKNOWN)  # noqa: SLF001
    assert next_state == SalahState.RUKU
    assert reason == "holding_ruku"


def test_qiyam_has_recovery_transition_for_direct_sujud() -> None:
    machine = SalahStateMachine(min_stable_frames=1)

    machine._current_state = SalahState.QIYAM  # noqa: SLF001
    next_state, reason = machine._transition(machine._current_state, BasePosture.SUJUD, StandingSubtype.UNKNOWN)  # noqa: SLF001
    assert next_state == SalahState.SUJUD_1
    assert reason == "qiyam_to_sujud_1_skipped_ruku_and_qauma"


def test_ruku_classification_is_tighter_during_downward_descent() -> None:
    machine = SalahStateMachine(min_stable_frames=1)

    true_ruku = PoseFeatures(
        nose_y=0.40,
        shoulder_mid_y=0.39,
        hip_mid_y=0.52,
        knee_mid_y=0.70,
        ankle_mid_y=0.92,
        wrist_mid_y=0.62,
        torso_from_vertical_deg=43.0,
        knee_angle_deg=160.0,
        nose_minus_hip_y=0.03,
        shoulder_minus_hip_y=-0.13,
        wrist_minus_knee_y=-0.08,
    )
    assert machine._classify_posture(true_ruku) == BasePosture.RUKU  # noqa: SLF001

    descending_to_sujud = PoseFeatures(
        nose_y=0.74,
        shoulder_mid_y=0.48,
        hip_mid_y=0.58,
        knee_mid_y=0.79,
        ankle_mid_y=0.94,
        wrist_mid_y=0.83,
        torso_from_vertical_deg=48.0,
        knee_angle_deg=151.0,
        nose_minus_hip_y=0.20,
        shoulder_minus_hip_y=-0.10,
        wrist_minus_knee_y=0.04,
    )
    assert machine._classify_posture(descending_to_sujud) != BasePosture.RUKU  # noqa: SLF001


def test_standing_subtype_detects_folded_vs_hands_down() -> None:
    machine = SalahStateMachine(min_stable_frames=1)

    folded = PoseFeatures(
        nose_y=0.12,
        shoulder_mid_y=0.16,
        hip_mid_y=0.48,
        knee_mid_y=0.74,
        wrist_mid_y=0.40,
        torso_from_vertical_deg=6.0,
        knee_angle_deg=176.0,
        elbow_angle_deg=140.0,
        wrist_minus_hip_y=-0.08,
        wrist_to_torso_center_x=0.04,
    )
    hands_down = PoseFeatures(
        nose_y=0.18,
        shoulder_mid_y=0.21,
        hip_mid_y=0.48,
        knee_mid_y=0.70,
        wrist_mid_y=0.48,
        torso_from_vertical_deg=2.0,
        knee_angle_deg=175.0,
        elbow_angle_deg=162.0,
        wrist_minus_hip_y=0.00,
        wrist_to_torso_center_x=0.06,
    )

    assert (
        machine._classify_standing_subtype(folded, BasePosture.STAND)  # noqa: SLF001
        == StandingSubtype.FOLDED
    )
    assert (
        machine._classify_standing_subtype(hands_down, BasePosture.STAND)  # noqa: SLF001
        == StandingSubtype.HANDS_DOWN
    )
