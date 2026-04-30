"""Tests for feature-based Salah FSM transitions."""

from salahsense.state_machine import BasePosture, PoseFeatures, SalahState, SalahStateMachine


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
    machine._completed_rakats = 1  # noqa: SLF001
    for _ in range(machine.min_sujud2_sit_frames_for_tashahhud):  # noqa: SLF001
        next_state, _ = machine._transition(machine._current_state, BasePosture.SIT)  # noqa: SLF001
        machine._current_state = next_state  # noqa: SLF001
    assert next_state == SalahState.TASHAHHUD
    assert machine._completed_rakats == 2  # noqa: SLF001
    assert machine._current_rakat == 2  # noqa: SLF001

    machine._current_state = SalahState.TASHAHHUD  # noqa: SLF001
    next_state, _ = machine._transition(machine._current_state, BasePosture.STAND)  # noqa: SLF001
    assert next_state == SalahState.QIYAM_NEXT
    assert machine._current_rakat == 3  # noqa: SLF001


def test_sujud2_brief_sit_does_not_transition_to_tashahhud() -> None:
    machine = SalahStateMachine(min_stable_frames=1)
    machine._current_state = SalahState.SUJUD_2  # noqa: SLF001

    for _ in range(machine.min_sujud2_sit_frames_for_tashahhud - 1):  # noqa: SLF001
        next_state, reason = machine._transition(machine._current_state, BasePosture.SIT)  # noqa: SLF001
        assert next_state == SalahState.SUJUD_2
        assert reason == "confirming_tashahhud_sit"

    # User stands up before sit is confirmed: this should count rakat and move on.
    next_state, reason = machine._transition(machine._current_state, BasePosture.STAND)  # noqa: SLF001
    assert next_state == SalahState.QIYAM_NEXT
    assert reason == "left_sujud_2_to_qiyam_next_rakat_counted"
    assert machine._completed_rakats == 1  # noqa: SLF001
    assert machine._current_rakat == 2  # noqa: SLF001


def test_sujud2_sit_in_first_rakat_does_not_trigger_tashahhud_for_4_rakat_prayer() -> None:
    machine = SalahStateMachine(min_stable_frames=1, tashahhud_after_rakats={2, 4})
    machine._current_state = SalahState.SUJUD_2  # noqa: SLF001
    machine._completed_rakats = 0  # noqa: SLF001

    for _ in range(machine.min_sujud2_sit_frames_for_tashahhud + 2):  # noqa: SLF001
        next_state, reason = machine._transition(machine._current_state, BasePosture.SIT)  # noqa: SLF001
        assert next_state == SalahState.SUJUD_2
        assert reason in {"confirming_tashahhud_sit", "sujud_2_sit_not_valid_tashahhud_rakat"}

    # Standing up should still complete rakat 1 correctly.
    next_state, reason = machine._transition(machine._current_state, BasePosture.STAND)  # noqa: SLF001
    assert next_state == SalahState.QIYAM_NEXT
    assert reason == "left_sujud_2_to_qiyam_next_rakat_counted"
    assert machine._completed_rakats == 1  # noqa: SLF001
    assert machine._current_rakat == 2  # noqa: SLF001


def test_sujud2_sit_without_hands_on_thighs_does_not_trigger_tashahhud() -> None:
    machine = SalahStateMachine(min_stable_frames=1)
    machine._current_state = SalahState.SUJUD_2  # noqa: SLF001
    machine._completed_rakats = 1  # noqa: SLF001

    sit_no_hands = PoseFeatures(
        nose_y=0.56,
        shoulder_mid_y=0.62,
        hip_mid_y=0.85,
        knee_mid_y=0.91,
        ankle_mid_y=0.92,
        wrist_mid_y=0.60,
        torso_from_vertical_deg=2.0,
        knee_angle_deg=38.0,
        nose_minus_hip_y=-0.29,
        shoulder_minus_hip_y=-0.23,
        wrist_minus_knee_y=-0.31,
        left_wrist_to_left_thigh_norm=1.6,
        right_wrist_to_right_thigh_norm=1.7,
        hip_to_ankle_y_gap=0.08,
        left_wrist_to_left_knee_x_abs=0.35,
        right_wrist_to_right_knee_x_abs=0.32,
        left_wrist_visibility=0.90,
        right_wrist_visibility=0.90,
    )

    for _ in range(machine.min_sujud2_sit_frames_for_tashahhud + 2):  # noqa: SLF001
        next_state, reason = machine._transition(machine._current_state, BasePosture.SIT, sit_no_hands)  # noqa: SLF001
        assert next_state == SalahState.SUJUD_2
        assert reason in {"confirming_tashahhud_sit", "sujud_2_sit_without_hands_on_thighs"}

def test_current_rakat_advances_when_standing_from_any_sajda_phase() -> None:
    machine = SalahStateMachine(min_stable_frames=1)

    machine._current_state = SalahState.SUJUD_1  # noqa: SLF001
    machine._current_rakat = 1  # noqa: SLF001
    next_state, _ = machine._transition(machine._current_state, BasePosture.STAND)  # noqa: SLF001
    assert next_state == SalahState.QIYAM_NEXT
    assert machine._current_rakat == 2  # noqa: SLF001

    machine._current_state = SalahState.JALSA  # noqa: SLF001
    machine._current_rakat = 2  # noqa: SLF001
    next_state, _ = machine._transition(machine._current_state, BasePosture.STAND)  # noqa: SLF001
    assert next_state == SalahState.QIYAM_NEXT
    assert machine._current_rakat == 3  # noqa: SLF001


def test_ruku_has_recovery_transitions_to_avoid_stuck_state() -> None:
    machine = SalahStateMachine(min_stable_frames=1)

    machine._current_state = SalahState.RUKU  # noqa: SLF001
    next_state, reason = machine._transition(machine._current_state, BasePosture.SUJUD)  # noqa: SLF001
    assert next_state == SalahState.SUJUD_1
    assert reason == "ruku_to_sujud_1_skipped_qauma"

    machine._current_state = SalahState.RUKU  # noqa: SLF001
    next_state, reason = machine._transition(machine._current_state, BasePosture.SIT)  # noqa: SLF001
    assert next_state == SalahState.RUKU
    assert reason == "holding_ruku"


def test_qiyam_has_recovery_transition_for_direct_sujud() -> None:
    machine = SalahStateMachine(min_stable_frames=1)

    machine._current_state = SalahState.QIYAM  # noqa: SLF001
    next_state, reason = machine._transition(machine._current_state, BasePosture.SUJUD)  # noqa: SLF001
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
        left_wrist_to_left_thigh_norm=0.9,
        right_wrist_to_right_thigh_norm=0.92,
        hip_to_ankle_y_gap=0.40,
        left_wrist_to_left_knee_x_abs=0.08,
        right_wrist_to_right_knee_x_abs=0.09,
        left_wrist_visibility=0.9,
        right_wrist_visibility=0.9,
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
        left_wrist_to_left_thigh_norm=1.6,
        right_wrist_to_right_thigh_norm=1.5,
        hip_to_ankle_y_gap=0.30,
        left_wrist_to_left_knee_x_abs=0.25,
        right_wrist_to_right_knee_x_abs=0.22,
        left_wrist_visibility=0.9,
        right_wrist_visibility=0.9,
    )
    assert machine._classify_posture(descending_to_sujud) != BasePosture.RUKU  # noqa: SLF001


def test_crouched_transition_is_not_classified_as_sit() -> None:
    machine = SalahStateMachine(min_stable_frames=1)
    crouched_transition = PoseFeatures(
        nose_y=0.67,
        shoulder_mid_y=0.60,
        hip_mid_y=0.70,
        knee_mid_y=0.82,
        ankle_mid_y=0.91,
        wrist_mid_y=0.84,
        torso_from_vertical_deg=48.0,
        knee_angle_deg=95.0,
        nose_minus_hip_y=-0.03,
        shoulder_minus_hip_y=-0.10,
        wrist_minus_knee_y=0.02,
        left_wrist_to_left_thigh_norm=1.25,
        right_wrist_to_right_thigh_norm=1.30,
        hip_to_ankle_y_gap=0.21,
        left_wrist_to_left_knee_x_abs=0.22,
        right_wrist_to_right_knee_x_abs=0.20,
        left_wrist_visibility=0.95,
        right_wrist_visibility=0.95,
    )
    assert machine._classify_posture(crouched_transition) != BasePosture.SIT  # noqa: SLF001
