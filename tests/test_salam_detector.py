"""Tests for salam end-of-prayer detector."""

from dataclasses import dataclass

from salahsense.counting import SalamDetector, SalamStage


@dataclass
class _Lm:
    x: float
    y: float = 0.0
    z: float = 0.0


@dataclass
class _Obs:
    pose_detected: bool
    landmarks: list[_Lm] | None


def _obs_with_yaw(yaw_score: float) -> _Obs:
    # ear_mid_x = 0.5, ear_distance = 0.2 -> nose_x = yaw * 0.2 + 0.5
    nose_x = 0.5 + (yaw_score * 0.2)
    landmarks = [_Lm(0.0) for _ in range(33)]
    landmarks[0] = _Lm(nose_x)
    landmarks[2] = _Lm(0.46)
    landmarks[5] = _Lm(0.54)
    landmarks[7] = _Lm(0.4)
    landmarks[8] = _Lm(0.6)
    return _Obs(pose_detected=True, landmarks=landmarks)


def test_salam_detector_marks_finished_after_opposite_turns() -> None:
    detector = SalamDetector(
        first_turn_threshold=0.12,
        recenter_threshold=0.20,
        opposite_turn_threshold=0.10,
        turn_hold_frames=1,
        recenter_hold_frames=2,
        smooth_window=3,
    )

    # Neutral tashahhud frames (build baseline).
    detector.update(_obs_with_yaw(0.00), enabled=True)
    detector.update(_obs_with_yaw(0.01), enabled=True)

    # First turn LEFT (use 2 frames because baseline/smoothing are active).
    detector.update(_obs_with_yaw(-0.35), enabled=True)
    first = detector.update(_obs_with_yaw(-0.45), enabled=True)
    assert first.stage == SalamStage.WAIT_RECENTER

    # Return close to center (may take a few frames because of smoothing).
    recentered = None
    for _ in range(8):
        recentered = detector.update(_obs_with_yaw(0.00), enabled=True)
        if recentered.stage == SalamStage.WAIT_OPPOSITE_TURN:
            break
    assert recentered is not None
    assert recentered.stage == SalamStage.WAIT_OPPOSITE_TURN

    # Opposite turn RIGHT should mark finished (may also need a few smoothed frames).
    done = None
    for _ in range(8):
        done = detector.update(_obs_with_yaw(0.55), enabled=True)
        if done.prayer_finished:
            break
    assert done is not None
    assert done.prayer_finished is True
    assert done.stage == SalamStage.FINISHED


def test_salam_detector_resets_when_not_enabled() -> None:
    detector = SalamDetector(first_turn_threshold=0.12, opposite_turn_threshold=0.10, turn_hold_frames=1)
    detector.update(_obs_with_yaw(0.20), enabled=True)
    reset = detector.update(_obs_with_yaw(0.00), enabled=False)
    assert reset.prayer_finished is False
    assert reset.stage == SalamStage.IDLE
