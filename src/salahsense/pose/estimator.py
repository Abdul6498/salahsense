"""Pose estimation wrapper around MediaPipe Tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks_python
from mediapipe.tasks.python import vision as mp_vision

from salahsense.capture import FramePacket


@dataclass(frozen=True)
class PoseObservation:
    """Pose output for a single frame."""

    frame_index: int
    timestamp_ms: int
    pose_detected: bool
    nose_y: float | None
    landmarks: list | None


class PoseEstimator:
    """Detect pose landmarks frame-by-frame in VIDEO mode."""

    def __init__(
        self,
        model_path: str,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        base_options = mp_tasks_python.BaseOptions(
            model_asset_path=str(model_file),
            delegate=mp_tasks_python.BaseOptions.Delegate.CPU,
        )
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    def detect(self, packet: FramePacket) -> PoseObservation:
        """Run pose detection on one frame."""
        frame_rgb = cv2.cvtColor(packet.frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect_for_video(mp_image, packet.timestamp_ms)

        if len(result.pose_landmarks) == 0:
            return PoseObservation(
                frame_index=packet.frame_index,
                timestamp_ms=packet.timestamp_ms,
                pose_detected=False,
                nose_y=None,
                landmarks=None,
            )

        first_pose = result.pose_landmarks[0]
        nose_y = float(first_pose[0].y)
        return PoseObservation(
            frame_index=packet.frame_index,
            timestamp_ms=packet.timestamp_ms,
            pose_detected=True,
            nose_y=nose_y,
            landmarks=first_pose,
        )

    def close(self) -> None:
        self._landmarker.close()
