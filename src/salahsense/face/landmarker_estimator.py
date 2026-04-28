"""Face Landmarker (MediaPipe Tasks) yaw estimator for salam detection."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks_python
from mediapipe.tasks.python import vision as mp_vision

from salahsense.capture import FramePacket


@dataclass(frozen=True)
class FaceYawObservation:
    yaw_score: float | None
    face_detected: bool
    landmarks: list | None


class FaceLandmarkerYawEstimator:
    """Estimate normalized head yaw from MediaPipe FaceLandmarker landmarks."""

    def __init__(
        self,
        model_path: str,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(
                f"Face landmarker model not found: {model_file}. "
                "Download face_landmarker.task into models/."
            )

        base_options = mp_tasks_python.BaseOptions(
            model_asset_path=str(model_file),
            delegate=mp_tasks_python.BaseOptions.Delegate.CPU,
        )
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def detect(self, packet: FramePacket) -> FaceYawObservation:
        """Return face yaw and landmarks for the current frame."""
        frame_rgb = cv2.cvtColor(packet.frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect_for_video(mp_image, packet.timestamp_ms)

        if len(result.face_landmarks) == 0:
            return FaceYawObservation(yaw_score=None, face_detected=False, landmarks=None)

        face = result.face_landmarks[0]

        # FaceLandmarker canonical indices:
        # 1=nose tip, 33=left eye outer corner, 263=right eye outer corner.
        nose = face[1]
        left_eye_outer = face[33]
        right_eye_outer = face[263]

        eye_distance = abs(left_eye_outer.x - right_eye_outer.x)
        if eye_distance < 1e-3:
            return FaceYawObservation(yaw_score=None, face_detected=True, landmarks=face)

        eye_mid_x = (left_eye_outer.x + right_eye_outer.x) / 2.0
        yaw_score = float((nose.x - eye_mid_x) / eye_distance)
        return FaceYawObservation(yaw_score=yaw_score, face_detected=True, landmarks=face)

    def close(self) -> None:
        self._landmarker.close()
