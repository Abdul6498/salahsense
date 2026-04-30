"""Pose estimation adapters (MediaPipe Tasks + YOLO pose + ViTPose)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import mediapipe as mp
import numpy as np
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


@dataclass(frozen=True)
class Landmark:
    """Minimal normalized landmark format used across backends."""

    x: float
    y: float
    z: float = 0.0
    visibility: float = 0.0
    presence: float = 0.0


class PoseEstimator(Protocol):
    def detect(self, packet: FramePacket) -> PoseObservation: ...
    def close(self) -> None: ...


class MediaPipePoseEstimator:
    """Detect pose landmarks frame-by-frame in VIDEO mode (MediaPipe Tasks)."""

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


class YoloPoseEstimator:
    """Detect pose landmarks frame-by-frame using YOLO pose models."""

    def __init__(self, model_path: str) -> None:
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"YOLO pose model file not found: {model_file}")
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on runtime deps
            raise RuntimeError(
                "YOLO backend requires ultralytics. Install it with: pip install ultralytics"
            ) from exc

        self._model = YOLO(str(model_file))

    def detect(self, packet: FramePacket) -> PoseObservation:
        h, w = packet.frame_bgr.shape[:2]
        result_list = self._model.predict(packet.frame_bgr, verbose=False)
        if not result_list:
            return PoseObservation(
                frame_index=packet.frame_index,
                timestamp_ms=packet.timestamp_ms,
                pose_detected=False,
                nose_y=None,
                landmarks=None,
            )

        result = result_list[0]
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
            return PoseObservation(
                frame_index=packet.frame_index,
                timestamp_ms=packet.timestamp_ms,
                pose_detected=False,
                nose_y=None,
                landmarks=None,
            )

        # Pick the highest-confidence person if available.
        person_index = 0
        boxes = getattr(result, "boxes", None)
        if boxes is not None and getattr(boxes, "conf", None) is not None and len(boxes.conf) > 0:
            person_index = int(boxes.conf.argmax().item())

        xy = keypoints.xy[person_index].cpu().numpy()
        conf = None
        if getattr(keypoints, "conf", None) is not None and len(keypoints.conf) > person_index:
            conf = keypoints.conf[person_index].cpu().numpy()

        coco_landmarks: list[Landmark] = []
        for idx in range(xy.shape[0]):
            x_px, y_px = float(xy[idx][0]), float(xy[idx][1])
            vis = float(conf[idx]) if conf is not None else 0.0
            coco_landmarks.append(
                Landmark(
                    x=max(0.0, min(1.0, x_px / max(1.0, w))),
                    y=max(0.0, min(1.0, y_px / max(1.0, h))),
                    z=0.0,
                    visibility=vis,
                    presence=vis,
                )
            )

        landmarks = _map_coco17_to_mediapipe33(coco_landmarks)
        nose_y = float(landmarks[0].y) if landmarks and landmarks[0].visibility > 0.01 else None
        return PoseObservation(
            frame_index=packet.frame_index,
            timestamp_ms=packet.timestamp_ms,
            pose_detected=bool(landmarks and landmarks[0].visibility > 0.01),
            nose_y=nose_y,
            landmarks=landmarks if landmarks else None,
        )

    def close(self) -> None:
        # Ultralytics model has no explicit close routine.
        return None


class VitPoseEstimator:
    """Detect pose landmarks using ViTPose with YOLO person detection."""

    def __init__(self, model_path: str, detector_model: str = "yolo11n.pt") -> None:
        try:
            import torch  # type: ignore
            from PIL import Image  # type: ignore
            from transformers import AutoImageProcessor, VitPoseForPoseEstimation  # type: ignore
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "ViTPose backend requires dependencies: torch, transformers, pillow, ultralytics. "
                "Install with: pip install torch transformers pillow ultralytics"
            ) from exc

        self._torch = torch
        self._pil_image = Image
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        model_ref = model_path.strip()
        self._processor = AutoImageProcessor.from_pretrained(model_ref)
        self._model = VitPoseForPoseEstimation.from_pretrained(model_ref)
        self._model.to(self._device)
        self._model.eval()

        self._person_detector = YOLO(detector_model)

    def detect(self, packet: FramePacket) -> PoseObservation:
        h, w = packet.frame_bgr.shape[:2]
        bbox_xyxy = self._best_person_bbox(packet.frame_bgr)
        if bbox_xyxy is None:
            return PoseObservation(
                frame_index=packet.frame_index,
                timestamp_ms=packet.timestamp_ms,
                pose_detected=False,
                nose_y=None,
                landmarks=None,
            )
        bbox_xywh = _xyxy_to_xywh(bbox_xyxy)

        frame_rgb = cv2.cvtColor(packet.frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._pil_image.fromarray(frame_rgb)

        # ViTPose expects boxes in image-list shape and XYWH format.
        # For one image with one person => boxes=[[x, y, w, h]] wrapped as [ ... ].
        inputs = self._processor(images=image, boxes=[[bbox_xywh]], return_tensors="pt")
        for key, value in inputs.items():
            if hasattr(value, "to"):
                inputs[key] = value.to(self._device)

        with self._torch.no_grad():
            outputs = self._model(**inputs)

        try:
            pose_results = self._processor.post_process_pose_estimation(
                outputs,
                boxes=[[bbox_xywh]],
                threshold=0.0,
            )
        except Exception as exc:
            raise RuntimeError(
                "ViTPose post-process failed. Ensure your transformers version supports "
                "VitPose post_process_pose_estimation."
            ) from exc

        if not pose_results or not pose_results[0]:
            return PoseObservation(
                frame_index=packet.frame_index,
                timestamp_ms=packet.timestamp_ms,
                pose_detected=False,
                nose_y=None,
                landmarks=None,
            )

        person = pose_results[0][0]
        keypoints = person.get("keypoints")
        scores = person.get("scores")
        if keypoints is None:
            return PoseObservation(
                frame_index=packet.frame_index,
                timestamp_ms=packet.timestamp_ms,
                pose_detected=False,
                nose_y=None,
                landmarks=None,
            )

        keypoints_np = np.asarray(keypoints, dtype=float)
        scores_np = np.asarray(scores, dtype=float) if scores is not None else None
        coco_landmarks: list[Landmark] = []
        for idx in range(keypoints_np.shape[0]):
            x_px, y_px = float(keypoints_np[idx][0]), float(keypoints_np[idx][1])
            vis = float(scores_np[idx]) if scores_np is not None and idx < len(scores_np) else 0.0
            coco_landmarks.append(
                Landmark(
                    x=max(0.0, min(1.0, x_px / max(1.0, w))),
                    y=max(0.0, min(1.0, y_px / max(1.0, h))),
                    z=0.0,
                    visibility=vis,
                    presence=vis,
                )
            )

        landmarks = _map_coco17_to_mediapipe33(coco_landmarks)
        nose_y = float(landmarks[0].y) if landmarks and landmarks[0].visibility > 0.01 else None
        return PoseObservation(
            frame_index=packet.frame_index,
            timestamp_ms=packet.timestamp_ms,
            pose_detected=bool(landmarks and landmarks[0].visibility > 0.01),
            nose_y=nose_y,
            landmarks=landmarks if landmarks else None,
        )

    def _best_person_bbox(self, frame_bgr) -> list[float] | None:
        results = self._person_detector.predict(frame_bgr, classes=[0], verbose=False)
        if not results:
            return None
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or getattr(boxes, "xyxy", None) is None or len(boxes.xyxy) == 0:
            return None

        best_idx = 0
        if getattr(boxes, "conf", None) is not None and len(boxes.conf) > 0:
            best_idx = int(boxes.conf.argmax().item())

        xyxy = boxes.xyxy[best_idx].cpu().numpy().tolist()
        if len(xyxy) != 4:
            return None
        return [float(v) for v in xyxy]

    def close(self) -> None:
        return None


def create_pose_estimator(backend: str, model_path: str) -> PoseEstimator:
    backend_norm = backend.strip().lower()
    if backend_norm == "mediapipe":
        return MediaPipePoseEstimator(model_path=model_path)
    if backend_norm == "yolo":
        return YoloPoseEstimator(model_path=model_path)
    if backend_norm == "vitpose":
        return VitPoseEstimator(model_path=model_path)
    raise ValueError(f"Unknown pose backend: {backend}. Expected: mediapipe|yolo|vitpose")


def _map_coco17_to_mediapipe33(coco: list[Landmark]) -> list[Landmark]:
    """Map YOLO COCO-17 keypoints into MediaPipe-like 33-index slots."""
    if len(coco) < 17:
        return []

    out: list[Landmark] = [Landmark(0.0, 0.0, 0.0, 0.0, 0.0) for _ in range(33)]

    def put(dst_idx: int, src_idx: int) -> None:
        out[dst_idx] = coco[src_idx]

    # Core face points
    put(0, 0)   # nose
    put(1, 1)   # left_eye_inner (approx)
    put(2, 1)   # left_eye
    put(3, 1)   # left_eye_outer (approx)
    put(4, 2)   # right_eye_inner (approx)
    put(5, 2)   # right_eye
    put(6, 2)   # right_eye_outer (approx)
    put(7, 3)   # left_ear
    put(8, 4)   # right_ear
    put(9, 0)   # mouth_left (approx)
    put(10, 0)  # mouth_right (approx)

    # Upper body
    put(11, 5)  # left_shoulder
    put(12, 6)  # right_shoulder
    put(13, 7)  # left_elbow
    put(14, 8)  # right_elbow
    put(15, 9)  # left_wrist
    put(16, 10) # right_wrist
    put(17, 9)  # left_pinky (approx)
    put(18, 10) # right_pinky (approx)
    put(19, 9)  # left_index (approx)
    put(20, 10) # right_index (approx)
    put(21, 9)  # left_thumb (approx)
    put(22, 10) # right_thumb (approx)

    # Lower body
    put(23, 11) # left_hip
    put(24, 12) # right_hip
    put(25, 13) # left_knee
    put(26, 14) # right_knee
    put(27, 15) # left_ankle
    put(28, 16) # right_ankle
    put(29, 15) # left_heel (approx)
    put(30, 16) # right_heel (approx)
    put(31, 15) # left_foot_index (approx)
    put(32, 16) # right_foot_index (approx)

    return out


def _xyxy_to_xywh(box_xyxy: list[float]) -> list[float]:
    x1, y1, x2, y2 = box_xyxy
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    return [float(x1), float(y1), float(w), float(h)]
