#!/usr/bin/env python3
"""Simple learning script: read a video and check pose detection.

Usage:
    python scripts/video_pose_reader.py --video /path/to/video.mp4 --model /path/to/model.task

Notes:
- Keep CLI simple: `--video` and optional `--model`
- `--model` is used only for MediaPipe Tasks backend
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Point Qt to system fonts to avoid cv2 Qt font warnings in many WSL setups.
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
import mediapipe as mp

DEFAULT_MODEL_PATH = Path("models/pose_landmarker.task")
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
PROCESS_WIDTH = 640

LANDMARK_NAMES = [
    "nose",
    "left eye (inner)",
    "left eye",
    "left eye (outer)",
    "right eye (inner)",
    "right eye",
    "right eye (outer)",
    "left ear",
    "right ear",
    "mouth (left)",
    "mouth (right)",
    "left shoulder",
    "right shoulder",
    "left elbow",
    "right elbow",
    "left wrist",
    "right wrist",
    "left pinky",
    "right pinky",
    "left index",
    "right index",
    "left thumb",
    "right thumb",
    "left hip",
    "right hip",
    "left knee",
    "right knee",
    "left ankle",
    "right ankle",
    "left heel",
    "right heel",
    "left foot index",
    "right foot index",
]

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (31, 32),
    (27, 31), (28, 32),
]


def build_parser() -> argparse.ArgumentParser:
    """Create minimal CLI arguments."""
    parser = argparse.ArgumentParser(description="Read a video and visualize pose detection")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to pose_landmarker.task (used by MediaPipe Tasks backend)",
    )
    return parser


def draw_status_overlay(
    frame: cv2.typing.MatLike,
    pose_detected: bool,
    nose_y: float | None,
    backend_name: str,
) -> None:
    """Draw a simple status panel on top of the frame."""
    status_text = f"Pose detected: {'YES' if pose_detected else 'NO'}"
    nose_text = (
        f"Nose Y (normalized): {nose_y:.3f}" if nose_y is not None else "Nose Y (normalized): N/A"
    )

    #cv2.rectangle(frame, (10, 10), (560, 120), (20, 20, 20), thickness=-1)
    #cv2.putText(frame, f"Backend: {backend_name}", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
    #cv2.putText(frame, status_text, (20, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    #cv2.putText(frame, nose_text, (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def print_landmark_reference() -> None:
    """Print landmark index-to-name mapping for learning/debug."""
    print("[INFO] Pose landmark index reference:")
    for idx, name in enumerate(LANDMARK_NAMES):
        print(f"  {idx:2d} -> {name}")


def print_landmark_values(landmarks: list) -> None:
    """Print detailed coordinates once to help understand landmark output."""
    print("[INFO] Landmark values from first detected pose:")
    for idx, landmark in enumerate(landmarks):
        name = LANDMARK_NAMES[idx] if idx < len(LANDMARK_NAMES) else f"point_{idx}"
        visibility = getattr(landmark, "visibility", None)
        if visibility is None:
            print(f"  {idx:2d} {name:18s} x={landmark.x:.3f} y={landmark.y:.3f} z={landmark.z:.3f}")
        else:
            print(
                f"  {idx:2d} {name:18s} "
                f"x={landmark.x:.3f} y={landmark.y:.3f} z={landmark.z:.3f} vis={visibility:.3f}"
            )


def draw_task_landmarks(frame: cv2.typing.MatLike, landmarks: list) -> None:
    """Draw skeleton lines + landmarks + index labels."""
    frame_h, frame_w = frame.shape[:2]
    points: list[tuple[int, int]] = []
    for landmark in landmarks:
        x_px = int(landmark.x * frame_w)
        y_px = int(landmark.y * frame_h)
        points.append((x_px, y_px))

    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(frame, points[start_idx], points[end_idx], (255, 255, 255), 2)

    for idx, (x_px, y_px) in enumerate(points):
        cv2.circle(frame, (x_px, y_px), 4, (0, 120, 255), thickness=-1)
        cv2.putText(
            frame,
            str(idx),
            (x_px + 4, y_px - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 0),
            1,
        )


def resize_frame(frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """Resize frame to a lower width while keeping aspect ratio."""
    height, width = frame.shape[:2]
    if width <= PROCESS_WIDTH:
        return frame

    scale = PROCESS_WIDTH / float(width)
    new_size = (PROCESS_WIDTH, int(height * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def run_with_legacy_solutions(cap: cv2.VideoCapture) -> None:
    """Run using legacy mp.solutions.pose API."""
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    frame_index = 0
    print("[INFO] Using backend: mediapipe.solutions")
    print("[INFO] Starting frame loop...")

    with mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[INFO] Reached end of video stream.")
                break

            frame = resize_frame(frame)
            frame_index += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            pose_detected = results.pose_landmarks is not None
            nose_y = None

            if pose_detected:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                nose_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                nose_y = float(nose_landmark.y)

            if frame_index % 15 == 0:
                print(
                    f"[FRAME {frame_index}] detected={pose_detected} "
                    f"nose_y={f'{nose_y:.3f}' if nose_y is not None else 'N/A'}"
                )

            draw_status_overlay(frame, pose_detected, nose_y, backend_name="mediapipe.solutions")
            cv2.imshow("Pose Check (press q to quit)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] User requested quit (q).")
                break


def create_task_landmarker(model_path: Path):
    """Create PoseLandmarker (CPU) in VIDEO mode for stable frame-by-frame processing."""
    from mediapipe.tasks import python as mp_tasks_python
    from mediapipe.tasks.python import vision as mp_vision

    print(f"[INFO] Preparing task model from: {model_path}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Download pose_landmarker.task and place it at models/pose_landmarker.task"
        )

    def build_options(delegate):
        base_options = mp_tasks_python.BaseOptions(
            model_asset_path=str(model_path),
            delegate=delegate,
        )
        return mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            min_pose_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )

    print("[INFO] Using CPU delegate for stable VIDEO-mode inference.")
    cpu_options = build_options(mp_tasks_python.BaseOptions.Delegate.CPU)
    landmarker = mp_vision.PoseLandmarker.create_from_options(cpu_options)
    print("[INFO] CPU delegate initialized successfully.")
    return landmarker, "mediapipe.tasks (CPU)"


def run_with_tasks_backend(cap: cv2.VideoCapture, model_path: Path) -> None:
    """Run using MediaPipe Tasks Pose Landmarker API."""
    landmarker, backend_name = create_task_landmarker(model_path)
    frame_index = 0
    printed_landmark_values = False
    print(f"[INFO] Using backend: {backend_name}")
    print("[INFO] Starting frame loop...")
    print_landmark_reference()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[INFO] Reached end of video stream.")
                break

            frame = resize_frame(frame)
            frame_index += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = frame_index * 33
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            pose_detected = len(result.pose_landmarks) > 0
            nose_y = None

            if pose_detected:
                first_pose = result.pose_landmarks[0]
                draw_task_landmarks(frame, first_pose)
                nose_y = float(first_pose[0].y)
                if not printed_landmark_values:
                    print_landmark_values(first_pose)
                    printed_landmark_values = True

            if frame_index % 15 == 0:
                print(
                    f"[FRAME {frame_index}] detected={pose_detected} "
                    f"nose_y={f'{nose_y:.3f}' if nose_y is not None else 'N/A'}"
                )

            draw_status_overlay(frame, pose_detected, nose_y, backend_name=backend_name)
            cv2.imshow("Pose Check (press q to quit)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] User requested quit (q).")
                break
    finally:
        print("[INFO] Closing PoseLandmarker.")
        landmarker.close()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    video_path = Path(args.video)
    model_path = Path(args.model)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    print(f"[INFO] Video path: {video_path}")
    print(f"[INFO] Model path: {model_path}")
    print(f"[INFO] Processing width: {PROCESS_WIDTH}px")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        if hasattr(mp, "solutions"):
            run_with_legacy_solutions(cap)
        else:
            run_with_tasks_backend(cap, model_path)
    finally:
        print("[INFO] Releasing video resources.")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
