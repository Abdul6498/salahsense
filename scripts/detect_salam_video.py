#!/usr/bin/env python3
"""Salam-only detector debug runner.

This isolates salam detection from rakat/state logic so you can tune quickly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from salahsense.capture import VideoReader
from salahsense.counting import SalamDetector
from salahsense.face import FaceLandmarkerYawEstimator
from salahsense.output import draw_face_landmarks


FACE_MODEL_PATH = "models/face_landmarker.task"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect salam from video using face landmarks")
    parser.add_argument("--video", required=True, help="Path to local input video")
    parser.add_argument(
        "--face-model",
        default=FACE_MODEL_PATH,
        help="Path to face landmarker .task model",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=720,
        help="Processing width",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not Path(args.face_model).exists():
        raise FileNotFoundError(
            f"Face model not found: {args.face_model}. "
            "Download it with: "
            "wget -O models/face_landmarker.task -q "
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )

    reader = VideoReader(video_path=args.video, process_width=args.width)
    face = FaceLandmarkerYawEstimator(model_path=args.face_model)
    salam = SalamDetector(expected_first_turn_direction="LEFT")

    paused = False
    should_stop = False

    print(f"[INFO] Video: {args.video}")
    print(f"[INFO] Face model: {args.face_model}")
    print("[INFO] Salam-only mode enabled")

    try:
        for packet in reader.frames():
            face_obs = face.detect(packet)
            update = salam.update(
                enabled=True,
                yaw_score=face_obs.yaw_score,
            )

            if face_obs.face_detected and face_obs.landmarks is not None:
                draw_face_landmarks(packet.frame_bgr, face_obs.landmarks)

            yaw_text = f"{update.yaw_score:.3f}" if update.yaw_score is not None else "N/A"
            lines = [
                f"Face Detected: {'YES' if face_obs.face_detected else 'NO'}",
                f"Yaw Score: {yaw_text}",
                f"Salam Stage: {update.stage.value}",
                f"First Turn: {update.first_turn_direction or 'N/A'}",
                f"Prayer Finished: {'YES' if update.prayer_finished else 'NO'}",
                "Space: Play/Pause | q: Quit",
            ]

            y = 24
            for line in lines:
                cv2.putText(
                    packet.frame_bgr,
                    line,
                    (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (245, 245, 245),
                    1,
                    cv2.LINE_AA,
                )
                y += 24

            cv2.imshow("SalahSense Salam Debug (press q to quit)", packet.frame_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                should_stop = True
                break
            if key == ord(" "):
                paused = not paused

            while paused:
                key = cv2.waitKey(80) & 0xFF
                if key == ord(" "):
                    paused = False
                elif key == ord("q"):
                    should_stop = True
                    paused = False
                    break

            if should_stop:
                break
    finally:
        face.close()
        reader.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
