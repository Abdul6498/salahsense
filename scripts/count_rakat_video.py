#!/usr/bin/env python3
"""Phase-1 runner: count rakats from a namaz video.

This script uses the modular SalahSense pipeline:
- capture (video reader)
- pose (MediaPipe estimator)
- state machine (vertical level + movement direction)
- counting (rakat sequence matcher)
- output (console logs)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from salahsense.capture import VideoReader
from salahsense.config.salah_states import SalahStateCatalog
from salahsense.config.settings import load_settings
from salahsense.output import (
    SessionLogger,
    draw_pose_skeleton,
    draw_top_overlay,
    print_frame_debug,
    print_rakat_completed,
    print_startup,
    print_summary,
    print_transition,
)
from salahsense.pose import PoseEstimator
from salahsense.state_machine import SalahStateMachine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Count rakats from a video")
    parser.add_argument("--video", required=True, help="Path to local input video")
    parser.add_argument(
        "--model",
        default="models/pose_landmarker_full.task",
        help="Path to pose landmarker .task model",
    )
    parser.add_argument(
        "--config",
        default="config/thresholds.toml",
        help="Path to threshold config TOML",
    )
    parser.add_argument(
        "--log-file",
        default="logs/rakat_run.jsonl",
        help="Path to JSONL debug log file",
    )
    parser.add_argument(
        "--salah-states",
        default="config/salah_states.json",
        help="Path to Salah state names/definitions JSON",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    settings = load_settings(args.config)
    print_startup(video_path=args.video, model_path=args.model, config_path=args.config)
    state_catalog = SalahStateCatalog.from_json(args.salah_states)

    if not Path(args.model).exists():
        raise FileNotFoundError(
            f"Model not found: {args.model}. Put a pose_landmarker .task model at that path."
        )

    reader = VideoReader(video_path=args.video, process_width=settings.runtime.process_width)
    estimator = PoseEstimator(model_path=args.model)
    # Lower stability window to keep transitions responsive in real-time use.
    state_machine = SalahStateMachine(min_stable_frames=2)
    logger = SessionLogger(log_path=args.log_file)
    previous_completed_rakats = 0
    paused = False
    should_stop = False
    print(f"[INFO] Logging to: {args.log_file}")
    logger.log_startup(video_path=args.video, model_path=args.model, config_path=args.config)

    try:
        for packet in reader.frames():
            observation = estimator.detect(packet)
            update = state_machine.update(observation)
            salah_state = state_catalog.resolve_from_fsm(update.state)

            if update.completed_rakats > previous_completed_rakats:
                print_rakat_completed(update.completed_rakats)
                previous_completed_rakats = update.completed_rakats

            if observation.pose_detected and observation.landmarks is not None:
                draw_pose_skeleton(packet.frame_bgr, observation.landmarks)

            if update.state_changed:
                logger.log_transition(
                    frame_index=packet.frame_index,
                    state_name=update.state.value,
                    rakat_count=update.completed_rakats,
                    current_rakat=update.current_rakat,
                    reason=update.reason,
                )
            if update.state_changed:
                print_transition(
                    state_name=update.state.value,
                    reason=update.reason,
                    completed_rakats=update.completed_rakats,
                    current_rakat=update.current_rakat,
                )

            draw_top_overlay(
                packet.frame_bgr,
                rakat_count=update.completed_rakats,
                current_rakat=update.current_rakat,
                fsm_state=update.state.value,
                posture=update.detected_posture.value,
                salah_state=f"{salah_state.english} ({salah_state.arabic})",
                reason=update.reason,
                nose_y=observation.nose_y,
            )
            cv2.imshow("SalahSense Phase-1 (press q to quit)", packet.frame_bgr)
            logger.log_frame(
                frame_index=packet.frame_index,
                timestamp_ms=packet.timestamp_ms,
                observation=observation,
                posture=update.detected_posture.value,
                fsm_state=update.state.value,
                state_changed=update.state_changed,
                transition_reason=update.reason,
                feature_snapshot={
                    "torso_from_vertical_deg": update.features.torso_from_vertical_deg,
                    "knee_angle_deg": update.features.knee_angle_deg,
                    "nose_minus_hip_y": update.features.nose_minus_hip_y,
                    "hip_mid_y": update.features.hip_mid_y,
                },
                salah_state_english=salah_state.english,
                salah_state_arabic=salah_state.arabic,
                rakat_count=update.completed_rakats,
                current_rakat=update.current_rakat,
            )

            if packet.frame_index % settings.runtime.frame_log_interval == 0:
                print_frame_debug(
                    frame_index=packet.frame_index,
                    nose_y=observation.nose_y,
                    posture=update.detected_posture.value,
                    fsm_state=update.state.value,
                    salah_state=f"{salah_state.english} ({salah_state.arabic})",
                )

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] User requested quit (q).")
                should_stop = True
                break
            if key == ord(" "):
                paused = not paused
                print("[INFO] Paused." if paused else "[INFO] Resumed.")

            while paused:
                key = cv2.waitKey(80) & 0xFF
                if key == ord(" "):
                    paused = False
                    print("[INFO] Resumed.")
                elif key == ord("q"):
                    print("[INFO] User requested quit (q).")
                    should_stop = True
                    paused = False
                    break

            if should_stop:
                break
    finally:
        logger.log_summary(final_rakat_count=update.completed_rakats if 'update' in locals() else 0)
        logger.close()
        estimator.close()
        reader.close()
        cv2.destroyAllWindows()

    print_summary(update.completed_rakats if 'update' in locals() else 0)


if __name__ == "__main__":
    main()
