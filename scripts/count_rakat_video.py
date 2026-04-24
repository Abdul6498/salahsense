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
from salahsense.counting import RakatCounter
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
from salahsense.state_machine import VerticalStateMachine


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
    state_machine = VerticalStateMachine(settings.thresholds)
    counter = RakatCounter()
    logger = SessionLogger(log_path=args.log_file)
    paused = False
    should_stop = False
    print(f"[INFO] Logging to: {args.log_file}")
    logger.log_startup(video_path=args.video, model_path=args.model, config_path=args.config)

    try:
        for packet in reader.frames():
            observation = estimator.detect(packet)
            state = state_machine.update(observation.nose_y)
            salah_state = state_catalog.resolve_runtime_state(
                level=state.level,
                direction=state.direction,
                stage=counter.stage,
            )

            if observation.pose_detected and observation.landmarks is not None:
                draw_pose_skeleton(packet.frame_bgr, observation.landmarks)

            if state.level_changed:
                update = counter.on_level_transition(state.level)
                print_transition(state.level, update)

                if update.completed_rakat:
                    print_rakat_completed(update.rakat_count)
                logger.log_transition(
                    frame_index=packet.frame_index,
                    level=state.level.value,
                    matched_pattern=[level.value for level in counter.matched_pattern],
                    rakat_count=counter.rakat_count,
                    current_rakat=counter.current_rakat,
                    stage=counter.stage.value,
                    reason=update.reason,
                )

            pattern_text = " -> ".join(level.value for level in counter.matched_pattern) or "(empty)"
            draw_top_overlay(
                packet.frame_bgr,
                rakat_count=counter.rakat_count,
                current_rakat=counter.current_rakat,
                stage=counter.stage.value,
                level=state.level.value,
                salah_state=f"{salah_state.english} ({salah_state.arabic})",
                direction=state.direction.value,
                nose_y=observation.nose_y,
                pattern=pattern_text,
            )
            cv2.imshow("SalahSense Phase-1 (press q to quit)", packet.frame_bgr)
            logger.log_frame(
                frame_index=packet.frame_index,
                timestamp_ms=packet.timestamp_ms,
                observation=observation,
                state=state,
                salah_state_english=salah_state.english,
                salah_state_arabic=salah_state.arabic,
                rakat_count=counter.rakat_count,
                current_rakat=counter.current_rakat,
                matched_pattern=[level.value for level in counter.matched_pattern],
            )

            if packet.frame_index % settings.runtime.frame_log_interval == 0:
                print_frame_debug(
                    frame_index=packet.frame_index,
                    nose_y=observation.nose_y,
                    level=state.level,
                    direction=state.direction,
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
        logger.log_summary(final_rakat_count=counter.rakat_count)
        logger.close()
        estimator.close()
        reader.close()
        cv2.destroyAllWindows()

    print_summary(counter.rakat_count)


if __name__ == "__main__":
    main()
