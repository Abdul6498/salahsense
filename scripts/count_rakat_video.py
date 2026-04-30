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
from salahsense.config.salah_sequences import SalahSequenceCatalog
from salahsense.config.salah_states import SalahStateCatalog
from salahsense.config.settings import load_settings
from salahsense.counting import SalahSequenceTracker
from salahsense.output import (
    SessionLogger,
    UdpTelemetrySender,
    draw_pose_skeleton,
    draw_top_overlay,
    print_frame_debug,
    print_missing_states,
    print_rakat_completed,
    print_startup,
    print_summary,
    print_transition,
)
from salahsense.pose import create_pose_estimator
from salahsense.state_machine import SalahStateMachine

UDP_INTERFACE_NAME = "eth1"
UDP_PORT = 5005


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Count rakats from a video")
    parser.add_argument("--video", required=True, help="Path to local input video")
    parser.add_argument(
        "--model",
        default="models/pose_landmarker_full.task",
        help="Pose model reference (.task for mediapipe, .pt for yolo, HF id/path for vitpose)",
    )
    parser.add_argument(
        "--pose-backend",
        default="mediapipe",
        choices=["mediapipe", "yolo", "vitpose"],
        help="Pose backend to use",
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
    parser.add_argument(
        "--salah-sequences",
        default="config/salah_sequences.json",
        help="Path to Salah sequence definitions JSON",
    )
    parser.add_argument(
        "--salah-type",
        default="2_rakat_prayer",
        choices=["2_rakat_prayer", "3_rakat_prayer", "4_rakat_prayer"],
        help="Which salah profile to run against sequence/target logic",
    )
    parser.add_argument(
        "--udp",
        action="store_true",
        help="Enable UDP telemetry sender on interface eth1 (broadcast, port 5005)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    settings = load_settings(args.config)
    print_startup(video_path=args.video, model_path=args.model, config_path=args.config)
    state_catalog = SalahStateCatalog.from_json(args.salah_states)
    sequence_catalog = SalahSequenceCatalog.from_json(args.salah_sequences)
    sequence_profile = sequence_catalog.get_profile(args.salah_type)
    sequence_tracker = SalahSequenceTracker(sequence_profile)
    sequence_progress = sequence_tracker.progress()
    print(
        f"[INFO] Salah type: {sequence_profile.profile_key} "
        f"({sequence_profile.profile_name}), target_rakats={sequence_profile.expected_rakats}"
    )

    if args.pose_backend in {"mediapipe", "yolo"} and not Path(args.model).exists():
        raise FileNotFoundError(
            f"Model not found: {args.model}. "
            "For mediapipe/yolo, pass a local model file path."
        )

    reader = VideoReader(video_path=args.video, process_width=settings.runtime.process_width)
    estimator = create_pose_estimator(backend=args.pose_backend, model_path=args.model)
    # Lower stability window to keep transitions responsive in real-time use.
    # Temporary debug mode:
    # allow tashahhud transition after any rakat so we can validate
    # joint/hand-placement logic in isolation.
    tashahhud_after_rakats = {1, 2, 3, 4}
    state_machine = SalahStateMachine(
        min_stable_frames=2,
        tashahhud_after_rakats=tashahhud_after_rakats,
    )
    logger = SessionLogger(log_path=args.log_file)
    udp_sender = UdpTelemetrySender(
        interface_name=UDP_INTERFACE_NAME,
        port=UDP_PORT,
        enabled=args.udp,
    )
    previous_completed_rakats = 0
    target_reached_announced = False
    session_missing_states: list[str] = []
    paused = False
    should_stop = False
    print(f"[INFO] Logging to: {args.log_file}")
    print(
        f"[INFO] UDP telemetry: {'enabled' if args.udp else 'disabled'} "
        f"-> iface={UDP_INTERFACE_NAME}, broadcast={udp_sender.broadcast_ip}, port={UDP_PORT}"
    )
    logger.log_startup(
        video_path=args.video,
        model_path=args.model,
        config_path=args.config,
        salah_type=sequence_profile.profile_key,
        salah_name=sequence_profile.profile_name,
        target_rakats=sequence_profile.expected_rakats,
    )

    try:
        for packet in reader.frames():
            observation = estimator.detect(packet)
            update = state_machine.update(observation)
            salah_state = state_catalog.resolve_from_fsm(update.state)

            if observation.pose_detected and observation.landmarks is not None:
                draw_pose_skeleton(packet.frame_bgr, observation.landmarks)

            if update.state_changed:
                sequence_progress = sequence_tracker.on_state_change(update.state)
                if sequence_progress.missing_state_entries:
                    missing_labels = [
                        f"{entry.rakat_number}_{entry.state.value}"
                        for entry in sequence_progress.missing_state_entries
                    ]
                    for missing_label in missing_labels:
                        if missing_label not in session_missing_states:
                            session_missing_states.append(missing_label)
                    print_missing_states(missing_labels)
                logger.log_transition(
                    frame_index=packet.frame_index,
                    state_name=update.state.value,
                    rakat_count=sequence_progress.completed_rakats,
                    current_rakat=sequence_progress.current_rakat,
                    reason=update.reason,
                )
                print_transition(
                    state_name=update.state.value,
                    reason=update.reason,
                    completed_rakats=sequence_progress.completed_rakats,
                    current_rakat=sequence_progress.current_rakat,
                )
                if sequence_progress.completed_rakats > previous_completed_rakats:
                    print_rakat_completed(sequence_progress.completed_rakats)
                    previous_completed_rakats = sequence_progress.completed_rakats

            next_expected = (
                sequence_progress.next_expected_state.value
                if sequence_progress.next_expected_state is not None
                else "DONE"
            )
            sequence_progress_text = f"{sequence_progress.current_index}/{sequence_progress.total_states}"
            missing_text = " -> ".join(session_missing_states) if session_missing_states else "-"

            draw_top_overlay(
                packet.frame_bgr,
                prayer_name=sequence_profile.profile_name,
                target_rakats=sequence_profile.expected_rakats,
                rakat_count=sequence_progress.completed_rakats,
                current_rakat=sequence_progress.current_rakat,
                fsm_state=update.state.value,
                posture=update.detected_posture.value,
                salah_state=f"{salah_state.english} ({salah_state.arabic})",
                next_expected_state=next_expected,
                sequence_progress_text=sequence_progress_text,
                reason=update.reason,
                nose_y=observation.nose_y,
                missing_states_text=missing_text,
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
                salah_type=sequence_profile.profile_key,
                target_rakats=sequence_profile.expected_rakats,
                sequence_index=sequence_progress.current_index,
                sequence_total=sequence_progress.total_states,
                next_expected_state=next_expected,
                rakat_count=sequence_progress.completed_rakats,
                current_rakat=sequence_progress.current_rakat,
            )
            udp_sender.send(
                {
                    "event": "overlay_frame",
                    "frame_index": packet.frame_index,
                    "timestamp_ms": packet.timestamp_ms,
                    "prayer_name": sequence_profile.profile_name,
                    "target_rakats": sequence_profile.expected_rakats,
                    "completed_rakats": sequence_progress.completed_rakats,
                    "current_rakat": sequence_progress.current_rakat,
                    "salah_state_english": salah_state.english,
                    "salah_state_arabic": salah_state.arabic,
                    "fsm_state": update.state.value,
                    "detected_posture": update.detected_posture.value,
                    "reason": update.reason,
                    "nose_y": observation.nose_y,
                    "next_expected_state": next_expected,
                    "sequence_progress": sequence_progress_text,
                }
            )

            if packet.frame_index % settings.runtime.frame_log_interval == 0:
                print_frame_debug(
                    frame_index=packet.frame_index,
                    nose_y=observation.nose_y,
                    posture=update.detected_posture.value,
                    fsm_state=update.state.value,
                    salah_state=f"{salah_state.english} ({salah_state.arabic})",
                )

            if (
                not target_reached_announced
                and sequence_progress.completed_rakats >= sequence_profile.expected_rakats
            ):
                print(
                    f"[INFO] Target reached for {sequence_profile.profile_name}: "
                    f"{sequence_progress.completed_rakats}/{sequence_profile.expected_rakats} rakats."
                )
                target_reached_announced = True

            if sequence_progress.completed_rakats > sequence_profile.expected_rakats:
                print(
                    f"[WARN] Extra rakat detected: {sequence_progress.completed_rakats} "
                    f"(target {sequence_profile.expected_rakats})."
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
        logger.log_summary(
            final_rakat_count=sequence_progress.completed_rakats if "sequence_progress" in locals() else 0
        )
        logger.close()
        udp_sender.close()
        estimator.close()
        reader.close()
        cv2.destroyAllWindows()

    print_summary(sequence_progress.completed_rakats if "sequence_progress" in locals() else 0)


if __name__ == "__main__":
    main()
