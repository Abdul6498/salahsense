"""Structured file logger for phase-1 debugging."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from salahsense.pose import PoseObservation
from salahsense.state_machine import VerticalState

LANDMARK_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]


class SessionLogger:
    """Write JSONL records for every frame and key events."""

    def __init__(self, log_path: str) -> None:
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")

    def log_startup(self, *, video_path: str, model_path: str, config_path: str) -> None:
        self._write(
            {
                "event": "startup",
                "video_path": video_path,
                "model_path": model_path,
                "config_path": config_path,
            }
        )

    def log_frame(
        self,
        *,
        frame_index: int,
        timestamp_ms: int,
        observation: PoseObservation,
        state: VerticalState,
        salah_state_english: str,
        salah_state_arabic: str,
        rakat_count: int,
        current_rakat: int,
        matched_pattern: list[str],
    ) -> None:
        self._write(
            {
                "event": "frame",
                "frame_index": frame_index,
                "timestamp_ms": timestamp_ms,
                "pose_detected": observation.pose_detected,
                "nose_y": observation.nose_y,
                "state_level": state.level.value,
                "state_direction": state.direction.value,
                "state_level_changed": state.level_changed,
                "salah_state_english": salah_state_english,
                "salah_state_arabic": salah_state_arabic,
                "rakat_count": rakat_count,
                "current_rakat": current_rakat,
                "matched_pattern": matched_pattern,
                "landmarks": self._serialize_landmarks(observation.landmarks),
            }
        )

    def log_transition(
        self,
        *,
        frame_index: int,
        level: str,
        matched_pattern: list[str],
        rakat_count: int,
        current_rakat: int,
        stage: str,
        reason: str,
    ) -> None:
        self._write(
            {
                "event": "transition",
                "frame_index": frame_index,
                "level": level,
                "matched_pattern": matched_pattern,
                "rakat_count": rakat_count,
                "current_rakat": current_rakat,
                "stage": stage,
                "reason": reason,
            }
        )

    def log_summary(self, *, final_rakat_count: int) -> None:
        self._write({"event": "summary", "final_rakat_count": final_rakat_count})

    def close(self) -> None:
        self._file.close()

    def _serialize_landmarks(self, landmarks: list | None) -> list[dict]:
        if not landmarks:
            return []

        rows: list[dict] = []
        for idx, lm in enumerate(landmarks):
            rows.append(
                {
                    "index": idx,
                    "name": LANDMARK_NAMES[idx] if idx < len(LANDMARK_NAMES) else f"point_{idx}",
                    "x": float(lm.x),
                    "y": float(lm.y),
                    "z": float(lm.z),
                    "visibility": float(getattr(lm, "visibility", 0.0)),
                    "presence": float(getattr(lm, "presence", 0.0)),
                }
            )
        return rows

    def _write(self, payload: dict) -> None:
        payload["logged_at_utc"] = datetime.now(timezone.utc).isoformat()
        self._file.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self._file.flush()
