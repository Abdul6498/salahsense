"""On-frame overlay helpers for video visualization."""

from __future__ import annotations

import cv2

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


def draw_pose_skeleton(frame: cv2.typing.MatLike, landmarks: list) -> None:
    """Draw pose lines and points on the frame."""
    frame_h, frame_w = frame.shape[:2]
    points: list[tuple[int, int] | None] = []
    for landmark in landmarks:
        visibility = float(getattr(landmark, "visibility", 1.0))
        if visibility < 0.01:
            points.append(None)
            continue
        x_px = int(landmark.x * frame_w)
        y_px = int(landmark.y * frame_h)
        points.append((x_px, y_px))

    for start_idx, end_idx in POSE_CONNECTIONS:
        if (
            start_idx < len(points)
            and end_idx < len(points)
            and points[start_idx] is not None
            and points[end_idx] is not None
        ):
            cv2.line(frame, points[start_idx], points[end_idx], (255, 255, 255), 2)

    for point in points:
        if point is None:
            continue
        x_px, y_px = point
        cv2.circle(frame, (x_px, y_px), 3, (0, 120, 255), thickness=-1)


def draw_top_overlay(
    frame: cv2.typing.MatLike,
    *,
    prayer_name: str,
    target_rakats: int,
    rakat_count: int,
    current_rakat: int,
    fsm_state: str,
    posture: str,
    salah_state: str,
    next_expected_state: str,
    sequence_progress_text: str,
    reason: str,
    nose_y: float | None,
    missing_states_text: str,
) -> None:
    """Draw small transparent text at the top of the frame."""
    nose_text = f"{nose_y:.3f}" if nose_y is not None else "N/A"
    target_color = (0, 220, 0) if rakat_count == target_rakats else (0, 0, 255)

    lines = [
        f"Prayer: {prayer_name}",
        f"Target Rakats: {target_rakats}",
        f"Completed Rakats: {rakat_count}",
        f"Current Rakat: {current_rakat}",
        f"Salah: {salah_state}",
        f"Next Expected: {next_expected_state}",
        f"Sequence: {sequence_progress_text}",
        f"FSM State: {fsm_state}",
        f"Detected Posture: {posture}",
        f"Reason: {reason}",
        f"Missing: {missing_states_text}",
        f"Nose Y: {nose_text}",
        "Space: Play/Pause | q: Quit",
    ]

    y = 18
    for idx, line in enumerate(lines):
        color = target_color if idx == 1 else (245, 245, 245)
        cv2.putText(
            frame,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )
        y += 18
