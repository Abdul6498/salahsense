"""Console output helpers for phase-1 pipeline."""

from __future__ import annotations

from salahsense.counting import CounterUpdate
from salahsense.state_machine import MovementDirection, VerticalLevel


def print_startup(video_path: str, model_path: str, config_path: str) -> None:
    print("[INFO] SalahSense Phase-1 Rakat Counter")
    print(f"[INFO] Video : {video_path}")
    print(f"[INFO] Model : {model_path}")
    print(f"[INFO] Config: {config_path}")


def print_frame_debug(
    frame_index: int,
    nose_y: float | None,
    level: VerticalLevel,
    direction: MovementDirection,
    salah_state: str,
) -> None:
    nose_text = f"{nose_y:.3f}" if nose_y is not None else "N/A"
    print(
        f"[FRAME {frame_index}] nose_y={nose_text} "
        f"level={level.value} direction={direction.value} salah_state={salah_state}"
    )


def print_transition(level: VerticalLevel, update: CounterUpdate) -> None:
    pattern_text = " -> ".join(item.value for item in update.matched_pattern) or "(empty)"
    print(
        f"[TRANSITION] level={level.value} stage={update.stage.value} "
        f"reason={update.reason} pattern={pattern_text} "
        f"completed={update.rakat_count} current={update.current_rakat}"
    )


def print_rakat_completed(rakat_count: int) -> None:
    print(f"[RAKAT] Completed rakat #{rakat_count}")


def print_summary(final_count: int) -> None:
    print(f"[SUMMARY] Final rakat count: {final_count}")
