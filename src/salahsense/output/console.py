"""Console output helpers for phase-1 pipeline."""

from __future__ import annotations


def print_startup(video_path: str, model_path: str, config_path: str) -> None:
    print("[INFO] SalahSense Phase-1 Rakat Counter")
    print(f"[INFO] Video : {video_path}")
    print(f"[INFO] Model : {model_path}")
    print(f"[INFO] Config: {config_path}")


def print_frame_debug(
    frame_index: int,
    nose_y: float | None,
    posture: str,
    fsm_state: str,
    salah_state: str,
) -> None:
    nose_text = f"{nose_y:.3f}" if nose_y is not None else "N/A"
    print(
        f"[FRAME {frame_index}] nose_y={nose_text} "
        f"posture={posture} fsm_state={fsm_state} salah_state={salah_state}"
    )


def print_transition(
    *,
    state_name: str,
    reason: str,
    completed_rakats: int,
    current_rakat: int,
) -> None:
    print(
        f"[TRANSITION] state={state_name} reason={reason} "
        f"completed={completed_rakats} current={current_rakat}"
    )


def print_rakat_completed(rakat_count: int) -> None:
    print(f"[RAKAT] Completed rakat #{rakat_count}")


def print_summary(final_count: int) -> None:
    print(f"[SUMMARY] Final rakat count: {final_count}")


def print_missing_states(missing_states: list[str]) -> None:
    if not missing_states:
        return
    print(f"[MISSING] {' -> '.join(missing_states)}")
