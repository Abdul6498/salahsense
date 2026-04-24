"""Configuration loading utilities for SalahSense.

The goal is to keep runtime behavior tunable from TOML while keeping code easy to read.
"""

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class ThresholdSettings:
    """Vertical threshold values in normalized image coordinates."""

    high_y: float
    mid_y: float
    low_y: float
    mid_tolerance: float
    direction_delta: float


@dataclass(frozen=True)
class RuntimeSettings:
    """Runtime controls for video processing and logging."""

    process_width: int
    frame_log_interval: int


@dataclass(frozen=True)
class AppSettings:
    """Top-level typed settings used by the phase-1 pipeline."""

    profile_name: str
    thresholds: ThresholdSettings
    runtime: RuntimeSettings


def load_settings(path: str) -> AppSettings:
    """Load app settings from a TOML file."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")

    with file_path.open("rb") as f:
        data = tomllib.load(f)

    thresholds_data = data.get("thresholds", {})
    runtime_data = data.get("runtime", {})
    profile_name = data.get("profile_name", "default")

    thresholds = ThresholdSettings(
        high_y=float(thresholds_data["high_y"]),
        mid_y=float(thresholds_data["mid_y"]),
        low_y=float(thresholds_data["low_y"]),
        mid_tolerance=float(thresholds_data.get("mid_tolerance", 0.06)),
        direction_delta=float(thresholds_data.get("direction_delta", 0.004)),
    )
    _validate_thresholds(thresholds)

    runtime = RuntimeSettings(
        process_width=int(runtime_data.get("process_width", 640)),
        frame_log_interval=int(runtime_data.get("frame_log_interval", 15)),
    )

    return AppSettings(profile_name=profile_name, thresholds=thresholds, runtime=runtime)


def _validate_thresholds(thresholds: ThresholdSettings) -> None:
    if not (thresholds.high_y < thresholds.mid_y < thresholds.low_y):
        raise ValueError("Invalid thresholds: expected high_y < mid_y < low_y")

    if thresholds.mid_tolerance <= 0:
        raise ValueError("Invalid thresholds: mid_tolerance must be > 0")

    if thresholds.direction_delta <= 0:
        raise ValueError("Invalid thresholds: direction_delta must be > 0")
