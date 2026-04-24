"""Configuration loading utilities for SalahSense.

We intentionally keep this simple so thresholds are easy to tweak while learning.
"""

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class AppSettings:
    """Typed settings used by the app."""

    profile_name: str
    high_y: float
    mid_y: float
    low_y: float


def load_settings(path: str) -> AppSettings:
    """Load app settings from a TOML file.

    Args:
        path: Path to threshold config file.

    Returns:
        AppSettings object with validated values.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")

    with file_path.open("rb") as f:
        data = tomllib.load(f)

    thresholds = data.get("thresholds", {})
    profile_name = data.get("profile_name", "default")

    high_y = float(thresholds["high_y"])
    mid_y = float(thresholds["mid_y"])
    low_y = float(thresholds["low_y"])

    if not (high_y < mid_y < low_y):
        raise ValueError(
            "Invalid thresholds: expected high_y < mid_y < low_y in image coordinates"
        )

    return AppSettings(
        profile_name=profile_name,
        high_y=high_y,
        mid_y=mid_y,
        low_y=low_y,
    )
