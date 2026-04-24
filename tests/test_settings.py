"""Basic tests for settings loading."""

from pathlib import Path

from salahsense.config.settings import load_settings


def test_load_settings_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    settings = load_settings(str(repo_root / "config" / "thresholds.toml"))

    assert settings.thresholds.high_y < settings.thresholds.mid_y < settings.thresholds.low_y
    assert settings.runtime.process_width > 0
