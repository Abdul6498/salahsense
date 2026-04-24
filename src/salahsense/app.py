"""Entry point for the SalahSense application.

For now this is a scaffold entry point. We will wire Phase 1 modules step-by-step.
"""

from salahsense.config.settings import AppSettings, load_settings


def main() -> None:
    """Run the basic startup flow and print current thresholds."""
    settings: AppSettings = load_settings("config/thresholds.toml")

    print("SalahSense initialized.")
    print(f"Profile: {settings.profile_name}")
    print(
        "Thresholds -> "
        f"HIGH: {settings.thresholds.high_y}, "
        f"MID: {settings.thresholds.mid_y}, "
        f"LOW: {settings.thresholds.low_y}"
    )
    print(f"Process width: {settings.runtime.process_width}")
    print("Phase 1 runner: python scripts/count_rakat_video.py --video ... --model ...")


if __name__ == "__main__":
    main()
