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
        f"HIGH: {settings.high_y}, MID: {settings.mid_y}, LOW: {settings.low_y}"
    )
    print("Next step: implement Phase 1A image calibration.")


if __name__ == "__main__":
    main()
