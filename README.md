# SalahSense

SalahSense is a modular, beginner-friendly computer vision project to support Salah improvement.

The first milestone is a proof-of-concept vertical state machine that counts rakats using head Y-axis thresholds.

## Current Phase

Phase 1 focuses on rakat counting with this sequence:

`HIGH -> LOW -> MID -> LOW -> HIGH`

## Project Structure

```text
salahsense/
  config/                 # Runtime threshold and app configuration files
  docs/                   # Phase notes and design decisions
  scripts/                # Small helper scripts
  src/salahsense/
    capture/              # Image/video/camera input handlers
    pose/                 # Pose landmark extraction
    state_machine/        # Up/down state logic
    counting/             # Rakat sequence validation
    config/               # Config loaders and typed settings
    output/               # Terminal/LCD output adapters
  tests/                  # Automated tests
```

## Environment (WSL2 + NVIDIA)

This repo uses Python `venv` by default for the easiest local GPU workflow.

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Upgrade pip:

```bash
python -m pip install --upgrade pip
```

3. Install dependencies:

```bash
pip install -e .
```

## Quick Start

```bash
python -m salahsense.app
```

## Step-by-Step Milestones

1. Phase 1A: image calibration for `HIGH`, `MID`, `LOW`
2. Phase 1B: video state transition detection
3. Phase 1C: live camera processing

## Notes

- Keep all thresholds in config files, not hardcoded.
- Keep logic simple, explicit, and well-commented.
- This is assistive software, not a replacement for traditional learning or guidance.
