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

## Phase 1 Video Counter

Use the first modular rakat counter pipeline on a local namaz video:

Before running, download both required MediaPipe task models:

```bash
mkdir -p models
wget -O models/pose_landmarker_lite.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
wget -O models/face_landmarker.task -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

```bash
python scripts/count_rakat_video.py \
  --video videos/namaz_1.mp4 \
  --model models/pose_landmarker_lite.task \
  --config config/thresholds.toml \
  --salah-states config/salah_states.json \
  --salah-sequences config/salah_sequences.json \
  --salah-type 2_rakat_prayer \
  --udp \
  --log-file logs/rakat_run.jsonl
```

`--log-file` writes detailed JSONL logs with per-frame state, transitions, rakat progress, and all pose landmarks (`index`, `name`, `x`, `y`, `z`, `visibility`, `presence`).

`--salah-type` chooses one of the configured prayer profiles (`2_rakat_prayer`, `3_rakat_prayer`, `4_rakat_prayer`).
The runner uses `salah_sequences.json` to track sequence progress and target rakat count.

UDP telemetry sends the same overlay fields every frame as JSON packets (`event=overlay_frame`) over interface `eth1` broadcast on port `5005`.
Telemetry is enabled only when `--udp` is present.

Quick UDP listener example:
```bash
nc -ul 5005
```

Pipeline modules used:
- `capture` -> frame reader and resizing
- `pose` -> MediaPipe pose estimation
- `state_machine` -> feature-based Salah FSM (`QIYAM -> RUKU -> QAUMA -> SUJUD_1 -> JALSA -> SUJUD_2 -> QIYAM_NEXT/TASHAHHUD`)
- `counting` -> rakat counting from validated Salah-state transitions
- `output` -> terminal logs and final rakat summary

## Salam-Only Debug Mode

To debug salam detection in isolation (without rakat/state logic), run:

```bash
python scripts/detect_salam_video.py \
  --video videos/namaz_1.mp4 \
  --face-model models/face_landmarker.task
```

This view shows:
- face landmarks
- yaw score
- salam stage progression
- prayer finished flag

## Step-by-Step Milestones

1. Phase 1A: image calibration for `HIGH`, `MID`, `LOW`
2. Phase 1B: video state transition detection
3. Phase 1C: live camera processing

## Notes

- Keep all thresholds in config files, not hardcoded.
- Keep logic simple, explicit, and well-commented.
- This is assistive software, not a replacement for traditional learning or guidance.
