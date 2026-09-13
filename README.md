# F1 Vision Telemetry

Computer vision pipeline to detect F1 cars, identify drivers/teams, track them across frames, and estimate real-time telemetry (position, speed) from race footage.

## Pipeline

1. **Detection** (`src/detection/`) — YOLO locates cars in each frame.
2. **Classification** (`src/classification/`) — ResNet50/EfficientNet identifies driver/team from cropped car images.
3. **Tracking** (`src/tracking/`) — assigns persistent IDs across frames.
4. **Telemetry** (`src/telemetry/`) — homography maps pixel coords to real-world track coords; speed and race position are derived from that.
5. **Pipeline** (`src/pipeline.py`) — orchestrates the full flow end to end.

## Project structure

```
configs/         model & pipeline configs (YAML)
data/            raw / interim / processed data, annotations (gitignored)
models/          checkpoints & pretrained weights (gitignored)
src/             core source code, organized by pipeline stage
scripts/         CLI entrypoints (data prep, full pipeline run)
notebooks/       exploratory work
tests/           unit tests
outputs/         logs, predictions, annotated videos (gitignored)
docs/            architecture notes
```

## Setup

```bash
python -m venv venv
source venv/bin/activate         # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage (once modules are filled in)

```bash
# Train detector
python src/detection/train_yolo.py

# Train classifier
python src/classification/train_classifier.py

# Run full pipeline on a video
python scripts/run_pipeline.py --video path/to/race_clip.mp4
```

## Status

Early scaffold — pipeline stages are stubbed out with clear interfaces. Detection and classification modules to be filled in first, tracking + telemetry after.
