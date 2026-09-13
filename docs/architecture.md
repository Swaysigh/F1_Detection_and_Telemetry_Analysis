# Architecture Notes

## Pipeline overview

```
Video frame
   │
   ▼
[YOLO Detection] ──► bounding boxes (car locations)
   │
   ▼
[Crop + Classifier (ResNet50/EfficientNet)] ──► driver/team ID per box
   │
   ▼
[Tracker] ──► persistent track ID across frames
   │
   ▼
[Homography] ──► pixel coords -> real-world track coords
   │
   ▼
[Speed / Position estimation] ──► km/h, race position
   │
   ▼
Telemetry output (json/csv) + annotated video
```

## Open design questions

- Single-stage YOLO for car-only detection, or multi-class (car + driver number/livery)?
- Classifier granularity: per-driver vs per-team (fewer classes, more robust if livery is main cue).
- Homography: fixed per camera angle, or recomputed per broadcast shot change?
- Tracker: build custom IoU/Kalman tracker vs wrap existing ByteTrack/DeepSORT implementation.

## Decisions log

(fill in as choices are made, with brief reasoning)
