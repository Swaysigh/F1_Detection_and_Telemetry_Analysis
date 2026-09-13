"""End-to-end inference pipeline: detection -> classification -> tracking -> telemetry."""


def run_pipeline(video_path, yolo_model, classifier_model, tracker, cfg):
    """
    High-level orchestration. TODO:
      1. Read frames from video_path
      2. Run YOLO detection per frame
      3. Crop detections, run classifier for driver/team ID
      4. Update tracker with detections -> persistent IDs
      5. Apply homography to get world coords -> speed/position
      6. Aggregate + output telemetry (json/csv) and/or annotated video
    """
    raise NotImplementedError
