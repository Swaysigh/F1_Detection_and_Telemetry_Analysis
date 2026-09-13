"""Run YOLO inference on frames/video to get car bounding boxes."""
from ultralytics import YOLO


def load_model(weights_path):
    return YOLO(weights_path)


def detect(model, frame, conf=0.4):
    """Returns list of detections: [{bbox, conf, class_id}, ...]"""
    results = model.predict(frame, conf=conf, verbose=False)
    return results


if __name__ == "__main__":
    pass
