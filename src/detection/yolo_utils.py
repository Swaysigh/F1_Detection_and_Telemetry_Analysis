"""Shared helpers for detection module (bbox conversion, NMS helpers, etc.)"""


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def crop_from_bbox(frame, bbox):
    """Crop a car region from a frame given a bounding box, for downstream classification."""
    x1, y1, x2, y2 = map(int, bbox)
    return frame[y1:y2, x1:x2]
