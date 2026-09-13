"""Multi-object tracker (ByteTrack-style) to assign persistent IDs to detected cars."""


class Track:
    def __init__(self, track_id, bbox):
        self.track_id = track_id
        self.bbox = bbox
        self.history = [bbox]

    def update(self, bbox):
        self.bbox = bbox
        self.history.append(bbox)


class Tracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tracks = {}
        self.next_id = 0

    def update(self, detections):
        """
        Takes current-frame detections, matches to existing tracks,
        returns list of active Track objects with updated bboxes/ids.
        TODO: implement IoU/Kalman-based matching (or wrap ByteTrack/DeepSORT).
        """
        raise NotImplementedError
