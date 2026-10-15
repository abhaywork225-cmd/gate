from __future__ import annotations
"""Class-aware temporal IoU association, replaceable independently of YOLO.

Deliberately small POC tracker; long occlusions require a calibrated stronger tracker.
Only update on measured inference frames, not skipped frames.
"""
from dataclasses import dataclass
from datetime import datetime
from app.domain import Detection


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0


@dataclass
class Track:
    id: int
    detection: Detection
    first_seen: datetime
    last_seen: datetime
    hits: int = 1
    direction: str = 'UNKNOWN'
    line_side: int = 0
    appearance: object = None


def appearance(image, bbox):
    if image is None:
        return None
    import cv2
    import numpy as np
    h, w = image.shape[:2]
    x1,y1,x2,y2 = map(int,bbox)
    crop = image[max(0,y1):min(h,y2),max(0,x1):min(w,x2)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(cv2.resize(crop,(48,48)),cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1,2],None,[12,4,4],[0,180,0,256,0,256]).reshape(-1)
    norm = float(np.linalg.norm(hist))
    return hist / norm if norm else None


def same_colour(first, second):
    if first is None or second is None:
        return False
    import numpy as np
    a, b = first.reshape(12,4,4).sum(axis=2).ravel(), second.reshape(12,4,4).sum(axis=2).ravel()
    denominator = float(np.linalg.norm(a)*np.linalg.norm(b))
    return denominator > 0 and float(a @ b)/denominator >= .98


def similar(first, second, minimum=.90):
    return first is None or second is None or float(first @ second) >= minimum


class TemporalTracker:
    def __init__(self, ttl_seconds=5, match_iou=0.15):
        self.ttl_seconds, self.match_iou = ttl_seconds, match_iou
        self.tracks = {}
        self._next = 1

    def update(self, detections: list[Detection], timestamp: datetime, image=None):
        self.tracks = {i: t for i, t in self.tracks.items() if (timestamp - t.last_seen).total_seconds() <= self.ttl_seconds}
        features = [appearance(image,d.bbox) for d in detections]
        pairs = sorted([
            (iou(track.detection.bbox, det.bbox), tid, index)
            for tid, track in self.tracks.items() for index, det in enumerate(detections)
            if (track.detection.class_name == det.class_name or
                {track.detection.class_name, det.class_name} <= {'truck', 'trailer', 'vehicle'})
            and (similar(track.appearance, features[index]) or
                 ((timestamp-track.last_seen).total_seconds() <= 1.0
                  and iou(track.detection.bbox, det.bbox) >= .85
                  and same_colour(track.appearance, features[index])))
        ], reverse=True)
        used_tracks, used_detections, observed = set(), set(), []
        for overlap, tid, index in pairs:
            if overlap < self.match_iou or tid in used_tracks or index in used_detections:
                continue
            track = self.tracks[tid]
            track.detection, track.last_seen = detections[index], timestamp
            track.appearance = features[index]
            track.hits += 1
            observed.append(track)
            used_tracks.add(tid)
            used_detections.add(index)
        for index, detection in enumerate(detections):
            if index not in used_detections:
                track = Track(self._next, detection, timestamp, timestamp, appearance=features[index])
                self.tracks[track.id] = track
                self._next += 1
                observed.append(track)
        return observed
