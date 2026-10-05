"""Lane-aware truck passages. A camera is an observer, never a lane identity.

Requires measured truck/trailer boxes. Text-only observations cannot establish
that two container numbers are on one chassis and are deliberately not fused.
"""
from collections import Counter
from dataclasses import dataclass, field, replace
from uuid import uuid4
import cv2
import numpy as np

from app.detection.tracker import TemporalTracker, iou


def lane_for_box(box, shape, lanes):
    h, w = shape[:2]
    point = ((box[0] + box[2]) / (2 * w), box[3] / h)
    matches = [lane.id for lane in lanes if cv2.pointPolygonTest(
        np.asarray(lane.polygon, dtype=np.float32), point, False) >= 0]
    return matches[0] if len(matches) == 1 else None


def contains_text(body, text):
    x, y = (text[0] + text[2]) / 2, (text[1] + text[3]) / 2
    return body[0] <= x <= body[2] and body[1] <= y <= body[3]


@dataclass
class Region:
    id: str = field(default_factory=lambda: uuid4().hex)
    bbox: tuple = ()
    candidates: list = field(default_factory=list)
    votes: Counter = field(default_factory=Counter)
    reads: dict = field(default_factory=dict)
    proof_quality: dict = field(default_factory=dict)
    attempts: int = 0
    pending: int = 0
    last_sample: object = None


@dataclass
class Passage:
    id: str
    camera_id: str
    source_id: str
    gate_id: str
    lane_id: str
    direction: str
    started: object
    last_seen: object
    track_id: int
    bbox: tuple
    regions: list = field(default_factory=list)
    closed: bool = False
    saved: bool = False
    ambiguous: bool = False
    source_type: str = 'rtsp'
    # Shared media-session ID. All cameras observing the same truck write into
    # one evidence folder, while `id` remains the per-camera passage key.
    visit_id: str | None = None
    clear_since: object = None
    last_measured: object = None


class PassageCollector:
    """Bounded candidate frames; independent OCR budgets for each text region."""
    def __init__(self, settings):
        self.settings = settings
        self.trackers = {}
        self.active = {}
        self.passages = {}
        self.health = {}

    def update(self, camera, frame, detections):
        key = (camera.id, frame.source_id)
        tracker = self.trackers.setdefault(key, TemporalTracker(self.settings.gate_clear_seconds))
        bodies = [d for d in detections if d.class_name in ('truck', 'trailer')]
        # Nested truck/trailer predictions must not become two passages.
        bodies = sorted(bodies, key=lambda d: (d.bbox[2]-d.bbox[0])*(d.bbox[3]-d.bbox[1]), reverse=True)
        unique = []
        for d in bodies:
            if not any(iou(b.bbox, d.bbox) > .8 for b in unique):
                unique.append(d)
        tracks = tracker.update(unique, frame.timestamp)
        self.health[camera.id] = 'TRACKING' if tracks else 'WAITING_FOR_TRUCK_OR_TRAILER_DETECTION'
        ready = []
        texts = [d for d in detections if d.class_name == 'container_number']
        for track in tracks:
            identity = (*key, track.id)
            passage = self.active.get(identity)
            if passage is None:
                lane = lane_for_box(track.detection.bbox, frame.image.shape, camera.lanes)
                if lane is None:
                    self.health[camera.id] = 'AMBIGUOUS_OR_OUTSIDE_LANE'
                    continue
                passage = Passage(str(uuid4()), camera.id, frame.source_id, camera.gate_id,
                                  lane, camera.direction, frame.timestamp, frame.timestamp,
                                  track.id, track.detection.bbox)
                # Reuse the active passage's visit ID for concurrent cameras at
                # the same gate/lane. This is the only folder key for the truck.
                for other in self.active.values():
                    if (other.camera_id != camera.id and other.gate_id == camera.gate_id
                            and other.lane_id == lane and other.direction == camera.direction
                            and other.source_type == camera.source_type
                            and abs((frame.timestamp - other.last_seen).total_seconds())
                            <= float(getattr(self.settings, 'cross_camera_view_window_seconds', 8.0))
                            and not other.closed):
                        passage.visit_id = other.visit_id or other.id
                        break
                passage.visit_id = passage.visit_id or passage.id
                self.active[identity] = passage
                self.passages[passage.id] = passage
                passage.source_type = camera.source_type
            # Preserve lane and event if this track straddles another polygon.
            passage.last_seen, passage.bbox = frame.timestamp, track.detection.bbox
            passage.clear_since = None
            for det in texts:
                owners = [t for t in tracks if contains_text(t.detection.bbox, det.bbox)]
                if len(owners) != 1 or owners[0].id != track.id:
                    if any(t.id == track.id for t in owners):
                        passage.ambiguous = True
                    continue
                bx = track.detection.bbox
                bw, bh = bx[2]-bx[0], bx[3]-bx[1]
                relative = ((det.bbox[0]-bx[0])/bw, (det.bbox[1]-bx[1])/bh,
                            (det.bbox[2]-bx[0])/bw, (det.bbox[3]-bx[1])/bh)
                region = max(passage.regions, key=lambda r: iou(r.bbox, relative), default=None)
                if region is None or iou(region.bbox, relative) < .15:
                    if len(passage.regions) >= 6:
                        passage.ambiguous = True
                        continue
                    region = Region(bbox=relative)
                    passage.regions.append(region)
                region.bbox = relative
                if region.attempts >= self.settings.ocr_max_attempts or max(region.votes.values(), default=0) >= self.settings.ocr_confirmations:
                    continue
                x1,y1,x2,y2 = map(int,det.bbox)
                crop = frame.image[max(0,y1):y2,max(0,x1):x2]
                if crop.size == 0:
                    continue
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                quality = float(cv2.Laplacian(gray, cv2.CV_64F).var()) * min(crop.shape[:2])
                if quality <= 0:
                    continue
                # Sample a few independent moments; retain only three best frames.
                if region.last_sample and (frame.timestamp-region.last_sample).total_seconds() < .15:
                    continue
                region.last_sample = frame.timestamp
                # Only this number and ISO boxes nearest to it enter its OCR job.
                # An adjacent container's high-confidence number must never win.
                def distance(a, b):
                    return ((a[0]+a[2]-b[0]-b[2])**2 + (a[1]+a[3]-b[1]-b[3])**2)
                relevant = [det]
                for iso in detections:
                    if iso.class_name == 'iso_code' and contains_text(track.detection.bbox, iso.bbox):
                        nearest = min(texts, key=lambda t:distance(t.bbox,iso.bbox),default=None)
                        if nearest is det:
                            relevant.append(iso)
                region.candidates.append((quality, frame, det, tuple(relevant)))
                region.candidates.sort(key=lambda item:item[0], reverse=True)
                del region.candidates[3:]
                # Trigger the first OCR immediately; later attempts choose the best
                # frame collected while the OCR worker was busy.
                if not region.pending:
                    ready.append((passage, region))
        for identity, passage in list(self.active.items()):
            if passage.camera_id != camera.id:
                continue
            if passage.source_id != frame.source_id:
                passage.closed = passage.ambiguous = True
                self.active.pop(identity)
                continue
            if passage.last_seen != frame.timestamp:
                if (passage.last_measured is None or
                        (frame.timestamp-passage.last_measured).total_seconds() > self.settings.gate_observation_gap_seconds):
                    passage.clear_since = None
                passage.clear_since = passage.clear_since or frame.timestamp
                if (frame.timestamp-passage.clear_since).total_seconds() >= self.settings.gate_clear_seconds:
                    passage.closed = True
                    self.active.pop(identity)
            passage.last_measured = frame.timestamp
        # Old source generations must not retain trackers and frames forever.
        for stale_key in list(self.trackers):
            if stale_key[0] == camera.id and stale_key != key:
                self.trackers.pop(stale_key)
        return ready

    def result(self, passage_id, region_id, validation, job):
        passage = self.passages.get(passage_id)
        if not passage:
            return
        region = next(r for r in passage.regions if r.id == region_id)
        region.pending = max(0, region.pending-1)
        if validation.valid_check_digit and validation.confidence >= self.settings.ocr_min_confidence:
            number = validation.normalized_text
            region.votes[number] += 1
            x1,y1,x2,y2 = map(int,job.detection.bbox)
            crop = job.frame.image[max(0,y1):y2,max(0,x1):x2]
            quality = float(cv2.Laplacian(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var()) if crop.size else 0
            previous = region.reads.get(number)
            if previous:
                old_validation, old_job = previous
                if old_validation.size_type and validation.size_type and old_validation.size_type != validation.size_type:
                    passage.ambiguous = True
                if quality <= region.proof_quality.get(number,0):
                    region.reads[number] = (replace(old_validation,size_type=old_validation.size_type or validation.size_type),old_job)
                    return
                validation = replace(validation,size_type=validation.size_type or old_validation.size_type)
            region.reads[number] = (validation, job)
            region.proof_quality[number] = quality

    def status(self, camera_id):
        return {'state': self.health.get(camera_id, 'WAITING_FOR_TRUCK_OR_TRAILER_DETECTION'),
                'passages': [{'event_id':p.visit_id or p.id,'passage_id':p.id,'lane_id':p.lane_id,'closed':p.closed,
                              'candidates':len(p.regions)} for p in self.passages.values()
                             if p.camera_id == camera_id and not p.saved]}

    def groups(self):
        """Associate camera views into one truck visit.

        Repeated identity is strong evidence. Different identities from two
        calibrated RTSP views can be collected into one visit, but that link is
        marked ambiguous so it cannot be silently confirmed or uploaded.
        Uploaded files never share a media clock and remain separate.
        """
        groups = []
        for passage in self.passages.values():
            identities = {n for r in passage.regions for n,c in r.votes.items()
                          if c >= self.settings.ocr_confirmations}
            matching = []
            for group in groups:
                for other in group:
                    other_numbers = {n for r in other.regions for n,c in r.votes.items()
                                     if c >= self.settings.ocr_confirmations}
                    same_identity = bool(identities & other_numbers)
                    start_gap = min(abs((passage.started - other.last_seen).total_seconds()),
                                    abs((other.started - passage.last_seen).total_seconds()))
                    cross_view_window = start_gap <= float(getattr(
                        self.settings, 'cross_camera_view_window_seconds', 8.0))
                    if (passage.camera_id != other.camera_id and
                            passage.source_type == other.source_type == 'rtsp' and
                            (passage.gate_id,passage.lane_id,passage.direction) ==
                            (other.gate_id,other.lane_id,other.direction) and
                            cross_view_window and
                            (same_identity or (identities and other_numbers and identities.isdisjoint(other_numbers)))):
                        if not same_identity:
                            passage.ambiguous = True
                            other.ambiguous = True
                        matching.append(group)
                        break
            merged = [passage]
            for group in matching:
                merged.extend(group)
                groups.remove(group)
            groups.append(merged)
        return groups
