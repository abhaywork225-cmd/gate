from __future__ import annotations
"""Lane-aware truck passages. A camera is an observer, never a lane identity.

Requires measured truck/trailer boxes. Text-only observations cannot establish
that two container numbers are on one chassis and are deliberately not fused.
"""
from collections import Counter
from datetime import timedelta
from dataclasses import dataclass, field, replace
from uuid import uuid4
import cv2
import numpy as np
import math

from app.detection.tracker import TemporalTracker, iou, similar, same_colour, appearance


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
    best_any: object = None
    source_proofs: dict = field(default_factory=dict)
    context_only: bool = False
    capture_sequences: set = field(default_factory=set)
    sequences: set = field(default_factory=set)
    vote_sequences: dict = field(default_factory=dict)

    def confirmed(self, confirmations):
        votes = self.votes.most_common(2)
        return bool(votes and votes[0][1] >= confirmations and
                    (len(votes)==1 or votes[0][1]>votes[1][1]))


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
    appearance: object = None
    context: object = None
    closed_at: object = None
    final_evidence: bool = False
    reversing: bool = False
    reverse_distance: float = 0
    body_observed: bool = True
    chassis_observed: bool = True
    last_body_seen: object = None
    last_body_bbox: tuple = ()
    last_body_appearance: object = None
    # A gate visit reserves an observer before that camera detects a body.
    capture_only: bool = False


class PassageCollector:
    """Bounded candidate frames; independent OCR budgets for each text region."""
    def __init__(self, settings):
        self.settings = settings
        self.trackers = {}
        self.active = {}
        self.passages = {}
        self.health = {}
        self.recent = {}

    def reserve_observer(self, owner, camera, frame):
        """Join a fresh calibrated live view before its independent OCR succeeds."""
        visit = owner.visit_id or owner.id
        members = [p for p in self.passages.values() if (p.visit_id or p.id) == visit
                   and p.camera_id == camera.id and not p.saved]
        if members:
            return members[0] if len(members) == 1 else None
        if (camera.source_type != 'rtsp' or owner.source_type != 'rtsp' or
                camera.gate_id != owner.gate_id or camera.direction != owner.direction or
                camera.lanes):
            return None  # Multi-lane views need an explicitly measured lane owner.
        h, w = frame.image.shape[:2]
        a, b, c, d = camera.occupancy_roi
        observer = Passage(str(uuid4()), camera.id, frame.source_id, camera.gate_id,
                           owner.lane_id, camera.direction, owner.started, frame.timestamp,
                           -1, (a*w,b*h,c*w,d*h), visit_id=visit,
                           body_observed=False, chassis_observed=False, capture_only=True)
        self.passages[observer.id] = observer
        return observer

    def close_untracked_observers(self):
        """Observer reservations follow measured visit lifetime, never extend it."""
        for observer in self.passages.values():
            if not observer.capture_only or observer.closed:
                continue
            measured = [p for p in self.passages.values() if not p.capture_only and
                        (p.visit_id or p.id) == observer.visit_id]
            if measured and all(p.closed for p in measured):
                observer.closed = True
                observer.closed_at = max((p.closed_at for p in measured if p.closed_at), default=None)

    def _number_matches_recent_body(self, passage, detection, frame):
        """Bridge a short body-detector dropout without shrinking to a label.

        Geometry alone is insufficient: require recent measured body evidence
        and agreement with the current pixels at that body location. The time
        bound is measured from the body detection, never renewed by text.
        """
        if (passage.last_body_seen is None or not passage.last_body_bbox or
                passage.last_body_appearance is None or
                not 0 <= (frame.timestamp-passage.last_body_seen).total_seconds()
                    <= self.settings.gate_observation_gap_seconds or
                not contains_text(passage.last_body_bbox, detection.bbox)):
            return False
        current = appearance(frame.image, passage.last_body_bbox)
        return current is not None and similar(passage.last_body_appearance, current, .9)

    def update(self, camera, frame, detections):
        self.recent = {key:p for key,p in self.recent.items() if p.closed_at is not None and
                       (frame.timestamp-p.closed_at).total_seconds() <= self.settings.reversal_grace_seconds}
        key = (camera.id, frame.source_id)
        tracker = self.trackers.setdefault(key, TemporalTracker(self.settings.gate_clear_seconds))
        h,w = frame.image.shape[:2]
        bodies = [d for d in detections if d.class_name in ('truck', 'trailer', 'vehicle')
                  and d.confidence >= self.settings.vehicle_presence_min_confidence
                  and (not self.settings.vehicle_visit_tracking or
                       ((d.bbox[2]-d.bbox[0])*(d.bbox[3]-d.bbox[1]) >= .005*w*h
                        and d.bbox[3]-d.bbox[1] >= .02*h))]
        if self.settings.vehicle_visit_tracking:
            from app.ocr.container_ocr import inside_gate
            # Container evidence is still captured if the auxiliary vehicle
            # detector misses the chassis. These are separate uncertain
            # container tracks, never invented twin/truck associations.
            for class_name in ('container','container_number'):
                bodies += [d for d in detections if d.class_name==class_name and
                           inside_gate(d.bbox,frame.image.shape,camera.occupancy_roi) and
                           not any(contains_text(b.bbox,d.bbox) for b in bodies)]
        # Nested truck/trailer predictions must not become two passages.
        bodies = sorted(bodies, key=lambda d: (d.bbox[2]-d.bbox[0])*(d.bbox[3]-d.bbox[1]), reverse=True)
        unique = []
        for d in bodies:
            if not any(iou(b.bbox, d.bbox) > .8 for b in unique):
                unique.append(d)
        tracks = tracker.update(unique, frame.timestamp, frame.image if self.settings.vehicle_visit_tracking else None)
        self.health[camera.id] = 'TRACKING' if tracks else 'WAITING_FOR_TRUCK_OR_TRAILER_DETECTION'
        ready = []
        from app.ocr.container_ocr import inside_gate
        container_evidence = [d for d in detections if d.class_name in ('container','container_number')
                              and inside_gate(d.bbox,frame.image.shape,camera.occupancy_roi)]
        texts = [d for d in detections if d.class_name == 'container_number' and
                 inside_gate(d.bbox, frame.image.shape, camera.occupancy_roi)]
        for track in tracks:
            if self.settings.vehicle_visit_tracking:
                h,w=frame.image.shape[:2]
                bx=track.detection.bbox
                cx,cy=(bx[0]+bx[2])/(2*w),(bx[1]+bx[3])/(2*h)
                rx1,ry1,rx2,ry2=camera.occupancy_roi
                if not (rx1<=cx<=rx2 and ry1<=cy<=ry2) and not any(
                        contains_text(bx,t.bbox) for t in container_evidence):
                    # A measured vehicle outside the capture ROI cannot keep
                    # the gate visit alive merely by remaining in the image.
                    continue
            identity = (*key, track.id)
            passage = self.active.get(identity)
            if passage is None:
                if self.settings.container_roi_trigger and not any(
                        d.class_name in ('container','container_number')
                        and inside_gate(d.bbox,frame.image.shape,camera.occupancy_roi)
                        and contains_text(track.detection.bbox,d.bbox) for d in detections):
                    continue  # A truck alone cannot open a container capture event.
                lane = lane_for_box(track.detection.bbox, frame.image.shape, camera.lanes) if camera.lanes else None
                if not camera.lanes:
                    h,w = frame.image.shape[:2]
                    bx = track.detection.bbox
                    cx,cy = (bx[0]+bx[2])/(2*w),(bx[1]+bx[3])/(2*h)
                    rx1,ry1,rx2,ry2 = camera.occupancy_roi
                    if (rx1 <= cx <= rx2 and ry1 <= cy <= ry2) or any(
                            contains_text(bx,d.bbox) for d in container_evidence):
                        lane = camera.gate_id
                if lane is None:
                    self.health[camera.id] = 'AMBIGUOUS_OR_OUTSIDE_LANE'
                    continue
                same_view = [p for p in self.active.values() if p.camera_id==camera.id and
                    p.source_id==frame.source_id and not p.closed and
                    ((not p.body_observed and contains_text(track.detection.bbox,p.bbox)) or
                     (track.detection.class_name in ('container_number', 'container') and
                      ((not p.body_observed and (contains_text(p.bbox,track.detection.bbox) or
                                                 iou(p.bbox,track.detection.bbox) >= .05 or
                                                 (len(tracks) == 1 and
                                                  0 <= (frame.timestamp-p.last_seen).total_seconds() <= self.settings.gate_observation_gap_seconds and
                                                  abs(track.detection.bbox[1]-p.bbox[1]) < .35*frame.image.shape[0] and
                                                  abs(track.detection.bbox[0]-p.bbox[0]) < .55*frame.image.shape[1]))) or
                       self._number_matches_recent_body(p,track.detection,frame))) or
                     (p.body_observed and track.detection.class_name in ('truck','trailer','vehicle','container')
                      and 0 <= (frame.timestamp-p.last_seen).total_seconds() <= 2
                      and iou(p.bbox,track.detection.bbox) >= .5
                      and (same_colour(p.appearance,track.appearance) or similar(p.appearance,track.appearance,.8))))]
                # Adopt the preassigned gate observer only when this view has
                # one measured target. Two simultaneous targets need ownership.
                if not same_view and len(tracks) == 1 and self.settings.shared_visit_folders:
                    same_view = [p for p in self.passages.values() if p.capture_only and
                                 not p.closed and not p.saved and p.camera_id == camera.id and
                                 p.source_id == frame.source_id and p.lane_id == lane and
                                 0 <= (frame.timestamp-p.last_seen).total_seconds() <=
                                 self.settings.cross_camera_view_window_seconds]
                if self.settings.vehicle_visit_tracking and len(same_view)==1:
                    passage=same_view[0]
                    passage.capture_only = False
                    for alias,other in list(self.active.items()):
                        if other is passage:self.active.pop(alias)
                    passage.track_id=track.id
                    self.active[identity]=passage
                # Reversal/short occlusion before finalization reopens the
                # existing passage, not a new movement. Appearance is a gate,
                # never sufficient for cross-camera identity association.
                returning = [p for p in [*self.passages.values(),*self.recent.values()] if p.closed and (not p.saved or p.reversing)
                    and p.camera_id == camera.id and p.source_id == frame.source_id
                    and p.lane_id == lane and p.closed_at is not None
                    and (frame.timestamp-p.closed_at).total_seconds() <= self.settings.reversal_grace_seconds
                    and p.appearance is not None and track.appearance is not None
                    and similar(p.appearance, track.appearance, .98)]
                if passage is not None:
                    pass
                elif self.settings.vehicle_visit_tracking and len(returning) == 1:
                    passage = returning[0]
                    passage.closed = False
                    passage.closed_at = passage.clear_since = None
                    passage.track_id = track.id
                    self.active[identity] = passage
                else:
                    passage = Passage(str(uuid4()), camera.id, frame.source_id, camera.gate_id,
                                  lane, camera.direction, frame.timestamp, frame.timestamp,
                                  track.id, track.detection.bbox,
                                  body_observed=track.detection.class_name in ('truck','trailer','vehicle','container'),
                                  chassis_observed=track.detection.class_name in ('truck','trailer','vehicle'))
                # Reuse the active passage's visit ID for concurrent cameras at
                # the same gate/lane. This is the only folder key for the truck.
                if not passage.visit_id:
                    # Stale tracker aliases cannot block camera association.
                    candidates = {}
                    for other in self.active.values():
                        if (other.camera_id != camera.id and other.gate_id == camera.gate_id
                                and other.lane_id == lane and other.direction == camera.direction
                                and other.source_type == camera.source_type == 'rtsp'
                                and not other.closed and not other.saved
                                and abs((frame.timestamp-other.started).total_seconds()) <= self.settings.cross_camera_arrival_window_seconds
                                and abs((frame.timestamp-other.last_seen).total_seconds()) <= self.settings.cross_camera_view_window_seconds):
                            visit = other.visit_id or other.id
                            # A following truck in the same camera cannot borrow
                            # an already observed visit from the other camera.
                            if not any(p.camera_id == camera.id and not p.closed
                                       and (p.visit_id or p.id) == visit for p in self.active.values()):
                                candidates[visit] = other
                    if candidates and len(tracks) == 1:
                        passage.visit_id = min(candidates.keys(), key=lambda v: candidates[v].started)
                passage.visit_id = passage.visit_id or passage.id
                self.active[identity] = passage
                if not passage.saved:
                    self.passages[passage.id] = passage
                passage.source_type = camera.source_type
            measured_body = track.detection.class_name in ('truck','trailer','vehicle','container')
            retained_body = (not measured_body and
                             self._number_matches_recent_body(passage,track.detection,frame))
            reference_box = passage.last_body_bbox if retained_body else track.detection.bbox
            # Changing from a body to one of its labels is not physical movement.
            if camera.gate_exit_axis is not None and not retained_body:
                axis = 0 if camera.gate_exit_axis == 'x' else 1
                extent = frame.image.shape[1-axis]
                old = (passage.bbox[axis]+passage.bbox[axis+2])/2
                new = (track.detection.bbox[axis]+track.detection.bbox[axis+2])/2
                delta = (new-old)/extent * (1 if camera.gate_exit_positive else -1)
                passage.reverse_distance = max(0,passage.reverse_distance-delta)
                passage.reversing = passage.reverse_distance > .05
            passage.last_seen, passage.bbox = frame.timestamp, reference_box
            passage.body_observed = measured_body or retained_body
            passage.chassis_observed=passage.chassis_observed or track.detection.class_name in ('truck','trailer','vehicle')
            passage.appearance = passage.last_body_appearance if retained_body else track.appearance
            if measured_body:
                passage.last_body_seen = frame.timestamp
                passage.last_body_bbox = track.detection.bbox
                passage.last_body_appearance = track.appearance
            passage.clear_since = None
            if passage.saved:
                continue  # Track movement; already published capture is immutable.
            # One bounded context frame survives even if no number is detected.
            body = track.detection
            x1,y1,x2,y2 = map(int,body.bbox)
            body_crop = frame.image[max(0,y1):y2,max(0,x1):x2]
            if body_crop.size and not retained_body:
                gray = cv2.cvtColor(cv2.resize(body_crop,(160,96)),cv2.COLOR_BGR2GRAY)
                quality = float(cv2.Laplacian(gray,cv2.CV_64F).var())
                if passage.context is None or quality > passage.context[0]:
                    passage.context = (quality,frame,body,(body,))
            for det in texts:
                owners = [t for t in tracks if contains_text(t.detection.bbox, det.bbox)]
                if len(owners) != 1 or owners[0].id != track.id:
                    if any(t.id == track.id for t in owners):
                        passage.ambiguous = True
                    continue
                bx = reference_box
                bw, bh = bx[2]-bx[0], bx[3]-bx[1]
                relative = ((det.bbox[0]-bx[0])/bw, (det.bbox[1]-bx[1])/bh,
                            (det.bbox[2]-bx[0])/bw, (det.bbox[3]-bx[1])/bh)
                region = max((r for r in passage.regions if r.bbox and not r.context_only),
                             key=lambda r: iou(r.bbox, relative), default=None)
                if region is None or iou(region.bbox, relative) < .15:
                    if len(passage.regions) >= 6:
                        passage.ambiguous = True
                        continue
                    region = Region(bbox=relative)
                    passage.regions.append(region)
                region.bbox = relative
                if region.confirmed(self.settings.ocr_confirmations):
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
                    passage.closed_at = frame.timestamp
                    self.active.pop(identity)
                    if passage.saved:
                        self.recent[passage.id] = passage
            passage.last_measured = frame.timestamp
        # Old source generations must not retain trackers and frames forever.
        for stale_key in list(self.trackers):
            if stale_key[0] == camera.id and stale_key != key:
                self.trackers.pop(stale_key)
        self.close_untracked_observers()
        return ready

    def result(self, passage_id, region_id, validation, job):
        passage = self.passages.get(passage_id)
        if not passage:
            return
        region = next(r for r in passage.regions if r.id == region_id)
        sequence = (job.detection.camera_id,job.frame.source_id,job.frame.sequence)
        if sequence in region.sequences:
            return
        region.sequences.add(sequence)
        region.pending = max(0, region.pending-1)
        if region.best_any is None or (validation.valid_check_digit,validation.valid_format,validation.confidence) > (
                region.best_any[0].valid_check_digit,region.best_any[0].valid_format,region.best_any[0].confidence):
            region.best_any = (validation,job)
        if validation.valid_check_digit and validation.confidence >= self.settings.ocr_min_confidence:
            number = validation.normalized_text
            region.votes[number] += 1
            region.vote_sequences.setdefault(number,set()).add(sequence)
            x1,y1,x2,y2 = map(int,job.detection.bbox)
            crop = job.frame.image[max(0,y1):y2,max(0,x1):x2]
            var = float(cv2.Laplacian(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var()) if crop.size else 0
            box_area = max(1.0, float((x2 - x1) * (y2 - y1)))
            quality = var * math.log1p(box_area)
            previous = region.reads.get(number)
            if previous:
                old_validation, old_job = previous
                if old_validation.size_type and validation.size_type and old_validation.size_type != validation.size_type:
                    passage.ambiguous = True
                if quality <= region.proof_quality.get(number,0):
                    region.reads[number] = (replace(old_validation,size_type=old_validation.size_type or validation.size_type),old_job)
                    region.source_proofs.setdefault(number,{})[job.detection.camera_id] = region.reads[number]
                    return
                validation = replace(validation,size_type=validation.size_type or old_validation.size_type)
            region.reads[number] = (validation, job)
            region.source_proofs.setdefault(number,{})[job.detection.camera_id] = (validation,job)
            region.proof_quality[number] = quality

    def status(self, camera_id):
        return {'state': self.health.get(camera_id, 'WAITING_FOR_TRUCK_OR_TRAILER_DETECTION'),
                'passages': [{'event_id':p.visit_id or p.id,'passage_id':p.id,'lane_id':p.lane_id,'closed':p.closed,
                              'candidates':len(p.regions)} for p in self.passages.values()
                             if p.camera_id == camera_id and not p.saved]}

    def _same_text_fragment(self, first, second, *, require_continuity=True):
        """Reconcile label tracks only with independent, unambiguous identity evidence.

        Measured vehicle/container tracks remain separate. This is not a timed
        deduplication rule: a cleared visit or a changed source cannot rejoin.
        """
        if (not self.settings.vehicle_visit_tracking or
                first.camera_id != second.camera_id or first.source_id != second.source_id or
                first.source_type != second.source_type or
                (first.gate_id,first.lane_id,first.direction) !=
                (second.gate_id,second.lane_id,second.direction) or
                any(p.body_observed or p.chassis_observed or p.last_body_seen is not None
                    or p.saved for p in (first,second))):
            return False
        earlier,later = sorted((first,second),key=lambda p:p.started)
        if ((earlier.closed_at is not None and earlier.closed_at < later.started) or
                (require_continuity and
                 (later.started-earlier.last_seen).total_seconds() > self.settings.gate_observation_gap_seconds)):
            return False
        def evidence(passage):
            observations = {}
            for region in passage.regions:
                for number,count in region.votes.items():
                    if count and number in region.reads:
                        validation,job = region.reads[number]
                        if validation.valid_check_digit and validation.confidence >= self.settings.ocr_min_confidence:
                            observations.setdefault(number,set()).update(region.vote_sequences.get(number,set()))
                            observations[number].add(
                                (job.detection.camera_id,job.frame.source_id,job.frame.sequence))
            return observations
        a,b = evidence(first),evidence(second)
        if a and b:
            return (len(a)==len(b)==1 and a.keys()==b.keys() and
                    len(next(iter(a.values())) | next(iter(b.values()))) >= 2)
        if (a and not b) or (b and not a) or (not a and not b):
            seqs_first = {s for r in first.regions for s in r.sequences}
            seqs_second = {s for r in second.regions for s in r.sequences}
            if seqs_first and seqs_second and seqs_first == seqs_second:
                return False
            return True
        return False

    def reconcile_text_visits(self):
        """Apply proven label continuity to subsequent shared-camera captures."""
        for group in self.groups():
            if len(group) < 2:
                continue
            owner = min(group,key=lambda p:(p.started,p.id))
            visit = owner.visit_id or owner.id
            for passage in group:
                passage.visit_id = visit

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
                          if c >= 1}
            matching = []
            for group in groups:
                # Physical same-camera tracks remain distinct. Label-only
                # fragments need an independent identity match and a continuous
                # chain of observations; timing or colour alone cannot join them.
                if any(other.camera_id == passage.camera_id and
                       (other.visit_id or other.id) != (passage.visit_id or passage.id)
                       and not (self._same_text_fragment(passage,other,require_continuity=False)
                                and any(self._same_text_fragment(passage,member) for member in group))
                       for other in group):
                    continue
                for other in group:
                    other_numbers = {n for r in other.regions for n,c in r.votes.items()
                                     if c >= 1}
                    # Votes already require checksum-valid text and minimum
                    # OCR confidence. Agreement across independent cameras is
                    # evidence too; do not require two reads in each camera.
                    overlap = max(passage.started,other.started) <= min(passage.last_seen,other.last_seen) + timedelta(seconds=self.settings.cross_camera_view_window_seconds)
                    same_identity = bool(identities & other_numbers) and overlap
                    start_gap = abs((passage.started - other.started).total_seconds())
                    cross_view_window = start_gap <= self.settings.cross_camera_arrival_window_seconds
                    if (passage.camera_id == other.camera_id and passage.source_id == other.source_id
                            and passage.visit_id and passage.visit_id == other.visit_id) or self._same_text_fragment(passage,other):
                        matching.append(group)
                        break
                    if (passage.camera_id != other.camera_id and
                            passage.source_type == other.source_type == 'rtsp' and
                            (passage.gate_id,passage.lane_id,passage.direction) ==
                            (other.gate_id,other.lane_id,other.direction) and
                            ((cross_view_window and same_identity) or
                             (passage.visit_id and passage.visit_id == other.visit_id))):
                        matching.append(group)
                        break
            merged = [passage]
            for group in matching:
                merged.extend(group)
                groups.remove(group)
            groups.append(merged)
        return groups


def final_captures(group, confirmations):
    """Plan immutable, identity-isolated records from a completed visit group."""
    from uuid import uuid5, NAMESPACE_URL
    numbers = {}
    unknown = []
    for passage in group:
        for region in passage.regions:
            votes = region.votes.most_common()
            if votes and not (len(votes)>1 and votes[0][1] == votes[1][1]):
                number,count = votes[0]
                numbers.setdefault(number,[]).append((passage,region,count))
            elif region.best_any:
                unknown.append((passage,region))
        if not passage.regions and passage.context:
            # Runtime persists this context as an explicit unreadable job before
            # finalization; a missing detector number must not drop the visit.
            raise ValueError('Departure context has not been persisted')
    # Two IDs are twins only if one observer assigned them to the same measured
    # truck and both have supported 20-foot ISO codes. Timing is not twin proof.
    reliable_twin = len(numbers)==2 and any(
        p.chassis_observed and not p.ambiguous and all(any(owner.id==p.id for owner,_,_ in records) for records in numbers.values())
        for p in group)
    reliable_twin = reliable_twin and all(
        any(str(region.reads[number][0].size_type or '').startswith('2') for _,region,_ in records)
        for number,records in numbers.items())
    base = min((p.visit_id or p.id for p in group))
    plans = []
    confirmed_numbers = {n: records for n, records in numbers.items() if sum(c for _, _, c in records) >= confirmations}
    if reliable_twin:
        bundles = [list(numbers)]
    elif confirmed_numbers:
        bundles = [[n] for n in sorted(confirmed_numbers)]
    elif numbers:
        best_candidate = max(numbers, key=lambda n: sum(c for _, _, c in numbers[n]))
        bundles = [[best_candidate]]
    else:
        bundles = []

    for bundle in bundles:
        regions = [r for n in bundle for _,r,_ in numbers[n]]
        owners = [p for n in bundle for p,_,_ in numbers[n]]
        first = min(owners,key=lambda p:(p.started,p.id))
        identity = str(uuid5(NAMESPACE_URL,'gate-container:'+base+':'+','.join(sorted(bundle))))
        plans.append(replace(first,id=identity,regions=regions,ambiguous=False,
                             final_evidence=True,closed=True))
    # If the only evidence is unreadable, keep one source per camera for a
    # concurrent visit. If known identities coexist, unreadable text on the same
    # passage is evidence of the container, not an additional container.
    unknown_bundles = [unknown] if (not numbers and unknown) else []
    for entries in unknown_bundles:
        first = min((p for p,_ in entries),key=lambda p:(p.started,p.id))
        key = ':'.join(sorted(p.id+':'+r.id for p,r in entries))
        identity = str(uuid5(NAMESPACE_URL,'gate-unreadable:'+base+':'+key))
        plans.append(replace(first,id=identity,regions=[r for _,r in entries],
                             ambiguous=True,final_evidence=True,closed=True))
    return plans
