from __future__ import annotations
"""View-space occupancy independent of short-lived tracker IDs.

Only measured inference frames advance clearance. Missing camera frames never
count as proof that a lane is clear. One calibrated zone represents one queue.
"""
from dataclasses import dataclass, field
from math import hypot
from uuid import uuid4
from app.detection.tracker import iou


def box_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def containment(b1, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    min_area = min(box_area(b1), box_area(b2))
    return inter / min_area if min_area > 0.0 else 0.0


def normalized(box, shape):
    h, w = shape[:2]
    return (box[0] / w, box[1] / h, box[2] / w, box[3] / h)


def center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


@dataclass
class ContainerSlot:
    slot_id: str  # 'FRONT' or 'REAR'
    origin: str
    bbox: tuple
    anchor: tuple
    first_seen: object
    last_measured: object
    stationary_since: object
    stationary: bool = False
    event_id: str | None = None
    container_number: str | None = None
    closed: bool = False
    clear_since: object = None


@dataclass
class Occupancy:
    source_id: str
    origin: str
    bbox: tuple
    anchor: tuple
    first_seen: object
    last_measured: object
    stationary_since: object
    clear_since: object = None
    stationary: bool = False
    event_id: str | None = None
    container_number: str | None = None
    closed: bool = False
    is_twin: bool = False
    slots: dict = field(default_factory=dict)
    active_slot: str = 'FRONT'
    provisional_event_id: str | None = None
    entrance_complete: bool = False


class ViewOccupancy:
    def __init__(self, settings):
        self.settings = settings
        self.current = None

    def _past_exit(self, box, camera):
        if camera.gate_exit_axis is None:
            return False
        axis = 0 if camera.gate_exit_axis == 'x' else 1
        # Entire bounding box, not its centroid, must clear the exit line.
        edge = box[axis] if camera.gate_exit_positive else box[axis + 2]
        return edge > camera.gate_exit_position if camera.gate_exit_positive else edge < camera.gate_exit_position

    def _order_slots(self, candidates_pair, camera):
        axis = 1 if camera.gate_exit_axis == 'y' else 0
        if camera.gate_exit_axis is not None:
            pos = camera.gate_exit_positive if camera.gate_exit_positive is not None else True
        else:
            # Default horizontal gate lane: exit is on left (smaller x)
            pos = False
        c1, c2 = candidates_pair[0], candidates_pair[1]
        coord1 = center(c1[1])[axis]
        coord2 = center(c2[1])[axis]
        if (coord1 > coord2) if pos else (coord1 < coord2):
            return [c1, c2]
        return [c2, c1]

    def _update_slot(self, slot, b, now):
        point = center(b)
        drift = hypot(point[0] - slot.anchor[0], point[1] - slot.anchor[1])
        if drift >= self.settings.stationary_drift_threshold:
            slot.anchor, slot.stationary_since, slot.stationary = point, now, False
        else:
            slot.stationary = (now - slot.stationary_since).total_seconds() >= self.settings.stationary_seconds
        slot.bbox = b
        slot.last_measured = now

    def update(self, camera, frame, detections):
        now = frame.timestamp
        dwell_classes = getattr(self.settings, 'dwell_container_classes', None) or ['container', 'container_number', 'iso_code']
        all_boxes = [(d, normalized(d.bbox, frame.image.shape)) for d in detections
                     if d.class_name in dwell_classes]
        zone = camera.occupancy_roi

        # 1. Filter raw candidates: must intersect occupancy ROI, center within ROI, not past exit, and realistic container/text size
        raw_candidates = []
        for d, b in all_boxes:
            cx, cy = center(b)
            if iou(b, zone) > 0 and (zone[0] <= cx <= zone[2] and zone[1] <= cy <= zone[3]) and not self._past_exit(b, camera):
                min_area = 0.005 if d.class_name == 'container' else 0.0001
                if box_area(b) >= min_area:
                    raw_candidates.append((d, b))

        # 2. Suppress nested sub-boxes (e.g. text strips, decals, doors detected inside container body)
        candidates = []
        for d, b in sorted(raw_candidates, key=lambda item: box_area(item[1]), reverse=True):
            if not any(containment(b, ex[1]) > 0.5 for ex in candidates):
                candidates.append((d, b))

        state = self.current
        if state and state.source_id != frame.source_id:
            self.current = state = None  # Explicit source/calibration generation change.

        if state:
            gap = (now - state.last_measured).total_seconds()
            if gap <= 0:
                return None  # Duplicate/backwards timestamps cannot advance clearance.
            if gap > self.settings.gate_observation_gap_seconds:
                state.clear_since = None
                state.stationary_since = now
                state.stationary = False
            state.last_measured = now

            # Check if all tracked container slots departed past the exit boundary
            if state.slots and all(self._past_exit(s.bbox, camera) for s in state.slots.values()):
                self.current = None
                return None

            # Backward-compat: single box departed past exit
            matched_dep = max(all_boxes, key=lambda item: iou(state.bbox, item[1]), default=None)
            if matched_dep and iou(state.bbox, matched_dep[1]) > 0.15:
                if not self._past_exit(state.bbox, camera) and self._past_exit(matched_dep[1], camera):
                    self.current = None
                    return None

            if not candidates:
                state.clear_since = state.clear_since or now
                if (now - state.clear_since).total_seconds() >= self.settings.gate_clear_seconds:
                    self.current = None
                return None

            state.clear_since = None

            # Update tracking for slots
            axis = 1 if camera.gate_exit_axis == 'y' else 0
            is_pos = camera.gate_exit_positive if camera.gate_exit_axis is not None else False
            if len(candidates) >= 2:
                best_pair = None
                max_sep = 0.0
                for i in range(len(candidates)):
                    for j in range(i + 1, len(candidates)):
                        c1, c2 = candidates[i], candidates[j]
                        if iou(c1[1], c2[1]) >= 0.15:
                            continue
                        sep = abs(center(c1[1])[axis] - center(c2[1])[axis])
                        align = abs(center(c1[1])[1 - axis] - center(c2[1])[1 - axis])
                        if sep < 0.18 or align > 0.35:
                            continue
                        area_1, area_2 = box_area(c1[1]), box_area(c2[1])
                        area_ratio = min(area_1, area_2) / max(area_1, area_2) if max(area_1, area_2) > 0 else 0
                        is_body_twin = (
                            c1[0].class_name not in ('container_number', 'iso_code', 'feet') and
                            c2[0].class_name not in ('container_number', 'iso_code', 'feet') and
                            area_1 >= 0.04 and area_2 >= 0.04 and area_ratio >= 0.25
                        )
                        # A body box plus its number box is one container, not
                        # a twin. Text-only twin evidence is accepted only when
                        # both boxes are independent number markings of useful
                        # size and confidence. OCR later confirms the identities.
                        is_text_twin = (
                            c1[0].class_name == 'container_number' and
                            c2[0].class_name == 'container_number' and
                            area_1 >= 0.0015 and area_2 >= 0.0015 and
                            c1[0].confidence >= 0.60 and c2[0].confidence >= 0.60
                        )
                        if is_body_twin or is_text_twin:
                            if sep > max_sep:
                                max_sep = sep
                                best_pair = (c1, c2)

                if best_pair is not None:
                    ordered = self._order_slots(best_pair, camera)
                    front_det, front_b = ordered[0]
                    rear_det, rear_b = ordered[1]
                    if not state.is_twin:
                        state.is_twin = True
                        state.closed = False
                        state.slots['REAR'] = ContainerSlot('REAR', f'{state.origin}:rear', rear_b, center(rear_b), now, now, now, event_id=state.event_id)
                    if 'FRONT' in state.slots:
                        self._update_slot(state.slots['FRONT'], front_b, now)
                    if 'REAR' in state.slots:
                        self._update_slot(state.slots['REAR'], rear_b, now)
                else:
                    # Multiple text boxes on the same single container: track the main box in FRONT slot!
                    best_det, best_b = max(candidates, key=lambda c: c[0].confidence * box_area(c[1]))
                    if 'FRONT' in state.slots:
                        self._update_slot(state.slots['FRONT'], best_b, now)
            elif len(candidates) == 1:
                det, b = candidates[0]
                front_slot = state.slots.get('FRONT')
                rear_slot = state.slots.get('REAR')

                if state.is_twin and rear_slot:
                    if front_slot and front_slot.closed:
                        self._update_slot(rear_slot, b, now)
                    elif rear_slot.closed:
                        self._update_slot(front_slot, b, now)
                    else:
                        front_coord = center(front_slot.bbox)[axis] if front_slot else 0
                        rear_coord = center(rear_slot.bbox)[axis] if rear_slot else 0
                        b_coord = center(b)[axis]
                        front_match = iou(front_slot.bbox, b) if front_slot else 0.0
                        rear_match = iou(rear_slot.bbox, b) if rear_slot else 0.0

                        if rear_match > front_match and rear_match >= 0.10:
                            self._update_slot(rear_slot, b, now)
                        elif front_match >= 0.10:
                            self._update_slot(front_slot, b, now)
                        else:
                            front_dist = abs(front_coord - b_coord)
                            rear_dist = abs(rear_coord - b_coord)
                            if rear_dist < front_dist:
                                self._update_slot(rear_slot, b, now)
                            else:
                                self._update_slot(front_slot, b, now)
                elif front_slot:
                    is_sequential_twin = (
                        camera.gate_exit_axis is not None and
                        self._past_exit(front_slot.bbox, camera) and
                        not self._past_exit(b, camera) and
                        iou(front_slot.bbox, b) < 0.20 and
                        abs(center(front_slot.bbox)[axis] - center(b)[axis]) >= 0.25
                    )
                    if is_sequential_twin:
                        state.is_twin = True
                        state.closed = False
                        state.slots['REAR'] = ContainerSlot('REAR', f'{state.origin}:rear', b, center(b), now, now, now, event_id=state.event_id)
                    else:
                        self._update_slot(front_slot, b, now)

            # Synchronize master state properties
            state.stationary = any(s.stationary for s in state.slots.values()) if state.slots else False
            state.entrance_complete = state.stationary or (
                (now - state.first_seen).total_seconds() >= self.settings.gate_observation_gap_seconds and
                (box_area(state.bbox) >= 0.005 or len(candidates) > 0)
            )
            state.closed = all(s.closed for s in state.slots.values()) if state.slots else False
            if 'FRONT' in state.slots:
                state.bbox = state.slots['FRONT'].bbox
                state.anchor = state.slots['FRONT'].anchor

            if state.closed:
                return None

            # Select which slot's detection to sample for OCR
            front_slot = state.slots.get('FRONT')
            rear_slot = state.slots.get('REAR')

            axis = 1 if camera.gate_exit_axis == 'y' else 0
            front_cand = max(((d, b) for d, b in candidates if front_slot and (iou(front_slot.bbox, b) >= 0.05 or abs(center(front_slot.bbox)[axis] - center(b)[axis]) < 0.25)),
                             key=lambda item: iou(front_slot.bbox, item[1]), default=None)
            rear_cand = max(((d, b) for d, b in candidates if rear_slot and (iou(rear_slot.bbox, b) >= 0.05 or abs(center(rear_slot.bbox)[axis] - center(b)[axis]) < 0.25)),
                            key=lambda item: iou(rear_slot.bbox, item[1]), default=None)

            target = None
            if rear_slot and not rear_slot.closed and rear_cand:
                if not front_slot or front_slot.closed:
                    target = ('REAR', rear_cand[0])
                elif front_cand and front_cand[0] is not rear_cand[0]:
                    target = ('FRONT', front_cand[0]) if getattr(frame, 'sequence', 0) % 2 == 0 else ('REAR', rear_cand[0])
                else:
                    target = ('REAR', rear_cand[0])
            elif front_slot and not front_slot.closed and front_cand:
                target = ('FRONT', front_cand[0])
            elif rear_cand and rear_slot and not rear_slot.closed:
                target = ('REAR', rear_cand[0])
            elif front_cand and front_slot and not front_slot.closed:
                target = ('FRONT', front_cand[0])

            if target:
                state.active_slot = target[0]
                return target[1]

            return None

        # State is None: fresh vehicle arrival
        if not candidates:
            return None

        # Determine if arrival carries Twin 20ft containers or Single (20ft/40ft) container
        non_overlapping = []
        for d, b in sorted(candidates, key=lambda item: box_area(item[1]), reverse=True):
            if not any(iou(b, ex[1]) > 0.15 for ex in non_overlapping):
                non_overlapping.append((d, b))
                if len(non_overlapping) == 2:
                    break

        base_origin = f'dwell:{camera.id}:{uuid4().hex}'
        prov_id = str(uuid4())
        axis = 1 if camera.gate_exit_axis == 'y' else 0
        best_pair = None
        max_sep = 0.0
        for i in range(len(non_overlapping)):
            for j in range(i + 1, len(non_overlapping)):
                c1, c2 = non_overlapping[i], non_overlapping[j]
                if iou(c1[1], c2[1]) >= 0.15:
                    continue
                sep = abs(center(c1[1])[axis] - center(c2[1])[axis])
                align = abs(center(c1[1])[1 - axis] - center(c2[1])[1 - axis])
                if sep < 0.18 or align > 0.35:
                    continue
                area_1, area_2 = box_area(c1[1]), box_area(c2[1])
                area_ratio = min(area_1, area_2) / max(area_1, area_2) if max(area_1, area_2) > 0 else 0
                is_body_twin = (
                    c1[0].class_name not in ('container_number', 'iso_code', 'feet') and
                    c2[0].class_name not in ('container_number', 'iso_code', 'feet') and
                    area_1 >= 0.04 and area_2 >= 0.04 and area_ratio >= 0.25
                )
                is_text_twin = (
                    c1[0].class_name == 'container_number' and
                    c2[0].class_name == 'container_number' and
                    area_1 >= 0.0015 and area_2 >= 0.0015 and
                    c1[0].confidence >= 0.60 and c2[0].confidence >= 0.60
                )
                if is_body_twin or is_text_twin:
                    if sep > max_sep:
                        max_sep = sep
                        best_pair = (c1, c2)

        if best_pair is not None:
            ordered = self._order_slots(best_pair, camera)
            front_det, front_b = ordered[0]
            rear_det, rear_b = ordered[1]
            is_twin = True
            if is_twin:
                slots = {
                    'FRONT': ContainerSlot('FRONT', f'{base_origin}:front', front_b, center(front_b), now, now, now, event_id=None),
                    'REAR': ContainerSlot('REAR', f'{base_origin}:rear', rear_b, center(rear_b), now, now, now, event_id=None),
                }
                self.current = Occupancy(frame.source_id, base_origin, front_b, center(front_b),
                                         now, now, now, event_id=None, is_twin=True, slots=slots, active_slot='FRONT',
                                         provisional_event_id=prov_id)
                return front_det
            else:
                detection, box = non_overlapping[0]
                slots = {
                    'FRONT': ContainerSlot('FRONT', f'{base_origin}:single', box, center(box), now, now, now, event_id=None)
                }
                self.current = Occupancy(frame.source_id, base_origin, box, center(box),
                                         now, now, now, event_id=None, is_twin=False, slots=slots, active_slot='FRONT',
                                         provisional_event_id=prov_id)
                return detection
        else:
            detection, box = non_overlapping[0]
            slots = {
                'FRONT': ContainerSlot('FRONT', f'{base_origin}:single', box, center(box), now, now, now, event_id=None)
            }
            self.current = Occupancy(frame.source_id, base_origin, box, center(box),
                                     now, now, now, event_id=None, is_twin=False, slots=slots, active_slot='FRONT',
                                     provisional_event_id=prov_id)
            return detection

    def confirm(self, origin, event_id, container_number=None):
        if not self.current:
            return
        state = self.current
        matched = False
        for s_id, slot in state.slots.items():
            if slot.origin == origin or (s_id == 'FRONT' and (origin == state.origin or origin.endswith(':front') or origin.endswith(':single'))) or (s_id == 'REAR' and origin.endswith(':rear')):
                slot.event_id = event_id
                slot.container_number = container_number
                slot.closed = True
                matched = True
                break
        if not matched:
            front = state.slots.get('FRONT')
            if front:
                front.event_id = event_id
                front.container_number = container_number
                front.closed = True
        state.event_id = event_id
        if not state.container_number:
            state.container_number = container_number
        state.closed = all(s.closed for s in state.slots.values()) if state.slots else True

    def status(self):
        state = self.current
        if not state:
            return {'state': 'CLEAR', 'state_machine': {'state': 'CLEAR', 'next': ['APPROACHING']}, 'event_id': None}
        if state.clear_since is not None:
            machine_state = 'CLEARING'
        elif state.stationary and state.event_id:
            machine_state = 'WAITING'
        elif state.event_id and getattr(state, 'is_twin', False):
            machine_state = 'OCCUPIED_TWIN'
        elif state.event_id:
            machine_state = 'OCCUPIED_SINGLE'
        elif getattr(state, 'entrance_complete', False):
            machine_state = 'EXITING'
        else:
            machine_state = 'APPROACHING'
        next_states = {
            'CLEAR': ['APPROACHING'], 'APPROACHING': ['OCCUPIED_SINGLE', 'OCCUPIED_TWIN', 'CLEAR'],
            'OCCUPIED_SINGLE': ['WAITING', 'EXITING'], 'OCCUPIED_TWIN': ['WAITING', 'EXITING'],
            'WAITING': ['EXITING', 'CLEARING'], 'EXITING': ['CLEARING', 'WAITING'],
            'CLEARING': ['CLEAR', 'OCCUPIED_SINGLE', 'OCCUPIED_TWIN'],
        }[machine_state]
        return {'state': 'WAITING' if state.stationary and state.event_id else
                'OCCUPIED' if state.event_id else 'TRACKING',
                'state_machine': {'state': machine_state, 'next': next_states},
                'stationary': state.stationary, 'event_id': state.event_id,
                'bbox_normalized': state.bbox, 'clear_pending': state.clear_since is not None,
                'is_twin': getattr(state, 'is_twin', False),
                'entrance_complete': getattr(state, 'entrance_complete', False),
                'slots': {key: {'closed': slot.closed, 'container_number': slot.container_number}
                          for key, slot in getattr(state, 'slots', {}).items()}}
