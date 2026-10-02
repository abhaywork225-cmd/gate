from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
import logging

from app.domain import Frame, Detection, utcnow
from app.events.gate_collector import GateSnapshot, CameraEvidence

log = logging.getLogger('gate.pipeline')


@dataclass
class ContainerTarget:
    slot: str  # 'FRONT', 'REAR', 'CONTAINER_1', 'CONTAINER_2'
    best_camera_id: str
    crop_frame: Frame
    crop_bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    text_bboxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    expected_size: str = 'UNKNOWN'  # '20FT', '40FT', 'UNKNOWN'
    confidence: float = 0.0


@dataclass
class SceneResult:
    scene_type: str  # 'SINGLE_40FT', 'TWIN_20FT', 'TWO_40FT', 'UNKNOWN'
    containers: list[ContainerTarget]  # 1 or 2 targets
    is_twin: bool
    gate_id: str
    event_id: str
    timestamp: datetime
    direction: str = 'UNKNOWN'
    metadata: dict = field(default_factory=dict)


class SceneAnalyzer:
    """Analyzes multi-camera GateSnapshot to classify the scene:

    - SINGLE_40FT: Standard single 40ft (or 20ft single) container on truck.
    - TWIN_20FT: Two 20ft containers on chassis (Front slot + Rear slot).
    - TWO_40FT: Two distinct 40ft containers at the gate (special tandem/double trailer scenario).
    """

    def __init__(self, settings=None):
        self.settings = settings

    @staticmethod
    def _box_center_x_norm(bbox: tuple[float, float, float, float], frame: Frame | None = None) -> float:
        mid_x = (bbox[0] + bbox[2]) / 2.0
        if mid_x <= 1.0:
            return mid_x
        w = float(frame.image.shape[1]) if frame and getattr(frame, 'image', None) is not None and len(frame.image.shape) > 1 else 1920.0
        return mid_x / max(w, 1.0)

    @staticmethod
    def _box_width_norm(bbox: tuple[float, float, float, float], frame: Frame | None = None) -> float:
        w = bbox[2] - bbox[0]
        if bbox[2] <= 1.0:
            return w
        img_w = float(frame.image.shape[1]) if frame and getattr(frame, 'image', None) is not None and len(frame.image.shape) > 1 else 1920.0
        return w / max(img_w, 1.0)

    def analyze(self, snapshot: GateSnapshot) -> SceneResult:
        cams = snapshot.camera_frames
        if not cams:
            return SceneResult(
                scene_type='UNKNOWN',
                containers=[],
                is_twin=False,
                gate_id=snapshot.gate_id,
                event_id=snapshot.event_id,
                timestamp=snapshot.last_activity,
                direction=snapshot.direction,
            )

        # Categorize available camera evidence by role
        rear_ev: CameraEvidence | None = None
        front_ev: CameraEvidence | None = None
        side_evs: list[CameraEvidence] = []

        for ev in cams.values():
            role = (ev.role or '').upper()
            if role == 'REAR' or 'rear' in ev.camera_id.lower():
                rear_ev = ev
            elif role in ('FRONT_TOP', 'FRONT') or 'front' in ev.camera_id.lower():
                front_ev = ev
            elif role in ('LEFT', 'RIGHT', 'SIDE') or any(s in ev.camera_id.lower() for s in ('left', 'right', 'side')):
                side_evs.append(ev)
            else:
                # Default fallback
                side_evs.append(ev)

        # Step 1: Count container objects and assess geometry across cameras
        multi_box_side = False
        wide_boxes_count = 0
        separated_text_clusters = False
        all_side_boxes: list[tuple[float, float, float, float]] = []

        for sev in side_evs:
            boxes = sev.container_boxes
            all_side_boxes.extend(boxes)
            if len(boxes) >= 2:
                # Check separation between boxes
                boxes_sorted = sorted(boxes, key=lambda b: b[0])
                gap = boxes_sorted[1][0] - boxes_sorted[0][2]
                norm_gap = self._box_width_norm((0, 0, gap, 0), sev.frame) if gap > 1.0 else gap
                if norm_gap >= -0.05:  # Distinct or slightly overlapping boxes
                    multi_box_side = True
            for b in boxes:
                w = b[2] - b[0]
                h = b[3] - b[1]
                aspect = w / max(h, 0.01)
                norm_w = self._box_width_norm(b, sev.frame)
                if aspect > 2.0 or norm_w > 0.6:
                    wide_boxes_count += 1

            # Check text clusters along x-axis
            if len(sev.text_boxes) >= 2:
                xs = [self._box_center_x_norm(d.bbox, sev.frame) for d in sev.text_boxes]
                if min(xs) < 0.45 and max(xs) > 0.55:
                    separated_text_clusters = True

        # Step 2: Determine scene classification
        scene_type = 'SINGLE_40FT'

        # Special Case: Two distinct 40ft containers (wide_boxes >= 2 with high separation)
        if wide_boxes_count >= 2:
            scene_type = 'TWO_40FT'
        # Twin 20ft: Side camera sees 2 separated boxes OR separated text clusters OR
        # both Front and Rear cameras have distinct container detections while side boxes are compact (<2.0 aspect)
        elif multi_box_side or separated_text_clusters:
            scene_type = 'TWIN_20FT'
        elif rear_ev and front_ev and len(rear_ev.container_boxes) > 0 and len(front_ev.container_boxes) > 0:
            # If side view shows non-wide boxes (or no side camera available)
            if wide_boxes_count == 0:
                scene_type = 'TWIN_20FT'
            else:
                scene_type = 'SINGLE_40FT'
        else:
            # Check single camera case if only 1 camera reported
            if len(cams) == 1:
                single_ev = next(iter(cams.values()))
                if len(single_ev.container_boxes) >= 2 or separated_text_clusters:
                    scene_type = 'TWIN_20FT'
                else:
                    scene_type = 'SINGLE_40FT'

        # Step 3: Build ContainerTarget list with best camera assignment
        containers: list[ContainerTarget] = []
        is_twin = scene_type in ('TWIN_20FT', 'TWO_40FT')

        if scene_type == 'TWIN_20FT':
            # Target 1: FRONT container
            front_target = self._build_front_target(front_ev, side_evs, cams)
            containers.append(front_target)

            # Target 2: REAR container
            rear_target = self._build_rear_target(rear_ev, side_evs, cams)
            containers.append(rear_target)

        elif scene_type == 'TWO_40FT':
            # Two 40ft containers: Container 1 (Front/Lead) and Container 2 (Rear/Trailing)
            front_target = self._build_front_target(front_ev, side_evs, cams, expected_size='40FT')
            containers.append(front_target)

            rear_target = self._build_rear_target(rear_ev, side_evs, cams, expected_size='40FT')
            containers.append(rear_target)

        else:  # SINGLE_40FT
            single_target = self._build_single_target(rear_ev, side_evs, front_ev, cams)
            containers.append(single_target)

        log.info(
            'scene_analyzed',
            extra={
                'gate_id': snapshot.gate_id,
                'event_id': snapshot.event_id,
                'scene_type': scene_type,
                'is_twin': str(is_twin),
            },
        )

        return SceneResult(
            scene_type=scene_type,
            containers=containers,
            is_twin=is_twin,
            gate_id=snapshot.gate_id,
            event_id=snapshot.event_id,
            timestamp=snapshot.last_activity,
            direction=snapshot.direction,
            metadata={
                'wide_boxes_count': wide_boxes_count,
                'multi_box_side': multi_box_side,
                'separated_text': separated_text_clusters,
                'camera_count': len(cams),
            },
        )

    def _build_front_target(
        self,
        front_ev: CameraEvidence | None,
        side_evs: list[CameraEvidence],
        cams: dict[str, CameraEvidence],
        expected_size: str = '20FT',
    ) -> ContainerTarget:
        # Best camera for front: Side camera front half or front_top camera
        # If front camera has text boxes, prefer front camera
        if front_ev and len(front_ev.text_boxes) > 0:
            best_ev = front_ev
            bbox = self._best_bbox(front_ev.container_boxes)
            text_boxes = [d.bbox for d in front_ev.text_boxes]
        else:
            # Check side camera front-half text boxes
            best_side = self._find_ev_with_front_text(side_evs)
            if best_side:
                best_ev = best_side
                bbox = self._get_split_bbox(best_side.container_boxes, slot='FRONT')
                text_boxes = [d.bbox for d in best_side.text_boxes if self._box_center_x_norm(d.bbox, best_side.frame) < 0.55]
            elif front_ev:
                best_ev = front_ev
                bbox = self._best_bbox(front_ev.container_boxes)
                text_boxes = [d.bbox for d in front_ev.text_boxes]
            else:
                best_ev = next(iter(cams.values()))
                bbox = self._best_bbox(best_ev.container_boxes)
                text_boxes = [d.bbox for d in best_ev.text_boxes]

        return ContainerTarget(
            slot='FRONT',
            best_camera_id=best_ev.camera_id,
            crop_frame=best_ev.frame,
            crop_bbox=bbox,
            text_bboxes=text_boxes,
            expected_size=expected_size,
            confidence=best_ev.best_confidence,
        )

    def _build_rear_target(
        self,
        rear_ev: CameraEvidence | None,
        side_evs: list[CameraEvidence],
        cams: dict[str, CameraEvidence],
        expected_size: str = '20FT',
    ) -> ContainerTarget:
        # Dedicated rear camera is best for rear container doors!
        if rear_ev:
            bbox = self._best_bbox(rear_ev.container_boxes)
            text_boxes = [d.bbox for d in rear_ev.text_boxes]
            return ContainerTarget(
                slot='REAR',
                best_camera_id=rear_ev.camera_id,
                crop_frame=rear_ev.frame,
                crop_bbox=bbox,
                text_bboxes=text_boxes,
                expected_size=expected_size,
                confidence=rear_ev.best_confidence,
            )

        # Fallback to rear half of side camera
        best_side = self._find_ev_with_rear_text(side_evs)
        if best_side:
            best_ev = best_side
            bbox = self._get_split_bbox(best_side.container_boxes, slot='REAR')
            text_boxes = [d.bbox for d in best_side.text_boxes if self._box_center_x_norm(d.bbox, best_side.frame) >= 0.45]
        else:
            best_ev = next(iter(cams.values()))
            bbox = self._best_bbox(best_ev.container_boxes)
            text_boxes = [d.bbox for d in best_ev.text_boxes]

        return ContainerTarget(
            slot='REAR',
            best_camera_id=best_ev.camera_id,
            crop_frame=best_ev.frame,
            crop_bbox=bbox,
            text_bboxes=text_boxes,
            expected_size=expected_size,
            confidence=best_ev.best_confidence,
        )

    def _build_single_target(
        self,
        rear_ev: CameraEvidence | None,
        side_evs: list[CameraEvidence],
        front_ev: CameraEvidence | None,
        cams: dict[str, CameraEvidence],
    ) -> ContainerTarget:
        # For single container, prefer rear camera if it has text detections (doors have clear numbers!)
        if rear_ev and len(rear_ev.text_boxes) > 0:
            return ContainerTarget(
                slot='CONTAINER_1',
                best_camera_id=rear_ev.camera_id,
                crop_frame=rear_ev.frame,
                crop_bbox=self._best_bbox(rear_ev.container_boxes),
                text_bboxes=[d.bbox for d in rear_ev.text_boxes],
                expected_size='40FT',
                confidence=rear_ev.best_confidence,
            )

        # Otherwise prefer whichever camera has the most text detections
        candidates = list(cams.values())
        best_ev = max(candidates, key=lambda ev: (len(ev.text_boxes), ev.best_confidence))
        return ContainerTarget(
            slot='CONTAINER_1',
            best_camera_id=best_ev.camera_id,
            crop_frame=best_ev.frame,
            crop_bbox=self._best_bbox(best_ev.container_boxes),
            text_bboxes=[d.bbox for d in best_ev.text_boxes],
            expected_size='40FT',
            confidence=best_ev.best_confidence,
        )

    def _best_bbox(self, boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
        if not boxes:
            return (0.0, 0.0, 1.0, 1.0)
        # Largest bounding box by area
        return max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))

    def _get_split_bbox(
        self, boxes: list[tuple[float, float, float, float]], slot: str
    ) -> tuple[float, float, float, float]:
        if not boxes:
            return (0.0, 0.0, 0.55, 1.0) if slot == 'FRONT' else (0.45, 0.0, 1.0, 1.0)

        boxes_sorted = sorted(boxes, key=lambda b: b[0])
        if len(boxes_sorted) >= 2:
            return boxes_sorted[0] if slot == 'FRONT' else boxes_sorted[-1]

        # Single large box, split horizontally
        b = boxes_sorted[0]
        mid_x = (b[0] + b[2]) / 2
        if slot == 'FRONT':
            return (b[0], b[1], mid_x, b[3])
        return (mid_x, b[1], b[2], b[3])

    def _find_ev_with_front_text(self, side_evs: list[CameraEvidence]) -> CameraEvidence | None:
        for ev in side_evs:
            if any(self._box_center_x_norm(d.bbox, ev.frame) < 0.55 for d in ev.text_boxes):
                return ev
        return side_evs[0] if side_evs else None

    def _find_ev_with_rear_text(self, side_evs: list[CameraEvidence]) -> CameraEvidence | None:
        for ev in side_evs:
            if any(self._box_center_x_norm(d.bbox, ev.frame) >= 0.45 for d in ev.text_boxes):
                return ev
        return side_evs[0] if side_evs else None
