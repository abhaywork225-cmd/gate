from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
import logging
import threading
from uuid import uuid4

from app.domain import Frame, Detection, utcnow

log = logging.getLogger('gate.pipeline')


@dataclass
class CameraEvidence:
    camera_id: str
    role: str
    frame: Frame
    detections: list[Detection]
    container_boxes: list[tuple[float, float, float, float]]  # [(x1, y1, x2, y2), ...]
    text_boxes: list[Detection]                               # 'cointainer' or 'feet' detections
    direction: str = 'UNKNOWN'
    stationary: bool = False
    best_confidence: float = 0.0
    received_at: datetime = field(default_factory=utcnow)


@dataclass
class GateSnapshot:
    gate_id: str
    event_id: str
    camera_frames: dict[str, CameraEvidence] = field(default_factory=dict)
    first_seen: datetime = field(default_factory=utcnow)
    last_activity: datetime = field(default_factory=utcnow)
    status: str = 'COLLECTING'  # 'COLLECTING', 'READY', 'PROCESSING', 'DONE'
    direction: str = 'UNKNOWN'


class GateCollector:
    """Aggregates multi-camera evidence for a physical gate lane.

    Collects frames from all cameras monitoring the same gate (e.g. Front/Top, Left, Rear).
    When sufficient evidence has accumulated (e.g. vehicle is stationary or collection window
    has elapsed), the snapshot is marked READY for SceneAnalyzer.
    """

    def __init__(
        self,
        gate_id: str,
        settings=None,
        collection_window_seconds: float = 2.5,
        clear_timeout_seconds: float = 6.0,
        min_cameras: int = 2,
    ):
        self.gate_id = gate_id
        self.settings = settings
        self.collection_window_seconds = collection_window_seconds
        self.clear_timeout_seconds = (
            getattr(settings, 'gate_clear_seconds', clear_timeout_seconds) if settings else clear_timeout_seconds
        )
        self.min_cameras = min_cameras
        self._lock = threading.Lock()
        self.current_snapshot: GateSnapshot | None = None
        self._last_processed_event_id: str | None = None
        self._cooldown_until: float = 0.0

    def add_evidence(
        self,
        camera_id: str,
        role: str,
        frame: Frame,
        detections: list[Detection],
        direction: str = 'UNKNOWN',
        stationary: bool = False,
        total_configured_cameras: int = 3,
    ) -> tuple[GateSnapshot | None, bool]:
        """Ingests frame detections from one camera.

        Returns:
            (snapshot, is_ready): The current GateSnapshot and True if ready for scene analysis.
        """
        container_classes = {'container', 'container_object'}
        text_classes = {'container_number', 'iso_code', 'cointainer', 'feet'}

        container_boxes = []
        text_boxes = []
        max_conf = 0.0

        for d in detections:
            cls = d.class_name.lower()
            if cls in container_classes:
                container_boxes.append(d.bbox)
                max_conf = max(max_conf, d.confidence)
            elif cls in text_classes:
                text_boxes.append(d)
                max_conf = max(max_conf, d.confidence)

        has_target = bool(container_boxes or text_boxes)

        with self._lock:
            now = frame.timestamp

            # Check if current snapshot has timed out (vehicle left gate)
            if self.current_snapshot is not None:
                idle_sec = (now - self.current_snapshot.last_activity).total_seconds()
                if idle_sec > self.clear_timeout_seconds:
                    log.info(
                        'gate_collector_cleared',
                        extra={'gate_id': self.gate_id, 'event_id': self.current_snapshot.event_id},
                    )
                    self.current_snapshot = None

            if not has_target:
                return self.current_snapshot, False

            # If no active snapshot, create one
            if self.current_snapshot is None or self.current_snapshot.status == 'DONE':
                new_event_id = str(uuid4())
                self.current_snapshot = GateSnapshot(
                    gate_id=self.gate_id,
                    event_id=new_event_id,
                    first_seen=now,
                    last_activity=now,
                    status='COLLECTING',
                    direction=direction if direction != 'UNKNOWN' else 'UNKNOWN',
                )
                log.info(
                    'gate_collector_started',
                    extra={'gate_id': self.gate_id, 'event_id': new_event_id, 'camera_id': camera_id},
                )

            snapshot = self.current_snapshot
            snapshot.last_activity = now
            if direction != 'UNKNOWN' and snapshot.direction == 'UNKNOWN':
                snapshot.direction = direction

            # Determine if this frame provides better evidence for this camera than existing
            new_evidence = CameraEvidence(
                camera_id=camera_id,
                role=role,
                frame=frame,
                detections=detections,
                container_boxes=container_boxes,
                text_boxes=text_boxes,
                direction=direction,
                stationary=stationary,
                best_confidence=max_conf,
                received_at=now,
            )

            existing = snapshot.camera_frames.get(camera_id)
            if existing is None or self._is_better_evidence(new_evidence, existing):
                snapshot.camera_frames[camera_id] = new_evidence

            # Check if snapshot is ready for scene analysis
            if snapshot.status == 'COLLECTING' and self._check_ready(snapshot, now, total_configured_cameras):
                snapshot.status = 'READY'
                return snapshot, True

            return snapshot, False

    def mark_processing(self, snapshot: GateSnapshot) -> None:
        with self._lock:
            if self.current_snapshot and self.current_snapshot.event_id == snapshot.event_id:
                self.current_snapshot.status = 'PROCESSING'

    def mark_done(self, snapshot: GateSnapshot) -> None:
        with self._lock:
            if self.current_snapshot and self.current_snapshot.event_id == snapshot.event_id:
                self.current_snapshot.status = 'DONE'
                self._last_processed_event_id = snapshot.event_id

    def reset(self) -> None:
        with self._lock:
            self.current_snapshot = None

    def _is_better_evidence(self, new: CameraEvidence, old: CameraEvidence) -> bool:
        # Prefer stationary frames
        if new.stationary and not old.stationary:
            return True
        if old.stationary and not new.stationary:
            return False

        # Prefer frames with text detections
        if len(new.text_boxes) > len(old.text_boxes):
            return True
        if len(new.text_boxes) < len(old.text_boxes):
            return False

        # Prefer frames with container detections
        if len(new.container_boxes) > len(old.container_boxes):
            return True

        # Prefer higher confidence
        return new.best_confidence > old.best_confidence

    def _check_ready(self, snapshot: GateSnapshot, now: datetime, total_configured: int) -> bool:
        cams_with_evidence = len(snapshot.camera_frames)
        if cams_with_evidence == 0:
            return False

        duration = (now - snapshot.first_seen).total_seconds()
        has_stationary = any(e.stationary for e in snapshot.camera_frames.values())
        has_text = any(len(e.text_boxes) > 0 for e in snapshot.camera_frames.values())

        # If only 1 camera is configured / online on this gate (single-camera setup or video replay)
        effective_min = min(self.min_cameras, max(1, total_configured))

        # Condition 1: Vehicle is stationary on any camera and we have at least effective_min cameras
        if has_stationary and cams_with_evidence >= effective_min:
            return True

        # Condition 2: Text detection found on any camera + enough cameras have contributed
        if has_text and cams_with_evidence >= effective_min and duration >= 1.0:
            return True

        # Condition 3: All configured cameras have provided evidence
        if cams_with_evidence >= total_configured and duration >= 1.0:
            return True

        # Condition 4: Collection window exceeded with at least 1 camera seeing containers
        if duration >= self.collection_window_seconds:
            return True

        return False
