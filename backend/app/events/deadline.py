"""Retain source-bound OCR evidence; publish only confirmed visits."""
import threading
import time
import re
from app.domain import Observation, Validation
from app.snapshots.snapshot_manager import SnapshotManager


class CaptureDeadlines:
    def __init__(self, settings, repository, snapshots, dispatcher):
        self.settings, self.repository, self.snapshots, self.dispatcher = settings, repository, snapshots, dispatcher
        self.visits = {}
        self.lock = threading.RLock()

    def record(self, job, validation=None):
        if not job.event_id or getattr(job, 'passage_id', None):
            return
        with self.lock:
            visit = self.visits.setdefault(job.event_id, {'samples': {}, 'sent': False})
            if visit['sent']:
                return
            current = self.repository.event(job.event_id)
            if current and current.get('cloud_synced'):
                return
            quality = SnapshotManager.compute_quality(job.frame.image)
            slot = (job.slot_position, job.detection.camera_id)
            value = validation or Validation('', '', 0, False, False, 'UNREADABLE')
            complete = value.valid_format and len(value.normalized_text) == 11
            partial = bool(re.fullmatch(r'\s*[A-Z]{3}[UJZ]\s+[0-9]{6}\s*', value.raw_text or ''))
            score = (bool(value.valid_check_digit), complete, partial,
                     job.detection.class_name == 'container_number', bool(value.raw_text), value.confidence, quality)
            old = visit['samples'].get(slot)
            if old is None or score > old[0]:
                visit['samples'][slot] = (score, job, value)
                # Keep improved OCR with its exact source frame throughout the visit.
                if validation is not None and value.raw_text:
                    origin = job.origin_key + ':late:' + job.frame.timestamp.isoformat() + ':deadline'
                    obs = Observation(origin, job.gate_id, job.track_id, job.detection,
                        value, job.direction, job.frame.image, job.started_at, False,
                        all_detections=list(job.all_detections), slot_position=job.slot_position,
                        event_id=job.event_id, is_twin=job.is_twin)
                    self.repository.save_observation(obs, self.snapshots,
                        self.settings.dedup_seconds, ongoing_event_id=job.event_id)

    def tick(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            for event_id, visit in list(self.visits.items()):
                if visit['sent']:
                    continue
                current = self.repository.event(event_id)
                if current and current.get('cloud_synced'):
                    del self.visits[event_id]
                    continue
                # Elapsed time cannot make an unreadable/partial visit ready.
                if not current or current.get('status') != 'CONFIRMED':
                    continue
                required = {current.get('container_number')}
                if current.get('is_twin'):
                    required.add(current.get('container_number_2'))
                supported = {value.normalized_text for _, _, value in visit['samples'].values()
                             if value.valid_check_digit}
                if None in required or not required.issubset(supported):
                    continue
                self.repository.mark_capture_ready(event_id)
                self.dispatcher.enqueue(None, event_id, delay_seconds=0)
                visit['sent'] = True
                # Durable source jobs own the proof. Do not retain full camera
                # arrays or repeatedly hydrate a completed event in the ticker.
                visit['samples'].clear()
