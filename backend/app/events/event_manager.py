from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
import logging
from app.domain import Observation

log = logging.getLogger('gate')


@dataclass
class Evidence:
    reads: list = field(default_factory=list)
    event_id: str | None = None
    last_seen: datetime | None = None
    last_sequence: int | None = None
    confirmed: bool = False


class EventManager:
    """Called only by the OCR worker; commit succeeds before marking a track done."""
    def __init__(self, settings, repository, snapshots, dispatcher=None, runtime=None):
        self.settings, self.repository, self.snapshots = settings, repository, snapshots
        self.dispatcher = dispatcher
        self.runtime = runtime
        self.evidence: dict[str, Evidence] = {}
        self._dispatched_events: set[str] = set()

    def prune(self, active_origins):
        self.evidence = {key: value for key, value in self.evidence.items() if key in active_origins}

    def observe(self, job, validation, additional_frames: dict | None = None, proof_job=None, ongoing_event_id=None):
        evidence = self.evidence.setdefault(job.origin_key, Evidence())
        # A persisted event is already committed. This also covers NEEDS_REVIEW
        # events: replaying the same spool job must not clear its event link or
        # create a second record after a restart.
        if evidence.event_id and evidence.confirmed:
            return evidence.event_id, False
        evidence.last_seen = job.detection.frame_timestamp
        # Retrying the same failed DB write is not another independent OCR vote.
        if evidence.last_sequence != job.frame.sequence:
            evidence.reads.append(validation)
            evidence.last_sequence = job.frame.sequence
        evidence.reads = evidence.reads[-self.settings.ocr_max_attempts:]
        good = [r for r in evidence.reads if r and r.valid_check_digit and getattr(r, 'confidence', 1.0) >= self.settings.ocr_min_confidence]
        votes = Counter(r.normalized_text for r in good)
        best, count = votes.most_common(1)[0] if votes else (None, 0)
        tied = sum(v == count for v in votes.values()) > 1
        confirmed = count >= self.settings.ocr_confirmations and not tied
        # A dwell track can end before the configured vote count is reached
        # (camera handoff, truck leaving the ROI, or a short queue passage).
        # If at least two samples were processed and one high-confidence,
        # check-digit-valid read exists, preserve it as the event identity
        # rather than silently dropping the completed track.
        dwell_end_fallback = (
            job.origin_key.startswith('dwell:') and
            getattr(job, 'entrance_complete', False) and
            len(evidence.reads) >= 2 and count >= 1 and not tied
        )
        if dwell_end_fallback:
            confirmed = True
        if not confirmed and len(evidence.reads) >= self.settings.ocr_max_attempts and count >= 1 and not tied:
            confirmed = True
        if not confirmed:
            # Let the truck complete its entrance before publishing an unconfirmed / NEEDS_REVIEW event!
            is_entrance_done = getattr(job, 'entrance_complete', True) or getattr(job, 'is_stationary', True)
            if len(evidence.reads) < self.settings.ocr_max_attempts:
                # If we have a valid check-digit candidate, wait for confirmation votes
                if count >= 1 or not is_entrance_done:
                    return None
            # If all attempts yielded no readable text, flag for review with an alert rather than silently dropping
            if not any(getattr(r, 'raw_text', None) for r in evidence.reads if r):
                from app.domain import Validation
                validation = Validation(
                    raw_text='UNREADABLE', normalized_text='', confidence=0.0,
                    valid_format=False, valid_check_digit=False, validation_status='UNREADABLE'
                )
                log.warning('passage_unreadable_flagged_for_review', extra={
                    'origin': job.origin_key, 'gate_id': job.gate_id, 'track_id': job.track_id
                })
        selected = max((r for r in good if r.normalized_text == best), key=lambda r: r.confidence) if confirmed else validation
        proof = proof_job or job
        obs = Observation(job.origin_key, job.gate_id, job.track_id, proof.detection, selected,
                          job.direction, proof.frame.image, job.started_at, confirmed,
                          all_detections=list(getattr(proof, 'all_detections', [])),
                          slot_position=getattr(job, 'slot_position', 'FRONT'),
                          event_id=getattr(job, 'event_id', None),
                          is_twin=getattr(job, 'is_twin', False))

        target_event_id = ongoing_event_id or evidence.event_id or getattr(job, 'event_id', None)
        options = {'ongoing_event_id': target_event_id} if target_event_id else {}
        try:
            event_id, created = self.repository.save_observation(
                obs, self.snapshots, self.settings.dedup_seconds, additional_frames,
                same_container_window_seconds=self.settings.same_container_event_window_seconds,
                **options)
        except TypeError as exc:
            # Keep compatibility with small repository test doubles/plugins
            # that still implement the pre-window method signature.
            if 'same_container_window_seconds' not in str(exc):
                raise
            event_id, created = self.repository.save_observation(
                obs, self.snapshots, self.settings.dedup_seconds, additional_frames, **options)
        evidence.event_id = event_id
        evidence.confirmed = confirmed

        # Only dispatch to cloud and buzz for new events or newly confirmed events that haven't been dispatched yet
        # A twin is one cloud transaction. Do not publish the front slot while
        # the rear slot or its proof image is still missing; doing so permanently
        # leaves Prosper with a single-container payload because the dispatcher
        # deduplicates already-sent event IDs.
        cloud_ready = True
        if getattr(obs, 'is_twin', False) and event_id:
            current = self.repository.event(event_id) or {}
            snapshots = current.get('snapshots') or []
            cloud_ready = bool(
                current.get('container_number') and current.get('container_number_2') and
                any(str(item.get('image_path', '')).endswith('_rear.jpg') for item in snapshots)
            )
        should_dispatch = obs.confirmed and cloud_ready and (event_id not in self._dispatched_events)
        if self.dispatcher and should_dispatch:
            self._dispatched_events.add(event_id)
            # Do not publish at the first valid OCR vote.  A second camera can
            # report the rear container a few seconds later; delaying the
            # immutable cloud POST lets repository.save_observation merge that
            # evidence into one twin visit.
            self.dispatcher.enqueue(
                obs,
                event_id,
                delay_seconds=getattr(self.settings, 'cloud_dispatch_grace_seconds', 0.0),
            )
        if self.runtime and obs.direction in ('EXIT', 'OUT', 'GATE_OUT') and should_dispatch:
            try:
                cntr = obs.ocr.normalized_text if (obs and obs.ocr and obs.confirmed) else 'Unconfirmed'
                self.runtime.emit_alert(
                    alert_type='GATE_OUT',
                    title='Gate OUT Movement',
                    message=f'Container {cntr} exiting on lane {obs.gate_id}',
                    level='WARNING',
                    container_number=cntr if obs.confirmed else None,
                    event_id=event_id,
                    buzzer=True
                )
            except Exception:
                pass
        log.info('gate_event_created' if created else 'gate_evidence_associated',
                 extra={'camera_id': job.detection.camera_id, 'event_id': event_id})
        return event_id, created
