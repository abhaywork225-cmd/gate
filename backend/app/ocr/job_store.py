from __future__ import annotations
"""Local transactional OCR evidence spool; separate from the event database.

PNG files are flushed and renamed before a metadata-only job is committed.
When image_root is shared with SnapshotManager, all captures and crops live in
the visit folder. The queue database stores references, never a second image copy.
"""
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
import json
import hashlib
import os
from pathlib import Path
import shutil
import sqlite3
import time
from uuid import uuid4

from app.domain import Detection, Frame, Validation, OCRRead
from app.ocr.validator import ContainerValidator


class OCRJobStore:
    def __init__(self, root, max_mb=10240, image_root=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'jobs.sqlite3'
        self.image_root = Path(image_root or self.root / 'images').resolve()
        self.shared_evidence = image_root is not None
        self.image_root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_mb * 1024 * 1024
        with self.connection() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY, origin TEXT NOT NULL, sequence INTEGER NOT NULL,
                metadata TEXT NOT NULL, crop TEXT NOT NULL, image TEXT NOT NULL,
                payload_bytes INTEGER NOT NULL DEFAULT 0,
                state TEXT NOT NULL DEFAULT 'PENDING', validation TEXT, event_id TEXT,
                attempts INTEGER NOT NULL DEFAULT 0, retry_at REAL NOT NULL DEFAULT 0,
                error_type TEXT, UNIQUE(origin, sequence))''')
            db.execute('CREATE INDEX IF NOT EXISTS pending_jobs ON jobs(state, retry_at, id)')
        self._migrate_blobs()
        if self.shared_evidence:
            self._migrate_legacy_images()
            self.reconcile_visit_folders()

    def _migrate_legacy_images(self):
        """Copy referenced legacy files before transactionally switching paths.

        Leave legacy originals recoverable; never remove evidence during startup.
        """
        legacy = self.root / 'images'
        with self.connection() as db:
            rows = db.execute('SELECT id,origin,sequence,metadata,crop,image FROM jobs').fetchall()
            for row in rows:
                meta = json.loads(row['metadata'])
                event = meta.get('event_id') or meta.get('passage_id')
                folder = str(event or uuid4())
                updates = {}
                refs = {'crop': row['crop'], 'image': row['image'],
                        **{f'companion:{cam}': path for cam, path in meta.get('companion_images', {}).items()}}
                # New jobs already point at the shared store.
                if self.image_path(row['image']).is_file():
                    continue
                for key, relative in refs.items():
                    source = (legacy / relative).resolve()
                    if not source.is_relative_to(legacy.resolve()) or not source.is_file():
                        break
                    name = hashlib.sha256(f'{row["origin"]}:{row["sequence"]}:{key}'.encode()).hexdigest()[:20]
                    target = f'{folder}/legacy_{name}{source.suffix}'
                    self._write_image(target, source.read_bytes())
                    updates[key] = target
                else:
                    meta['event_id'] = event or folder
                    meta['companion_images'] = {cam: updates[f'companion:{cam}']
                                                for cam in meta.get('companion_images', {})}
                    db.execute('UPDATE jobs SET crop=?,image=?,metadata=? WHERE id=?',
                               (updates['crop'], updates['image'], json.dumps(meta), row['id']))

    def reconcile_visit_folders(self):
        """Move provisional OCR files into the final event folder after restart."""
        if not self.shared_evidence:
            return
        with self.connection() as db:
            rows = db.execute('SELECT id,origin,event_id,image FROM jobs').fetchall()
        canonical = {row['origin']: row['event_id'] for row in rows if row['event_id']}
        for row in rows:
            event_id = row['event_id'] or canonical.get(row['origin'])
            if not event_id:
                continue
            current = str(row['image']).replace('\\', '/').split('/', 1)[0]
            if current != str(event_id):
                self.complete(row['id'], event_id)

    def image_path(self, relative):
        path = (self.image_root / relative).resolve()
        if not path.is_relative_to(self.image_root) or path == self.image_root:
            raise ValueError('Invalid OCR image path')
        return path

    def _write_image(self, relative, data):
        path = self.image_path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() == data:
                return
        temporary = path.with_name(f'{uuid4().hex[:8]}.tmp')
        try:
            with temporary.open('wb') as file:
                file.write(data)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _save_images(self, origin, sequence, crop, image, folder=None):
        if folder:
            paths = f'{folder}/candidate_{sequence}_crop.png', f'{folder}/candidate_{sequence}_frame.png'
        else:
            folder = hashlib.sha256(json.dumps([origin, sequence]).encode()).hexdigest()
            paths = f'{folder}/crop.png', f'{folder}/frame.png'
        self._write_image(paths[0], crop)
        self._write_image(paths[1], image)
        return paths

    def _migrate_blobs(self):
        # Original database remains recoverable if export fails: all metadata
        # changes roll back. Already exported files are reused on the next run.
        with self.connection() as db:
            columns = {row['name']: row['type'] for row in db.execute('PRAGMA table_info(jobs)')}
            if columns['crop'] != 'BLOB':
                return
            db.execute('BEGIN IMMEDIATE')
            db.execute('ALTER TABLE jobs ADD COLUMN payload_bytes INTEGER NOT NULL DEFAULT 0')
            for row in db.execute('SELECT id,origin,sequence,metadata,crop,image FROM jobs'):
                paths = self._save_images(row['origin'], row['sequence'], row['crop'], row['image'])
                size = len(row['crop']) + len(row['image']) + len(row['metadata'].encode())
                db.execute('UPDATE jobs SET crop=?, image=?, payload_bytes=? WHERE id=?', (*paths, size, row['id']))
            db.execute('''CREATE TABLE jobs_files (
                id INTEGER PRIMARY KEY, origin TEXT NOT NULL, sequence INTEGER NOT NULL,
                metadata TEXT NOT NULL, crop TEXT NOT NULL, image TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'PENDING', validation TEXT, event_id TEXT,
                attempts INTEGER NOT NULL DEFAULT 0, retry_at REAL NOT NULL DEFAULT 0,
                error_type TEXT, payload_bytes INTEGER NOT NULL DEFAULT 0,
                UNIQUE(origin, sequence))''')
            db.execute('INSERT INTO jobs_files SELECT * FROM jobs')
            db.execute('DROP TABLE jobs')
            db.execute('ALTER TABLE jobs_files RENAME TO jobs')
            db.execute('CREATE INDEX pending_jobs ON jobs(state, retry_at, id)')
        # Reclaim pages previously occupied by image blobs, only after commit.
        with self.connection() as db:
            db.execute('VACUUM')

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA synchronous=NORMAL')
        db.execute('''CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY, origin TEXT NOT NULL, sequence INTEGER NOT NULL,
            metadata TEXT NOT NULL, crop TEXT NOT NULL, image TEXT NOT NULL,
            payload_bytes INTEGER NOT NULL DEFAULT 0,
            state TEXT NOT NULL DEFAULT 'PENDING', validation TEXT, event_id TEXT,
            attempts INTEGER NOT NULL DEFAULT 0, retry_at REAL NOT NULL DEFAULT 0,
            error_type TEXT, UNIQUE(origin, sequence))''')
        db.execute('CREATE INDEX IF NOT EXISTS pending_jobs ON jobs(state, retry_at, id)')
        try:
            with db:
                yield db
        finally:
            db.close()

    def save(self, job, crop):
        import cv2
        def encode_img(image, crop_image=False):
            # Preserve full-frame proof pixels exactly; JPEG recompression can
            # replace the sharp accepted frame during cleanup/retry checks.
            extension = '.png'
            options = [cv2.IMWRITE_PNG_COMPRESSION, 1]
            ok, data = cv2.imencode(extension, image, options)
            if not ok:
                raise OSError('OCR evidence encoding failed')
            return data.tobytes()
        crop_data, image_data = encode_img(crop, True), encode_img(job.frame.image)
        import math
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        exposure = float(((gray > 10) & (gray < 245)).mean())
        quality = math.log1p(float(cv2.Laplacian(gray, cv2.CV_64F).var())) * exposure
        folder = str(getattr(job, 'event_id', None) or getattr(job, 'passage_id', None) or uuid4()) if self.shared_evidence else f"{job.origin_key.replace(':', '_')}_{job.frame.sequence}"
        # Flat visit directory, unique names across cameras, tracks and retries.
        sample = hashlib.sha256(f'{job.origin_key}:{job.frame.source_id}:{job.frame.sequence}'.encode()).hexdigest()[:16]
        camera = job.frame.camera_id
        prefix = f'{camera}_{job.frame.sequence}_{sample}'
        companion_paths = {}
        companion_bytes = 0
        if getattr(job, 'companion_frames', None):
            for cam_id, comp_img in job.companion_frames.items():
                if comp_img is not None:
                    try:
                        c_data = encode_img(comp_img)
                        c_path = f'{folder}/{prefix}_companion_{cam_id}.jpg' if self.shared_evidence else f'{folder}/companion_{cam_id}.png'
                        self._write_image(c_path, c_data)
                        companion_paths[cam_id] = c_path
                        companion_bytes += len(c_data)
                    except Exception:
                        pass
        metadata = {
            'origin_key': job.origin_key, 'gate_id': job.gate_id, 'track_id': job.track_id,
            'event_id': getattr(job, 'event_id', None) or (folder if self.shared_evidence else None),
            'passage_id': getattr(job, 'passage_id', None),
            'region_id': getattr(job, 'region_id', None),
            'passage_metadata': getattr(job, 'passage_metadata', {}),
            'proof_quality': quality,
            'direction': job.direction, 'role': job.role, 'started_at': job.started_at.isoformat(),
            'slot_position': getattr(job, 'slot_position', 'FRONT'),
            'is_stationary': getattr(job, 'is_stationary', True),
            'entrance_complete': getattr(job, 'entrance_complete', True),
            'is_twin': getattr(job, 'is_twin', False),
            'roi': job.roi, 'detection': asdict(job.detection),
            'all_detections': [asdict(d) for d in job.all_detections],
            'frame': {'camera_id': job.frame.camera_id, 'timestamp': job.frame.timestamp.isoformat(),
                      'sequence': job.frame.sequence, 'source_id': job.frame.source_id},
            'companion_images': companion_paths}
        encoded = json.dumps(metadata, default=lambda value: value.isoformat())
        payload_bytes = len(crop_data) + len(image_data) + len(encoded.encode()) + companion_bytes
        self._prune_completed(payload_bytes, protected_origins=(job.origin_key,))
        if payload_bytes > self.max_bytes or shutil.disk_usage(self.root).free < payload_bytes + 64 * 1024 * 1024:
            raise OSError('OCR spool lacks free disk space')
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            existing = db.execute('SELECT id FROM jobs WHERE origin=? AND sequence=?',
                                  (job.origin_key, job.frame.sequence)).fetchone()
            if existing:
                return existing['id']
            if self.shared_evidence:
                paths = f'{folder}/{prefix}_crop.png', f'{folder}/{prefix}_frame.png'
                self._write_image(paths[0], crop_data)
                self._write_image(paths[1], image_data)
            else:
                paths = self._save_images(job.origin_key, job.frame.sequence, crop_data, image_data, folder=folder)
            return db.execute('INSERT INTO jobs(origin,sequence,metadata,crop,image,payload_bytes) VALUES(?,?,?,?,?,?)',
                              (job.origin_key, job.frame.sequence, encoded, *paths, payload_bytes)).lastrowid

    def next_id(self):
        with self.connection() as db:
            row = db.execute("SELECT id FROM jobs WHERE state='PENDING' AND retry_at<=? ORDER BY id DESC LIMIT 1",
                             (time.time(),)).fetchone()
            return row['id'] if row else None

    def passage_jobs(self):
        with self.connection() as db:
            return [dict(row) for row in db.execute(
                "SELECT id,state,metadata,event_id FROM jobs WHERE metadata LIKE '%\"passage_id\": \"%' ORDER BY id")]

    def link_passage(self, passage_id, event_id):
        with self.connection() as db:
            db.execute('UPDATE jobs SET event_id=? WHERE origin LIKE ?',
                       (event_id, f'passage:{passage_id}:%'))

    def passage_pending(self, passage_id):
        with self.connection() as db:
            return db.execute("SELECT COUNT(*) FROM jobs WHERE origin LIKE ? AND state='PENDING'",
                              (f'passage:{passage_id}:%',)).fetchone()[0]

    def load(self, job_id):
        import cv2
        import numpy as np
        from app.runtime import OCRJob
        with self.connection() as db:
            row = db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
        data = json.loads(row['metadata'])
        def detection(value):
            return Detection(**{**value, 'frame_timestamp': datetime.fromisoformat(value['frame_timestamp']),
                                'bbox': tuple(value['bbox'])})
        def decode(relative):
            blob = self.image_path(relative).read_bytes()
            result = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
            if result is None:
                raise ValueError('Unreadable persisted OCR image')
            return result
        comp_frames = {}
        for cam_id, rel in data.get('companion_images', {}).items():
            try:
                comp_frames[cam_id] = decode(rel)
            except Exception:
                pass
        frame = Frame(**{**data['frame'], 'timestamp': datetime.fromisoformat(data['frame']['timestamp']),
                         'image': decode(row['image'])})
        job = OCRJob(data['origin_key'], data['gate_id'], data['track_id'], detection(data['detection']),
                     frame, data['direction'], datetime.fromisoformat(data['started_at']), tuple(data['roi']),
                     tuple(detection(d) for d in data['all_detections']), data['role'],
                     companion_frames=comp_frames, slot_position=data.get('slot_position', 'FRONT'),
                     event_id=data.get('event_id'),
                     passage_id=data.get('passage_id'), region_id=data.get('region_id'),
                     passage_metadata=data.get('passage_metadata', {}),
                     is_stationary=data.get('is_stationary', True),
                     entrance_complete=data.get('entrance_complete', True),
                     is_twin=data.get('is_twin', False))
        validation = Validation(**json.loads(row['validation'])) if row['validation'] else None
        return job, decode(row['crop']), validation

    def save_validation(self, job_id, validation):
        with self.connection() as db:
            db.execute('UPDATE jobs SET validation=? WHERE id=?', (json.dumps(asdict(validation)), job_id))

    def best_proof(self, origin, through_id):
        with self.connection() as db:
            rows = db.execute('SELECT id, metadata, validation FROM jobs WHERE origin=? AND id<=? ORDER BY id',
                              (origin, through_id)).fetchall()
        def score(row):
            meta = json.loads(row['metadata']) if row['metadata'] else {}
            val = json.loads(row['validation']) if row['validation'] else {}
            q = float(meta.get('proof_quality', 0))
            checked = ContainerValidator().validate(OCRRead(val.get('raw_text', ''), val.get('confidence', 0)))
            if checked.valid_check_digit:
                q += 1000.0 + (float(val.get('confidence', 0)) * 100.0)
            return q

        rows.sort(key=score, reverse=True)
        for row in rows:
            try:
                return self.load(row['id'])[0]
            except (OSError, ValueError):
                continue
        return None

    def event_crops(self, event_id):
        """Return persisted OCR crops linked to an event for operator review."""
        with self.connection() as db:
            rows = db.execute(
                "SELECT id, crop, metadata, validation FROM jobs "
                "WHERE event_id=? OR metadata LIKE ? ORDER BY id",
                (event_id, f'%\"event_id\": \"{event_id}\"%'),
            ).fetchall()
        result = []
        for row in rows:
            try:
                meta = json.loads(row['metadata'])
                crop_path = self.image_path(row['crop'])
                if not crop_path.is_file():
                    continue
                validation = json.loads(row['validation']) if row['validation'] else {}
                result.append({
                    'id': str(row['id']),
                    'camera_id': (meta.get('frame') or {}).get('camera_id'),
                    'timestamp': (meta.get('frame') or {}).get('timestamp'),
                    'slot_position': meta.get('slot_position', 'FRONT'),
                    'path': row['crop'],
                    'normalized_text': validation.get('normalized_text'),
                })
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        return result

    def crop_path(self, job_id):
        with self.connection() as db:
            row = db.execute('SELECT crop FROM jobs WHERE id=?', (job_id,)).fetchone()
        if not row:
            return None
        path = self.image_path(row['crop'])
        return path if path.is_file() else None

    def evidence(self, origin, limit):
        from app.events.event_manager import Evidence
        with self.connection() as db:
            rows = db.execute("SELECT * FROM jobs WHERE origin=? AND state='DONE' ORDER BY id DESC LIMIT ?",
                              (origin, limit)).fetchall()
        rows = list(reversed(rows))
        return Evidence(reads=[ContainerValidator().validate(OCRRead(
                            json.loads(row['validation']).get('raw_text', ''),
                            json.loads(row['validation']).get('confidence', 0)))
                        for row in rows if row['validation']],
                        event_id=next((row['event_id'] for row in rows if row['event_id']), None),
                        last_sequence=rows[-1]['sequence'] if rows else None)

    def complete(self, job_id, event_id=None):
        with self.connection() as db:
            row = db.execute('SELECT crop,image,metadata FROM jobs WHERE id=?', (job_id,)).fetchone()
            if row and event_id and self.shared_evidence:
                metadata = json.loads(row['metadata'])
                updates = {}
                target_dir = self.image_root / str(event_id)
                target_dir.mkdir(parents=True, exist_ok=True)

                def rehome(relative, label):
                    if not relative:
                        return relative
                    source = self.image_path(relative)
                    if not source.is_file():
                        return relative
                    # Preserve camera/sequence names while avoiding collisions
                    # when two provisional visits contain the same basename.
                    target = target_dir / source.name
                    if source.resolve() != target.resolve() and target.exists():
                        target = target_dir / f'{job_id}_{source.name}'
                    if source.resolve() != target.resolve():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            os.replace(source, target)
                        except PermissionError:
                            # A camera/preview process can briefly hold a
                            # provisional crop open on Windows.  Startup
                            # reconciliation must not fail the whole backend;
                            # copy the immutable evidence and leave the source
                            # for the next cleanup pass.
                            import shutil
                            shutil.copy2(source, target)
                    return str(target.relative_to(self.image_root)).replace('\\', '/')

                crop_path = rehome(row['crop'], 'crop')
                image_path = rehome(row['image'], 'image')
                companions = metadata.get('companion_images') or {}
                metadata['companion_images'] = {
                    cam: rehome(path, f'companion:{cam}') for cam, path in companions.items()
                }
                metadata['event_id'] = str(event_id)
                db.execute("""UPDATE jobs SET crop=?, image=?, metadata=?, state='DONE',
                             event_id=?, error_type=NULL, retry_at=? WHERE id=?""",
                           (crop_path, image_path, json.dumps(metadata), event_id, time.time(), job_id))
                # Remove only an empty provisional visit directory; never delete
                # files belonging to another job.
                old_dirs = {self.image_path(row['crop']).parent, self.image_path(row['image']).parent}
                for old_dir in old_dirs:
                    if old_dir != target_dir:
                        try:
                            old_dir.rmdir()
                        except OSError:
                            pass
            else:
                db.execute("UPDATE jobs SET state='DONE', event_id=?, error_type=NULL, retry_at=? WHERE id=?",
                           (event_id, time.time(), job_id))

    def cancel_pending(self, origin, event_id=None):
        with self.connection() as db:
            db.execute("UPDATE jobs SET state='DONE', event_id=?, error_type=NULL, retry_at=? WHERE origin=? AND state='PENDING'",
                       (event_id, time.time(), origin))

    def clean_completed(self, max_age_seconds=10, protected_origins=()):
        self._prune_completed(0, protected_origins)

    def _prune_completed(self, incoming_bytes, protected_origins=()):
        """Remove oldest completed spool copies only when the size/free-space limit requires it."""
        # Visit evidence is authoritative for UI and cloud retries, not a cache.
        # Never delete those files merely because an OCR job completed.
        if self.shared_evidence:
            return
        with self.connection() as db:
            total = db.execute('SELECT COALESCE(SUM(payload_bytes),0) FROM jobs').fetchone()[0]
            free = shutil.disk_usage(self.root).free
            if total + incoming_bytes <= self.max_bytes and free >= incoming_bytes + 128 * 1024 * 1024:
                return
            rows = db.execute("SELECT id,origin,crop,image,metadata,payload_bytes FROM jobs WHERE state='DONE' ORDER BY id").fetchall()
            protected = set(protected_origins)
            for row in sorted(rows, key=lambda r: (r['origin'] in protected, r['id'])):
                if total + incoming_bytes <= self.max_bytes and free >= incoming_bytes + 128 * 1024 * 1024:
                    break
                db.execute('DELETE FROM jobs WHERE id=? AND state=?', (row['id'], 'DONE'))
                relatives = [row['crop'], row['image']]
                try:
                    relatives += list(json.loads(row['metadata']).get('companion_images', {}).values())
                except (ValueError, TypeError):
                    pass
                for relative in relatives:
                    path = self.image_path(relative)
                    size = path.stat().st_size if path.exists() else 0
                    path.unlink(missing_ok=True)
                    free += size
                total -= row['payload_bytes']
                parent = self.image_path(row['crop']).parent
                if parent != self.image_root:
                    try:
                        parent.rmdir()
                    except OSError:
                        pass

    def retry(self, job_id, error_type):
        with self.connection() as db:
            db.execute('UPDATE jobs SET attempts=attempts+1, retry_at=?, error_type=? WHERE id=?',
                       (time.time() + 5, error_type, job_id))

    def status(self):
        with self.connection() as db:
            row = db.execute("SELECT COUNT(*) total, SUM(state='PENDING') pending, SUM(error_type IS NOT NULL) failed FROM jobs").fetchone()
        return {key: row[key] or 0 for key in ('total', 'pending', 'failed')}
