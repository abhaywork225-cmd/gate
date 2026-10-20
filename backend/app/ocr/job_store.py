from __future__ import annotations
"""Local transactional OCR evidence spool; separate from the event database.

PNG files are flushed and renamed before a metadata-only job is committed.
When image_root is shared with SnapshotManager, capture references live in
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
    def __init__(self, root, max_mb=10240, image_root=None, staged_visits=False, shared_visit_folders=False):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'jobs.sqlite3'
        self.image_root = Path(image_root or self.root / 'images').resolve()
        self.shared_evidence = image_root is not None
        self.staged_visits = staged_visits
        self.shared_visit_folders = shared_visit_folders
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
                    target = f'waiting/{folder}/legacy_{name}{source.suffix}'
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
            rows = db.execute('SELECT id,origin,event_id,image,metadata,state FROM jobs').fetchall()
        canonical = {row['origin']: row['event_id'] for row in rows if row['event_id']}
        for row in rows:
            metadata = json.loads(row['metadata'])
            if metadata.get('storage_visit_id'):
                continue  # Final record IDs never rename shared visit folders.
            if metadata.get('passage_id') and not metadata.get('finalized'):
                continue  # Recovery must not promote or complete pending visits.
            event_id = row['event_id'] or canonical.get(row['origin'])
            if not event_id:
                continue
            clean_event = str(event_id).replace('\\', '/').strip('/')
            for c in ('waiting', 'detected_in', 'detected_out'):
                if clean_event.startswith(f'{c}/'):
                    clean_event = clean_event[len(c)+1:]
                    break
            direction = metadata.get('direction')
            gate_id = metadata.get('gate_id')
            cat = 'detected_in'
            d = str(direction or '').upper()
            g = str(gate_id or '').lower()
            if d in ('EXIT', 'OUT', 'GATE_OUT') or any(k in g for k in ('out', 'exit')):
                cat = 'detected_out'
            target_rel = f'{cat}/{clean_event}'
            current = str(row['image']).replace('\\', '/').strip('/')
            if current.startswith(target_rel + '/'):
                continue
            self.complete(row['id'], event_id)

    def image_path(self, relative):
        rel_str = str(relative).replace('\\', '/').strip('/')
        direct = (self.image_root / rel_str).resolve()
        if direct.is_relative_to(self.image_root) and direct.exists():
            return direct
        clean_rel = rel_str
        has_cat = False
        for cat in ('waiting', 'detected_in', 'detected_out'):
            if clean_rel.startswith(f'{cat}/'):
                clean_rel = clean_rel[len(cat)+1:]
                has_cat = True
                break
        for cat in ('waiting', 'detected_in', 'detected_out'):
            cand = (self.image_root / cat / clean_rel).resolve()
            if cand.is_relative_to(self.image_root) and cand.exists():
                return cand
        if not has_cat:
            fallback = (self.image_root / 'waiting' / clean_rel).resolve()
            if fallback.is_relative_to(self.image_root) and fallback != self.image_root:
                return fallback
        if not direct.is_relative_to(self.image_root) or direct == self.image_root:
            raise ValueError('Invalid OCR image path')
        return direct

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
            paths = f'waiting/{folder}/crop.png', f'waiting/{folder}/frame.png'
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
        visit_folder = folder
        if self.shared_visit_folders and getattr(job,'passage_id',None):
            folder = folder  # Camera and sequence are encoded in the filename.
        elif self.staged_visits and getattr(job,'passage_id',None):
            folder = f'_processing/{folder}/{job.frame.camera_id}'
        if self.shared_evidence:
            clean_f = folder.replace('\\', '/').strip('/')
            if not any(clean_f.startswith(f'{c}/') for c in ('waiting', 'detected_in', 'detected_out')):
                folder = f'waiting/{clean_f}'
        # Unique names across cameras, tracks and retries.
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
            'event_id': getattr(job, 'event_id', None) or (visit_folder if self.shared_evidence else None),
            'storage_visit_id': visit_folder if self.shared_visit_folders else None,
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
            job_id = db.execute('INSERT INTO jobs(origin,sequence,metadata,crop,image,payload_bytes) VALUES(?,?,?,?,?,?)',
                              (job.origin_key, job.frame.sequence, encoded, *paths, payload_bytes)).lastrowid
            # Keep source files, but don't repeatedly OCR every obsolete
            # frame of one waiting truck or passage. Retain three independent samples.
            db.execute("UPDATE jobs SET state='SUPERSEDED' WHERE origin=? AND state='PENDING' "
                       "AND id NOT IN (SELECT id FROM jobs WHERE origin=? AND state='PENDING' ORDER BY id DESC LIMIT 3)",
                       (job.origin_key, job.origin_key))
            return job_id

    def clean_stale_pending(self, max_age_seconds: float = 60.0):
        """Mark uncompleted jobs from expired/historical passages as superseded."""
        with self.connection() as db:
            cutoff = time.time() - max_age_seconds
            rows = db.execute("SELECT id, metadata FROM jobs WHERE state='PENDING'").fetchall()
            stale_ids = []
            for r in rows:
                try:
                    meta = json.loads(r['metadata'])
                    ts_str = meta.get('frame', {}).get('timestamp') or meta.get('started_at')
                    if ts_str:
                        if not ts_str.endswith('Z') and '+' not in ts_str:
                            ts_str += '+00:00'
                        dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                        if dt.timestamp() < cutoff:
                            stale_ids.append(r['id'])
                    else:
                        stale_ids.append(r['id'])
                except Exception:
                    stale_ids.append(r['id'])
            if stale_ids:
                db.executemany("UPDATE jobs SET state='SUPERSEDED' WHERE id=?", [(i,) for i in stale_ids])

    def completed_event(self, job_id):
        with self.connection() as db:
            row = db.execute("SELECT event_id FROM jobs WHERE id=? AND state='DONE'", (job_id,)).fetchone()
            return row['event_id'] if row else None

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

    def join_visit_folder(self, passage_id, visit_id):
        """Unify proven camera views without changing job state or ownership."""
        if not self.shared_visit_folders:
            return
        with self.connection() as db:
            rows = db.execute('SELECT * FROM jobs WHERE origin LIKE ?',
                              (f'passage:{passage_id}:%',)).fetchall()
            for row in rows:
                meta = json.loads(row['metadata'])
                if meta.get('finalized'):
                    continue
                camera = meta['detection']['camera_id']
                clean_visit = str(visit_id).replace('\\', '/').strip('/')
                for c in ('waiting', 'detected_in', 'detected_out'):
                    if clean_visit.startswith(f'{c}/'):
                        clean_visit = clean_visit[len(c)+1:]
                        break
                target_dir = self.image_root / 'waiting' / clean_visit
                previous_visit = meta.get('storage_visit_id')
                if previous_visit and previous_visit != clean_visit:
                    clean_prev = str(previous_visit).replace('\\', '/').strip('/')
                    for c in ('waiting', 'detected_in', 'detected_out'):
                        if clean_prev.startswith(f'{c}/'):
                            clean_prev = clean_prev[len(c)+1:]
                            break
                    previous_dir = self.image_root / 'waiting' / clean_prev
                    if not previous_dir.exists():
                        previous_dir = self.image_root / clean_prev
                    for context in previous_dir.glob('*_context_*.jpg'):
                        target_dir.mkdir(parents=True,exist_ok=True)
                        target = target_dir / context.name
                        if target.exists():target = target_dir / f'{clean_prev[:8]}_{context.name}'
                        os.replace(context,target)
                        sidecar = context.with_suffix('.json')
                        if sidecar.is_file():
                            os.replace(sidecar,target.with_suffix('.json'))
                references = {'crop':row['crop'],'image':row['image']}
                if meta.get('original_crop'):
                    references['original_crop'] = meta['original_crop']
                references.update({f'companion:{cam}':path for cam,path in meta.get('companion_images',{}).items()})
                updates = {}
                old_dirs = set()
                for label,relative in references.items():
                    source = self.image_path(relative)
                    target = target_dir / source.name
                    old_dirs.add(source.parent)
                    if source.resolve() != target.resolve() and source.is_file():
                        target_dir.mkdir(parents=True,exist_ok=True)
                        if target.exists():
                            target = target_dir / f'{row["id"]}_{source.name}'
                        os.replace(source,target)
                    updates[label] = str(target.relative_to(self.image_root)).replace('\\', '/') if target.exists() else relative
                meta['storage_visit_id'] = clean_visit
                meta['event_id'] = clean_visit
                if 'original_crop' in updates:meta['original_crop'] = updates['original_crop']
                meta['companion_images'] = {cam:updates[f'companion:{cam}'] for cam in meta.get('companion_images',{})}
                db.execute('UPDATE jobs SET crop=?,image=?,metadata=?,event_id=? WHERE id=?',
                           (updates['crop'],updates['image'],json.dumps(meta),clean_visit,row['id']))
                for directory in old_dirs:
                    while directory != self.image_root and directory != target_dir:
                        try:directory.rmdir()
                        except OSError:break
                        directory = directory.parent

    def link_region(self, passage_id, region_id, event_id):
        """Split container records cannot relabel another region's jobs."""
        origin = f'passage:{passage_id}:{region_id}'
        with self.connection() as db:
            rows = db.execute('SELECT id,metadata FROM jobs WHERE origin=?',(origin,)).fetchall()
            for row in rows:
                meta = json.loads(row['metadata'])
                meta['event_id'] = event_id
                meta['finalized'] = True
                db.execute('UPDATE jobs SET event_id=?,metadata=? WHERE id=?',
                           (event_id,json.dumps(meta),row['id']))
        for row in rows:
            self.complete(row['id'],event_id)

    def _remove_empty_staging_parents(self, path):
        staging = self.image_root / '_processing'
        parent = path.parent
        if not parent.is_relative_to(staging):
            return
        while parent != self.image_root:
            try:
                parent.rmdir()
            except OSError:
                break  # Another camera/job still owns files in this directory.
            parent = parent.parent

    def discard_region(self, passage_id, region_id):
        """Finish non-proof companion work without resurrecting it on restart."""
        origin = f'passage:{passage_id}:{region_id}'
        with self.connection() as db:
            rows = db.execute('SELECT id,metadata,crop,image FROM jobs WHERE origin=?',(origin,)).fetchall()
            for row in rows:
                meta = json.loads(row['metadata'])
                meta['finalized'] = True
                db.execute("UPDATE jobs SET state='SUPERSEDED',metadata=? WHERE id=?",(json.dumps(meta),row['id']))
                for relative in {row['crop'],row['image'],meta.get('original_crop')} - {None,''}:
                    path = self.image_path(relative)
                    if path.is_file():path.unlink()
                    self._remove_empty_staging_parents(path)

    def retarget_context(self, job_id, job, crop):
        """Persist recovered text coordinates with the same immutable capture."""
        import cv2
        ok, encoded = cv2.imencode('.png', crop)
        if not ok:
            raise OSError('Recovered OCR crop encoding failed')
        with self.connection() as db:
            row = db.execute('SELECT metadata,crop FROM jobs WHERE id=?',(job_id,)).fetchone()
            metadata = json.loads(row['metadata'])
            metadata['detection'] = asdict(job.detection)
            metadata['all_detections'] = [asdict(d) for d in job.all_detections]
            metadata['passage_metadata'] = job.passage_metadata
            self._write_image(row['crop'],encoded.tobytes())
            db.execute('UPDATE jobs SET metadata=?,validation=NULL WHERE id=?',
                       (json.dumps(metadata,default=lambda value:value.isoformat()),job_id))

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
            try:
                p = self.image_path(relative)
                if not p.is_file():
                    return np.zeros((100, 100, 3), dtype=np.uint8)
                blob = p.read_bytes()
                result = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
                if result is None:
                    return np.zeros((100, 100, 3), dtype=np.uint8)
                return result
            except Exception:
                return np.zeros((100, 100, 3), dtype=np.uint8)
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
                     event_id=data.get('storage_visit_id') or data.get('event_id'),
                     passage_id=data.get('passage_id'), region_id=data.get('region_id'),
                     passage_metadata=data.get('passage_metadata', {}),
                     is_stationary=data.get('is_stationary', True),
                     entrance_complete=data.get('entrance_complete', True),
                     is_twin=data.get('is_twin', False))
        validation = Validation(**json.loads(row['validation'])) if row['validation'] else None
        return job, decode(row['crop']), validation

    def save_read_crop(self, job_id, crop):
        """Keep the exact accepted OCR crop and retain its original crop."""
        import cv2
        ok, encoded = cv2.imencode('.png', crop)
        if not ok:
            raise OSError('OCR crop encoding failed')
        with self.connection() as db:
            row = db.execute('SELECT crop,metadata,payload_bytes FROM jobs WHERE id=?', (job_id,)).fetchone()
            metadata = json.loads(row['metadata'])
            original = metadata.setdefault('original_crop', row['crop'])
            base = Path(original)
            target = str(base.with_name(base.stem + '_expanded.png'))
            target_path = self.image_path(target)
            previous_bytes = target_path.stat().st_size if target_path.exists() else 0
            additional_bytes = len(encoded) - previous_bytes
            if shutil.disk_usage(self.root).free < max(0, additional_bytes) + 64*1024*1024:
                raise OSError('OCR spool lacks free space for accepted crop')
            self._write_image(target, encoded.tobytes())
            metadata['accepted_crop_variant'] = 'expanded_inside_gate_roi'
            new_metadata = json.dumps(metadata)
            size = row['payload_bytes'] + additional_bytes + len(new_metadata.encode()) - len(row['metadata'].encode())
            db.execute('UPDATE jobs SET crop=?,metadata=?,payload_bytes=? WHERE id=?', (target, new_metadata, size, job_id))

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

    def event_proof_evidence(self, event_id):
        """Expose each camera's own immutable OCR source, never companion frames."""
        with self.connection() as db:
            rows = db.execute('SELECT * FROM jobs WHERE event_id=? OR metadata LIKE ?',
                              (event_id, f'%"event_id": "{event_id}"%')).fetchall()
        result = {'detections': [], 'ocr_results': [], 'snapshots': []}
        for row in rows:
            meta = json.loads(row['metadata'])
            if (row['event_id'] or meta.get('event_id')) != event_id or not row['validation']:
                continue
            validation = json.loads(row['validation'])
            value = ContainerValidator().validate(OCRRead(validation.get('raw_text', ''),
                                                         validation.get('confidence', 0)))
            if not value.valid_check_digit or not self.image_path(row['image']).is_file():
                continue
            detection = meta['detection']
            if detection.get('class_name') != 'container_number':
                continue
            key = 'ocr-job-' + str(row['id'])
            timestamp = meta['frame']['timestamp']
            if not timestamp.endswith('Z') and '+' not in timestamp:
                timestamp += 'Z'
            camera = meta['frame']['camera_id']
            origin = meta['origin_key'] + ':proof:deadline'
            if meta.get('slot_position') == 'REAR' and ':rear' not in origin:
                origin = meta['origin_key'] + ':rear:proof:deadline'
            result['detections'].append({'id': key, 'origin_key': origin,
                'camera_id': camera, 'timestamp': timestamp, 'class_name': 'container_number',
                'bbox': detection['bbox'], 'confidence': detection.get('confidence', 0)})
            result['ocr_results'].append({'detection_id': key, 'raw_text': value.raw_text,
                                         'confidence': value.confidence, 'normalized_text': value.normalized_text,
                                         'valid_check_digit': True, 'valid_format': True})
            result['snapshots'].append({'id': key, 'camera_id': camera,
                'timestamp': timestamp, 'image_path': row['image']})
        return result

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
            metadata = json.loads(row['metadata']) if row else {}
            provisional = self.staged_visits and metadata.get('passage_id') and not metadata.get('finalized')
            if row and event_id and self.shared_evidence and not provisional and not metadata.get('storage_visit_id'):
                updates = {}
                direction = metadata.get('direction')
                gate_id = metadata.get('gate_id')
                cat = 'detected_in'
                d = str(direction or '').upper()
                g = str(gate_id or '').lower()
                if d in ('EXIT', 'OUT', 'GATE_OUT') or any(k in g for k in ('out', 'exit')):
                    cat = 'detected_out'
                clean_event = str(event_id).replace('\\', '/').strip('/')
                for c in ('waiting', 'detected_in', 'detected_out'):
                    if clean_event.startswith(f'{c}/'):
                        clean_event = clean_event[len(c)+1:]
                        break
                target_dir = self.image_root / cat / clean_event
                if self.staged_visits:
                    target_dir = target_dir / 'evidence' / metadata['detection']['camera_id']
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
                        if self.staged_visits:
                            parent = old_dir.parent
                            while parent != self.image_root and '_processing' in parent.parts:
                                try: parent.rmdir()
                                except OSError: break
                                parent = parent.parent
            else:
                if row and metadata.get('storage_visit_id') and event_id:
                    metadata['event_id'] = event_id
                    db.execute('UPDATE jobs SET metadata=? WHERE id=?',(json.dumps(metadata),job_id))
                db.execute("UPDATE jobs SET state='DONE', event_id=?, error_type=NULL, retry_at=? WHERE id=?",
                           (event_id, time.time(), job_id))

    def cancel_pending(self, origin, event_id=None):
        with self.connection() as db:
            if '%' in origin:
                db.execute("UPDATE jobs SET state='DONE', event_id=?, error_type=NULL, retry_at=? WHERE origin LIKE ? AND state='PENDING'",
                           (event_id, time.time(), origin))
            else:
                db.execute("UPDATE jobs SET state='DONE', event_id=?, error_type=NULL, retry_at=? WHERE origin=? AND state='PENDING'",
                           (event_id, time.time(), origin))

    def retain_origin_proof(self, origin, sequence):
        """Delete surplus completed captures while retaining OCR audit rows."""
        with self.connection() as db:
            rows = db.execute("SELECT * FROM jobs WHERE origin=? AND state IN ('DONE','SUPERSEDED')",
                              (origin,)).fetchall()
        def paths(row):
            meta = json.loads(row['metadata'])
            return {row['crop'], row['image'], meta.get('original_crop'),
                    *meta.get('companion_images', {}).values()} - {None, ''}
        protected = set()
        for row in rows:
            if row['sequence'] == sequence:
                protected.update(paths(row))
        if not protected:
            return
        for row in rows:
            if row['sequence'] == sequence:
                continue
            for relative in paths(row) - protected:
                path = self.image_path(relative)
                if path.is_file():
                    path.unlink()
                self._remove_empty_staging_parents(path)

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

    def retry(self, job_id, error_type, max_attempts=5):
        with self.connection() as db:
            row = db.execute('SELECT attempts FROM jobs WHERE id=?', (job_id,)).fetchone()
            curr = (row['attempts'] if row else 0) + 1
            if curr >= max_attempts:
                db.execute("UPDATE jobs SET attempts=?, state='FAILED', error_type=? WHERE id=?",
                           (curr, error_type, job_id))
            else:
                db.execute('UPDATE jobs SET attempts=?, retry_at=?, error_type=? WHERE id=?',
                           (curr, time.time() + 5, error_type, job_id))

    def status(self):
        with self.connection() as db:
            row = db.execute("SELECT COUNT(*) total, SUM(state='PENDING') pending, SUM(state='FAILED' OR (error_type IS NOT NULL AND state!='DONE')) failed FROM jobs").fetchone()
        return {key: row[key] or 0 for key in ('total', 'pending', 'failed')}
