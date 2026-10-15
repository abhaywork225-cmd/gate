from dataclasses import replace
from unittest.mock import Mock
import numpy as np
import pytest
from app.ocr.job_store import OCRJobStore
from app.ocr.validator import ContainerValidator
from app.domain import OCRRead
from app.runtime import GateRuntime


def test_atomic_images_and_metadata_survive_reopen(tmp_path, make_job):
    job = replace(make_job(), role='LEFT')
    crop = np.full((30, 40, 3), 123, dtype=np.uint8)
    store = OCRJobStore(tmp_path)
    key = store.save(job, crop)
    assert store.save(job, crop) == key
    reopened = OCRJobStore(tmp_path)
    restored, pixels, validation = reopened.load(reopened.next_id())
    assert restored.origin_key == job.origin_key
    assert restored.role == 'LEFT' and restored.frame.timestamp == job.frame.timestamp
    assert np.array_equal(pixels, crop)
    assert np.array_equal(restored.frame.image, job.frame.image)
    assert validation is None
    with store.connection() as db:
        row = db.execute('SELECT crop,image,typeof(crop) kind FROM jobs WHERE id=?', (key,)).fetchone()
    assert row['kind'] == 'text'
    assert store.image_path(row['crop']).is_file()
    assert store.image_path(row['image']).is_file()
    with pytest.raises(Exception):
        store.save(replace(job, frame=replace(job.frame, sequence=2)), np.empty((0, 0, 3), dtype=np.uint8))
    assert store.status()['total'] == 1


def test_capacity_failure_never_marks_job_saved(tmp_path, make_job):
    store = OCRJobStore(tmp_path)
    store.max_bytes = 1
    with pytest.raises(OSError):
        store.save(make_job(), np.zeros((10, 10, 3), dtype=np.uint8))
    assert store.next_id() is None


def test_votes_and_validation_recover_and_complete_idempotently(settings, repository, snapshots, camera, make_job):
    first = make_job(sequence=1)
    second = make_job(sequence=2)
    crop = np.zeros((30, 40, 3), dtype=np.uint8)
    runtime = GateRuntime(settings, [camera], repository, snapshots)
    runtime.ocr = Mock()
    runtime.ocr.read.return_value = OCRRead('CSQU3054383', .99)
    key1 = runtime.ocr_store.save(first, crop)
    key2 = runtime.ocr_store.save(second, crop)
    runtime._process_saved_ocr(key1)
    assert runtime.ocr_store.status()['pending'] == 1

    # Simulate crash after OCR result was saved but before event commit.
    validation = ContainerValidator().validate(OCRRead('CSQU3054383', .99))
    runtime.ocr_store.save_validation(key2, validation)
    recovered = GateRuntime(settings, [camera], repository, snapshots)
    recovered.ocr = Mock()
    recovered._process_saved_ocr(key2)
    recovered.ocr.read.assert_not_called()
    assert recovered.ocr_store.status()['pending'] == 0
    evidence = recovered.ocr_store.evidence(first.origin_key, 4)
    assert evidence.event_id is not None
    assert len(evidence.reads) == 2
    recovered._process_saved_ocr(key2)
    assert recovered.ocr_store.evidence(first.origin_key, 4).event_id == evidence.event_id


def test_failed_ocr_preserves_evidence_for_retry(settings, repository, snapshots, camera, make_job):
    runtime = GateRuntime(settings, [camera], repository, snapshots)
    runtime.ocr = Mock()
    runtime.ocr.read.side_effect = RuntimeError('OCR unavailable')
    key = runtime.ocr_store.save(make_job(), np.zeros((20, 20, 3), dtype=np.uint8))
    with pytest.raises(RuntimeError):
        runtime._process_saved_ocr(key)
    assert OCRJobStore(runtime.ocr_store.root).next_id() == key
    runtime.ocr_store.retry(key, 'RuntimeError')
    assert runtime.ocr_store.status() == {'total': 1, 'pending': 1, 'failed': 1}


def test_event_commit_before_spool_completion_is_safe(settings, repository, snapshots, camera, make_job, monkeypatch):
    settings.ocr_confirmations = 1
    runtime = GateRuntime(settings, [camera], repository, snapshots)
    runtime.ocr = Mock()
    runtime.ocr.read.return_value = OCRRead('CSQU3054383', .99)
    key = runtime.ocr_store.save(make_job(), np.zeros((20, 20, 3), dtype=np.uint8))
    monkeypatch.setattr(runtime.ocr_store, 'complete', Mock(side_effect=OSError('Simulated crash')))
    with pytest.raises(OSError):
        runtime._process_saved_ocr(key)
    from sqlalchemy import select, func
    from app.database.models import GateEvent
    with repository.db.sessions() as session:
        assert session.scalar(select(func.count()).select_from(GateEvent)) == 1
    recovered = GateRuntime(settings, [camera], repository, snapshots)
    recovered.ocr = Mock()
    recovered._process_saved_ocr(key)
    recovered.ocr.read.assert_not_called()
    with repository.db.sessions() as session:
        assert session.scalar(select(func.count()).select_from(GateEvent)) == 1


def test_failed_disk_write_retries_without_running_ocr(settings, repository, snapshots, camera, make_job, monkeypatch):
    import threading
    import time
    runtime = GateRuntime(settings, [camera], repository, snapshots)
    runtime.ocr = Mock()
    save = runtime.ocr_store.save
    attempts = []
    def transient(job, crop):
        attempts.append(True)
        if len(attempts) == 1:
            raise OSError('Disk temporarily unavailable')
        return save(job, crop)
    monkeypatch.setattr(runtime.ocr_store, 'save', transient)
    runtime.queue.put(make_job())
    thread = threading.Thread(target=runtime._save_ocr_loop)
    thread.start()
    try:
        deadline = time.monotonic() + 4
        while runtime.ocr_store.next_id() is None and time.monotonic() < deadline:
            time.sleep(.02)
        assert runtime.ocr_store.next_id() is not None
        assert len(attempts) >= 2
        runtime.ocr.read.assert_not_called()
    finally:
        runtime._stop.set()
        thread.join(2)


def legacy_store(root, job):
    """Recreate the previous BLOB schema from a real saved job."""
    import cv2
    store = OCRJobStore(root)
    key = store.save(job, np.zeros((20, 20, 3), dtype=np.uint8))
    with store.connection() as db:
        row = dict(db.execute('SELECT * FROM jobs WHERE id=?', (key,)).fetchone())
        db.execute('DROP TABLE jobs')
        db.execute('''CREATE TABLE jobs (
            id INTEGER PRIMARY KEY, origin TEXT NOT NULL, sequence INTEGER NOT NULL,
            metadata TEXT NOT NULL, crop BLOB NOT NULL, image BLOB NOT NULL,
            state TEXT NOT NULL DEFAULT 'PENDING', validation TEXT, event_id TEXT,
            attempts INTEGER NOT NULL DEFAULT 0, retry_at REAL NOT NULL DEFAULT 0,
            error_type TEXT, UNIQUE(origin,sequence))''')
        db.execute('INSERT INTO jobs(id,origin,sequence,metadata,crop,image,attempts) VALUES(?,?,?,?,?,?,?)',
                   (key, row['origin'], row['sequence'], row['metadata'],
                    cv2.imencode('.png', np.zeros((20,20,3), dtype=np.uint8))[1].tobytes(),
                    cv2.imencode('.png', job.frame.image)[1].tobytes(), 2))
    return key


def test_migrate_existing_blobs_without_losing_pending_job(tmp_path, make_job):
    key = legacy_store(tmp_path / 'spool', make_job())
    store = OCRJobStore(tmp_path / 'spool', image_root=tmp_path / 'snapshots' / 'ocr-pending')
    assert store.next_id() == key
    job, crop, validation = store.load(key)
    assert np.array_equal(job.frame.image, make_job().frame.image)
    assert crop.shape == (20,20,3)
    with store.connection() as db:
        row = db.execute('SELECT crop, image, attempts, typeof(crop) kind FROM jobs').fetchone()
        assert row['kind'] == 'text' and row['attempts'] == 2
        assert all(r['type'] != 'BLOB' for r in db.execute('PRAGMA table_info(jobs)'))
    assert store.image_path(row['crop']).is_file()


def test_interrupted_migration_preserves_original_and_retries(tmp_path, make_job, monkeypatch):
    import sqlite3
    root = tmp_path / 'spool'
    key = legacy_store(root, make_job())
    original = OCRJobStore._write_image
    def fail_frame(self, relative, data):
        if relative.endswith('frame.png'):
            raise OSError('Disk full')
        original(self, relative, data)
    with monkeypatch.context() as patch:
        patch.setattr(OCRJobStore, '_write_image', fail_frame)
        with pytest.raises(OSError):
            OCRJobStore(root, image_root=tmp_path / 'snapshots')
    with sqlite3.connect(root / 'jobs.sqlite3') as db:
        assert db.execute('SELECT typeof(crop) FROM jobs').fetchone()[0] == 'blob'
    store = OCRJobStore(root, image_root=tmp_path / 'snapshots')
    assert store.next_id() == key
    assert store.load(key)[1].shape == (20,20,3)


def test_partial_image_write_never_exposes_job(tmp_path, make_job, monkeypatch):
    store = OCRJobStore(tmp_path)
    original = store._write_image
    def fail_frame(relative, data):
        if relative.endswith('frame.png'):
            raise OSError('Disk full')
        original(relative, data)
    with monkeypatch.context() as patch:
        patch.setattr(store, '_write_image', fail_frame)
        with pytest.raises(OSError):
            store.save(make_job(), np.zeros((20,20,3), dtype=np.uint8))
    assert store.next_id() is None
    key = store.save(make_job(), np.zeros((20,20,3), dtype=np.uint8))
    assert store.next_id() == key
    with pytest.raises(ValueError):
        store.image_path('../outside.png')


def test_waiting_visit_keeps_three_pending_samples_and_preserves_all_images(tmp_path, make_job):
    from app.ocr.job_store import OCRJobStore
    store = OCRJobStore(tmp_path / 'spool')
    ids = []
    for sequence in range(1, 11):
        job = make_job(sequence)
        ids.append(store.save(job, job.frame.image))
    with store.connection() as db:
        rows = db.execute('SELECT id,state,crop,image FROM jobs ORDER BY id').fetchall()
    assert [r['id'] for r in rows if r['state'] == 'PENDING'] == ids[-3:]
    assert all(store.image_path(r['crop']).is_file() and store.image_path(r['image']).is_file() for r in rows)
    assert store.next_id() == ids[-3]


def test_finished_origin_removes_surplus_images_but_retains_selected_proof(tmp_path, make_job):
    from app.ocr.job_store import OCRJobStore
    store = OCRJobStore(tmp_path / 'spool')
    for sequence in (1, 2, 3):
        job = make_job(sequence)
        key = store.save(job, job.frame.image)
        store.complete(key)
    with store.connection() as db:
        rows = db.execute('SELECT * FROM jobs ORDER BY id').fetchall()
    store.retain_origin_proof(make_job().origin_key, 2)
    for row in rows:
        assert store.image_path(row['image']).is_file() == (row['sequence'] == 2)
        assert store.image_path(row['crop']).is_file() == (row['sequence'] == 2)
    with store.connection() as db:
        assert db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] == 3


def test_staged_visit_restart_and_final_camera_folder(tmp_path,make_job):
    job=replace(make_job(origin='passage:passage:number'),event_id='visit',passage_id='passage',region_id='number')
    root=tmp_path/'snapshots';spool=tmp_path/'spool'
    store=OCRJobStore(spool,image_root=root,staged_visits=True)
    key=store.save(job,np.full((30,40,3),120,dtype=np.uint8))
    store=OCRJobStore(spool,image_root=root,staged_visits=True)
    with store.connection() as db:
        row=db.execute('SELECT * FROM jobs WHERE id=?',(key,)).fetchone()
    assert row['state']=='PENDING'
    assert row['image'].startswith('_processing/visit/'+job.frame.camera_id+'/')
    assert not (root/'visit').exists()
    store.complete(key,'visit')
    assert not (root/'visit').exists()
    store.link_region('passage','number','final-event')
    with store.connection() as db:
        row=db.execute('SELECT * FROM jobs WHERE id=?',(key,)).fetchone()
    assert row['image'].startswith('final-event/evidence/'+job.frame.camera_id+'/')
    assert store.image_path(row['image']).is_file()
    assert not (root/'_processing'/'visit').exists()
    store=OCRJobStore(spool,image_root=root,staged_visits=True)
    _,_,_=store.load(key)


def test_discarded_staging_region_keeps_other_camera_files(tmp_path,make_job):
    store=OCRJobStore(tmp_path/'spool',image_root=tmp_path/'snapshots',staged_visits=True)
    for cam,region in [('camera-1','one'),('camera-2','two')]:
        job=replace(make_job(camera_id=cam,origin='passage:p:'+region),
                    event_id='visit',passage_id='p',region_id=region)
        store.save(job,np.full((20,30,3),120,dtype=np.uint8))
    root=store.image_root/'_processing'/'visit'
    store.discard_region('p','one')
    assert not (root/'camera-1').exists()
    assert (root/'camera-2').exists()
    store.discard_region('p','two')
    assert not root.exists()


def test_shared_visit_completion_never_moves_capture_files(tmp_path,make_job):
    store=OCRJobStore(tmp_path/'spool',image_root=tmp_path/'snapshots',shared_visit_folders=True)
    job=replace(make_job(origin='passage:p:r'),event_id='visit',passage_id='p',region_id='r')
    key=store.save(job,np.full((30,40,3),120,dtype=np.uint8))
    with store.connection() as db:before=dict(db.execute('select * from jobs where id=?',(key,)).fetchone())
    assert before['image'].startswith('visit/') and len(before['image'].split('/'))==2
    store.complete(key,'p');store.link_region('p','r','container-record')
    reopened=OCRJobStore(tmp_path/'spool',image_root=tmp_path/'snapshots',shared_visit_folders=True)
    with reopened.connection() as db:after=dict(db.execute('select * from jobs where id=?',(key,)).fetchone())
    assert before['image']==after['image'] and before['crop']==after['crop']
    assert after['event_id']=='container-record'
    assert not (store.image_root/'container-record').exists()
    assert reopened.image_path(after['image']).is_file()


def test_proven_camera_join_preserves_pending_jobs(tmp_path,make_job):
    store=OCRJobStore(tmp_path/'spool',image_root=tmp_path/'snapshots',shared_visit_folders=True)
    job=replace(make_job(origin='passage:p:r'),event_id='separate',passage_id='p',region_id='r')
    key=store.save(job,np.full((30,40,3),120,dtype=np.uint8))
    store.join_visit_folder('p','shared')
    with store.connection() as db:row=dict(db.execute('select * from jobs where id=?',(key,)).fetchone())
    assert row['state']=='PENDING' and row['event_id']=='shared'
    assert row['image'].startswith('shared/') and len(row['image'].split('/'))==2
    assert not (store.image_root/'separate').exists()
    assert store.image_path(row['image']).is_file()
