import numpy as np
from datetime import datetime, timezone, timedelta
from app.snapshots.snapshot_manager import SnapshotManager
from app.database.database import Database
from app.database.repositories.gate import GateRepository, ocr_similarity_score
from app.database.models import GateEvent
from app.domain import Observation, Detection, Validation, OCRRead, Frame


def test_compute_similarity_identical_and_different(tmp_path):
    mgr = SnapshotManager(tmp_path)
    rng = np.random.default_rng(42)
    img1 = rng.integers(50, 200, (480, 640, 3), dtype=np.uint8)
    # Identical image
    assert SnapshotManager.compute_similarity(img1, img1) >= 0.999

    # Slightly perturbed image (e.g. minor noise)
    noise = rng.integers(-5, 5, (480, 640, 3), dtype=np.int16)
    img1_noisy = np.clip(img1.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    assert SnapshotManager.compute_similarity(img1, img1_noisy) >= 0.95

    # Completely different image
    img2 = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    assert SnapshotManager.compute_similarity(img1, img2) < 0.30


def test_ocr_similarity_score():
    # Exact match
    assert ocr_similarity_score("MSMU5388338", "MSMU5388338") == 1.0
    # Duplicate pair: MSMU5388338 vs MSMU5386335 (9 out of 11 matching characters)
    score = ocr_similarity_score("MSMU5388338", "MSMU5386335")
    assert score >= 0.80
    # Completely different
    assert ocr_similarity_score("MSMU5388338", "TCLU1234567") < 0.20


def test_save_observation_deduplicates_by_image_and_timestamp(tmp_path):
    from app.config.settings import CameraConfig
    db = Database("sqlite:///:memory:")
    db.initialize()
    repo = GateRepository(db)
    snapshots = SnapshotManager(tmp_path)
    repo.sync_cameras([CameraConfig(id='gate-in-1', name='Gate In 1', gate_id='lane-in', source_env='CAM1')])

    now = datetime(2026, 9, 25, 6, 41, 27, tzinfo=timezone.utc)
    rng = np.random.default_rng(100)
    truck_scene = rng.integers(40, 220, (480, 640, 3), dtype=np.uint8)

    # First event: camera gate-in-1, OCR read MSMU5386335
    det1 = Detection("gate-in-1", now, "container", 0.9, (10, 10, 100, 100))
    val1 = Validation("MSMU5386335", "MSMU5386335", 0.95, True, True, "VALID", size_type="45G1")
    obs1 = Observation(
        origin_key="dwell:gate-in-1:orig1",
        gate_id="lane-in",
        track_id=1,
        detection=det1,
        ocr=val1,
        direction="ENTRY",
        image=truck_scene,
        started_at=now,
        confirmed=True
    )
    event_id1, created1 = repo.save_observation(obs1, snapshots, dedup_seconds=30)
    assert created1 is True

    # Second event 0.5s later: camera gate-in-1, OCR misread MSMU5388338 (different string!)
    # But identical scene / same timestamp
    now2 = now + timedelta(milliseconds=480)
    det2 = Detection("gate-in-1", now2, "container", 0.9, (10, 10, 100, 100))
    val2 = Validation("MSMU5388338", "MSMU5388338", 0.90, True, True, "VALID", size_type="45G1")
    obs2 = Observation(
        origin_key="dwell:gate-in-1:orig2",
        gate_id="lane-in",
        track_id=2,
        detection=det2,
        ocr=val2,
        direction="ENTRY",
        image=truck_scene,
        started_at=now2,
        confirmed=True
    )
    event_id2, created2 = repo.save_observation(obs2, snapshots, dedup_seconds=30)

    # Must be merged into the first event despite the OCR character error!
    assert created2 is False
    assert event_id2 == event_id1


def test_canonical_event_id_resolves_duplicate_by_image(tmp_path):
    from app.config.settings import CameraConfig
    db = Database("sqlite:///:memory:")
    db.initialize()
    repo = GateRepository(db)
    snapshots = SnapshotManager(tmp_path)
    repo.sync_cameras([CameraConfig(id='gate-in-1', name='Gate In 1', gate_id='lane-in', source_env='CAM1')])

    now = datetime(2026, 9, 25, 6, 41, 27, tzinfo=timezone.utc)
    rng = np.random.default_rng(200)
    truck_scene = rng.integers(40, 220, (480, 640, 3), dtype=np.uint8)

    # Save two events manually into the database simulating an earlier unmerged duplicate
    with repo.db.sessions.begin() as session:
        ev1 = GateEvent(
            id="ev-1-primary",
            gate_id="lane-in",
            timestamp=now,
            container_number="MSMU5386335",
            event_type="ENTRY",
            status="CONFIRMED"
        )
        ev2 = GateEvent(
            id="ev-2-duplicate",
            gate_id="lane-in",
            timestamp=now + timedelta(milliseconds=480),
            container_number="MSMU5388338",
            event_type="ENTRY",
            status="CONFIRMED"
        )
        session.add_all([ev1, ev2])

    # Save snapshots on disk for both events
    snapshots.save_side_snapshot("ev-1-primary", "gate-in-1", truck_scene)
    snapshots.save_side_snapshot("ev-2-duplicate", "gate-in-1", truck_scene)

    # canonical_event_id should resolve ev-2-duplicate to ev-1-primary
    canonical = repo.canonical_event_id("ev-2-duplicate", max_gap_seconds=300, snapshots=snapshots)
    assert canonical == "ev-1-primary"


def test_congestion_dwell_does_not_merge_different_container_ids(tmp_path):
    from app.config.settings import CameraConfig
    db = Database("sqlite:///:memory:")
    db.initialize()
    repo = GateRepository(db)
    snapshots = SnapshotManager(tmp_path)
    repo.sync_cameras([CameraConfig(id='gate-in-1', name='Gate In 1', gate_id='lane-in', source_env='CAM1')])

    base_time = datetime(2026, 10, 6, 6, 50, 0, tzinfo=timezone.utc)
    rng = np.random.default_rng(300)
    img = rng.integers(50, 200, (480, 640, 3), dtype=np.uint8)

    # 1. First dwell observation: MSNU9152637 at t=0
    det1 = Detection("gate-in-1", base_time, "container", 0.9, (10, 10, 100, 100))
    val1 = Validation("MSNU 915263 7", "MSNU9152637", 0.98, True, True, "VALID", size_type="45G1")
    obs1 = Observation(
        origin_key="dwell:gate-in-1:orig_dwell_1",
        gate_id="lane-in",
        track_id=1,
        detection=det1,
        ocr=val1,
        direction="ENTRY",
        image=img,
        started_at=base_time,
        confirmed=True
    )
    ev_id1, created1 = repo.save_observation(obs1, snapshots, dedup_seconds=30, same_container_window_seconds=2400)
    assert created1 is True

    # 2. Five minutes later a different valid ID must remain a new visit.
    # Same-line containers are common in a queue; OCR similarity is not a
    # physical identity and must not fuse them.
    t2 = base_time + timedelta(minutes=5)
    det2 = Detection("gate-in-1", t2, "container", 0.9, (15, 12, 105, 102))
    val2 = Validation("MSNU 115263 0", "MSNU1152630", 0.94, True, True, "VALID", size_type="45G1")
    obs2 = Observation(
        origin_key="dwell:gate-in-1:orig_dwell_2",
        gate_id="lane-in",
        track_id=2,
        detection=det2,
        ocr=val2,
        direction="ENTRY",
        image=img,
        started_at=t2,
        confirmed=True
    )
    ev_id2, created2 = repo.save_observation(obs2, snapshots, dedup_seconds=30, same_container_window_seconds=2400)
    assert created2 is True
    assert ev_id2 != ev_id1

    # 3. Another different ID after 12 minutes is also a separate visit.
    t3 = base_time + timedelta(minutes=12)
    det3 = Detection("gate-in-1", t3, "container", 0.88, (18, 14, 108, 104))
    val3 = Validation("MSNU 152631 9", "MSNU1526319", 0.91, True, True, "VALID", size_type="45G1")
    obs3 = Observation(
        origin_key="dwell:gate-in-1:orig_dwell_3",
        gate_id="lane-in",
        track_id=3,
        detection=det3,
        ocr=val3,
        direction="ENTRY",
        image=img,
        started_at=t3,
        confirmed=True
    )
    ev_id3, created3 = repo.save_observation(obs3, snapshots, dedup_seconds=30, same_container_window_seconds=2400)
    assert created3 is True
    assert ev_id3 not in {ev_id1, ev_id2}
