import pytest
import numpy as np
from datetime import datetime, timezone
from dataclasses import asdict

from app.domain import Frame, Detection, OCRRead, Validation
from app.ocr.validator import ContainerValidator, extract_iso_size_code
from app.database.repositories.gate import GateRepository
from app.database.database import Database
from app.config.settings import Settings
from app.snapshots.snapshot_manager import SnapshotManager


def test_targeted_subcrop_ocr_priority(tmp_path):
    validator = ContainerValidator()

    # Create dummy frame with synthetic image
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame = Frame(
        camera_id='cam1',
        timestamp=datetime.now(timezone.utc),
        sequence=1,
        image=img
    )

    # Main container box (large)
    c_det = Detection(
        camera_id='cam1',
        frame_timestamp=frame.timestamp,
        class_name='container',
        confidence=0.88,
        bbox=(100.0, 100.0, 1800.0, 900.0)
    )

    # Sub-box cointainer (small text box)
    sub_cointainer = Detection(
        camera_id='cam1',
        frame_timestamp=frame.timestamp,
        class_name='cointainer',
        confidence=0.92,
        bbox=(400.0, 200.0, 700.0, 280.0)
    )

    # Sub-box feet (small size marking)
    sub_feet = Detection(
        camera_id='cam1',
        frame_timestamp=frame.timestamp,
        class_name='feet',
        confidence=0.90,
        bbox=(750.0, 200.0, 900.0, 280.0)
    )

    # Test validator on text extraction
    read_text = validator.validate(OCRRead('BMOU 412845 5', 0.95))
    assert read_text.valid_check_digit is True
    assert read_text.normalized_text == 'BMOU4128455'

    feet_code, human_size = extract_iso_size_code('45G1')
    assert feet_code == '45G1'
    assert human_size == '40 FT HC'


def test_twin_atomic_publishing(tmp_path):
    from app.config.settings import CameraConfig
    db_path = tmp_path / 'test_gate.db'
    db = Database(f'sqlite:///{db_path}')
    db.initialize()
    settings = Settings()
    repo = GateRepository(db)
    repo.sync_cameras([CameraConfig(id='cam1', name='Cam 1', gate_id='LANE_1', source_env='CAM1', source_type='rtsp')])
    snapshots = SnapshotManager(tmp_path / 'snapshots')

    # Front container observation
    from app.domain import Observation
    front_det = Detection('cam1', datetime.now(timezone.utc), 'container', 0.9, (100, 100, 500, 500))
    front_ocr = Validation('MSCU 654602 1', 'MSCU6546021', 0.95, True, True, 'VALID', '22G1')
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    front_obs = Observation(
        origin_key='cam1:front',
        gate_id='LANE_1',
        track_id=1,
        detection=front_det,
        ocr=front_ocr,
        direction='ENTRY',
        image=img,
        started_at=datetime.now(timezone.utc),
        confirmed=True,
        slot_position='FRONT',
        is_twin=True
    )

    event_id, created = repo.save_observation(front_obs, snapshots, 30)
    assert created is True

    # Front container is saved, but because it's a twin event, status is held at NEEDS_REVIEW
    ev1 = repo.event(event_id)
    assert ev1['is_twin'] is False
    assert ev1['container_number'] == 'MSCU6546021'
    assert ev1['container_number_2'] is None
    assert ev1['status'] == 'NEEDS_REVIEW'

    # Now Rear container observation arrives for the same event
    rear_det = Detection('cam1', datetime.now(timezone.utc), 'container', 0.9, (600, 100, 1000, 500))
    rear_ocr = Validation('WHSU 216671 7', 'WHSU2166717', 0.95, True, True, 'VALID', '22G1')
    rear_obs = Observation(
        origin_key='cam1:rear',
        gate_id='LANE_1',
        track_id=2,
        detection=rear_det,
        ocr=rear_ocr,
        direction='ENTRY',
        image=img,
        started_at=datetime.now(timezone.utc),
        confirmed=True,
        slot_position='REAR'
    )

    event_id_2, created_2 = repo.save_observation(rear_obs, snapshots, 30, ongoing_event_id=event_id)
    assert event_id_2 == event_id
    assert created_2 is False

    # Now both containers are populated and confirmed together atomically
    ev2 = repo.event(event_id)
    assert ev2['is_twin'] is True
    assert ev2['container_number'] == 'MSCU6546021'
    assert ev2['container_number_2'] == 'WHSU2166717'
    assert ev2['status'] == 'CONFIRMED'
