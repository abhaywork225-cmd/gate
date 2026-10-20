from datetime import datetime
import numpy as np
import pytest
from app.config.settings import CameraConfig, Settings
from app.database.database import Database
from app.database.repositories.gate import GateRepository
from app.domain import Detection, Frame
from app.runtime import OCRJob
from app.snapshots.snapshot_manager import SnapshotManager


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, deployment_mode='test', database_url='sqlite+pysqlite:///' + str(tmp_path / 'test.db'),
                    snapshot_directory=tmp_path / 'snapshots', categorized_snapshot_folders=False, pipeline_enabled=False, vehicle_visit_tracking=False, minimum_number_proof_cameras=1, shared_visit_folders=False, container_roi_trigger=False,
                    reversal_grace_seconds=0,
                    upload_directory=tmp_path / 'uploads',
                    ocr_interval_seconds=.001, reconnect_initial_seconds=.01, reconnect_max_seconds=.02)


@pytest.fixture
def camera():
    return CameraConfig(id='camera-1', name='View 1', gate_id='lane-1', source_env='CAMERA_1_RTSP')


@pytest.fixture
def repository(settings, camera):
    db = Database(settings.database_url.get_secret_value())
    db.initialize()
    repo = GateRepository(db)
    repo.sync_cameras([camera])
    yield repo
    db.close()


@pytest.fixture
def snapshots(settings):
    return SnapshotManager(settings.snapshot_directory, categorized=settings.categorized_snapshot_folders)


@pytest.fixture
def make_job():
    def factory(sequence=1, camera_id='camera-1', origin='run:camera-1:1', timestamp=None, direction='UNKNOWN', gate_id='lane-1'):
        ts = timestamp or datetime(2026, 9, 8, 12, 0, 0)
        frame = Frame(camera_id, ts, sequence, np.zeros((100, 150, 3), dtype=np.uint8))
        detection = Detection(camera_id, ts, 'container', .92, (10, 10, 130, 90))
        return OCRJob(origin, gate_id, 1, detection, frame, direction, ts, (0, 0, 1, 1))
    return factory
