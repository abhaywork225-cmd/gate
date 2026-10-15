import time
import numpy as np
from app.camera.camera_manager import CameraManager
from app.camera.camera_worker import CameraWorker
from app.camera.rtsp_stream import OpenCVStream
from app.config.settings import CameraConfig
from app.domain import Detection, OCRRead
from app.runtime import GateRuntime


def wait_until(predicate, seconds=5):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError('Expected condition did not occur')


class FakeStream:
    fps = 60
    def __init__(self, *args):
        self.closed = False
    def read(self):
        time.sleep(.005)
        return True, np.zeros((100, 150, 3), dtype=np.uint8)
    def close(self):
        self.closed = True


def test_open_failure_reconnect_and_other_camera_isolation(settings):
    configs = [CameraConfig(id=f'camera-{i}', name=f'View {i}', gate_id='lane-1', source_env=f'CAMERA_{i}_RTSP') for i in range(1, 5)]
    calls = {'broken': 0}
    def factory(source, *_):
        if source == 'broken':
            calls['broken'] += 1
            raise ConnectionError('Do not log rtsp://user:password@example/')
        return FakeStream()
    manager = CameraManager(configs, settings, factory, lambda c: 'broken' if c.id == 'camera-1' else 'ok')
    manager.start()
    try:
        wait_until(lambda: all(w.status()['status'] == 'ONLINE' for key, w in manager.workers.items() if key != 'camera-1'))
        wait_until(lambda: calls['broken'] >= 2)
        assert manager.workers['camera-1'].status()['reconnect_count'] >= 1
        assert 'password' not in str(manager.status())
    finally:
        assert manager.stop()


def test_read_failure_reconnects(settings, camera):
    class Broken(FakeStream):
        def read(self):
            return False, None
    streams = []
    def factory(*args):
        stream = Broken() if not streams else FakeStream()
        streams.append(stream)
        return stream
    worker = CameraWorker(camera, 'source', settings, factory)
    worker.start()
    try:
        wait_until(lambda: worker.status()['status'] == 'ONLINE')
        assert streams[0].closed and worker.status()['reconnect_count'] == 1
    finally:
        worker.stop()
        assert worker.join()


def test_latest_frame_buffer_is_bounded(settings, camera):
    worker = CameraWorker(camera, 'source', settings, FakeStream)
    for _ in range(100):
        worker._publish(np.zeros((10, 10, 3), dtype=np.uint8))
    assert worker.status()['dropped_frames'] == 99
    assert worker.take().sequence == 100
    assert worker.take() is None


def test_real_opencv_recorded_video_eof(settings, camera, tmp_path):
    import cv2
    path = str(tmp_path / 'replay.avi')
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 20, (100, 100))
    assert writer.isOpened()
    for _ in range(4):
        writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
    writer.release()
    camera.source_type = 'file'
    worker = CameraWorker(camera, path, settings, OpenCVStream)
    worker.start()
    try:
        wait_until(lambda: worker.status()['reason'] == 'file_completed')
        assert worker.status()['frames_captured'] > 0
        assert worker.status()['reconnect_count'] == 0
    finally:
        worker.stop()
        assert worker.join()


def test_real_opencv_recorded_video_looping(settings, camera, tmp_path):
    import cv2
    path = str(tmp_path / 'loop.avi')
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 20, (100, 100))
    assert writer.isOpened()
    for _ in range(4):
        writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
    writer.release()
    camera.source_type = 'file'
    camera.loop_file = True
    worker = CameraWorker(camera, path, settings, OpenCVStream)
    worker.start()
    try:
        wait_until(lambda: worker.status()['frames_captured'] >= 8)
        assert worker.status()['status'] == 'ONLINE'
        assert worker.status()['reconnect_count'] == 0
    finally:
        worker.stop()
        assert worker.join()


def test_four_camera_frames_to_database_and_snapshot(settings, repository, snapshots):
    configs = [CameraConfig(id=f'camera-{i}', name=f'View {i}', gate_id=f'lane-{(i-1)//2}', source_env=f'CAMERA_{i}_RTSP', capture_fps=60) for i in range(1, 5)]
    repository.sync_cameras(configs)
    settings.pipeline_enabled = True
    settings.inference_fps = 60
    settings.min_track_hits = 2
    class Detector:
        def __init__(self, *_): pass
        def detect(self, frame):
            return [Detection(frame.camera_id, frame.timestamp, 'container', .95, (10, 10, 130, 90))]
    class OCR:
        def __init__(self, *_): pass
        def read(self, image): return OCRRead('CSQU3054383', .96)
    manager = CameraManager(configs, settings, FakeStream, lambda _: 'synthetic')
    runtime = GateRuntime(settings, configs, repository, snapshots, manager, Detector, OCR)
    runtime.start()
    try:
        wait_until(lambda: len(list(snapshots.root.rglob('*.jpg'))) >= 4)
        wait_until(lambda: repository.events()['total'] == 2)
        time.sleep(.15)
        assert repository.events()['total'] == 2
        # A camera can contribute both captured context and its own OCR proof.
        # Check camera coverage and file integrity, not a timing-dependent count.
        def camera_evidence():
            event_rows = repository.events()['items']
            return [s for event in event_rows for s in repository.event(event['id'])['snapshots']]
        # Events can exist before the other camera's asynchronous OCR finishes.
        wait_until(lambda: {s['camera_id'] for s in camera_evidence()} == {c.id for c in configs})
        evidence = camera_evidence()
        assert {s['camera_id'] for s in evidence} == {c.id for c in configs}
        assert all(snapshots.path(s['image_path']).is_file() for s in evidence)
        assert len(runtime.metrics.snapshot()['inference_fps']) == 4
    finally:
        assert runtime.stop()
