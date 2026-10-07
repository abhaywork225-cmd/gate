from datetime import timedelta
from types import SimpleNamespace
import numpy as np
import pytest
from app.detection.tracker import TemporalTracker
from app.detection.yolo_detector import YoloDetector
from app.domain import Detection
from app.events.gate_logic import update_direction
from app.ocr.container_ocr import crop_identification


def test_detector_contract_and_aliases(settings, make_job):
    class Box:
        cls = np.array(0)
        conf = np.array(.92)
        xyxy = np.array([[-10, -20, 200, 200]])
    class Model:
        def predict(self, **kwargs):
            assert kwargs['rect'] is False
            return [SimpleNamespace(names={0: 'Cointainer'}, boxes=[Box()])]
    settings.class_aliases = {'cointainer': 'container'}
    frame = make_job().frame
    output = YoloDetector(settings, Model()).detect(frame)
    assert output == [Detection(frame.camera_id, frame.timestamp, 'container', .92, (0, 0, 150, 100))]


def test_no_model_fallback(settings, tmp_path):
    cfg = settings.model_copy(update={'model_path': tmp_path / 'missing.pt'})
    with pytest.raises(ValueError):
        YoloDetector(cfg)


def test_tracking_expires_by_time_and_never_matches_other_classes(make_job):
    tracker = TemporalTracker(ttl_seconds=2)
    detection = make_job().detection
    first = tracker.update([detection], detection.frame_timestamp)[0]
    again = tracker.update([detection], detection.frame_timestamp + timedelta(seconds=.5))[0]
    assert first.id == again.id and again.hits == 2
    truck = Detection(detection.camera_id, detection.frame_timestamp, 'truck', .9, detection.bbox)
    assert tracker.update([truck], detection.frame_timestamp + timedelta(seconds=1))[0].id != first.id
    assert tracker.update([detection], detection.frame_timestamp + timedelta(seconds=5))[0].id != first.id


def test_direction_uses_configuration_not_camera_name(camera, make_job):
    camera.id = 'gate-out-front'
    track = TemporalTracker().update([make_job().detection], make_job().frame.timestamp)[0]
    assert update_direction(camera, track, (100, 150, 3)) == 'UNKNOWN'
    camera.direction = 'EXIT'
    assert update_direction(camera, track, (100, 150, 3)) == 'EXIT'


def test_line_crossing_requires_two_sides(camera, make_job):
    camera.line_axis = 'x'
    camera.line_deadband = .05
    detection = make_job().detection
    track = TemporalTracker().update([detection], detection.frame_timestamp)[0]
    for box, expected in [((5, 10, 35, 60), 'UNKNOWN'), ((35, 10, 65, 60), 'UNKNOWN'), ((65, 10, 95, 60), 'ENTRY')]:
        track.detection = Detection(detection.camera_id, detection.frame_timestamp, 'container', .9, box)
        assert update_direction(camera, track, (100, 100, 3)) == expected


def test_clipped_identification_crop():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    assert crop_identification(image, (-30, -10, 250, 150), (.5, .5, 1, 1)).shape == (50, 100, 3)
    with pytest.raises(ValueError):
        crop_identification(image, (300, 100, 350, 120))


def test_legitimate_vertical_number_is_not_discarded(settings, make_job):
    class Box:
        cls = np.array(0)
        conf = np.array(.92)
        xyxy = np.array([[10, 5, 45, 95]])
    class Model:
        def predict(self, **kwargs):
            return [SimpleNamespace(names={0:'container_number'}, boxes=[Box()])]
    assert len(YoloDetector(settings, Model()).detect(make_job().frame)) == 1
