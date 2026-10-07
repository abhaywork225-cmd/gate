"""Production-style abuse tests for the gate pipeline.

These tests intentionally use malformed OCR, tracker churn, late camera
evidence, blank frames, cloud outages and boundary geometry.  They are kept
separate from the smoke tests so a deployment can run the cheap suite first
and this gate before accepting a release candidate.
"""

from dataclasses import replace
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.domain import Detection, Frame, Observation, Validation, OCRRead
from app.events.event_manager import EventManager
from app.events.occupancy import ViewOccupancy
from app.events.cloud_dispatcher import CloudOutboxDispatcher


def _frame(camera, second=0, value=0, source="harsh"):
    ts = datetime(2026, 10, 2, 10, 0, 0) + timedelta(seconds=second)
    image = np.full((240, 320, 3), value, dtype=np.uint8)
    return Frame(camera, ts, second + 1, image, source)


def _valid(text="CSQU3054383", size="22G1"):
    return Validation(text, text, .99, True, True, "VALID", size_type=size)


@pytest.fixture
def mock_settings(settings):
    return settings.model_copy(update={"prosper_enabled": True, "prosper_auto_dispatch": True, "prosper_site_id": "site-test"})


def _job(make_job, sequence, origin, camera="camera-1", timestamp=None):
    job = make_job(sequence=sequence, origin=origin, camera_id=camera, timestamp=timestamp)
    return job


def test_invalid_partial_read_never_becomes_a_twin(repository, snapshots, camera, make_job):
    """A partial second OCR result must not fabricate a second container."""
    repository.sync_cameras([camera, camera.model_copy(update={"id": "camera-2", "source_env": "CAM2"})])
    first = _job(make_job, 1, "dwell:camera-1:front")
    first = replace(first, detection=Detection(camera.id, first.frame.timestamp, "container", .98, (10, 10, 140, 220)))
    obs = Observation(first.origin_key, camera.gate_id, 1, first.detection, _valid(), "EXIT",
                      first.frame.image, first.frame.timestamp, True)
    event_id, _ = repository.save_observation(obs, snapshots, 30)

    # This is the shape of the late read observed in eb73...: CMAU72583.
    late = _job(make_job, 2, "dwell:camera-2:rear", camera="camera-2",
                timestamp=first.frame.timestamp + timedelta(seconds=9))
    late = replace(late, detection=Detection("camera-2", late.frame.timestamp, "container_number", .95, (20, 20, 120, 100)))
    partial = Validation("CMAU72583", "CMAU72583", .95, False, False, "INVALID_FORMAT")
    late_obs = Observation(late.origin_key, camera.gate_id, 2, late.detection, partial, "EXIT",
                           late.frame.image, late.frame.timestamp, False)
    repository.save_observation(late_obs, snapshots, 30, ongoing_event_id=event_id)

    event = repository.event(event_id)
    assert event["container_number"] == "CSQU3054383"
    assert event["container_number_2"] is None
    assert event["is_twin"] is False


def test_occupancy_boundary_and_tracker_churn_do_not_reopen_lane(settings, camera):
    """Tiny edge detections and a new tracker ID cannot create occupancy."""
    view = ViewOccupancy(settings)
    camera.occupancy_roi = (.1, .1, .9, .9)
    frame = _frame(camera.id)
    outside = Detection(camera.id, frame.timestamp, "container", .99, (0, 0, 20, 20))
    assert view.update(camera, frame, [outside]) is None

    inside = Detection(camera.id, frame.timestamp + timedelta(seconds=1), "container", .99, (50, 50, 250, 220))
    assert view.update(camera, _frame(camera.id, 1), [inside]) is not None
    # Re-identification with a shifted box is still the same physical lane.
    churned = Detection(camera.id, frame.timestamp + timedelta(seconds=2), "container", .99, (52, 50, 252, 220))
    assert view.update(camera, _frame(camera.id, 2), [churned]) is not None
    assert view.current is not None


def test_two_valid_cameras_merge_into_one_event(repository, snapshots, camera, make_job):
    """Two independent camera origins can be joined without duplicate events."""
    repository.sync_cameras([camera, camera.model_copy(update={"id": "camera-2", "source_env": "CAM2"})])
    a = _job(make_job, 1, "dwell:camera-1:front", "camera-1")
    a = replace(a, detection=Detection("camera-1", a.frame.timestamp, "container", .97, (10, 10, 140, 220)))
    obs_a = Observation(a.origin_key, camera.gate_id, 1, a.detection, _valid("CSQU3054383"), "EXIT",
                        a.frame.image, a.frame.timestamp, True, slot_position="FRONT")
    event_id, created = repository.save_observation(obs_a, snapshots, 30)
    assert created

    b = _job(make_job, 2, "dwell:camera-2:rear", "camera-2", a.frame.timestamp + timedelta(seconds=3))
    b = replace(b, detection=Detection("camera-2", b.frame.timestamp, "container", .96, (20, 20, 150, 220)))
    obs_b = Observation(b.origin_key, camera.gate_id, 2, b.detection, _valid("MSCU6639870"), "EXIT",
                        b.frame.image, b.frame.timestamp, True, slot_position="REAR")
    merged, created_second = repository.save_observation(obs_b, snapshots, 30, ongoing_event_id=event_id)
    assert merged == event_id
    assert not created_second
    event = repository.event(event_id)
    assert event["is_twin"] is True
    assert {event["container_number"], event["container_number_2"]} == {"CSQU3054383", "MSCU6639870"}


def test_blank_proof_is_replaced_with_safe_fallback(mock_settings):
    """Cloud dispatcher must not upload an all-black proof frame."""
    dispatcher = CloudOutboxDispatcher(mock_settings)
    det = Detection("camera-1", datetime.utcnow(), "container", .95, (1, 1, 100, 100))
    obs = Observation("origin", "lane-1", 1, det, _valid(), "ENTRY",
                      np.zeros((240, 320, 3), dtype=np.uint8), datetime.utcnow(), True)
    with patch.object(dispatcher.session, "upload_image", return_value="https://img.test/blank.jpg"), \
         patch.object(dispatcher.session, "post_gate_event", return_value=True) as post:
        dispatcher.start()
        dispatcher.enqueue(obs, "blank-event")
        dispatcher.queue.join()
        dispatcher.stop()
    payload = post.call_args.args[0]
    image = payload["Containers"][0]["Images"][0]["Image"]
    assert not image.endswith("AAAA")  # no raw all-zero JPEG/data is accepted
    assert image.startswith("data:image/jpeg;base64,")


def test_cloud_grace_delays_publication(mock_settings):
    """The worker must hold a cloud POST until the evidence grace expires."""
    dispatcher = CloudOutboxDispatcher(mock_settings)
    det = Detection("camera-1", datetime.utcnow(), "container", .95, (1, 1, 100, 100))
    obs = Observation("origin", "lane-1", 1, det, _valid(), "ENTRY",
                      np.full((240, 320, 3), 180, dtype=np.uint8), datetime.utcnow(), True)
    with patch.object(dispatcher.session, "post_gate_event", return_value=True) as post:
        dispatcher.start()
        dispatcher.enqueue(obs, "grace-event", delay_seconds=.15)
        assert not post.called
        dispatcher.queue.join()
        dispatcher.stop()
        assert post.called


@pytest.mark.parametrize("bad", ["", "MSCU123", "MSCU1234568", "12341234567", "MSCU 1234O67"])
def test_validator_rejects_corrupted_container_numbers(bad):
    from app.ocr.validator import ContainerValidator
    result = ContainerValidator().validate(OCRRead(raw_text=bad, confidence=.99))
    assert (not result.valid_check_digit) or result.normalized_text != bad.replace(" ", "").upper()
