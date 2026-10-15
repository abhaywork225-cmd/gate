from dataclasses import replace
from uuid import uuid4
from unittest.mock import Mock
import numpy as np
from app.events.deadline import CaptureDeadlines
from app.events.cloud_dispatcher import CloudOutboxDispatcher
from app.domain import OCRRead
from app.ocr.validator import ContainerValidator



def test_elapsed_time_does_not_finalize_unreadable_or_partial_visit(settings, repository, snapshots, make_job):
    for reading in (None, ContainerValidator().validate(OCRRead('CMAU 062325', .88))):
        dispatcher = Mock()
        controller = CaptureDeadlines(settings, repository, snapshots, dispatcher)
        job = replace(make_job(1), event_id=str(uuid4()))
        controller.record(job, reading)
        controller.tick(10**12)
        event = repository.event(job.event_id)
        assert not event or event.get('capture_policy') != 'BEST_AVAILABLE'
        dispatcher.enqueue.assert_not_called()


def test_late_correct_ocr_is_saved_without_time_limit(settings, repository, snapshots, make_job):
    dispatcher = Mock()
    controller = CaptureDeadlines(settings, repository, snapshots, dispatcher)
    job = replace(make_job(1), event_id=str(uuid4()))
    image = job.frame.image.copy()
    image[10:40, 10:80] = 255
    job = replace(job, frame=replace(job.frame, image=image))
    controller.record(job)
    controller.tick(10**12)
    valid = ContainerValidator().validate(OCRRead('T C K U 6 8 3 6 5 2 9', .94261))
    controller.record(job, valid)
    event = repository.event(job.event_id)
    assert event['container_number'] == 'TCKU6836529'
    assert any(o['raw_text'] == valid.raw_text for o in event['ocr_results'])
    controller.record(job, ContainerValidator().validate(OCRRead('garbage', .99)))
    assert repository.event(job.event_id)['container_number'] == 'TCKU6836529'
    controller.tick(10**12)
    dispatcher.enqueue.assert_not_called()
    repository.mark_cloud_synced(job.event_id)
    controller.record(job, ContainerValidator().validate(OCRRead('MSCU1234566', .99)))
    assert repository.event(job.event_id)['container_number'] == 'TCKU6836529'


def test_confirmed_visit_dispatches_once_with_matching_proof(settings, repository, snapshots, make_job):
    from app.domain import Observation
    dispatcher = Mock()
    controller = CaptureDeadlines(settings, repository, snapshots, dispatcher)
    job = replace(make_job(1), event_id=str(uuid4()))
    image = job.frame.image.copy()
    image[10:40, 10:80] = 255
    job = replace(job, frame=replace(job.frame, image=image))
    value = ContainerValidator().validate(OCRRead('TCKU6836529', .94))
    controller.record(job, value)
    obs = Observation(job.origin_key, job.gate_id, job.track_id, job.detection,
                      value, job.direction, image, job.started_at, True,
                      event_id=job.event_id)
    repository.save_observation(obs, snapshots, settings.dedup_seconds, ongoing_event_id=job.event_id)
    controller.tick()
    controller.tick()
    dispatcher.enqueue.assert_called_once_with(None, job.event_id, delay_seconds=0)
    payload = CloudOutboxDispatcher(settings, repository=repository, snapshots=snapshots).build_event_payload(job.event_id)
    assert payload['Containers'][0]['ContainerNumber'] == 'TCKU6836529'
    assert payload['Containers'][0]['Images']


def test_payload_uses_matching_number_source_from_each_camera(settings, repository, snapshots, make_job):
    from types import SimpleNamespace
    from app.ocr.job_store import OCRJobStore
    store = OCRJobStore(settings.upload_directory / 'ocr-spool', image_root=snapshots.root)
    event_id = str(uuid4())
    controller = CaptureDeadlines(settings, repository, snapshots, Mock())
    valid = ContainerValidator().validate(OCRRead('TCKU6836529', .94))
    for index, camera in enumerate(('camera-1', 'camera-2', 'camera-3'), 1):
        job = replace(make_job(index, camera_id=camera, origin='visit:' + camera), event_id=event_id)
        image = job.frame.image.copy()
        image[10:40,10:80] = 255
        job = replace(job, frame=replace(job.frame, image=image),
                      detection=replace(job.detection, class_name='container_number'))
        read = valid if index < 3 else ContainerValidator().validate(OCRRead('partial', .99))
        job_id = store.save(job, image[10:40,10:80])
        store.save_validation(job_id, read)
        if index == 1:
            controller.record(job, read)
    repository.mark_capture_ready(event_id)
    dispatcher = CloudOutboxDispatcher(settings, repository=repository, snapshots=snapshots,
                                       runtime=SimpleNamespace(ocr_store=store))
    payload = dispatcher.build_event_payload(event_id)
    assert len(payload['Containers']) == 1
    assert payload['Containers'][0]['ContainerNumber'] == 'TCKU6836529'
    assert {i['CameraId'] for i in payload['Containers'][0]['Images']} == {'camera-1', 'camera-2'}
