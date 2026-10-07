from app.domain import OCRRead
from app.ocr.validator import ContainerValidator
from app.ocr.container_ocr import inside_gate
from app.events.event_manager import EventManager
from app.events.cloud_dispatcher import identity_confidence, detection_confidence


def test_outside_or_mostly_outside_gate_is_rejected():
    assert not inside_gate((5, 20, 40, 80), (100,100,3), (.5,0,1,1))
    assert not inside_gate((0, 20, 100, 80), (100,100,3), (.5,0,1,1))
    assert inside_gate((55,20,95,80), (100,100,3), (.5,0,1,1))


def test_dwell_cannot_bypass_confirmation_count(settings, repository, snapshots, make_job):
    settings.ocr_confirmations = 2
    manager = EventManager(settings, repository, snapshots)
    read = ContainerValidator().validate(OCRRead('CSQU3054383', .94))
    assert manager.observe(make_job(1, origin='dwell:camera-1:visit'), read) is None
    assert manager.observe(make_job(1, origin='dwell:camera-1:visit'), read) is None
    event_id, _ = manager.observe(make_job(2, origin='dwell:camera-1:visit'), read)
    assert repository.event(event_id)['status'] == 'CONFIRMED'


def test_low_confidence_checksum_stays_review_but_number_visible(settings, repository, snapshots, make_job):
    manager = EventManager(settings, repository, snapshots)
    read = ContainerValidator().validate(OCRRead('CSQU3054383', .31))
    for i in range(settings.ocr_max_attempts):
        result = manager.observe(make_job(i), read)
    event = repository.event(result[0])
    assert event['status'] == 'NEEDS_REVIEW'
    assert event['container_number'] == 'CSQU3054383'
    assert event['ocr_results'][0]['confidence'] == .31
    assert event['review_reason']


def test_cloud_restart_preserves_identity_confidence():
    event = {'ocr_results':[{'normalized_text':'CSQU3054383','valid_check_digit':True,
                             'confidence':.71,'detection_id':'one'},
                            {'normalized_text':'22G1','confidence':.99}],
             'detections':[{'id':'one','confidence':.82}]}
    assert identity_confidence(event, None, 'CSQU3054383') == .71
    assert detection_confidence(event, None, 'CSQU3054383') == .82
    assert identity_confidence({}, None, 'CSQU3054383') == 0


def test_moving_truck_is_not_confirmed(settings, repository, snapshots, make_job):
    from dataclasses import replace
    manager = EventManager(settings, repository, snapshots)
    read = ContainerValidator().validate(OCRRead('CSQU3054383', .94))
    for i in range(settings.ocr_max_attempts):
        result = manager.observe(replace(make_job(i), is_stationary=False), read)
    assert repository.event(result[0])['status'] == 'NEEDS_REVIEW'
    result = manager.observe(replace(make_job(10), is_stationary=True), read)
    assert repository.event(result[0])['status'] == 'CONFIRMED'


def test_cloud_revalidates_actual_text_not_synthetic_valid_flag():
    from app.events.cloud_dispatcher import verified_cloud_identity
    event = {'ocr_results':[{'raw_text':'CSQU305438','normalized_text':'CSQU3054383',
                             'valid_check_digit':True,'confidence':.99}]}
    assert not verified_cloud_identity(event, None, 'CSQU3054383', .6)
    event['ocr_results'][0]['raw_text'] = 'CSQU 305438 [3]'
    assert verified_cloud_identity(event, None, 'CSQU3054383', .6)


def test_prebuilt_cloud_payload_cannot_bypass_validation(settings):
    import pytest
    from app.events.cloud_dispatcher import CloudOutboxDispatcher
    dispatcher = CloudOutboxDispatcher(settings)
    with pytest.raises(ValueError):
        dispatcher.build_event_payload('missing', pre_payload={'Containers':[{'ContainerNumber':'CSQU3054383'}]})


def test_runtime_only_queues_stopped_inside_roi(settings, repository, snapshots, camera, make_job):
    from dataclasses import replace
    from datetime import timedelta
    from unittest.mock import Mock
    from app.runtime import GateRuntime
    from app.domain import Detection
    from app.ocr.container_ocr import crop_identification
    camera.occupancy_roi = (.5, 0, 1, 1)
    settings.min_track_hits = 1
    runtime = GateRuntime(settings, [camera], repository, snapshots)
    runtime.detector = Mock()
    runtime.model_status['detector'] = 'READY'
    runtime.ocr = Mock()
    runtime.ocr.read.return_value = OCRRead('CSQU3054383', .94)
    frame = replace(make_job().frame, source_id='test')
    def run(second, bbox):
        f = replace(frame, timestamp=frame.timestamp+timedelta(seconds=second), sequence=second+1)
        runtime.detector.detect.return_value = [Detection(camera.id, f.timestamp, 'container', .95, bbox)]
        runtime.process_frame(camera, f)
        while not runtime.queue.empty():
            job = runtime.queue.get_nowait()
            key = runtime.ocr_store.save(job, crop_identification(job.frame.image, job.detection.bbox, job.roi))
            runtime._process_saved_ocr(key)
            runtime.queue.task_done()
    for t in range(3):
        run(t, (5, 10, 50, 90))
    assert runtime.ocr.read.call_count == 0
    run(3, (80, 10, 140, 90))
    assert runtime.ocr.read.call_count == 0
    run(4, (80, 10, 140, 90))
    run(5, (80, 10, 140, 90))
    assert repository.events()['total'] == 1
    assert repository.events()['items'][0]['status'] == 'CONFIRMED'
