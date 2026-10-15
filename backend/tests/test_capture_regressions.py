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


def test_matching_reads_confirm_without_waiting_for_truck_stop(settings, repository, snapshots, make_job):
    from dataclasses import replace
    manager = EventManager(settings, repository, snapshots)
    read = ContainerValidator().validate(OCRRead('CSQU3054383', .94))
    for i in range(settings.ocr_max_attempts):
        result = manager.observe(replace(make_job(i), is_stationary=False), read)
    assert repository.event(result[0])['status'] == 'CONFIRMED'
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


def test_runtime_queues_moving_inside_roi_but_rejects_outside(settings, repository, snapshots, camera, make_job):
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
    assert runtime.ocr.read.call_count > 0
    run(4, (80, 10, 140, 90))
    run(5, (80, 10, 140, 90))
    assert repository.events()['total'] == 1
    assert repository.events()['items'][0]['status'] == 'CONFIRMED'


def test_expanded_text_crop_stays_inside_gate():
    import numpy as np
    from app.ocr.container_ocr import expand_number_crop
    image = np.zeros((100,100,3), dtype=np.uint8)
    image[:,50:] = 255
    crop = expand_number_crop(image, (50,20,85,40), (.5,0,.9,1))
    assert crop.shape[1] <= 40
    assert np.all(crop == 255)


def test_accepted_expanded_crop_preserves_original(settings, make_job):
    import numpy as np
    from app.ocr.job_store import OCRJobStore
    store = OCRJobStore(settings.upload_directory/'ocr-spool')
    original = np.zeros((20,30,3), dtype=np.uint8)
    key = store.save(make_job(), original)
    before = store.crop_path(key)
    expanded = np.full((25,40,3), 180, dtype=np.uint8)
    store.save_read_crop(key, expanded)
    assert before.is_file()
    assert store.crop_path(key) != before
    assert np.array_equal(store.load(key)[1], expanded)


def test_historical_synthesized_digit_is_not_displayed_as_confirmed():
    from app.database.repositories.gate import review_feedback
    event = {'status': 'CONFIRMED', 'container_number': 'MODU0038420'}
    review_feedback(event, [{'raw_text': 'MODU 003842', 'confidence': .8448,
                             'normalized_text': 'MODU0038420', 'valid_check_digit': True}])
    assert event['status'] == 'NEEDS_REVIEW'
    assert event['container_number'] is None
    assert event['historical_container_number'] == 'MODU0038420'
    assert 'Historical confirmation' in event['review_reason']


def test_complete_printed_digit_keeps_supported_historical_number():
    from app.database.repositories.gate import review_feedback
    event = {'status': 'CONFIRMED', 'container_number': 'MODU0038420'}
    review_feedback(event, [{'raw_text': 'MODU 003842 [0]', 'confidence': .84}])
    assert event['status'] == 'CONFIRMED'
    assert event['container_number'] == 'MODU0038420'


def test_partial_owner_serial_is_visible_without_inventing_check_digit():
    from app.database.repositories.gate import review_feedback
    event = {'status': 'NEEDS_REVIEW', 'container_number': None}
    review_feedback(event, [{'raw_text': 'CMAU 062325', 'confidence': .99}])
    assert event['container_number'] is None
    assert event['candidate_number'] == 'CMAU062325'
    assert event['candidate_is_partial']
    assert 'incomplete' in event['review_reason']


def test_cloud_proof_requires_identity_in_same_camera_frame():
    from app.events.cloud_dispatcher import snapshot_has_identity
    event = {'ocr_results': [{'detection_id':'read', 'raw_text':'CSQU3054383', 'confidence':.9}],
             'detections':[{'id':'read','camera_id':'one','timestamp':'2026-10-07T00:00:00Z'}]}
    snap = {'camera_id':'one','timestamp':'2026-10-07T00:00:00Z'}
    assert snapshot_has_identity(snap,event,'CSQU3054383',.6)
    assert not snapshot_has_identity({**snap,'camera_id':'companion'},event,'CSQU3054383',.6)
    assert not snapshot_has_identity({**snap,'timestamp':'2026-10-07T00:00:01Z'},event,'CSQU3054383',.6)


def test_cloud_proof_declared_jpeg_contains_jpeg_bytes():
    import cv2
    import numpy as np
    from app.snapshots.proof import proof_jpeg
    ok, png = cv2.imencode('.png', np.zeros((100,150,3), dtype=np.uint8))
    assert ok
    jpeg = proof_jpeg(png.tobytes(), {}, {})
    assert jpeg.startswith(b'\xff\xd8')
    assert cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), 1).shape == (100,150,3)


def test_two_camera_views_share_provisional_visit(settings, repository, snapshots, camera, make_job):
    from dataclasses import replace
    from unittest.mock import Mock
    from app.runtime import GateRuntime
    from app.domain import Detection
    second = camera.model_copy(update={'id':'second'})
    settings.min_track_hits = 1
    runtime = GateRuntime(settings,[camera,second],repository,snapshots)
    runtime.detector = Mock()
    runtime.model_status['detector'] = 'READY'
    for cam in (camera,second):
        frame = replace(make_job().frame,camera_id=cam.id,source_id='test')
        runtime.detector.detect.return_value = [Detection(cam.id,frame.timestamp,'container',.9,(20,20,70,80))]
        runtime.process_frame(cam,frame)
    assert runtime.occupancy[camera.id].current.provisional_event_id == runtime.occupancy[second.id].current.provisional_event_id
    # Detection assigns the shared ID without scoring/hydrating evidence on
    # every frame; the admitted OCR jobs supply durable source evidence.
    assert len(runtime.capture_deadlines.visits) == 0


def test_next_number_after_boundary_starts_new_visit(settings, camera):
    from dataclasses import replace
    from datetime import datetime, timedelta
    import numpy as np
    from app.domain import Detection, Frame
    from app.events.occupancy import ViewOccupancy
    camera.gate_exit_axis = 'x'
    camera.gate_exit_positive = True
    camera.gate_exit_position = .62
    view = ViewOccupancy(settings)
    timestamp = datetime(2026, 10, 7)
    frame = Frame(camera.id, timestamp, 1, np.zeros((100,100,3), dtype=np.uint8))
    first = Detection(camera.id, timestamp, 'container_number', .9, (45,30,60,70))
    view.update(camera, frame, [first])
    origin = view.current.origin
    departed_frame = replace(frame, timestamp=timestamp + timedelta(seconds=.5), sequence=2)
    departed = replace(first, frame_timestamp=departed_frame.timestamp, bbox=(75,30,90,70))
    view.update(camera, departed_frame, [departed])
    next_frame = replace(frame, timestamp=timestamp + timedelta(seconds=1), sequence=3)
    second = replace(first, frame_timestamp=next_frame.timestamp, bbox=(10,30,25,70))
    view.update(camera, next_frame, [second])
    assert view.current.origin != origin
    assert not view.current.is_twin


def test_unconfirmed_rear_read_cannot_promote_single_to_twin(settings, repository, snapshots, make_job):
    from dataclasses import replace
    from app.domain import Observation
    first = make_job(1)
    valid = ContainerValidator().validate(OCRRead('FCIU8060114', .9))
    event_id, _ = repository.save_observation(Observation(first.origin_key, first.gate_id,
        first.track_id, first.detection, valid, first.direction, first.frame.image,
        first.started_at, True), snapshots, settings.dedup_seconds)
    other = ContainerValidator().validate(OCRRead('MSKU7519800', .49))
    repository.save_observation(Observation(first.origin_key + ':rear', first.gate_id,
        first.track_id, first.detection, other, first.direction, first.frame.image,
        first.started_at, False, slot_position='REAR', event_id=event_id), snapshots,
        settings.dedup_seconds, ongoing_event_id=event_id)
    event = repository.event(event_id)
    assert event['container_number'] == 'FCIU8060114'
    assert not event['container_number_2']
    assert not event['is_twin']


def test_review_prefers_stronger_supported_number_without_confirmation():
    from app.database.repositories.gate import review_feedback
    event = {'status':'NEEDS_REVIEW', 'container_number':'CAQU5609761', 'is_twin':False}
    reads = [{'raw_text':'CAQU 560976 1','confidence':.2595},
             {'raw_text':'CAIU 560976 9','confidence':.8585}]
    review_feedback(event, reads)
    assert event['container_number'] == 'CAIU5609769'
    assert event['historical_container_number'] == 'CAQU5609761'
    assert event['status'] == 'NEEDS_REVIEW'


def test_improved_sample_keeps_original_visit_event(settings, repository, snapshots, make_job):
    from dataclasses import replace
    from uuid import uuid4
    from app.domain import Observation
    job = make_job()
    number = ContainerValidator().validate(OCRRead('CAJU5473444', .89))
    first_id, _ = repository.save_observation(Observation('dwell:camera-1:visit:late:one:deadline',
        job.gate_id, job.track_id, job.detection, number, job.direction, job.frame.image,
        job.started_at, False), snapshots, 30)
    requested_id = str(uuid4())
    actual_id, _ = repository.save_observation(Observation('dwell:camera-1:visit:late:two:deadline',
        job.gate_id, job.track_id, job.detection, number, job.direction, job.frame.image,
        job.started_at, False, event_id=requested_id), snapshots, 30, ongoing_event_id=requested_id)
    assert actual_id == first_id
    assert repository.events()['total'] == 1


def test_recent_activity_does_not_join_old_visit():
    from datetime import datetime, timedelta
    from types import SimpleNamespace
    from app.events.occupancy import concurrent_visits
    now = datetime(2026, 10, 8, 10)
    old = SimpleNamespace(first_seen=now-timedelta(minutes=14), last_measured=now, clear_since=None)
    new = SimpleNamespace(first_seen=now, last_measured=now, clear_since=None)
    assert not concurrent_visits(old, new, 8)
    old.first_seen = now-timedelta(seconds=5)
    assert concurrent_visits(old, new, 8)
    old.clear_since = now
    assert not concurrent_visits(old, new, 8)


def test_borrowed_event_id_cannot_undo_new_visit(repository, snapshots, make_job):
    from dataclasses import replace
    from datetime import timedelta
    from tests.test_events_database import observation
    first = make_job(origin='dwell:camera-1:old')
    one = repository.save_observation(observation(first, text='CSQU3054383'), snapshots, 30)[0]
    second = make_job(origin='dwell:camera-1:new', timestamp=first.frame.timestamp+timedelta(minutes=14))
    obs = replace(observation(second, text='TEMU4518441'), event_id=one)
    two = repository.save_observation(obs, snapshots, 30, ongoing_event_id=one)[0]
    assert two != one
    assert repository.event(one)['container_number'] == 'CSQU3054383'


def test_finished_text_visit_hands_off_without_empty_roi(settings, camera):
    from dataclasses import replace
    from tests.test_twin_occupancy import make_frame_detections
    from app.events.occupancy import ViewOccupancy
    camera.gate_exit_axis = 'x'
    camera.gate_exit_positive = True
    camera.gate_exit_position = 1.0
    view = ViewOccupancy(settings)
    frame, dets = make_frame_detections(0, [(60,20,70,80)])
    dets = [replace(d, class_name='container_number') for d in dets]
    view.update(camera, frame, dets)
    old = view.current.origin
    view.confirm(old, 'old-event', 'CSQU3054383')
    frame, dets = make_frame_detections(10, [(10,20,20,80)])
    dets = [replace(d, class_name='container_number') for d in dets]
    assert view.update(camera, frame, dets) is not None
    assert view.current.origin != old
    assert not view.current.is_twin


def test_text_pair_requires_common_vehicle():
    from datetime import datetime
    from app.domain import Detection
    from app.events.occupancy import same_vehicle_pair
    a, b = (.1,.2,.2,.8), (.6,.2,.7,.8)
    assert not same_vehicle_pair(a, b, [], (100,100,3))
    separate = [Detection('cam', datetime.now(), 'truck', .95, (0,0,30,100)),
                Detection('cam', datetime.now(), 'truck', .95, (50,0,90,100))]
    assert not same_vehicle_pair(a, b, separate, (100,100,3))
    common = [Detection('cam', datetime.now(), 'truck', .95, (0,0,90,100))]
    assert same_vehicle_pair(a, b, common, (100,100,3))


def test_checksum_valid_low_confidence_read_still_tries_enhancement():
    from types import SimpleNamespace
    from unittest.mock import Mock
    import numpy as np
    from app.runtime import GateRuntime

    runtime = GateRuntime.__new__(GateRuntime)
    runtime.settings = SimpleNamespace(ocr_min_confidence=.4)
    runtime.validator = ContainerValidator()
    runtime.ocr = Mock()
    low = OCRRead('HLHU 851675 1', .066802288)
    stronger = OCRRead('HLHU 851675 1', .88)
    runtime.ocr.read.side_effect = [low, stronger]
    crop = np.zeros((80,120,3), dtype=np.uint8)
    assert runtime._read_ocr(crop) == stronger
    assert runtime.ocr.read.call_count == 2

    # The real enhanced replay lost the check digit. Its high confidence must
    # not replace a complete observation or be presented as 99% identification.
    runtime.ocr.read.reset_mock()
    runtime.ocr.read.side_effect = [low, OCRRead('HLHU 851675', .996)]
    assert runtime._read_ocr(crop) == low
    assert runtime.ocr.read.call_count == 2

    runtime.ocr.read.reset_mock()
    runtime.ocr.read.side_effect = [stronger]
    assert runtime._read_ocr(crop) == stronger
    assert runtime.ocr.read.call_count == 1


def test_container_headroom_accepted_inside_gate():
    from app.ocr.container_ocr import expand_number_crop
    import numpy as np
    zone = [0.0, 0.34, 0.85, 1.0]
    shape = (1440, 2560, 3)
    # Container number in upper half of frame (headroom) within calibrated lane x
    headroom_bbox = (1500, 150, 1650, 600)
    assert inside_gate(headroom_bbox, shape, zone)
    # Detection outside lane horizontally is rejected
    outside_x = (2300, 150, 2450, 600)
    assert not inside_gate(outside_x, shape, zone)
    # expand_number_crop does not crash or clamp top to road zone
    image = np.zeros(shape, dtype=np.uint8)
    crop = expand_number_crop(image, headroom_bbox, zone)
    assert crop.shape[0] > 0 and crop.shape[1] > 0

