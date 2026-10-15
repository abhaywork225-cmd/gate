from datetime import datetime, timedelta
import numpy as np
import pytest
from app.domain import Frame, Detection, Validation, Observation
from app.events.occupancy import ViewOccupancy


def make_frame_detections(t, boxes, camera_id='cam-1', source='source'):
    ts = datetime(2026, 9, 15, 10, 0, 0) + timedelta(seconds=t)
    frame = Frame(camera_id, ts, int(t * 100) + 1, np.zeros((100, 100, 3), dtype=np.uint8), source)
    dets = [Detection(camera_id, ts, 'container', 0.95, box) for box in boxes]
    return frame, dets


def test_single_container_lifecycle(settings, camera):
    view = ViewOccupancy(settings)
    # 1 box in camera zone: 20ft or 40ft
    frame, dets = make_frame_detections(0, [((20, 20, 70, 80))])
    det = view.update(camera, frame, dets)
    assert det is not None
    assert view.current is not None
    assert not view.current.is_twin
    assert len(view.current.slots) == 1
    assert 'FRONT' in view.current.slots

    # Confirm event
    view.confirm(view.current.origin, 'evt-single', 'MSCU1234566')
    assert view.current.closed is True
    assert view.current.container_number == 'MSCU1234566'


def test_twin_containers_simultaneous_arrival(settings, camera):
    view = ViewOccupancy(settings)
    camera.gate_exit_axis = 'x'
    camera.gate_exit_positive = True

    # Two non-overlapping 20ft containers on trailer:
    # Front container box: x from 55 to 90
    # Rear container box: x from 10 to 45
    box_front = (55, 20, 90, 80)
    box_rear = (10, 20, 45, 80)

    frame, dets = make_frame_detections(0, [box_front, box_rear])
    det = view.update(camera, frame, dets)
    assert det is not None
    assert view.current is not None
    assert view.current.is_twin is True
    assert len(view.current.slots) == 2
    assert 'FRONT' in view.current.slots
    assert 'REAR' in view.current.slots

    front_slot = view.current.slots['FRONT']
    rear_slot = view.current.slots['REAR']
    assert front_slot.origin != rear_slot.origin

    # Confirm front container first
    view.confirm(front_slot.origin, 'evt-twin-1', 'MSCU1111111')
    assert front_slot.closed is True
    # Master occupancy must remain OPEN for the rear container!
    assert view.current.closed is False

    # Confirm rear container
    view.confirm(rear_slot.origin, 'evt-twin-1', 'TGHU2222222')
    assert rear_slot.closed is True
    # Now that BOTH are confirmed, master occupancy is closed!
    assert view.current.closed is True


def test_twin_containers_second_box_enters_later(settings, camera):
    view = ViewOccupancy(settings)
    # Frame 0: Only front container is in the zone
    box_front = (55, 20, 90, 80)
    frame0, dets0 = make_frame_detections(0, [box_front])
    view.update(camera, frame0, dets0)
    assert view.current is not None
    assert not view.current.is_twin

    # Frame 1: Truck moves forward, rear container now also in the zone
    box_rear = (10, 20, 45, 80)
    frame1, dets1 = make_frame_detections(1, [box_front, box_rear])
    view.update(camera, frame1, dets1)

    # Automatically upgraded to twin!
    assert view.current.is_twin is True
    assert 'REAR' in view.current.slots


def test_twin_containers_saved_under_single_event_id(repository, snapshots, camera):
    repo = repository

    frame1, (det1,) = make_frame_detections(0, [(55, 20, 90, 80)], camera_id=camera.id)
    val1 = Validation('MSCU1234566', 'MSCU1234566', 0.99, True, True, 'VALID', size_type='22G1')
    obs1 = Observation(f'dwell:{camera.id}:uuid1:front', camera.gate_id, 1, det1, val1, 'IN',
                       frame1.image, frame1.timestamp, True, slot_position='FRONT')

    # 1. Front container creates the gate event
    event_id, created = repo.save_observation(obs1, snapshots, 30)
    assert created is True
    assert event_id is not None

    # 2. Rear container arrives on same truck (ongoing_event_id = event_id)
    frame2, (det2,) = make_frame_detections(1, [(10, 20, 45, 80)], camera_id=camera.id)
    val2 = Validation('TGHU7654320', 'TGHU7654320', 0.98, True, True, 'VALID', size_type='22G1')
    obs2 = Observation(f'dwell:{camera.id}:uuid1:rear', camera.gate_id, 2, det2, val2, 'IN',
                       frame2.image, frame2.timestamp, True, slot_position='REAR')

    event_id2, created2 = repo.save_observation(obs2, snapshots, 30, ongoing_event_id=event_id)
    assert created2 is False
    assert event_id2 == event_id  # SAME EVENT ID!

    # 3. Retrieve event and verify both containers exist under that single event
    full_event = repo.event(event_id)
    assert full_event is not None
    assert full_event['id'] == event_id
    assert full_event['is_twin'] is True
    assert full_event['container_number'] == 'MSCU1234566'
    assert full_event['container_number_2'] == 'TGHU7654320'
    assert full_event['container_size'] == '20 FT'
    assert full_event['container_size_2'] == '20 FT'
    assert full_event['size_code'] == '22G1'
    assert full_event['size_code_2'] == '22G1'


def test_twin_containers_detected_via_container_number_markings(settings, camera):
    view = ViewOccupancy(settings)
    camera.gate_exit_axis = None

    # Two text markings detected (TEMU and SEGU) separated along the X axis
    ts = datetime(2026, 9, 30, 11, 45, 0)
    frame = Frame(camera.id, ts, 1, np.zeros((720, 1280, 3), dtype=np.uint8), 'source')
    # Normalized coords: X in 0..1280, Y in 0..720
    # Front container marking at x=1033..1088 (norm: 0.81..0.85)
    # Rear container marking at x=336..359 (norm: 0.26..0.28)
    det_front = Detection(camera.id, ts, 'container_number', 0.85, (1033.0, 416.0, 1088.0, 688.0))
    det_rear = Detection(camera.id, ts, 'container_number', 0.76, (336.0, 311.0, 359.0, 482.0))

    det = view.update(camera, frame, [det_front, det_rear])
    assert det is not None
    assert not view.current.is_twin  # Two queued trucks can show two markings.
    view = ViewOccupancy(settings)
    truck = Detection(camera.id, ts, 'truck', .95, (200, 200, 1200, 720))
    det = view.update(camera, frame, [det_front, det_rear, truck])
    assert det is not None
    assert view.current is not None
    assert view.current.is_twin is True
    assert len(view.current.slots) == 2
    assert 'FRONT' in view.current.slots
    assert 'REAR' in view.current.slots
