"""
Tests for Prosper Smart Yard Cloud Outbox Dispatcher
"""

import time
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from pydantic import SecretStr

from app.domain import Detection, Observation, Validation, utcnow
from app.events.cloud_dispatcher import CloudOutboxDispatcher, ProsperSession


@pytest.fixture
def mock_settings(settings):
    cfg = settings.model_copy(update={
        'prosper_enabled': True,
        'prosper_auto_dispatch': True,
        'prosper_api_base_url': 'http://syapi.test.prosper.com',
        'prosper_site_id': 'site-uuid-1234',
        'prosper_bearer_token': SecretStr('test-bearer-token'),
        'prosper_site_code': 'SITE1',
        'prosper_email': 'test@prosper.com',
        'prosper_password': SecretStr('secret123'),
        'prosper_max_retries': 2,
        'prosper_queue_size': 10,
    })
    return cfg


def test_prosper_session_bearer_headers(mock_settings):
    session = ProsperSession(mock_settings)
    headers = session.get_auth_headers()
    assert headers.get('Authorization') == 'Bearer test-bearer-token'


def test_prosper_session_api_key_headers(mock_settings):
    cfg = mock_settings.model_copy(update={
        'prosper_bearer_token': None,
        'prosper_api_key': SecretStr('my-api-key-999'),
    })
    session = ProsperSession(cfg)
    headers = session.get_auth_headers()
    assert headers.get('x-api-key') == 'my-api-key-999'


def test_prosper_session_auto_login(mock_settings):
    cfg = mock_settings.model_copy(update={
        'prosper_bearer_token': None,
        'prosper_api_key': None,
    })
    session = ProsperSession(cfg)

    with patch.object(session.session, 'post') as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {'token': 'auto-login-jwt-token', 'expiresIn': 3600}

        headers = session.get_auth_headers()
        assert headers.get('Authorization') == 'Bearer auto-login-jwt-token'
        assert mock_post.called
        assert '/api/auth/login' in mock_post.call_args[0][0]


def test_prosper_session_upload_image(mock_settings):
    session = ProsperSession(mock_settings)
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch.object(session.session, 'post') as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {'url': 'https://s3.prosper.com/trailers/abc.jpg'}

        url = session.upload_image(dummy_img)
        assert url == 'https://s3.prosper.com/trailers/abc.jpg'
        assert mock_post.called
        assert '/api/sites/site-uuid-1234/images/trailers' in mock_post.call_args[0][0]


def test_prosper_session_post_gate_event(mock_settings):
    session = ProsperSession(mock_settings)
    payload = {
        'id': 'event-1',
        'gateId': 'gate-lane-1',
        'timestamp': '2026-09-09T12:00:00.000Z',
        'containerNumber': 'CSQU3054383',
        'eventType': 'ENTRY',
        'deviceId': 'device-uuid-1',
        'confidence': 0.95,
        'imageUrl': 'https://s3.prosper.com/trailers/abc.jpg',
    }

    with patch.object(session.session, 'post') as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.content = b'{"visitDatabaseId": "abc-123", "eventId": "evt-456"}'
        mock_post.return_value.json.return_value = {'visitDatabaseId': 'abc-123', 'eventId': 'evt-456'}
        ok = session.post_gate_event(payload)
        # post_gate_event now returns the response dict on success (for visitDatabaseId extraction)
        assert isinstance(ok, dict) or ok is True
        assert '/api/sites/site-uuid-1234/gate-events' in mock_post.call_args[0][0]


def test_prosper_session_skips_when_no_auth(mock_settings):
    """Ensure post_gate_event does not send unauthenticated requests that trigger 401."""
    cfg = mock_settings.model_copy(update={'prosper_bearer_token': None, 'prosper_api_key': None})
    session = ProsperSession(cfg)
    # Simulate auto-login failing (no token)
    with patch.object(session, 'get_auth_headers', return_value={'X-Site-Id': 'site-1'}):
        with patch.object(session.session, 'post') as mock_post:
            ok = session.post_gate_event({'id': 'ev-1', 'Containers': []})
            assert ok == 'OFFLINE'
            assert not mock_post.called


def test_prosper_session_invalidates_token_on_401(mock_settings):
    """Ensure cached token is invalidated when cloud returns 401."""
    session = ProsperSession(mock_settings)
    session._token = 'cached-token'
    session._token_expires_at = time.time() + 3600

    with patch.object(session.session, 'post') as mock_post:
        mock_post.return_value.status_code = 401
        mock_post.return_value.text = 'Unauthorized'
        ok = session.post_gate_event({'id': 'ev-1', 'Containers': []})
        assert ok == 'OFFLINE'
        assert session._token is None
        assert session._token_expires_at == 0.0


def test_prosper_session_buffers_on_503_service_unavailable(mock_settings):
    """Ensure 503 service unavailable returns OFFLINE to buffer locally."""
    session = ProsperSession(mock_settings)
    with patch.object(session.session, 'post') as mock_post:
        mock_post.return_value.status_code = 503
        mock_post.return_value.text = 'The service is unavailable.'
        ok = session.post_gate_event({'id': 'ev-1', 'Containers': []})
        assert ok == 'OFFLINE'


def test_cloud_dispatcher_disabled_does_nothing(settings):
    cfg = settings.model_copy(update={'prosper_enabled': False})
    dispatcher = CloudOutboxDispatcher(cfg)
    dispatcher.start()
    assert dispatcher._worker_thread is None

    # Enqueue should safely no-op
    dummy_obs = MagicMock(confirmed=True)
    dispatcher.enqueue(dummy_obs, 'event-1')
    assert dispatcher.queue.empty()
    dispatcher.stop()


def test_cloud_dispatcher_enqueue_and_sync(mock_settings):
    metrics = MagicMock()
    dispatcher = CloudOutboxDispatcher(mock_settings, metrics=metrics)

    dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
    det = Detection(
        camera_id='gate-1',
        frame_timestamp=utcnow(),
        class_name='container',
        confidence=0.95,
        bbox=(10.0, 10.0, 50.0, 50.0),
    )
    val = Validation(raw_text='CSQU 305438-3', normalized_text='CSQU3054383',
                     confidence=0.95, valid_format=True, valid_check_digit=True,
                     validation_status='VALID')
    obs = Observation('origin-1', 'gate-lane-1', 1, det, val, 'ENTRY', dummy_img, utcnow(), confirmed=True)

    with patch.object(dispatcher.session, 'upload_image', return_value='https://img.test/1.jpg') as mock_up, \
         patch.object(dispatcher.session, 'post_gate_event', return_value={'visitDatabaseId': 'vdb-1', 'eventId': 'eid-1'}) as mock_post:

        dispatcher.start()
        dispatcher.enqueue(obs, 'evt-uuid-123')

        # Wait briefly for background worker
        for _ in range(20):
            if dispatcher.queue.empty() and mock_post.called:
                break
            time.sleep(0.05)

        dispatcher.stop()

        assert mock_post.called
        call_payload = mock_post.call_args[0][0]
        assert call_payload['id'] == 'evt-uuid-123'
        assert call_payload['containerNumber'] == 'CSQU3054383'
        assert call_payload['EventType'] in ('GATE_IN', 'GATE_OUT')
        assert metrics.increment.called


def test_syapi_gate_event_schema(mock_settings):
    dispatcher = CloudOutboxDispatcher(mock_settings)
    dummy_img = np.zeros((32, 32, 3), dtype=np.uint8)
    det = Detection(
        camera_id='CAM-01',
        frame_timestamp=utcnow(),
        class_name='container',
        confidence=0.96,
        bbox=(5.0, 5.0, 25.0, 25.0),
    )
    val = Validation(raw_text='MSCU 663987 0', normalized_text='MSCU6639870',
                     confidence=0.96, valid_format=True, valid_check_digit=True,
                     validation_status='VALID', size_type='22G1')
    obs = Observation('origin-syapi', 'LANE-1', 10, det, val, 'ENTRY', dummy_img, utcnow(), confirmed=True)

    with patch.object(dispatcher.session, 'post_gate_event', return_value=True) as mock_post:
        dispatcher.start()
        dispatcher.enqueue(obs, 'event-uuid-888')

        for _ in range(20):
            if dispatcher.queue.empty() and mock_post.called:
                break
            time.sleep(0.05)

        dispatcher.stop()

        assert mock_post.called
        payload = mock_post.call_args[0][0]
        assert 'VisitId' in payload
        assert payload['EventType'] == 'GATE_IN'
        assert payload['DeviceId'] == 'EDGE-GATE-01'
        assert payload['Truck'] == {'TruckNumber': 'NA'}
        assert payload['Driver'] is None
        assert len(payload['Containers']) == 1
        container = payload['Containers'][0]
        assert container['ContainerNumber'] == 'MSCU6639870'
        assert container['ContainerNumberConfidence'] == 0.96
        assert container['Size'] == '20'
        assert container['Position'] == 'SINGLE'
        assert len(container['Images']) >= 1
        assert container['Images'][0]['CameraId'] == 'CAM-01'
        assert container['Images'][0]['Image'].startswith('data:image/jpeg;base64,')


def test_each_camera_sends_one_image_without_duplicate_views(mock_settings, tmp_path):
    import cv2
    from pathlib import Path

    # Create dummy sharp images
    snap_dir = tmp_path / "snaps"
    snap_dir.mkdir()
    event_folder = snap_dir / "evt-123"
    event_folder.mkdir()

    img = np.ones((300, 400, 3), dtype=np.uint8) * 128
    # Add high contrast edges for high Laplacian quality
    img[50:250, 50:350] = 255
    cv2.putText(img, "TEST CONTAINER", (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    _, buf = cv2.imencode('.jpg', img)

    p1 = event_folder / "gate-in-1.jpg"
    p1.write_bytes(buf.tobytes())
    p2a = event_folder / "gate-in-2-a.jpg"
    p2a.write_bytes(buf.tobytes())
    p2b = event_folder / "gate-in-2-b.jpg"
    p2b.write_bytes(buf.tobytes())
    p3 = event_folder / "gate-in-3.jpg"
    p3.write_bytes(buf.tobytes())

    mock_snaps = MagicMock()
    mock_snaps.path.side_effect = lambda rel: snap_dir / rel
    mock_snaps.compute_quality.return_value = 150.0

    mock_repo = MagicMock()
    mock_repo.event.return_value = {
        'id': 'evt-123',
        'status': 'CONFIRMED',
        'container_number': 'BSIU3391518',
        'ocr_results': [{'raw_text': 'BSIU3391518', 'normalized_text': 'BSIU3391518', 'valid_check_digit': True, 'confidence': .91, 'detection_id': 'det-main'}],
        'detections': [{'id': 'det-main', 'camera_id': 'gate-in-1', 'timestamp': '2026-09-30T10:00:00Z'},
                       {'id': 'det-side', 'camera_id': 'gate-in-2', 'timestamp': '2026-09-30T10:00:01Z'},
                       {'id': 'det-rear', 'camera_id': 'gate-in-3', 'timestamp': '2026-09-30T10:00:03Z'}],
        'container_number_2': None,
        'is_twin': 0,
        'iso_code_1': '22G1',
        'primary_snapshot_id': 'snap-2a',
        'cloud_synced': 0,
        'snapshots': [
            {'id': 'snap-1', 'camera_id': 'gate-in-1', 'image_path': 'evt-123/gate-in-1.jpg', 'timestamp': '2026-09-30T10:00:00Z'},
            {'id': 'snap-2a', 'camera_id': 'gate-in-2', 'image_path': 'evt-123/gate-in-2-a.jpg', 'timestamp': '2026-09-30T10:00:01Z'},
            {'id': 'snap-2b', 'camera_id': 'gate-in-2', 'image_path': 'evt-123/gate-in-2-b.jpg', 'timestamp': '2026-09-30T10:00:02Z'},
            {'id': 'snap-3', 'camera_id': 'gate-in-3', 'image_path': 'evt-123/gate-in-3.jpg', 'timestamp': '2026-09-30T10:00:03Z'},
        ]
    }
    mock_repo.canonical_event_id.return_value = 'evt-123'

    dispatcher = CloudOutboxDispatcher(mock_settings, repository=mock_repo, snapshots=mock_snaps)

    with patch.object(dispatcher.session, 'post_gate_event', return_value=True) as mock_post:
        dispatcher.start()
        dispatcher.enqueue(None, 'evt-123')

        for _ in range(20):
            if dispatcher.queue.empty() and mock_post.called:
                break
            time.sleep(0.05)

        dispatcher.stop()

        assert mock_post.called
        payload = mock_post.call_args[0][0]
        assert len(payload['Containers']) == 1
        container = payload['Containers'][0]
        assert container['ContainerNumber'] == 'BSIU3391518'
        assert container['Position'] == 'SINGLE'

        # Only the camera frame with a validated identity read is upload proof
        camera_ids = [img['CameraId'] for img in container['Images']]
        assert camera_ids == ['gate-in-1']
        assert [img['ImageType'] for img in container['Images']] == ['FRONT']


def test_dispatch_audit_logging(mock_settings):
    audit_calls = []
    session = ProsperSession(mock_settings, audit_callback=lambda entry: audit_calls.append(entry))

    payload = {
        'id': 'evt-audit-1',
        'VisitId': 'VISIT-evt-audit-1',
        'EventType': 'GATE_OUT',
        'containerNumber': 'CSQU3054383',
        'Containers': [
            {
                'ContainerNumber': 'CSQU3054383',
                'Position': 'SINGLE',
                'Images': [
                    {'ImageType': 'FRONT', 'CameraId': 'cam-1', 'CapturedAt': '2026-10-15T12:00:00Z', 'Image': 'data:image/jpeg;base64,1234567890'}
                ]
            }
        ]
    }

    with patch.object(session.session, 'post') as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.content = b'{"eventId": "cloud-ev-999", "visitDatabaseId": "v-888"}'
        mock_post.return_value.json.return_value = {'eventId': 'cloud-ev-999', 'visitDatabaseId': 'v-888'}

        res = session.post_gate_event(payload)
        assert res == {'eventId': 'cloud-ev-999', 'visitDatabaseId': 'v-888'}
        assert len(audit_calls) == 1

        entry = audit_calls[0]
        assert entry['event_id'] == 'evt-audit-1'
        assert entry['visit_id'] == 'VISIT-evt-audit-1'
        assert entry['status_code'] == 200
        assert entry['status'] == 'SYNCED'
        assert entry['cloud_event_id'] == 'cloud-ev-999'
        assert entry['visit_database_id'] == 'v-888'
        assert entry['api_response'] == {'eventId': 'cloud-ev-999', 'visitDatabaseId': 'v-888'}
        # Image base64 must be sanitized and not stored raw
        assert 'Image' not in entry['event_sent']['Containers'][0]['Images'][0]
        assert entry['event_sent']['Containers'][0]['Images'][0]['image_bytes_len'] > 0


