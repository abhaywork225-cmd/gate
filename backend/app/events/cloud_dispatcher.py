"""
Prosper Smart Yard Cloud Outbox Dispatcher
Asynchronously synchronizes confirmed GateVision events and evidence images
to the remote Prosper Smart Yard API (syapi.prosperassettracking.com).
"""

from __future__ import annotations

import base64
import hashlib
from datetime import datetime, timezone
import io
import json
import logging
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time
from typing import Any, Dict, Optional
import uuid

import cv2
import numpy as np
import requests
from app.snapshots.proof import proof_jpeg

log = logging.getLogger('gate.cloud')

FALLBACK_IMAGE_DATA_URI = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='


def _iso_utc(dt: Any = None) -> str:
    if dt is None:
        d = datetime.now(timezone.utc)
    elif isinstance(dt, (int, float)):
        try:
            d = datetime.fromtimestamp(dt, tz=timezone.utc)
        except Exception:
            d = datetime.now(timezone.utc)
    elif isinstance(dt, str):
        return dt
    elif isinstance(dt, datetime):
        if dt.tzinfo is None:
            d = dt.replace(tzinfo=timezone.utc)
        else:
            d = dt.astimezone(timezone.utc)
    else:
        d = datetime.now(timezone.utc)
    return d.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'


def _within_capture_window(left: Any, right: Any, seconds: float = 3.0) -> bool:
    """Compare camera timestamps safely when joining companion proof frames."""
    try:
        l = datetime.fromisoformat(str(left).replace('Z', '+00:00'))
        r = datetime.fromisoformat(str(right).replace('Z', '+00:00'))
        if l.tzinfo is None:
            l = l.replace(tzinfo=timezone.utc)
        if r.tzinfo is None:
            r = r.replace(tzinfo=timezone.utc)
        return abs((l - r).total_seconds()) <= seconds
    except Exception:
        return False


def verified_cloud_identity(event, observation, number, minimum):
    from app.domain import OCRRead
    from app.ocr.validator import ContainerValidator
    sources = [(o.get('raw_text', ''), o.get('confidence', 0))
               for o in (event or {}).get('ocr_results', [])]
    ocr = getattr(observation, 'ocr', None)
    if ocr:
        sources.append((ocr.raw_text, ocr.confidence))
    threshold = min(float(minimum or 0.50), 0.50)
    for raw, confidence in sources:
        value = ContainerValidator().validate(OCRRead(raw, float(confidence or 0)))
        if (value.valid_check_digit or (value.valid_format and float(value.confidence or 0) >= 0.50)) and value.normalized_text == number and float(value.confidence or 0) >= threshold:
            return True
    return False


def snapshot_has_identity(snapshot, event, number, minimum):
    """Only the source frame of an actual validated OCR read is identity proof."""
    from app.domain import OCRRead
    from app.ocr.validator import ContainerValidator
    ids = set()
    for reading in event.get('ocr_results', []):
        value = ContainerValidator().validate(OCRRead(reading.get('raw_text', ''), reading.get('confidence', 0)))
        if value.valid_check_digit and value.normalized_text == number and value.confidence >= minimum:
            ids.add(reading.get('detection_id'))
    return any(d.get('id') in ids and d.get('camera_id') == snapshot.get('camera_id')
               and _within_capture_window(d.get('timestamp'), snapshot.get('timestamp'), 3.0)
               for d in event.get('detections', []))


def identity_confidence(event, observation, number):
    """Use the actual reading for this identity, including outbox restarts."""
    readings = [o for o in (event or {}).get('ocr_results', [])
                if o.get('valid_check_digit') and o.get('normalized_text') == number]
    scores = [float(o.get('confidence') or 0) for o in readings]
    ocr = getattr(observation, 'ocr', None)
    if ocr and ocr.normalized_text == number:
        scores.append(float(ocr.confidence or 0))
    return max(scores, default=0.0)


def detection_confidence(event, observation, number):
    ids = {o.get('detection_id') for o in (event or {}).get('ocr_results', [])
           if o.get('normalized_text') == number and o.get('valid_check_digit')}
    scores = [float(d.get('confidence') or 0) for d in (event or {}).get('detections', [])
              if d.get('id') in ids]
    ocr = getattr(observation, 'ocr', None)
    det = getattr(observation, 'detection', None)
    if det and ocr and ocr.normalized_text == number:
        scores.append(float(det.confidence or 0))
    return max(scores, default=0.0)


def sanitize_payload_for_audit(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {'raw': str(payload)}
    sanitized = {}
    for k, v in payload.items():
        if k == 'Containers' and isinstance(v, list):
            containers_sanitized = []
            for c in v:
                if not isinstance(c, dict):
                    containers_sanitized.append(c)
                    continue
                c_copy = {}
                for ck, cv in c.items():
                    if ck == 'Images' and isinstance(cv, list):
                        images_summary = []
                        for img in cv:
                            if isinstance(img, dict):
                                img_copy = {ik: iv for ik, iv in img.items() if ik != 'Image'}
                                raw_img = img.get('Image')
                                if isinstance(raw_img, (str, bytes)):
                                    img_copy['image_bytes_len'] = len(raw_img)
                                    if isinstance(raw_img, str) and ',' in raw_img:
                                        img_copy['image_format'] = raw_img.split(',', 1)[0]
                                images_summary.append(img_copy)
                            else:
                                images_summary.append(str(img))
                        c_copy['Images'] = images_summary
                    else:
                        c_copy[ck] = cv
                containers_sanitized.append(c_copy)
            sanitized['Containers'] = containers_sanitized
        elif k in ('Image', 'imagePath', 'imageUrl') and isinstance(v, str) and len(v) > 200:
            sanitized[k] = f"<data len={len(v)}>"
        else:
            sanitized[k] = v
    return sanitized


def resolve_camera_image_type(camera_id: str, camera_configs: Optional[dict] = None) -> str:
    """Map camera ID or role to FRONT or REAR.
    gate-out-1 / gate-in-3 are REAR container door cameras.
    gate-out-2 / gate-in-1 / gate-in-2 are FRONT / cabin / approaching cameras.
    """
    c = (camera_id or '').lower()
    if camera_configs and camera_id in camera_configs:
        cfg = camera_configs[camera_id]
        role = str(getattr(cfg, 'role', None) or (cfg.get('role') if isinstance(cfg, dict) else '')).upper()
        if 'REAR' in role or 'BACK' in role:
            return 'REAR'
        if 'FRONT' in role or 'CABIN' in role or 'TOP' in role:
            return 'FRONT'
    if 'out-1' in c or 'in-3' in c or 'in-2' in c or any(k in c for k in ('rear', 'back', '-3')):
        return 'REAR'
    return 'FRONT'


def resolve_container_size(iso_code: Optional[str], container_size: Optional[str] = None,
                           is_twin: bool = False, aspect_ratio: Optional[float] = None) -> str:
    """Determine container size ('20', '40', '45'). Never returns None."""
    if is_twin:
        return '20'
    iso = (iso_code or '').strip().upper()
    cs = (container_size or '').strip().upper()
    if '20' in cs or iso.startswith('2'):
        return '20'
    if '45' in cs or iso.startswith('L'):
        return '45'
    if '40' in cs or iso.startswith('4'):
        return '40'
    if aspect_ratio is not None and aspect_ratio > 0:
        return '20' if aspect_ratio < 1.85 else '40'
    return '40'


class ProsperSession:
    """Manages authentication tokens and HTTP requests to Prosper Smart Yard."""

    def __init__(self, settings, audit_callback: Optional[Any] = None):
        self.settings = settings
        self.audit_callback = audit_callback
        self.last_dispatch_audit: Optional[Dict[str, Any]] = None
        base = settings.prosper_api_base_url.rstrip('/')
        if base.endswith('/api/v1/gate/events'):
            self.events_url = base
            self.base_url = base[:-len('/api/v1/gate/events')].rstrip('/')
        else:
            self.base_url = base
            self.events_url = f'{self.base_url}/api/v1/gate/events'
        self.auth_url = getattr(settings, 'prosper_auth_url', None) or 'https://cfsapi.prosperassettracking.com/api/auth/login'
        self.site_id = (settings.prosper_site_id or '').strip()
        self.client_id = (getattr(settings, 'prosper_client_id', None) or '').strip()
        self.session = requests.Session()
        self._token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._last_auth_attempt: float = 0.0
        self._auth_cooldown_seconds: float = 30.0
        # A configured bearer token can expire.  After a 401, fall back to the
        # username/password login flow instead of retrying the same stale token
        # forever and leaving the outbox permanently offline.
        self._force_dynamic_auth: bool = False
        self._lock = threading.Lock()

    def get_auth_headers(self) -> Dict[str, str]:
        """Return Authorization or API key headers, along with X-Site-Id and X-Client-Id."""
        headers: Dict[str, str] = {}
        if self.settings.prosper_bearer_token and not self._force_dynamic_auth:
            headers['Authorization'] = f'Bearer {self.settings.prosper_bearer_token.get_secret_value()}'
        elif self.settings.prosper_api_key and not self._force_dynamic_auth:
            headers['x-api-key'] = self.settings.prosper_api_key.get_secret_value()
        else:
            # Auto-login credential flow
            with self._lock:
                now = time.monotonic()
                if self._token and now < self._token_expires_at - 30:
                    headers['Authorization'] = f'Bearer {self._token}'
                elif now - self._last_auth_attempt < self._auth_cooldown_seconds:
                    # In cooldown period after a failed login attempt; avoid blocking worker threads
                    pass
                else:
                    self._last_auth_attempt = now
                    username = getattr(self.settings, 'prosper_username', None) or self.settings.prosper_email or 'abhay@prosperinfotech.com'
                    configured_pw = self.settings.prosper_password.get_secret_value() if self.settings.prosper_password else ''
                    passwords_to_try = [configured_pw] if configured_pw else []

                    login_urls = [self.auth_url]
                    alt_url = f'{self.base_url}/api/auth/login'
                    if alt_url not in login_urls:
                        login_urls.append(alt_url)

                    res = None
                    for l_url in login_urls:
                        for pw in passwords_to_try:
                            for payload in [{'UserName': username, 'Password': pw}, {'username': username, 'password': pw}]:
                                try:
                                    res = self.session.post(l_url, json=payload, timeout=6)
                                    if res.status_code in (200, 201):
                                        data = res.json()
                                        self._token = data.get('accessToken') or data.get('token')
                                        expires_in = data.get('expiresIn', 3600)
                                        self._token_expires_at = time.monotonic() + float(expires_in)
                                        # Parse JWT claims for site_ids and client_ids if not configured
                                        try:
                                            parts = self._token.split('.')
                                            if len(parts) >= 2:
                                                padded = parts[1] + '=' * (-len(parts[1]) % 4)
                                                claims = json.loads(base64.urlsafe_b64decode(padded))
                                                if not self.site_id and claims.get('site_ids'):
                                                    self.site_id = str(claims['site_ids'])
                                                if not self.client_id and claims.get('client_ids'):
                                                    self.client_id = str(claims['client_ids'])
                                        except Exception:
                                            pass
                                        log.info('prosper_auth_token_refreshed', extra={'username': username})
                                        break
                                    elif res.status_code == 204:
                                        # FastEndpoints cookie/header token check
                                        tok = res.cookies.get('token') or res.cookies.get('accessToken') or res.headers.get('Authorization')
                                        if tok:
                                            self._token = tok.replace('Bearer ', '')
                                            self._token_expires_at = time.monotonic() + 3600
                                            break
                                except Exception as e:
                                    log.debug('prosper_auth_login_error', extra={'error': str(e)})
                            if self._token:
                                break
                        if self._token:
                            break

                    if self._token:
                        self._auth_error = None
                        headers['Authorization'] = f'Bearer {self._token}'
                    elif res is not None:
                        if res.status_code == 423:
                            self._last_auth_attempt = now + 60
                            self._auth_error = 'Cloud account temporarily locked by auth server (HTTP 423). Auto-refreshing...'
                        elif res.status_code in (401, 403):
                            self._auth_error = f'Cloud authentication failed ({res.status_code}). Invalid credentials.'
                        else:
                            self._auth_error = f'Auth server returned HTTP {res.status_code}'
                        log.warning('prosper_auth_login_failed', extra={'status_code': res.status_code, 'body': res.text[:200]})
                    else:
                        self._auth_error = 'Unable to connect to auth server'
                        log.warning('prosper_auth_login_unreachable: Unable to connect to auth server (%s)', self.auth_url)

        if self.site_id:
            headers['X-Site-Id'] = self.site_id
        if self.client_id:
            headers['X-Client-Id'] = self.client_id
        return headers

    def upload_image(self, image_data: bytes | Any, trailer_number: str = '', camera_id: str = '',
                     device_id: str = '') -> Optional[str]:
        """Upload snapshot to Prosper and return public/stored URL."""
        if not self.site_id:
            return None

        headers = self.get_auth_headers()
        if 'Authorization' not in headers and 'x-api-key' not in headers:
            log.debug('prosper_image_upload_skipped_no_auth')
            return None

        if not isinstance(image_data, bytes):
            ok, buf = cv2.imencode('.jpg', image_data, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                return None
            image_data = buf.tobytes()

        headers = self.get_auth_headers()
        base64_str = base64.b64encode(image_data).decode('utf-8')
        base64_uri = f'data:image/jpeg;base64,{base64_str}'

        # 1. Try S3 Base64 upload endpoint
        s3_url = f'{self.base_url}/api/sites/{self.site_id}/images/trailers/upload-s3'
        s3_payload = {
            'trailerNumber': trailer_number or 'UNKNOWN',
            'scac': trailer_number[:4] if len(trailer_number) >= 4 and trailer_number[:4].isalpha() else 'UNKN',
            'deviceId': device_id,
            'cameraId': camera_id,
            'imagePath': base64_uri,
        }
        try:
            res = self.session.post(s3_url, json=s3_payload, headers={**headers, 'Content-Type': 'application/json'}, timeout=15)
            if res.status_code in (200, 201):
                data = res.json() if res.content else {}
                nested = data.get('data') if isinstance(data.get('data'), dict) else {}
                img_id = data.get('id') or nested.get('id') or nested.get('imageMetadataId') or data.get('imageMetadataId')
                img_url = nested.get('fileUrl') or nested.get('imagePath') or data.get('fileUrl') or data.get('imagePath') or data.get('url')
                self.last_image_metadata_id = img_id
                log.info('prosper_image_s3_uploaded', extra={'img_id': img_id, 'url': img_url})
                return img_url
        except Exception as e:
            log.warning('prosper_image_s3_upload_error', extra={'error': str(e)})

        # 2. Fallback to multipart upload
        url = f'{self.base_url}/api/sites/{self.site_id}/images/trailers'
        files = {'file': ('evidence.jpg', io.BytesIO(image_data), 'image/jpeg')}
        try:
            res = self.session.post(url, files=files, headers=headers, timeout=15)
            if res.status_code in (200, 201):
                data = res.json() if res.content else {}
                img_id = data.get('id') or data.get('imageMetadataId')
                img_url = data.get('url') or data.get('imageUrl') or data.get('fileUrl')
                self.last_image_metadata_id = img_id
                log.info('prosper_image_uploaded', extra={'url': img_url})
                return img_url
            log.warning('prosper_image_upload_failed', extra={'status_code': res.status_code})
        except Exception as e:
            log.warning('prosper_image_upload_error', extra={'error': str(e)})

        return None

    def _record_audit(self, payload: Dict[str, Any], url: str, status_code: Optional[int],
                      response_data: Any, duration_ms: float, status_label: str, error_msg: Optional[str] = None):
        cloud_event_id = None
        visit_database_id = None
        if isinstance(response_data, dict):
            cloud_event_id = response_data.get('eventId') or response_data.get('id')
            visit_database_id = response_data.get('visitDatabaseId')
        event_id = payload.get('id') or payload.get('VisitId')
        visit_id = payload.get('VisitId')

        audit_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event_id': event_id,
            'visit_id': visit_id,
            'event_sent': sanitize_payload_for_audit(payload),
            'api_endpoint': url,
            'status_code': status_code,
            'api_response': response_data,
            'cloud_event_id': cloud_event_id,
            'visit_database_id': visit_database_id,
            'duration_ms': duration_ms,
            'status': status_label,
            'error': error_msg,
        }
        self.last_dispatch_audit = audit_entry

        audit_log = logging.getLogger('gate.cloud.audit')
        extra = {
            'event_id': event_id,
            'visit_id': visit_id,
            'status_code': status_code,
            'duration_ms': duration_ms,
            'cloud_event_id': cloud_event_id,
            'audit_data': audit_entry,
        }
        summary_msg = f"Cloud dispatch to {url} - Status {status_code} ({status_label}) in {duration_ms}ms"
        if status_code in (200, 201):
            audit_log.info(summary_msg, extra=extra)
        else:
            audit_log.warning(summary_msg, extra=extra)

        if self.audit_callback:
            try:
                self.audit_callback(audit_entry)
            except Exception as e:
                log.warning('prosper_audit_callback_failed', extra={'error': str(e)})

        try:
            log_dir = Path(getattr(self.settings, 'log_directory', 'logs'))
            log_dir.mkdir(parents=True, exist_ok=True)
            sync_log_file = log_dir / 'prosper_cloud_sync.log'
            with sync_log_file.open('a', encoding='utf-8') as f:
                f.write(json.dumps(audit_entry, default=str) + '\n')
        except Exception as e:
            log.warning('prosper_sync_file_log_failed', extra={'error': str(e)})

    def post_gate_event(self, payload: Dict[str, Any]) -> bool | str:
        """Send Gate Event to Prosper Smart Yard."""
        headers = self.get_auth_headers()
        if 'Authorization' not in headers and 'x-api-key' not in headers:
            auth_msg = getattr(self, '_auth_error', None) or 'Cloud authentication unavailable'
            log.info('prosper_gate_event_postponed_no_auth: %s. Dispatch buffered locally.', auth_msg)
            self._record_audit(payload, getattr(self, 'events_url', self.base_url), None,
                               auth_msg, 0.0, 'OFFLINE', auth_msg)
            return 'OFFLINE'

        headers['Content-Type'] = 'application/json'

        # Check whether this is the new /api/v1/gate/events schema or legacy /api/sites/{site_id}/gate-events
        if 'VisitId' in payload or 'Containers' in payload:
            url = getattr(self, 'events_url', f'{self.base_url}/api/v1/gate/events')
        else:
            if not self.site_id:
                log.warning('prosper_sync_skipped_missing_site_id')
                self._record_audit(payload, f'{self.base_url}/api/sites/UNKNOWN/gate-events', None,
                                   'Missing site_id', 0.0, 'REJECTED', 'Missing site_id')
                return 'REJECTED'
            url = f'{self.base_url}/api/sites/{self.site_id}/gate-events'

        start_time = time.monotonic()
        try:
            res = self.session.post(url, json=payload, headers=headers, timeout=15)
            duration_ms = round((time.monotonic() - start_time) * 1000, 2)
        except requests.exceptions.RequestException as e:
            duration_ms = round((time.monotonic() - start_time) * 1000, 2)
            log.info('prosper_gate_event_network_offline: %s. Dispatch buffered locally in database.', e)
            self._record_audit(payload, url, None, str(e), duration_ms, 'OFFLINE', str(e))
            return 'OFFLINE'
        except Exception as e:
            duration_ms = round((time.monotonic() - start_time) * 1000, 2)
            log.warning('prosper_gate_event_post_exception: %s', e)
            self._record_audit(payload, url, None, str(e), duration_ms, 'OFFLINE', str(e))
            return 'OFFLINE'

        # Parse response body
        try:
            resp_body = res.json() if res.content else {}
        except Exception:
            resp_body = res.text[:2000]

        if res.status_code in (200, 201):
            resp_data = resp_body if isinstance(resp_body, dict) else {}
            log.info('prosper_gate_event_synced', extra={
                'event_id': payload.get('VisitId') or payload.get('id'),
                'visit_db_id': resp_data.get('visitDatabaseId', ''),
                'cloud_event_id': resp_data.get('eventId', ''),
            })
            self._record_audit(payload, url, res.status_code, resp_body, duration_ms, 'SYNCED')
            return resp_data if resp_data else True
        if res.status_code == 401:
            # 401 Unauthorized: token expired or rejected by cloud
            log.warning('prosper_gate_event_rejected_401: %s %s', res.status_code, res.text[:300],
                        extra={'status_code': res.status_code, 'body': res.text[:300]})
            with self._lock:
                self._token = None
                self._token_expires_at = 0.0
                self._force_dynamic_auth = True
            self._record_audit(payload, url, res.status_code, resp_body, duration_ms, 'OFFLINE', 'Unauthorized 401')
            return 'OFFLINE'
        if res.status_code == 409:
            # 409 Conflict: E.g., "Unknown visit for GATE_OUT."
            log.warning('prosper_gate_event_conflict_skipped: %s %s', res.status_code, res.text[:300],
                        extra={'status_code': res.status_code, 'body': res.text[:300]})
            self._record_audit(payload, url, res.status_code, resp_body, duration_ms, 'CONFLICT', 'Conflict 409')
            return 'CONFLICT'
        if res.status_code in (500, 502, 503, 504):
            body_text = res.text[:400].strip()
            if 'DbUpdateConcurrencyException' in body_text:
                event_type = payload.get('EventType', '')
                log.warning(
                    'prosper_gate_event_concurrency_retry: GATE_OUT concurrency conflict on cloud (known server bug). '
                    'Will retry. EventType=%s VisitId=%s',
                    event_type, payload.get('VisitId', ''),
                    extra={'status_code': 500, 'exception': 'DbUpdateConcurrencyException'}
                )
            elif 'ck_departure' in body_text:
                log.warning(
                    'prosper_gate_event_constraint_retry: visit constraint violation (ck_departure). '
                    'GATE_IN may not be committed yet. Will retry. VisitId=%s',
                    payload.get('VisitId', ''),
                    extra={'status_code': 500, 'exception': 'ck_departure'}
                )
            else:
                log.info('prosper_gate_event_cloud_service_unavailable: HTTP %s (%s). Dispatch buffered locally in database.',
                         res.status_code, body_text[:200])
            self._record_audit(payload, url, res.status_code, resp_body, duration_ms, 'OFFLINE', f'Server error {res.status_code}')
            return 'OFFLINE'
        if res.status_code in (400, 422):
            log.warning('prosper_gate_event_rejected: %s %s', res.status_code, res.text[:300],
                        extra={'status_code': res.status_code, 'body': res.text[:300]})
            self._record_audit(payload, url, res.status_code, resp_body, duration_ms, 'REJECTED', f'Rejected {res.status_code}')
            return 'REJECTED'

        log.warning('prosper_gate_event_failed: %s %s', res.status_code, res.text[:300],
                    extra={'status_code': res.status_code, 'body': res.text[:300]})
        self._record_audit(payload, url, res.status_code, resp_body, duration_ms, 'FAILED', f'Failed {res.status_code}')
        return False


class CloudOutboxDispatcher:
    """
    Dedicated non-blocking worker thread that serializes confirmed gate events,
    encodes snapshot evidence images, and dispatches gate events to Prosper Smart Yard.
    """

    def __init__(self, settings, metrics=None, repository=None, snapshots=None, runtime=None):
        self.settings = settings
        self.metrics = metrics
        self.repository = repository
        self.snapshots = snapshots
        self.runtime = runtime
        self.enabled = bool(settings.prosper_enabled)
        self.auto_dispatch: bool = bool(getattr(settings, 'prosper_auto_dispatch', False))
        self.session = ProsperSession(settings, audit_callback=self._on_dispatch_audit) if self.enabled else None
        self.queue: Queue = Queue(maxsize=settings.prosper_queue_size)
        self._stop = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        self._is_online: bool = True
        self._last_sync_time: Optional[str] = None
        self._last_error: Optional[str] = None

        self.device_uuid = (
            settings.prosper_device_uuid
            or 'EDGE-GATE-01'
        )

    def _on_dispatch_audit(self, audit_entry: Dict[str, Any]):
        """Persist dispatch audit entry to system_logs database table."""
        if not self.repository:
            return
        try:
            event_id = audit_entry.get('event_id')
            status_code = audit_entry.get('status_code')
            level = 'INFO' if status_code in (200, 201) else ('WARN' if status_code in (409, 500, 502, 503, 504) else 'ERROR')
            if hasattr(self.repository, 'log_system_event'):
                self.repository.log_system_event(
                    event_name='cloud_dispatch_audit',
                    level=level,
                    event_id=event_id,
                    context=audit_entry
                )
        except Exception as e:
            log.warning('prosper_audit_persistence_error', extra={'error': str(e)})

    def pause(self):
        """Pause automatic background dispatch to remote cloud site."""
        self.auto_dispatch = False
        log.info('prosper_cloud_auto_dispatch_paused')

    def resume(self):
        """Resume automatic background dispatch to remote cloud site."""
        self.auto_dispatch = True
        log.info('prosper_cloud_auto_dispatch_resumed')

    @property
    def is_paused(self) -> bool:
        return not self.auto_dispatch

    def status(self) -> dict:
        pending_count = 0
        if self.repository and hasattr(self.repository, 'count_unsynced_confirmed_events'):
            try:
                pending_count = self.repository.count_unsynced_confirmed_events()
            except Exception:
                pass
        return {
            'enabled': self.enabled,
            'auto_dispatch': self.auto_dispatch,
            'is_paused': not self.auto_dispatch,
            'is_online': self._is_online if self.enabled else False,
            'pending_count': pending_count,
            'queue_depth': self.queue.qsize(),
            'last_sync_time': self._last_sync_time,
            'last_error': self._last_error,
            'site_id': self.session.site_id if (self.session and self.session.site_id) else None,
        }

    def trigger_retry(self) -> int:
        """Trigger an immediate offline backlog sweep and return the number of pending items queued."""
        if not self.enabled or not self.repository:
            return 0
        self._sweep_offline_backlog()
        return self.status()['pending_count']

    def start(self):
        if not self.enabled:
            return
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._stop.clear()
        self._worker_thread = threading.Thread(target=self._run_loop, name='prosper-dispatcher', daemon=True)
        self._worker_thread.start()
        log.info('prosper_cloud_dispatcher_started')

    def stop(self, timeout: float = 2.0):
        if not self.enabled:
            return
        self._stop.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)
        log.info('prosper_cloud_dispatcher_stopped')

    def is_dispatchable_event(self, ev: dict | None, min_confidence: float = 0.50, allow_synced: bool = False) -> bool:
        """
        Check if an event meets requirements to be dispatched to Prosper site:
        1. Must not be already synced to cloud (unless allow_synced=True for re-dispatch).
        2. Must NOT be an unreadable event (status != 'UNREADABLE').
        3. Must have a valid container number (not None/empty/'UNREADABLE'/'UNCONFIRMED').
        4. If twin event, must have both valid distinct container numbers.
        5. Must have OCR confidence >= min_confidence (50% / 0.50).
        """
        if not ev:
            return False
        if not allow_synced and ev.get('cloud_synced'):
            return False
        if ev.get('status') == 'UNREADABLE':
            return False

        c1 = ev.get('container_number')
        if not c1 or str(c1).strip().upper() in ('', 'NONE', 'UNREADABLE', 'UNCONFIRMED'):
            return False

        if ev.get('is_twin'):
            c2 = ev.get('container_number_2')
            if not c2 or str(c2).strip().upper() in ('', 'NONE', 'UNREADABLE', 'UNCONFIRMED') or c2 == c1:
                return False

        ocr_results = ev.get('ocr_results', [])
        if ocr_results:
            c1_confs = [
                float(o.get('confidence', 0.0)) for o in ocr_results
                if o.get('normalized_text') == c1 or (o.get('raw_text', '').replace(' ', '') == c1)
            ]
            best_conf = max(c1_confs, default=max((float(o.get('confidence', 0.0)) for o in ocr_results), default=0.0))
            if best_conf < min_confidence:
                return False
            if ev.get('is_twin'):
                c2_confs = [
                    float(o.get('confidence', 0.0)) for o in ocr_results
                    if o.get('normalized_text') == c2 or (o.get('raw_text', '').replace(' ', '') == c2)
                ]
                best_conf_2 = max(c2_confs, default=0.0)
                if best_conf_2 < min_confidence:
                    return False
        else:
            if ev.get('status') != 'CONFIRMED':
                return False

        return True

    def enqueue(self, obs: Any, event_id: str, image_bytes: Optional[bytes] = None,
                payload: Optional[dict] = None, delay_seconds: float = 0.0):
        """Enqueue gate event with >= 50% OCR accuracy for background dispatch."""
        if not self.enabled:
            return
        if obs is not None:
            conf = getattr(getattr(obs, 'ocr', None), 'confidence', 0.0) or 0.0
            if not getattr(obs, 'confirmed', False) and float(conf) < 0.50:
                return
        if not self.auto_dispatch:
            log.info('prosper_event_held_auto_dispatch_paused', extra={'event_id': event_id})
            return

        # Do not re-enqueue if already synced in database, unreadable, or below 50% OCR confidence
        if self.repository:
            try:
                ev = self.repository.event(event_id)
                if ev and not self.is_dispatchable_event(ev):
                    return
                canonical = self.repository.canonical_event_id(event_id)
                if canonical and canonical != event_id:
                    # A later tracker/camera event for the same waiting
                    # container is represented by the original visit.
                    if hasattr(self.repository, 'mark_duplicate_event'):
                        self.repository.mark_duplicate_event(event_id, canonical)
                    else:
                        self.repository.mark_cloud_synced(event_id)
                    log.info('prosper_duplicate_event_suppressed', extra={
                        'event_id': event_id, 'canonical_event_id': canonical})
                    return
            except Exception:
                pass

        item = {
            'event_id': event_id,
            'obs': obs,
            'image_bytes': image_bytes if image_bytes is not None else getattr(obs, 'image', None),
            'payload': payload,
            'enqueued_at': time.monotonic(),
            'not_before': time.monotonic() + max(0.0, float(delay_seconds or 0.0)),
            'retries': 0,
        }

        try:
            self.queue.put_nowait(item)
            if self.metrics:
                self.metrics.increment('prosper_events_queued')
        except Full:
            log.warning('prosper_cloud_queue_full_event_dropped', extra={'event_id': event_id})
            if self.metrics:
                self.metrics.increment('prosper_queue_dropped')

    def _proof_is_usable(self, image, quality: float) -> bool:
        """Reject blank, tiny, badly exposed or blurred frames before cloud POST."""
        if image is None or getattr(image, 'size', 0) == 0:
            return False
        try:
            h, w = image.shape[:2]
            if h < 240 or w < 320:
                return False
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if float(gray.std()) < 8.0:
                return False
            minimum = float(getattr(self.settings, 'cloud_min_proof_quality', 2.5))
            return float(quality) >= minimum
        except Exception:
            return False

    def _sweep_offline_backlog(self):
        """
        Periodically checks SQLite database for confirmed events that were saved while offline.
        As soon as internet connectivity returns, dispatches them in chronological order.
        """
        if not self.enabled or not self.repository or not self.auto_dispatch:
            return
        try:
            unsynced = self.repository.get_unsynced_confirmed_events(limit=20)
            if not unsynced:
                return

            if self.queue.qsize() > 0:
                return

            dispatchable_items = []
            for ev in unsynced:
                event_id = ev['id']
                full_ev = self.repository.event(event_id) if hasattr(self.repository, 'event') else ev
                if self.is_dispatchable_event(full_ev):
                    dispatchable_items.append(event_id)

            if not dispatchable_items:
                return

            log.info('prosper_sweep_offline_backlog_found', extra={'count': len(dispatchable_items)})
            for event_id in dispatchable_items:
                if self._stop.is_set():
                    break
                try:
                    self.queue.put_nowait({
                        'event_id': event_id,
                        'obs': None,
                        'image_bytes': None,
                        'payload': None,
                        'enqueued_at': time.monotonic(),
                        'retries': 0,
                    })
                except Full:
                    break
        except Exception as e:
            log.warning('prosper_backlog_sweep_error', extra={'error': str(e)})

    def _best_available_payload(self, event):
        import re
        from app.domain import OCRRead
        from app.ocr.validator import ContainerValidator
        event = {**event, **{key: list(event.get(key, [])) for key in ('detections', 'ocr_results', 'snapshots')}}
        store = getattr(self.runtime, 'ocr_store', None)
        if store:
            evidence = store.event_proof_evidence(event['id'])
            for key in evidence:
                event[key].extend(evidence[key])
        containers = []
        # Each deadline slot has its own saved OCR observation and source frame.
        for slot in (('FRONT', 'REAR') if event.get('is_twin') else ('FRONT',)):
            choices = []
            for d in event.get('detections', []):
                origin = d.get('origin_key', '')
                if not origin.endswith(':deadline'):
                    continue
                is_rear = ':rear' in origin or origin.endswith(':rear:deadline')
                if event.get('is_twin') and is_rear != (slot == 'REAR'):
                    continue
                read = next((o for o in event.get('ocr_results', []) if o.get('detection_id') == d.get('id')), {})
                value = ContainerValidator().validate(OCRRead(read.get('raw_text', ''), read.get('confidence', 0)))
                if not value.valid_check_digit:
                    continue
                number = value.normalized_text
                expected = event.get('container_number_2') if slot == 'REAR' else event.get('container_number')
                if expected and number != expected:
                    continue
                partial = re.fullmatch(r'\s*([A-Z]{3}[UJZ])\s+([0-9]{6})\s*', read.get('raw_text', '').upper())
                if not number and partial:
                    number = ''.join(partial.groups())
                for snap in event.get('snapshots', []):
                    if snap.get('camera_id') != d.get('camera_id') or not _within_capture_window(d.get('timestamp'), snap.get('timestamp'), .05):
                        continue
                    data = self.snapshots.path(snap['image_path']).read_bytes()
                    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if image is None or image.size == 0 or float(image.std()) < 2:
                        continue
                    quality = self.snapshots.compute_quality(image)
                    choices.append(((value.valid_check_digit, bool(number), value.confidence, quality), number, value, snap, data))
            if not choices:
                continue
            _, number, value, snap, data = max(choices, key=lambda x:x[0])
            by_camera = {}
            for choice in choices:
                if choice[1] != number:
                    continue
                camera_id = choice[3]['camera_id']
                if camera_id not in by_camera or choice[0] > by_camera[camera_id][0]:
                    by_camera[camera_id] = choice
            images = [{'ImageType': resolve_camera_image_type(entry[3]['camera_id']), 'CameraId': entry[3]['camera_id'], 'CapturedAt': entry[3]['timestamp'],
                       'Image': 'data:image/jpeg;base64,' + base64.b64encode(proof_jpeg(entry[4],entry[3],event)).decode()}
                      for entry in by_camera.values()]
            resolved_size = resolve_container_size(None, None, is_twin=bool(event.get('is_twin')))
            containers.append({'ContainerNumber': number, 'ContainerNumberConfidence': value.confidence,
                'DetectionConfidence': 0.0, 'ValidationStatus': value.validation_status,
                'Position': slot if event.get('is_twin') else 'SINGLE', 'Size': resolved_size, 'SizeCode': '22G1' if resolved_size == '20' else '42G1',
                'Images': images})
        if not containers:
            raise ValueError('No captured source proof is available; retry after evidence is saved')
        return {'id':event['id'], 'VisitId':'VISIT-' + event['id'],
            'EventType':'GATE_OUT' if event.get('event_type') in ('EXIT','OUT','GATE_OUT') else 'GATE_IN',
            'DeviceId':self.device_uuid, 'CapturedAt':event.get('timestamp'),
            'containerNumber':containers[0]['ContainerNumber'], 'Truck':{'TruckNumber':'NA'},
            'Driver':None, 'Containers':containers}

    def _final_capture_payload(self,event):
        """Final source-bound proofs, including explicitly unreadable departures."""
        numbers = [event.get('container_number')]
        if event.get('is_twin'):
            numbers.append(event.get('container_number_2'))
        cam_cfgs = getattr(self.settings, 'cameras', lambda: {})()
        containers = []
        for index,number in enumerate(numbers):
            by_camera = {}
            target_snap_id = event.get('rear_snapshot_id') if (index == 1 and event.get('is_twin')) else event.get('primary_snapshot_id')
            id_snaps = [s for s in event.get('snapshots', []) if number and snapshot_has_identity(s, event, number, 0.0)]
            ref_ts = id_snaps[0].get('timestamp') if id_snaps else event.get('timestamp')

            for snap in event.get('snapshots',[]):
                if event.get('is_twin') and number:
                    other_num = event.get('container_number_2' if index == 0 else 'container_number')
                    if other_num and snapshot_has_identity(snap, event, other_num, 0.0):
                        continue
                path = self.snapshots.path(snap['image_path'])
                if not path.is_file(): continue
                data = path.read_bytes()
                decoded = cv2.imdecode(np.frombuffer(data,dtype=np.uint8),cv2.IMREAD_COLOR)
                if decoded is None or decoded.size==0: continue
                quality = self.snapshots.compute_quality(decoded)
                score = float(quality)
                if number and snapshot_has_identity(snap, event, number, 0.0):
                    score += 2000.0
                elif target_snap_id and snap.get('id') == target_snap_id:
                    score += 1000.0
                if ref_ts and not _within_capture_window(snap.get('timestamp'), ref_ts, float(getattr(self.settings, 'cross_camera_view_window_seconds', 8.0) or 8.0)):
                    continue
                camera = snap['camera_id']
                if camera not in by_camera or score > by_camera[camera][0]:
                    by_camera[camera] = (score,snap,data)
            if not by_camera:
                raise ValueError('Final capture has no retained source proof')
            position = ('FRONT' if index==0 else 'REAR') if event.get('is_twin') else 'SINGLE'
            images = [{'ImageType': resolve_camera_image_type(snap['camera_id'], cam_cfgs),
                       'CameraId': snap['camera_id'],
                       'CapturedAt': snap['timestamp'],
                       'Image': 'data:image/jpeg;base64,' + base64.b64encode(data).decode()}
                      for _,snap,data in by_camera.values()]
            matching = [o for o in event.get('ocr_results',[]) if number and o.get('normalized_text')==number]
            confidence = max((o.get('confidence',0) for o in matching),default=0)
            iso_code = event.get('iso_code_1' if index==0 else 'iso_code_2')
            c_size_str = event.get('container_size' if index==0 else 'container_size_2')
            resolved_size = resolve_container_size(iso_code, c_size_str, is_twin=bool(event.get('is_twin')))
            containers.append({'ContainerNumber':number,'ContainerNumberConfidence':confidence,
                'ValidationStatus':'VALID' if number else 'UNREADABLE',
                'RecognitionStatus':('CONFIRMED' if event.get('status')=='CONFIRMED' else 'UNCONFIRMED') if number else 'UNREADABLE',
                'Position':position,
                'Size':resolved_size,
                'ContainerSize':f"{resolved_size} FT",
                'SizeCode':iso_code or ('22G1' if resolved_size == '20' else '42G1'),
                'Images':images})
        return {'id':event['id'],'VisitId':'VISIT-'+event['id'],
            'EventType':'GATE_OUT' if event.get('event_type') in ('EXIT','OUT','GATE_OUT') else 'GATE_IN',
            'DeviceId':self.device_uuid,'CapturedAt':event.get('timestamp'),
            'containerNumber':numbers[0],'Truck':{'TruckNumber':'NA'},'Driver':None,'Containers':containers}

    def build_proof_preview(self, event_id):
        """Show matching source evidence without making a visit dispatchable."""
        event = self.repository.event(event_id) if self.repository else None
        if event and event.get('capture_policy') in ('FINAL_CAPTURE','UNREADABLE_CAPTURE'):
            return self._final_capture_payload(event)  # Same bytes; coverage gate does not substitute images.
        try:
            return self.build_event_payload(event_id)
        except ValueError:
            event = self.repository.event(event_id) if self.repository else None
            if not event:
                raise
            return self._best_available_payload(event)

    def build_event_payload(self, event_id: str, obs: Any = None, img_fallback: Any = None, pre_payload: Optional[dict] = None) -> dict:
        """
        Constructs the Prosper Gate Event payload.
        Ensures both Container 1 (Front) and Container 2 (Rear in twin events)
        have clear, verified, distinct proof images where their container numbers are visible.
        """
        # Build from observed identity and stored proof; a prebuilt payload
        # cannot bypass validation or substitute another visit's images.

        event_record = self.repository.event(event_id) if self.repository else None
        if event_record and event_record.get('capture_policy') in ('FINAL_CAPTURE','UNREADABLE_CAPTURE'):
            payload = self._final_capture_payload(event_record)
            for container in payload['Containers']:
                cameras = {image['CameraId'] for image in container['Images']}
                if container.get('ContainerNumber') and len(cameras) < self.settings.minimum_number_proof_cameras:
                    raise ValueError(f'Number proof incomplete: {len(cameras)} of {self.settings.minimum_number_proof_cameras} cameras for {container["ContainerNumber"]}')
            return payload
        if event_record and event_record.get('capture_policy') == 'BEST_AVAILABLE':
            return self._best_available_payload(event_record)
        if event_record and self.snapshots:
            store = getattr(self.runtime, 'ocr_store', None)
            if store:
                source_evidence = store.event_proof_evidence(event_id)
                for key in source_evidence:
                    event_record[key] = list(event_record.get(key, [])) + source_evidence[key]
            numbers = [event_record.get('container_number'), event_record.get('container_number_2')]
            manifest = self.snapshots.build_evidence_manifest(
                event_record,
                min_quality=getattr(self.settings, 'cloud_min_proof_quality', 2.5),
            )
            if isinstance(manifest, dict):
                event_record['evidence_manifest'] = manifest
                event_record['snapshots'] = manifest['captures']
                event_record['primary_snapshot_id'] = manifest['primary_snapshot_id']
                event_record['rear_snapshot_id'] = manifest['rear_snapshot_id']

        cntr1 = (event_record.get('container_number') if event_record else None) or (obs.ocr.normalized_text if (obs and obs.ocr) else '')
        cntr2 = event_record.get('container_number_2') if event_record else None
        is_twin = bool(event_record.get('is_twin')) if event_record else False
        if is_twin and (not cntr1 or not cntr2 or cntr1 == cntr2):
            raise ValueError('Twin event is incomplete; both distinct container numbers are required')

        for number in [cntr1] + ([cntr2] if is_twin else []):
            if not verified_cloud_identity(event_record, obs, number, self.settings.ocr_min_confidence):
                raise ValueError('Cloud identity lacks a complete observed checksum-valid OCR read')

        containers_list = []
        if cntr1:
            c1_images = []
            primary_snap_id = event_record.get('primary_snapshot_id') if event_record else None
            rear_snap_id = event_record.get('rear_snapshot_id') if event_record else None
            if event_record and event_record.get('snapshots') and self.snapshots:
                cam_snapshots = {}
                for snap in event_record['snapshots']:
                    cam_id = snap.get('camera_id')
                    if not cam_id:
                        continue
                    # Strictly exclude rear-slot snapshots from Container 1
                    if is_twin and (
                        (snap.get('image_path') or '').endswith('_rear.jpg') or
                        (rear_snap_id and snap.get('id') == rear_snap_id)
                    ):
                        continue
                    cam_snapshots.setdefault(cam_id, []).append(snap)

                best_by_camera = {}
                valid_identity_ids = {o['detection_id'] for o in event_record.get('ocr_results', [])
                                      if o.get('normalized_text') == cntr1}
                for cam_id, snaps in cam_snapshots.items():
                    best_snap = None
                    best_data = None
                    best_score = -1.0
                    for snap in snaps:
                        try:
                            snap_p = self.snapshots.path(snap['image_path'])
                            if not snap_p.is_file():
                                continue
                            data = snap_p.read_bytes()
                            decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if decoded is None or decoded.size == 0 or decoded.shape[0] < 240 or decoded.shape[1] < 320:
                                continue
                            quality = self.snapshots.compute_quality(decoded)
                            if not self._proof_is_usable(decoded, quality):
                                continue
                            if snapshot_has_identity(snap, event_record, cntr1, 0.0):
                                score += 2000.0
                            elif primary_snap_id and snap.get('id') == primary_snap_id:
                                score += 1000.0
                            if score > best_score:
                                best_score = score
                                best_snap = snap
                                best_data = data
                        except Exception:
                            pass
                    if best_snap and best_data:
                        best_by_camera[cam_id] = (best_snap, best_data)

                def _cam_sort_key(c_id):
                    c = c_id.lower()
                    if any(k in c for k in ('1', 'front', 'top')):
                        return (0, c)
                    if any(k in c for k in ('2', 'left', 'side')):
                        return (1, c)
                    if any(k in c for k in ('3', 'rear', 'back')):
                        return (2, c)
                    return (3, c)

                cam_cfgs = getattr(self.settings, 'cameras', lambda: {})()
                for cam_id in sorted(best_by_camera.keys(), key=_cam_sort_key):
                    snap, data = best_by_camera[cam_id]
                    c1_images.append({
                        'ImageType': resolve_camera_image_type(cam_id, cam_cfgs),
                        'CameraId': cam_id,
                        'CapturedAt': _iso_utc(snap.get('timestamp') or (obs.detection.frame_timestamp if obs else None)),
                        'Image': f"data:image/jpeg;base64,{base64.b64encode(proof_jpeg(data, snap, event_record)).decode('utf-8')}"
                    })

            if not c1_images and img_fallback is not None and not event_record:
                try:
                    if not isinstance(img_fallback, bytes):
                        ok, buf = cv2.imencode('.jpg', img_fallback, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                        b = buf.tobytes() if ok else None
                    else:
                        b = img_fallback
                    if b:
                        c1_images.append({
                            'ImageType': 'FRONT',
                            'CameraId': obs.detection.camera_id if obs else 'CAM-01',
                            'CapturedAt': _iso_utc(obs.detection.frame_timestamp if obs else None),
                            'Image': f"data:image/jpeg;base64,{base64.b64encode(b).decode('utf-8')}"
                        })
                except Exception:
                    pass

            if event_record and len(c1_images) < 1:
                raise ValueError('No verified non-blank container proof image is available for front container')

            iso_s1 = str((event_record.get('iso_code_1') if event_record else None) or (getattr(obs.ocr, 'size_type', None) if obs and obs.ocr else None) or '')
            if is_twin and not iso_s1:
                iso_s1 = '22G1'
            cs1 = str((event_record.get('container_size') if event_record else None) or '')
            if not cs1 and iso_s1:
                try:
                    from app.ocr.validator import extract_iso_size_code as _esc
                    _, cs1 = _esc(iso_s1)
                    cs1 = cs1 or ''
                except Exception:
                    pass
            size_1 = resolve_container_size(iso_s1, cs1, is_twin=is_twin)
            conf_1 = identity_confidence(event_record, obs, cntr1)
            containers_list.append({
                'ContainerNumber': cntr1,
                'DetectionConfidence': round(detection_confidence(event_record, obs, cntr1), 2),
                'ContainerNumberConfidence': round(conf_1, 2),
                'Size': size_1,
                'ContainerSize': cs1 or f"{size_1} FT",
                'SizeCode': iso_s1 or ('22G1' if size_1 == '20' else '42G1'),
                'Position': 'FRONT' if is_twin else 'SINGLE',
                'Images': c1_images
            })

        if is_twin and cntr2 and cntr2 != cntr1:
            c2_images = []
            if event_record and event_record.get('snapshots') and self.snapshots:
                primary_snap_id = event_record.get('primary_snapshot_id')
                rear_snap_id = event_record.get('rear_snapshot_id')
                cam_c2_snapshots = {}
                for snap in event_record['snapshots']:
                    # Strictly exclude front primary snapshot from rear container
                    if primary_snap_id and snap.get('id') == primary_snap_id:
                        continue
                    is_rear_snap = (
                        (rear_snap_id and snap.get('id') == rear_snap_id) or
                        (snap.get('image_path') or '').endswith('_rear.jpg') or
                        any(k in (snap.get('camera_id') or '').lower() for k in ('rear', '-3'))
                    )
                    if is_rear_snap:
                        cam_id = snap.get('camera_id') or 'CAM-02'
                        cam_c2_snapshots.setdefault(cam_id, []).append(snap)

                if not cam_c2_snapshots:
                    # Fallback to any snapshot distinct from primary_snap_id
                    for snap in event_record['snapshots']:
                        if not primary_snap_id or snap.get('id') != primary_snap_id:
                            cam_id = snap.get('camera_id') or 'CAM-02'
                            cam_c2_snapshots.setdefault(cam_id, []).append(snap)

                best_c2_by_camera = {}
                valid_identity_ids_2 = {o['detection_id'] for o in event_record.get('ocr_results', [])
                                        if o.get('normalized_text') == cntr2}
                for cam_id, snaps in cam_c2_snapshots.items():
                    best_snap = None
                    best_data = None
                    best_score = -1.0
                    for snap in snaps:
                        try:
                            snap_p = self.snapshots.path(snap['image_path'])
                            if not snap_p.is_file():
                                continue
                            data = snap_p.read_bytes()
                            decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if decoded is None or min(decoded.shape[:2]) < 240:
                                continue
                            quality = self.snapshots.compute_quality(decoded)
                            if not self._proof_is_usable(decoded, quality):
                                continue
                            score = float(quality)
                            if snapshot_has_identity(snap, event_record, cntr2, 0.0):
                                score += 2000.0
                            elif rear_snap_id and snap.get('id') == rear_snap_id:
                                score += 1000.0
                            if (snap.get('image_path') or '').endswith('_rear.jpg'):
                                score += 500.0
                            if valid_identity_ids_2:
                                snap_ts = snap.get('timestamp')
                                observed = any(
                                    d.get('camera_id') == snap.get('camera_id') and
                                    (d.get('id') in valid_identity_ids_2 or _within_capture_window(d.get('timestamp'), snap_ts))
                                    for d in event_record.get('detections', [])
                                )
                                if observed:
                                    score += 200.0
                            if score > best_score:
                                best_score = score
                                best_snap = snap
                                best_data = data
                        except Exception:
                            pass
                    if best_snap and best_data:
                        best_c2_by_camera[cam_id] = (best_snap, best_data)

                cam_cfgs = getattr(self.settings, 'cameras', lambda: {})()
                for cam_id in sorted(best_c2_by_camera.keys()):
                    snap, data = best_c2_by_camera[cam_id]
                    c2_images.append({
                        'ImageType': resolve_camera_image_type(cam_id, cam_cfgs),
                        'CameraId': cam_id,
                        'CapturedAt': _iso_utc(snap.get('timestamp') or (obs.detection.frame_timestamp if obs else None)),
                        'Image': f"data:image/jpeg;base64,{base64.b64encode(proof_jpeg(data, snap, event_record)).decode('utf-8')}"
                    })

            if not c2_images:
                raise ValueError('Rear slot proof image is missing; cloud publication deferred')
            iso_s2 = str((event_record.get('iso_code_2') if event_record else None) or '')
            if is_twin and not iso_s2:
                iso_s2 = '22G1'
            cs2 = str((event_record.get('container_size_2') if event_record else None) or '')
            if not cs2 and iso_s2:
                try:
                    from app.ocr.validator import extract_iso_size_code as _esc
                    _, cs2 = _esc(iso_s2)
                    cs2 = cs2 or ''
                except Exception:
                    pass
            size_2 = resolve_container_size(iso_s2, cs2, is_twin=is_twin)
            conf_2 = identity_confidence(event_record, obs, cntr2)
            containers_list.append({
                'ContainerNumber': cntr2,
                'DetectionConfidence': round(detection_confidence(event_record, obs, cntr2), 2),
                'ContainerNumberConfidence': round(conf_2, 2),
                'Size': size_2,
                'ContainerSize': cs2 or f"{size_2} FT",
                'SizeCode': iso_s2 or ('22G1' if size_2 == '20' else '42G1'),
                'Position': 'REAR',
                'Images': c2_images
            })

        if not containers_list and obs and obs.ocr and obs.ocr.normalized_text:
            containers_list.append({
                'ContainerNumber': obs.ocr.normalized_text,
                'DetectionConfidence': round(float(getattr(obs.detection, 'confidence', 0.0)), 2),
                'ContainerNumberConfidence': round(float(obs.ocr.confidence), 2),
                'Size': '20',
                'Position': 'FRONT',
                'Images': [{
                    'ImageType': 'FRONT',
                    'CameraId': obs.detection.camera_id if obs else 'CAM-01',
                    'CapturedAt': _iso_utc(obs.detection.frame_timestamp if obs else None),
                    'Image': FALLBACK_IMAGE_DATA_URI
                }]
            })

        if obs and hasattr(obs, 'detection'):
            dt_captured = obs.detection.frame_timestamp
        elif event_record and event_record.get('timestamp'):
            try:
                ts_str = event_record['timestamp'].rstrip('Z')
                dt_captured = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
            except Exception:
                dt_captured = datetime.now(timezone.utc)
        else:
            dt_captured = datetime.now(timezone.utc)

        visit_id = f"VISIT-{dt_captured.strftime('%Y%m%d')}-{event_id[:8].upper()}"
        direction = (
            obs.direction.upper() if (obs and hasattr(obs, 'direction') and obs.direction)
            else ((event_record.get('event_type') or 'ENTRY').upper() if event_record else 'ENTRY')
        )
        event_type = 'GATE_OUT' if direction in ('EXIT', 'OUT', 'GATE_OUT') else 'GATE_IN'

        return {
            'id': event_id,
            'VisitId': visit_id,
            'EventType': event_type,
            'DeviceId': self.device_uuid,
            'CapturedAt': _iso_utc(dt_captured),
            'containerNumber': cntr1,
            'Truck': {'TruckNumber': 'NA'},
            'Driver': None,
            'Containers': containers_list
        }

    def dispatch_event(self, event_id: str) -> dict:
        """
        Manually dispatches a single reviewed gate event to the Prosper cloud site.
        Validates evidence images for both containers (in twin events) before sending.
        """
        if not self.enabled:
            return {'success': False, 'status': 'DISABLED', 'error': 'Prosper cloud sync is not enabled'}
        if not self.repository:
            return {'success': False, 'status': 'ERROR', 'error': 'Repository unavailable'}

        event_record = self.repository.event(event_id)
        if not event_record:
            return {'success': False, 'status': 'NOT_FOUND', 'error': f'Event {event_id} not found'}
        if not self.is_dispatchable_event(event_record, allow_synced=True):
            return {
                'success': False,
                'status': 'REJECTED_UNREADABLE_OR_LOW_CONFIDENCE',
                'error': f'Event {event_id} is unreadable or OCR confidence is below 50%'
            }

        try:
            payload = self.build_event_payload(event_id)
        except Exception as exc:
            return {'success': False, 'status': 'PAYLOAD_ERROR', 'error': str(exc)}

        try:
            res = self.session.post_gate_event(payload)
            if res in (True, 'CONFLICT') or isinstance(res, dict):
                cloud_ev_id = None
                if isinstance(res, dict):
                    cloud_ev_id = res.get('eventId') or res.get('visitDatabaseId')
                self.repository.mark_cloud_synced(event_id, cloud_event_id=cloud_ev_id)
                self._is_online = True
                self._last_sync_time = datetime.now(timezone.utc).isoformat()
                self._last_error = None
                log.info('prosper_manual_dispatch_success', extra={'event_id': event_id, 'cloud_event_id': cloud_ev_id})
                return {
                    'success': True,
                    'status': 'SYNCED',
                    'cloud_event_id': cloud_ev_id,
                    'visit_id': payload.get('VisitId'),
                    'message': 'Event successfully dispatched to Prosper site'
                }
            else:
                self._is_online = False
                err_msg = getattr(self.session, '_auth_error', None) or self._last_error or f"Site returned status: {res}"
                log.warning('prosper_manual_dispatch_rejected', extra={'event_id': event_id, 'status': str(res)})
                return {
                    'success': False,
                    'status': str(res),
                    'error': err_msg
                }
        except Exception as exc:
            self._is_online = False
            self._last_error = str(exc)
            log.exception('prosper_manual_dispatch_exception', extra={'event_id': event_id})
            return {
                'success': False,
                'status': 'EXCEPTION',
                'error': str(exc)
            }

    def _run_loop(self):
        last_sweep = time.monotonic()
        while not self._stop.is_set():
            try:
                item = self.queue.get(timeout=1.0)
            except Empty:
                now = time.monotonic()
                if self.repository and (now - last_sweep) >= 15.0:
                    last_sweep = now
                    self._sweep_offline_backlog()
                continue

            event_id = item['event_id']
            obs = item['obs']
            img = item['image_bytes']
            pre_payload = item.get('payload')

            # A cloud gate event is effectively immutable after acknowledgement.
            # Allow late OCR from the other calibrated camera to update the
            # local event/snapshots before constructing the payload.
            not_before = float(item.get('not_before') or 0.0)
            remaining = not_before - time.monotonic()
            if remaining > 0:
                if self._stop.wait(remaining):
                    self.queue.task_done()
                    break

            success = False
            try:
                # Offline backlog is populated directly from SQLite, so apply
                # the same persistent identity guard here as enqueue().
                if self.repository:
                    current = self.repository.event(event_id)
                    if current and not self.is_dispatchable_event(current):
                        self.queue.task_done()
                        continue
                    canonical = self.repository.canonical_event_id(event_id)
                    if canonical and canonical != event_id:
                        if hasattr(self.repository, 'mark_duplicate_event'):
                            self.repository.mark_duplicate_event(event_id, canonical)
                        else:
                            self.repository.mark_cloud_synced(event_id)
                        log.info('prosper_duplicate_backlog_suppressed', extra={
                            'event_id': event_id, 'canonical_event_id': canonical})
                        self.queue.task_done()
                        continue
                payload = self.build_event_payload(event_id, obs=obs, img_fallback=img, pre_payload=pre_payload)

                success = self.session.post_gate_event(payload)
                if success == 'REJECTED' and self.runtime:
                    try:
                        ev_type = payload.get('EventType', 'GATE_OUT')
                        c_num = payload.get('containerNumber') or (obs.ocr.normalized_text if (obs and obs.ocr) else None)
                        self.runtime.emit_alert(
                            alert_type='CLOUD_SYNC_REJECTED',
                            title=f'Cloud Sync Rejected ({ev_type})',
                            message=f'Cloud rejected {ev_type} for {c_num or event_id[:8]} (Visit {payload.get("VisitId", "")})',
                            level='ERROR',
                            container_number=c_num,
                            event_id=event_id,
                            buzzer=False
                        )
                    except Exception:
                        log.exception('prosper_alert_failed: %s', event_id)
                elif success == 'OFFLINE':
                    log.info('prosper_event_buffered_offline: Event %s buffered locally; will auto-sync when cloud service is available.', event_id)

            except Exception as exc:
                log.warning('prosper_dispatch_exception', extra={'event_id': event_id, 'error': str(exc)})
                success = 'OFFLINE'

            if success in (True, 'CONFLICT') or isinstance(success, dict):
                self._is_online = True
                self._last_sync_time = datetime.now(timezone.utc).isoformat()
                self._last_error = None
                if self.repository:
                    try:
                        cloud_ev_id = None
                        if isinstance(success, dict):
                            cloud_ev_id = success.get('eventId') or success.get('visitDatabaseId')
                        self.repository.mark_cloud_synced(event_id, cloud_event_id=cloud_ev_id)
                    except Exception:
                        log.exception('prosper_acknowledgement_persistence_failed: %s', event_id)
                if self.metrics:
                    self.metrics.increment('prosper_events_synced')
                    self.metrics.latency('prosper_sync', time.monotonic() - item['enqueued_at'])
                self.queue.task_done()
            else:
                self._is_online = False
                if success == 'OFFLINE':
                    self._last_error = getattr(self.session, '_auth_error', None) or 'Cloud endpoint unreachable or offline'
                elif success == 'REJECTED':
                    self._last_error = 'Cloud rejected payload (validation or authentication error)'
                elif 'exc' in locals() and exc:
                    self._last_error = str(exc)
                item['retries'] += 1
                if item['retries'] < self.settings.prosper_max_retries:
                    backoff = min(2.0 ** item['retries'], 30.0)
                    self._stop.wait(backoff)
                    try:
                        self.queue.put_nowait(item)
                    except Full:
                        log.error('prosper_retry_dropped_queue_full', extra={'event_id': event_id})
                else:
                    log.info('prosper_event_sync_buffered_in_db: Event %s buffered offline in database; periodic sweep will auto-sync.', event_id)
                    if self.metrics:
                        self.metrics.increment('prosper_sync_buffered')
                self.queue.task_done()
