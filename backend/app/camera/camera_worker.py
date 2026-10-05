from __future__ import annotations
from collections import deque
import logging
import threading
import time
from uuid import uuid4

from app.domain import Frame, utcnow

log = logging.getLogger('gate.camera')


class CameraWorker:
    def __init__(self, config, source, settings, stream_factory):
        self.config, self._source = config, source
        self.settings, self.stream_factory = settings, stream_factory
        self._stop = threading.Event()
        self._paused = threading.Event()
        self.source_id = str(uuid4())
        self.video_position_seconds = 0.
        self.video_duration_seconds = None
        self._lock = threading.Lock()
        self._thread = None
        self._test_lock = threading.Lock()
        self._pending = None
        self._latest = None
        self._status = 'OFFLINE'
        self._reason = 'not_started'
        self._last_frame = None
        self._captured = deque(maxlen=1000)
        self._dropped = 0
        self._reconnects = 0
        self._sequence = 0
        self._cached_jpeg = None
        self._cached_seq = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        if not self.config.enabled or not self._source:
            self._set_status('OFFLINE', 'disabled' if not self.config.enabled else 'source_not_configured')
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f'capture-{self.config.id}', daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def prepare_video(self):
        stream = self.stream_factory(self._source, self.config.source_type, self.settings)
        try:
            ok, image = stream.read()
            if not ok or image is None:
                raise ValueError('Video has no readable frame')
            self.video_duration_seconds = getattr(stream, 'frame_count', 0) / stream.fps or None
            self._publish(image)
            self.take()  # Preview is not an inference observation until playback starts.
            self._set_status('OFFLINE', 'video_ready')
        finally:
            stream.close()

    def pause(self):
        if self.config.source_type != 'file':
            raise ValueError('Pause is available only for video files')
        if self._thread and self._thread.is_alive():
            self._paused.set()
            self._set_status('ONLINE', 'video_paused')

    def play(self):
        if self._paused.is_set():
            self._paused.clear()
        else:
            self.start()

    def join(self, timeout=10):
        if self._thread:
            self._thread.join(timeout)
        return not self._thread or not self._thread.is_alive()

    def _set_status(self, status, reason):
        with self._lock:
            changed = self._status != status or self._reason != reason
            self._status, self._reason = status, reason
        if changed:
            log.info('camera_' + status.lower(), extra={'camera_id': self.config.id})

    def _publish(self, image):
        if image.shape[1] > self.settings.max_frame_width:
            import cv2
            width = self.settings.max_frame_width
            image = cv2.resize(image, (width, round(image.shape[0] * width / image.shape[1])))
        timestamp = utcnow()
        with self._lock:
            self._sequence += 1
            if self._pending is not None:
                self._dropped += 1
            self._pending = Frame(self.config.id, timestamp, self._sequence, image, self.source_id)
            self._latest = self._pending
            self._cached_jpeg = None
            self._cached_seq = None
            self._last_frame = timestamp.isoformat() + 'Z'
            self._captured.append(time.monotonic())

    def get_jpeg(self):
        with self._lock:
            frame = self._latest
            if frame is None or frame.image is None:
                return None
            if self._cached_jpeg is not None and self._cached_seq == frame.sequence:
                return self._cached_jpeg
            img = frame.image
            seq = frame.sequence
        import cv2
        if img.shape[1] > 640:
            img = cv2.resize(img, (640, round(img.shape[0] * 640 / img.shape[1])))
        ok, jpeg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            return None
        data = jpeg.tobytes()
        with self._lock:
            if self._latest is not None and self._latest.sequence == seq:
                self._cached_jpeg = data
                self._cached_seq = seq
        return data

    def take(self):
        with self._lock:
            frame, self._pending = self._pending, None
            return frame

    def latest(self):
        with self._lock:
            return self._latest

    def status(self):
        with self._lock:
            now = time.monotonic()
            recent = [t for t in self._captured if now - t < 5]
            return {
                'camera_id': self.config.id, 'status': self._status, 'reason': self._reason,
                'last_frame_at': self._last_frame, 'capture_fps': len(recent) / 5,
                'dropped_frames': self._dropped, 'reconnect_count': self._reconnects,
                'frames_captured': self._sequence,
                'has_frame': self._latest is not None,
                'playback_status': ('PAUSED' if self._paused.is_set() else 'PLAYING' if self._status == 'ONLINE' else
                                    'COMPLETED' if self._reason == 'file_completed' else 'READY' if self._reason == 'video_ready' else 'STOPPED') if self.config.source_type == 'file' else None,
                'video_position_seconds': self.video_position_seconds,
                'video_duration_seconds': self.video_duration_seconds,
            }

    def test(self):
        if not self._test_lock.acquire(blocking=False):
            return {'success': False, 'reason': 'test_already_running'}
        try:
            return self._test_source()
        finally:
            self._test_lock.release()

    def _test_source(self):
        if not self._source:
            return {'success': False, 'reason': 'source_not_configured'}
        stream = None
        try:
            stream = self.stream_factory(self._source, self.config.source_type, self.settings)
            ok, frame = stream.read()
            return {'success': bool(ok and frame is not None), 'reason': 'frame_received' if ok else 'read_failed'}
        except Exception as exc:
            return {'success': False, 'reason': type(exc).__name__}
        finally:
            if stream:
                stream.close()

    def _run(self):
        backoff = self.settings.reconnect_initial_seconds
        try:
            while not self._stop.is_set():
                stream = None
                self._set_status('RECONNECTING', 'connecting')
                try:
                    stream = self.stream_factory(self._source, self.config.source_type, self.settings)
                    next_publish = 0.0
                    video_frame = 0
                    while not self._stop.is_set():
                        if self._paused.is_set():
                            self._stop.wait(.05)
                            continue
                        if self.config.source_type == 'file':
                            started = time.monotonic()
                            ok, image = stream.read()
                            if not ok or image is None:
                                if self.config.loop_file:
                                    if hasattr(stream, 'rewind') and stream.rewind():
                                        video_frame = 0
                                        ok, image = stream.read()
                                    if not ok or image is None:
                                        self._set_status('OFFLINE', 'file_completed')
                                        return
                                else:
                                    self._set_status('OFFLINE', 'file_completed')
                                    return
                            video_frame += 1
                            self.video_position_seconds = video_frame / stream.fps
                            self._publish(image)
                            self._set_status('ONLINE', 'receiving')
                            backoff = self.settings.reconnect_initial_seconds
                            self._stop.wait(max(0.005, 1 / (self.config.capture_fps * 1.5)))
                        else:
                            started = time.monotonic()
                            if started >= next_publish:
                                ok, image = stream.read()
                                if not ok or image is None:
                                    raise ConnectionError('Read failed')
                                self._publish(image)
                                next_publish = started + 1.0 / self.config.capture_fps
                            else:
                                if hasattr(stream, 'grab'):
                                    if not stream.grab():
                                        raise ConnectionError('Grab failed')
                                else:
                                    ok, _ = stream.read()
                                    if not ok:
                                        raise ConnectionError('Read failed')
                            self._set_status('ONLINE', 'receiving')
                            backoff = self.settings.reconnect_initial_seconds
                except Exception as exc:
                    log.warning('camera_capture_failed', extra={'camera_id': self.config.id, 'error_type': type(exc).__name__}, exc_info=True)
                    self._set_status('ERROR', f'capture_failed: {type(exc).__name__}')
                finally:
                    if stream is not None:
                        try:
                            stream.close()
                        except Exception:
                            pass
                if not self._stop.is_set():
                    with self._lock:
                        self._reconnects += 1
                    self._set_status('RECONNECTING', 'retry_backoff')
                    self._stop.wait(backoff)
                    backoff = min(backoff * 2, self.settings.reconnect_max_seconds)
        finally:
            if self._stop.is_set():
                self._set_status('OFFLINE', 'stopped')
