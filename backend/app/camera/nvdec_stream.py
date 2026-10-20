from __future__ import annotations
import logging
import threading
import time
from typing import Optional, Tuple
import numpy as np

log = logging.getLogger('gate.camera.nvdec')

_gst_initialized = False
_gst_init_lock = threading.Lock()


def _ensure_gst_init():
    global _gst_initialized
    with _gst_init_lock:
        if not _gst_initialized:
            import gi
            gi.require_version('Gst', '1.0')
            from gi.repository import Gst
            Gst.init(None)
            _gst_initialized = True


class JetsonNVDECStream:
    """NVIDIA Jetson Hardware NVDEC H.264/H.265 RTSP Video Stream via GStreamer."""
    def __init__(self, source: str, settings):
        _ensure_gst_init()
        from gi.repository import Gst
        self.Gst = Gst

        self.source = source
        self.settings = settings
        self.fps = 25.0
        self.frame_count = 0
        self._last_frame: Optional[np.ndarray] = None
        self._last_frame_time = 0.0
        self._lock = threading.Lock()

        pipeline_str = (
            f'rtspsrc location="{source}" tls-validation-flags=0 protocols=tcp latency=100 ! '
            'rtph264depay ! h264parse ! nvv4l2decoder enable-max-performance=1 ! '
            'nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! '
            'appsink name=sink emit-signals=True max-buffers=1 drop=True sync=False'
        )
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.appsink = self.pipeline.get_by_name('sink')
        self.pipeline.set_state(Gst.State.PLAYING)

        # Verify initial frame delivery
        timeout_sec = max(6.0, getattr(settings, 'open_timeout_ms', 5000) / 1000.0)
        t0 = time.time()
        ok = False
        while time.time() - t0 < timeout_sec:
            sample = self.appsink.emit('try-pull-sample', 200 * Gst.MSECOND)
            if sample:
                buf = sample.get_buffer()
                caps = sample.get_caps()
                s = caps.get_structure(0)
                w, h = s.get_value('width'), s.get_value('height')
                data = buf.extract_dup(0, buf.get_size())
                self._last_frame = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3)).copy()
                self._last_frame_time = time.time()
                ok = True
                break
            time.sleep(0.02)

        if not ok:
            self.close()
            raise ConnectionError('NVDEC GStreamer pipeline timed out opening RTSP source')
        log.info('nvdec_stream_opened', extra={'source': source.split('@')[-1] if '@' in source else source})

    def is_alive(self) -> bool:
        if self.pipeline is None:
            return False
        bus = self.pipeline.get_bus()
        msg = bus.pop_filtered(self.Gst.MessageType.ERROR | self.Gst.MessageType.EOS)
        if msg:
            if msg.type == self.Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                log.warning('nvdec_bus_error', extra={'error': str(err), 'debug': str(debug)})
            return False
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self.pipeline is None or not self.is_alive():
                return False, None
            sample = self.appsink.emit('try-pull-sample', 500 * self.Gst.MSECOND)
            if sample:
                buf = sample.get_buffer()
                caps = sample.get_caps()
                s = caps.get_structure(0)
                w, h = s.get_value('width'), s.get_value('height')
                data = buf.extract_dup(0, buf.get_size())
                self._last_frame = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3)).copy()
                self._last_frame_time = time.time()
                return True, self._last_frame
            if self._last_frame is not None and (time.time() - self._last_frame_time < 2.0):
                return True, self._last_frame
            return False, None

    def grab(self) -> bool:
        if not self.is_alive():
            return False
        time.sleep(0.01)
        return True

    def retrieve(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self._last_frame is not None and (time.time() - self._last_frame_time < 2.0):
                return True, self._last_frame
            return self.read()

    def rewind(self) -> bool:
        return False

    def close(self):
        with self._lock:
            if hasattr(self, 'pipeline') and self.pipeline:
                try:
                    self.pipeline.set_state(self.Gst.State.NULL)
                except Exception:
                    pass
                self.pipeline = None
                self.appsink = None
                self._last_frame = None
