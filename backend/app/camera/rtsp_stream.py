from __future__ import annotations
"""OpenCV boundary. Native libraries are imported only when capture is requested."""
import os
from pathlib import Path


class OpenCVStream:
    def __init__(self, source, source_type, settings):
        # Suppress native decoder logs that can include credential-bearing URLs.
        os.environ.setdefault('OPENCV_FFMPEG_LOGLEVEL', '-8')
        os.environ.setdefault('OPENCV_LOG_LEVEL', 'SILENT')
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
            'rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000|probesize;262144|analyzeduration;1000000'
        )
        import cv2
        if hasattr(cv2, 'setLogLevel'):
            cv2.setLogLevel(0)
        self.cv2 = cv2
        if source_type == 'rtsp':
            if not source.lower().startswith(('rtsp://', 'rtsps://')):
                raise ValueError('Configured camera source is not RTSP')
            cap = None
            use_nvdec = os.getenv('JETSON_NVDEC', 'auto').lower() in ('1', 'true', 'auto')
            is_tegra = os.path.isfile('/etc/nv_tegra_release')
            if use_nvdec and is_tegra and hasattr(cv2, 'CAP_GSTREAMER'):
                try:
                    gst = (
                        f'rtspsrc location="{source}" tls-validation-flags=0 protocols=tcp latency=100 ! '
                        'rtph264depay ! h264parse ! nvv4l2decoder ! '
                        'nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1'
                    )
                    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
                    if not cap.isOpened():
                        cap.release()
                        cap = None
                except Exception:
                    cap = None
            if cap is None:
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG, [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, settings.open_timeout_ms,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, settings.read_timeout_ms,
                ])
            self.cap = cap
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            if not Path(source).is_file():
                raise FileNotFoundError('Configured replay file is unavailable')
            self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.cap.release()
            raise ConnectionError('Capture unavailable')
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if 0 < fps < 240 else 25.0
        self.frame_count = max(0, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))) if source_type == 'file' else 0

    def read(self):
        return self.cap.read()

    def grab(self):
        return self.cap.grab()

    def retrieve(self):
        return self.cap.retrieve()

    def rewind(self):
        if getattr(self, 'cap', None) and self.cap.isOpened():
            self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, 0)
            return True
        return False

    def close(self):
        self.cap.release()
