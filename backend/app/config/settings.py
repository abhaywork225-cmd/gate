from __future__ import annotations
from pathlib import Path
from typing import Literal
import os

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Direction = Literal['ENTRY', 'EXIT', 'UNKNOWN']


class LaneRegion(BaseModel):
    id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,32}$')
    polygon: list[tuple[float, float]] = Field(min_length=3, max_length=12)

    @model_validator(mode='after')
    def validate_polygon(self):
        if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in self.polygon):
            raise ValueError('Lane vertices must be inside the frame')
        import cv2
        import numpy as np
        points = np.array(self.polygon, dtype=np.float32)
        if not cv2.isContourConvex(points) or cv2.contourArea(points) < .001:
            raise ValueError('Draw a convex lane polygon with nonzero area')
        return self


class CameraConfig(BaseModel):
    id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,64}$')
    name: str = Field(max_length=100)
    gate_id: str = Field(max_length=64, min_length=1)
    source_env: str = Field(pattern=r'^[A-Z][A-Z0-9_]+$', max_length=100)
    source_type: Literal['rtsp', 'file'] = 'rtsp'
    role: Literal['UNASSIGNED', 'FRONT_TOP', 'LEFT', 'RIGHT', 'REAR'] = 'UNASSIGNED'
    enabled: bool = True
    direction: Direction = 'UNKNOWN'
    capture_fps: float = Field(default=15, gt=0, le=120)
    inference_fps: float | None = Field(default=None, gt=0, le=60)
    frame_skip: int = Field(default=0, ge=0, le=120)
    loop_file: bool = False
    # Normalized region inside a detected box; whole box by default.
    ocr_roi: tuple[float, float, float, float] = (0, 0, 1, 1)
    # Optional crossing of normalized x or y, with a deadband.
    line_axis: Literal['x', 'y'] | None = None
    line_position: float = Field(default=0.5, gt=0, lt=1)
    line_deadband: float = Field(default=0.03, ge=0, lt=0.5)
    positive_crossing: Literal['ENTRY', 'EXIT'] = 'ENTRY'
    occupancy_roi: tuple[float, float, float, float] = (0, 0, 1, 1)
    lanes: list[LaneRegion] = Field(default_factory=list, max_length=2)
    gate_exit_axis: Literal['x', 'y'] | None = None
    gate_exit_position: float = Field(default=.95, gt=0, lt=1)
    gate_exit_positive: bool = True

    @model_validator(mode='after')
    def validate_geometry(self):
        x1, y1, x2, y2 = self.ocr_roi
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError('ocr_roi must be a normalized nonempty rectangle')
        x1, y1, x2, y2 = self.occupancy_roi
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError('occupancy_roi must be a normalized nonempty rectangle')
        return self


class CameraSourceConfig(BaseModel):
    id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,64}$')
    name: str = Field(max_length=100, min_length=1)
    source_env: str = Field(pattern=r'^[A-Z][A-Z0-9_]+$', max_length=100)
    enabled: bool = True


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')
    deployment_mode: Literal['development', 'edge', 'test'] = 'development'
    database_url: SecretStr = SecretStr('sqlite+pysqlite:///data/gate.db')
    cameras_file: Path = Path('config/cameras.yaml')
    pipeline_enabled: bool = True
    vehicle_visit_tracking: bool = True
    minimum_number_proof_cameras: int = Field(default=2, ge=1, le=16)
    shared_visit_folders: bool = True
    container_roi_trigger: bool = True
    reversal_grace_seconds: float = Field(default=30, ge=0, le=300)
    cross_camera_arrival_window_seconds: float = Field(default=30, ge=1, le=120)
    model_backend: Literal['pytorch', 'onnx', 'tensorrt'] = 'pytorch'
    model_path: Path = Path('models/best.pt')
    vehicle_presence_min_confidence: float = Field(default=0.5, ge=0.5, le=1)
    passage_vehicle_model_path: Path = Path('models/yolov8x.pt')
    model_device: str = 'cpu'
    model_image_size: int = Field(default=640, ge=32)
    class_aliases: dict[str, str] = {}
    target_classes: list[str] = ['container', 'container_number', 'iso_code']
    ocr_classes: list[str] = ['container', 'container_number', 'iso_code']

    # PaddleOCR runs in its own CUDA worker so it does not contend with the
    # PyTorch YOLO process. Qwen remains a last-resort reader only.
    ocr_engine: Literal['paddleocr', 'olmocr', 'easyocr', 'qwen2_vl', 'qwen2vl'] = 'paddleocr'
    ocr_fallback_engine: Literal['olmocr', 'easyocr', 'none'] = 'olmocr'
    paddle_ocr_detection_model: str = 'PP-OCRv6_medium_det'
    paddle_ocr_recognition_model: str = 'PP-OCRv6_medium_rec'
    ocr_model_name: str = 'Qwen/Qwen2-VL-2B-Instruct'
    ocr_gpu: bool = True
    ocr_download_enabled: bool = False
    ocr_model_directory: Path = Path('models/easyocr')
    inference_fps: float = Field(default=3, gt=0, le=60)
    confidence_threshold: float = Field(default=0.35, ge=0, le=1)
    ocr_min_confidence: float = Field(default=0.6, ge=0, le=1)
    min_track_hits: int = Field(default=3, ge=1)
    # One valid check-digit read is sufficient once a truck is stationary;
    # additional samples remain available for twin confirmation and review.
    ocr_confirmations: int = Field(default=2, ge=1)
    ocr_max_attempts: int = Field(default=4, ge=1)
    ocr_interval_seconds: float = Field(default=.5, gt=0)
    ocr_queue_size: int = Field(default=8, ge=1, le=100)
    # The spool is a temporary OCR work area. Definitive event evidence is
    # stored under snapshots, so this must stay bounded on an edge device.
    ocr_spool_max_mb: int = Field(default=2048, ge=256)
    gate_clear_seconds: float = Field(default=7, ge=1, le=120)
    # Keep a confirmed event local briefly so OCR from a second calibrated
    # camera can join the same visit before the immutable cloud POST happens.
    # This is especially important for twin containers where the rear view
    # often arrives after the front view.
    # Give the other calibrated gate cameras time to contribute a rear/twin
    # identity before the cloud transaction becomes immutable.  Operators can
    # tune this per site through CLOUD_DISPATCH_GRACE_SECONDS.
    cloud_dispatch_grace_seconds: float = Field(default=10, ge=0, le=60)
    # Minimum sharpness/exposure score for a frame sent as cloud proof.
    # Frames below this are retained locally for diagnostics but never sent.
    cloud_min_proof_quality: float = Field(default=2.5, ge=0, le=1000)
    # Maximum timestamp distance used when joining two calibrated RTSP views
    # into one candidate truck visit. Keep this bounded for queued traffic.
    # Cameras on one gate are timestamp-joined only briefly.  A long window
    # causes the next queued truck to be treated as another view of this one.
    cross_camera_view_window_seconds: float = Field(default=8, ge=1, le=60)
    # Uploaded videos usually have independent clocks and may show different
    # trucks. Do not fuse their cameras unless an operator explicitly enables
    # synchronized-file mode.
    allow_file_cross_camera_fusion: bool = False
    stationary_drift_threshold: float = Field(default=.02, gt=0, lt=.05)
    stationary_match_iou: float = Field(default=.45, ge=.45, lt=1)
    stationary_seconds: float = Field(default=.8, gt=0)
    gate_observation_gap_seconds: float = Field(default=2, gt=0)
    dwell_container_classes: list[str] = ['container', 'container_number', 'iso_code']
    dedup_seconds: float = Field(default=30, gt=0)
    # A tracker can be re-created when a queued truck remains in view.  Keep
    # one persisted event for the same confirmed container/lane/direction for
    # this long, while leaving normal (unidentified) passage dedup unchanged.
    # This is a recovery window for tracker churn, not a visit identity.  The
    # active lane/visit ID is the primary key; this fallback must stay short.
    same_container_event_window_seconds: float = Field(default=600, ge=60, le=7200)
    track_ttl_seconds: float = Field(default=5, gt=0)
    snapshot_directory: Path = Path('snapshots')
    upload_directory: Path = Path('uploads')
    log_directory: Path = Path('logs')
    max_upload_mb: int = Field(default=2048, ge=1, le=20480)
    camera_discovery_enabled: bool = True
    camera_discovery_interval_seconds: int = Field(default=60, ge=15, le=3600)
    camera_onvif_username: str = ''
    camera_onvif_password: SecretStr = SecretStr('')
    cors_origins: list[str] = ['http://localhost:4200', 'http://127.0.0.1:4200']
    open_timeout_ms: int = Field(default=15000, ge=100, le=60000)
    read_timeout_ms: int = Field(default=8000, ge=100, le=60000)
    reconnect_initial_seconds: float = Field(default=1, gt=0)
    reconnect_max_seconds: float = Field(default=30, gt=0)
    max_frame_width: int = Field(default=1920, ge=320, le=4096)
    prosper_enabled: bool = False
    prosper_auto_dispatch: bool = False
    prosper_api_base_url: str = 'https://syapi.prosperassettracking.com'
    prosper_auth_url: str = 'https://cfsapi.prosperassettracking.com/api/auth/login'
    prosper_site_id: str = ''
    prosper_client_id: str = ''
    prosper_api_key: SecretStr | None = None
    prosper_bearer_token: SecretStr | None = None
    prosper_site_code: str | None = None
    prosper_username: str | None = None
    prosper_email: str | None = None
    prosper_password: SecretStr | None = None
    prosper_device_uuid: str | None = None
    prosper_max_retries: int = Field(default=5, ge=1, le=20)
    prosper_queue_size: int = Field(default=200, ge=1, le=1000)

    @model_validator(mode='after')
    def validate_settings(self):
        if self.ocr_max_attempts < self.ocr_confirmations:
            raise ValueError('ocr_max_attempts must cover ocr_confirmations')
        from sqlalchemy.engine import make_url
        database = make_url(self.database_url.get_secret_value())
        if database.drivername not in ('sqlite', 'sqlite+pysqlite', 'mysql+pymysql'):
            raise ValueError('Use a local SQLite file or an explicitly configured MySQL database')
        if database.drivername.startswith('sqlite') and self.deployment_mode != 'test':
            if not database.database or database.database == ':memory:' or database.query:
                raise ValueError('Use a persistent SQLite file without URI query options')
        return self

    def cameras(self) -> list[CameraConfig]:
        data = yaml.safe_load(self.cameras_file.read_text(encoding='utf-8')) or {}
        cameras = [CameraConfig.model_validate(c) for c in data.get('cameras', [])]
        if len(cameras) > 8 or len({c.id for c in cameras}) != len(cameras):
            raise ValueError('Configure at most eight cameras with unique IDs')
        return cameras

    def camera_sources(self, cameras: list[CameraConfig] | None = None) -> list[CameraSourceConfig]:
        data = yaml.safe_load(self.cameras_file.read_text(encoding='utf-8')) or {}
        configured = data.get('camera_sources')
        if configured is None:
            configured = [{'id': c.id, 'name': c.name, 'source_env': c.source_env, 'enabled': c.enabled}
                          for c in (cameras or self.cameras())]
        sources = [CameraSourceConfig.model_validate(item) for item in configured]
        if len(sources) > 32 or len({source.id for source in sources}) != len(sources):
            raise ValueError('Configure at most 32 camera sources with unique IDs')
        return sources


def resolve_source(camera: CameraConfig | CameraSourceConfig) -> str:
    # Explicit lookup avoids requiring every possible camera key in Settings.
    if camera.source_env in os.environ:
        return os.environ[camera.source_env]
    if 'PYTEST_CURRENT_TEST' in os.environ:
        return ''
    return dotenv_values('.env').get(camera.source_env) or ''
