from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


def utcnow() -> datetime:
    # Naive UTC is used consistently in SQL (MySQL DATETIME) and domain objects.
    return datetime.now(timezone.utc).replace(tzinfo=None)


@dataclass(frozen=True)
class Frame:
    camera_id: str
    timestamp: datetime
    sequence: int
    image: Any
    source_id: str = ''


@dataclass(frozen=True)
class Detection:
    camera_id: str
    frame_timestamp: datetime
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class OCRRead:
    raw_text: str
    confidence: float


@dataclass(frozen=True)
class Validation:
    raw_text: str
    normalized_text: str
    confidence: float
    valid_format: bool
    valid_check_digit: bool
    validation_status: str
    size_type: str | None = None


@dataclass(frozen=True)
class Observation:
    origin_key: str
    gate_id: str
    track_id: int
    detection: Detection
    ocr: Validation
    direction: str
    image: Any
    started_at: datetime
    confirmed: bool
    all_detections: list[Detection] | None = None
    slot_position: str = 'FRONT'
    event_id: str | None = None
    is_twin: bool = False

