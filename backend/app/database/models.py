from __future__ import annotations
from datetime import datetime
from typing import Optional
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from app.domain import utcnow


class Base(DeclarativeBase):
    pass


class SchemaVersion(Base):
    __tablename__ = 'gate_schema_version'
    id: Mapped[int] = mapped_column(primary_key=True)
    version: Mapped[int]


class Camera(Base):
    __tablename__ = 'cameras'
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    gate_id: Mapped[str] = mapped_column(String(64))
    source_env: Mapped[str] = mapped_column(String(100))
    enabled: Mapped[bool] = mapped_column(Boolean)
    configuration: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class GateEvent(Base):
    __tablename__ = 'gate_events'
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    gate_id: Mapped[str] = mapped_column(String(64))
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    lane_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    container_number: Mapped[Optional[str]] = mapped_column(String(11))
    event_type: Mapped[str] = mapped_column(String(10))
    status: Mapped[str] = mapped_column(String(20))
    is_twin: Mapped[bool] = mapped_column(Boolean, default=False)
    container_number_2: Mapped[Optional[str]] = mapped_column(String(11), nullable=True)
    iso_code_1: Mapped[Optional[str]] = mapped_column(String(4), nullable=True)
    iso_code_2: Mapped[Optional[str]] = mapped_column(String(4), nullable=True)
    cloud_synced: Mapped[bool] = mapped_column(Boolean, default=False)
    cloud_synced_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    cloud_event_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    __table_args__ = (Index('ix_gate_identity_time', 'gate_id', 'container_number', 'event_type', 'timestamp'),)


class DetectionRecord(Base):
    __tablename__ = 'detections'
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    event_id: Mapped[str] = mapped_column(ForeignKey('gate_events.id'), index=True)
    camera_id: Mapped[str] = mapped_column(ForeignKey('cameras.id'), index=True)
    origin_key: Mapped[str] = mapped_column(String(160), unique=True)
    track_id: Mapped[int] = mapped_column(Integer)
    class_name: Mapped[str] = mapped_column(String(100))
    confidence: Mapped[float] = mapped_column(Float)
    bbox: Mapped[list] = mapped_column(JSON)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)


class OCRResult(Base):
    __tablename__ = 'ocr_results'
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    detection_id: Mapped[str] = mapped_column(ForeignKey('detections.id'), unique=True)
    raw_text: Mapped[str] = mapped_column(Text)
    normalized_text: Mapped[str] = mapped_column(String(256))
    confidence: Mapped[float] = mapped_column(Float)
    valid_format: Mapped[bool] = mapped_column(Boolean)
    valid_check_digit: Mapped[bool] = mapped_column(Boolean)
    validation_status: Mapped[str] = mapped_column(String(30))
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Snapshot(Base):
    __tablename__ = 'snapshots'
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    event_id: Mapped[str] = mapped_column(ForeignKey('gate_events.id'), index=True)
    camera_id: Mapped[str] = mapped_column(ForeignKey('cameras.id'))
    image_path: Mapped[str] = mapped_column(String(256), unique=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime)


class SystemLog(Base):
    __tablename__ = 'system_logs'
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    event_id: Mapped[Optional[str]] = mapped_column(ForeignKey('gate_events.id'))
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    level: Mapped[str] = mapped_column(String(10))
    event_name: Mapped[str] = mapped_column(String(100))
    context: Mapped[dict] = mapped_column(JSON)
