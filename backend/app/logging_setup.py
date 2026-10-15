from __future__ import annotations
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record):
        # Internal event names and allowlisted scalar fields only. Exception strings
        # may contain DSNs/URLs, so only exception TYPE is emitted by callers in JSON.
        data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'event': record.getMessage(),
            **{k: getattr(record, k) for k in ('camera_id', 'gate_id', 'event_id', 'error_type') if hasattr(record, k)},
        }
        if record.exc_info:
            data['error_type'] = data.get('error_type') or (record.exc_info[0].__name__ if record.exc_info[0] else None)
        return json.dumps(data)


class TextErrorFormatter(logging.Formatter):
    def format(self, record):
        timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        details = []
        for k in ('camera_id', 'gate_id', 'event_id', 'error_type'):
            if hasattr(record, k):
                details.append(f"{k}={getattr(record, k)}")
        details_str = f" [{', '.join(details)}]" if details else ""
        msg = f"[{timestamp}] [{record.levelname}] [{record.name}]{details_str} {record.getMessage()}"
        if record.exc_info and record.exc_info[0] is not None:
            msg += "\n" + self.formatException(record.exc_info)
        return msg


class CloudDispatchFormatter(logging.Formatter):
    def format(self, record):
        timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        details = []
        for k in ('event_id', 'visit_id', 'status_code', 'duration_ms', 'cloud_event_id'):
            if hasattr(record, k) and getattr(record, k) is not None:
                details.append(f"{k}={getattr(record, k)}")
        details_str = f" [{', '.join(details)}]" if details else ""
        msg = f"[{timestamp}] [{record.levelname}] [{record.name}]{details_str} {record.getMessage()}"
        if hasattr(record, 'audit_data') and record.audit_data:
            try:
                formatted_audit = json.dumps(record.audit_data, indent=2)
                msg += f"\n--- DISPATCH AUDIT ---\n{formatted_audit}\n----------------------"
            except Exception:
                pass
        return msg


def _add_file_handler(logger, filepath: Path, level: int, formatter: logging.Formatter, max_bytes=10 * 1024 * 1024, backup_count=5):
    resolved = str(filepath.resolve())
    for h in logger.handlers:
        if isinstance(h, RotatingFileHandler) and getattr(h, 'baseFilename', None) == resolved:
            return
    try:
        handler = RotatingFileHandler(filepath, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8')
    except OSError:
        # Logging must never prevent capture/OCR startup. This can happen on
        # Windows when a previous crashed process still holds a rotated log.
        fallback = logging.getLogger('gate.logging')
        fallback.warning('file_log_unavailable', extra={'error_type': 'PermissionError'})
        return
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def configure_logging(log_dir: Path | str = "logs"):
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    root_gate = logging.getLogger('gate')
    root_gate.setLevel(logging.INFO)
    root_gate.propagate = False

    # Console output using JSON formatter
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler) for h in root_gate.handlers):
        console = logging.StreamHandler()
        console.setFormatter(JsonFormatter())
        root_gate.addHandler(console)

    text_fmt = TextErrorFormatter()

    # All error log for entire gate system
    _add_file_handler(root_gate, log_path / "errors.log", logging.WARNING, text_fmt)

    # Subsystem specific loggers
    cam_logger = logging.getLogger('gate.camera')
    _add_file_handler(cam_logger, log_path / "camera_errors.log", logging.WARNING, text_fmt)

    ocr_logger = logging.getLogger('gate.ocr')
    _add_file_handler(ocr_logger, log_path / "ocr_errors.log", logging.WARNING, text_fmt)

    pipe_logger = logging.getLogger('gate.pipeline')
    _add_file_handler(pipe_logger, log_path / "pipeline_errors.log", logging.WARNING, text_fmt)

    # Cloud dispatch audit logger
    cloud_audit_logger = logging.getLogger('gate.cloud.audit')
    cloud_audit_logger.setLevel(logging.INFO)
    _add_file_handler(cloud_audit_logger, log_path / "cloud_dispatch.log", logging.INFO, CloudDispatchFormatter())

