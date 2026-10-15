"""Offline detector audit of retained OCR jobs; never starts cameras or dispatch.

Run from backend with its GPU environment. Writes only the requested report.
Example: .venv/bin/python scripts/audit_live_evidence.py --output /tmp/audit.json
"""
import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import cv2
    from app.config.settings import Settings
    from app.domain import Frame
    from app.detection.yolo_detector import YoloDetector

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    settings = Settings(prosper_enabled=False)
    detector = YoloDetector(settings)
    spool = settings.upload_directory / 'ocr-spool/jobs.sqlite3'
    rows = []
    with sqlite3.connect(spool.resolve().as_uri() + '?mode=ro', uri=True) as db:
        jobs = list(db.execute('select metadata,image from jobs'))
    jobs.sort(key=lambda row: json.loads(row[0])['frame']['timestamp'])
    for metadata, relative in jobs:
        meta = json.loads(metadata)
        path = settings.snapshot_directory / relative
        row = {'file': relative, 'retained': path.is_file()}
        if path.is_file():
            image = cv2.imread(str(path))
            if image is None:
                row['error'] = 'Image decode failed'
            else:
                f = meta['frame']
                frame = Frame(meta['detection']['camera_id'], datetime.fromisoformat(f['timestamp']),
                              f['sequence'], image, f['source_id'])
                start = time.monotonic()
                main_detections = detector.detect(frame)
                vehicle_detections = detector.detect_vehicles(frame)
                row.update(camera=frame.camera_id, sequence=frame.sequence,
                           seconds=time.monotonic()-start,
                           main=[asdict(d) for d in main_detections],
                           auxiliary=[asdict(d) for d in vehicle_detections])
        rows.append(row)
        print(relative, 'retained' if row['retained'] else 'not retained', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'model': str(settings.model_path), 'frames': rows},
                                      indent=2, default=str))


if __name__ == '__main__':
    main()
