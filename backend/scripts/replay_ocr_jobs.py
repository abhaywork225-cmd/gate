"""Offline GPU EasyOCR replay of persisted number jobs. No camera/cloud/event writes.

Run from backend with the same GPU library environment as start.sh, for example:
  .venv/bin/python scripts/replay_ocr_jobs.py --report /tmp/ocr-replay.json
A checksum-valid count is NOT an accuracy score; compare text to source images.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    import cv2
    from app.config.settings import Settings
    from app.ocr.ocr_engine import EasyOCREngine
    from app.ocr.validator import ContainerValidator
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--padding',type=int,action='append')
    args = parser.parse_args()
    pads = args.padding if args.padding is not None else [0,10]
    if any(p < 0 or p > 50 for p in pads):
        parser.error('Padding must be between 0 and 50 pixels')
    settings = Settings(ocr_gpu=True,ocr_download_enabled=False,prosper_enabled=False)
    engine, validator = EasyOCREngine(settings), ContainerValidator()
    spool = settings.upload_directory/'ocr-spool/jobs.sqlite3'
    with sqlite3.connect(spool.resolve().as_uri()+'?mode=ro',uri=True) as db:
        jobs = list(db.execute('select metadata,image from jobs'))
    results = []
    for metadata,relative in jobs:
        meta = json.loads(metadata)
        if meta['detection']['class_name'] != 'container_number':
            continue
        path = settings.snapshot_directory/relative
        if not path.is_file():
            results.append({'file':relative,'skipped':'source no longer retained'})
            continue
        image = cv2.imread(str(path))
        if image is None:
            results.append({'file':relative,'skipped':'image decoding failed'})
            continue
        h,w = image.shape[:2]
        x1,y1,x2,y2 = map(int,meta['detection']['bbox'])
        for pad in pads:
            crop = image[max(0,y1-pad):min(h,y2+pad),max(0,x1-pad):min(w,x2+pad)]
            started = time.monotonic()
            read = engine.read(crop)
            validation = validator.validate(read)
            row = {'file':relative,'camera':meta['detection']['camera_id'],
                   'sequence':meta['frame']['sequence'],'padding':pad,
                   'read':asdict(read),'validation':asdict(validation),
                   'seconds':time.monotonic()-started}
            results.append(row)
            print(relative,pad,read.raw_text,round(read.confidence,4),flush=True)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps({'engine':'easyocr','gpu':True,'results':results},indent=2))


if __name__ == '__main__':
    main()
