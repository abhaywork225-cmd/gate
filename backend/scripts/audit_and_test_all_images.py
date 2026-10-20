"""
Comprehensive Offline Image Testing & Audit Module
Analyzes all captured images and events across backend/snapshots/ to diagnose:
1. Container Number Detection & OCR (check-digit accuracy, character confusions)
2. ISO Code Detection & Feet Sizing (20ft vs 40ft vs 45ft)
3. Twin Container Detection (dual containers on single chassis)
4. Camera View & ImageType Alignment (FRONT vs REAR, empty yard detection)
5. False Trigger / Unreadable Analysis (empty trucks, box vans, reach stackers)
"""
from __future__ import annotations

import os
import sys
import json
import re
import time
from pathlib import Path
from collections import Counter, defaultdict
import cv2
import numpy as np

# Ensure backend root is in python path
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config.settings import Settings
from app.ocr.validator import ContainerValidator, extract_iso_size_code, check_digit, normalize
from app.ocr.ocr_engine import EasyOCREngine

def build_tester(settings: Settings):
    print("Initializing EasyOCREngine...")
    ocr = EasyOCREngine(settings)
    validator = ContainerValidator()
    
    # Load YOLO detector
    print("Loading YOLO model...")
    from ultralytics import YOLO
    model_path = BACKEND_ROOT / "models" / "best.engine"
    if not model_path.exists():
        model_path = BACKEND_ROOT / "models" / "best.pt"
    print(f"Using detector model: {model_path}")
    detector = YOLO(str(model_path))
    return ocr, validator, detector

def test_single_image(img_path: Path, detector, ocr_engine, validator):
    """Run full detection and OCR on a single image and report detailed diagnostics."""
    img = cv2.imread(str(img_path))
    if img is None:
        return {'status': 'ERROR_LOAD_FAILED'}
    
    h, w = img.shape[:2]
    # Check if image is blank / low variance
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    # Run YOLO detection
    # classes: 0: container, 1: container_number, 2: iso_code
    results = detector(img, verbose=False, conf=0.20)
    boxes = []
    if results and len(results) > 0:
        det = results[0].boxes
        for i in range(len(det)):
            cls_id = int(det.cls[i].item())
            raw_name = str(detector.names.get(cls_id, str(cls_id))).lower()
            if raw_name in ('class0', '0', 'container_object'):
                cls_name = 'container'
            elif raw_name in ('class1', '1', 'cointainer'):
                cls_name = 'container_number'
            elif raw_name in ('class2', '2', 'feet'):
                cls_name = 'iso_code'
            else:
                cls_name = raw_name
            conf = float(det.conf[i].item())
            xyxy = det.xyxy[i].cpu().numpy().tolist()
            boxes.append({
                'class_name': cls_name,
                'class_id': cls_id,
                'confidence': conf,
                'bbox': [round(x, 1) for x in xyxy]
            })

    container_numbers = []
    iso_codes = []

    # Process container_number boxes
    num_boxes = [b for b in boxes if b['class_name'] == 'container_number']
    iso_boxes = [b for b in boxes if b['class_name'] == 'iso_code']

    for nb in num_boxes:
        bx1, by1, bx2, by2 = [int(v) for v in nb['bbox']]
        # Add 15px padding
        pad = 15
        crop = img[max(0, by1 - pad):min(h, by2 + pad), max(0, bx1 - pad):min(w, bx2 + pad)]
        if crop.size == 0:
            continue
        read = ocr_engine.read(crop)
        val = validator.validate(read)
        
        # Also try rotating 90 deg if vertical
        if not val.valid_check_digit and crop.shape[0] >= 1.25 * crop.shape[1]:
            rot = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
            rot_read = ocr_engine.read(rot)
            rot_val = validator.validate(rot_read)
            if rot_val.valid_check_digit or (not val.valid_check_digit and rot_val.confidence > val.confidence and len(rot_val.normalized_text) >= 4):
                read, val = rot_read, rot_val

        container_numbers.append({
            'raw_text': read.raw_text,
            'normalized_text': val.normalized_text,
            'confidence': read.confidence,
            'valid_format': val.valid_format,
            'valid_check_digit': val.valid_check_digit,
            'validation_status': val.validation_status,
            'bbox': nb['bbox']
        })

    # Process iso_code boxes (or nearby regions)
    for ib in iso_boxes:
        bx1, by1, bx2, by2 = [int(v) for v in ib['bbox']]
        pad = 12
        crop = img[max(0, by1 - pad):min(h, by2 + pad), max(0, bx1 - pad):min(w, bx2 + pad)]
        if crop.size == 0:
            continue
        read = ocr_engine.read_iso(crop)
        iso_c, human_s = extract_iso_size_code(read.raw_text)
        iso_codes.append({
            'raw_text': read.raw_text,
            'iso_code': iso_c,
            'human_size': human_s,
            'confidence': read.confidence,
            'bbox': ib['bbox']
        })

    # If iso_boxes not detected by YOLO, try searching directly below the container number box!
    if not iso_codes and num_boxes:
        for nb in num_boxes:
            bx1, by1, bx2, by2 = [int(v) for v in nb['bbox']]
            nw = bx2 - bx1
            nh = by2 - by1
            # Search below container number: 0 to 2.5x height below, 0.5x width left/right
            sy1 = by2
            sy2 = min(h, by2 + int(nh * 2.5))
            sx1 = max(0, bx1 - int(nw * 0.2))
            sx2 = min(w, bx2 + int(nw * 0.2))
            sub_below = img[sy1:sy2, sx1:sx2]
            if sub_below.size > 0:
                read = ocr_engine.read_iso(sub_below)
                iso_c, human_s = extract_iso_size_code(read.raw_text)
                if iso_c:
                    iso_codes.append({
                        'raw_text': read.raw_text,
                        'iso_code': iso_c,
                        'human_size': human_s,
                        'confidence': read.confidence,
                        'source': 'neighbourhood_below'
                    })

    # Determine derived feet
    derived_feet = None
    derived_iso = None
    for item in iso_codes:
        if item.get('iso_code'):
            code = item['iso_code']
            derived_iso = code
            if code.startswith('2'):
                derived_feet = '20'
            elif code.startswith('4'):
                derived_feet = '40'
            elif code.startswith('L'):
                derived_feet = '45'
            break

    return {
        'status': 'OK',
        'resolution': [w, h],
        'sharpness': round(variance, 1),
        'boxes_detected': len(boxes),
        'box_summary': Counter([b['class_name'] for b in boxes]),
        'container_numbers': container_numbers,
        'iso_codes': iso_codes,
        'derived_feet': derived_feet,
        'derived_iso': derived_iso
    }

def run_study(limit: int = 150):
    settings = Settings(_env_file=str(BACKEND_ROOT / ".env"))
    ocr_engine, validator, detector = build_tester(settings)

    snapshots_root = BACKEND_ROOT / "snapshots"
    visit_folders = [d for d in snapshots_root.iterdir() if d.is_dir()]
    print(f"\nTotal visit folders in {snapshots_root}: {len(visit_folders)}")
    print(f"Sampling up to {limit} folders for in-depth analysis...\n")

    # Sample representative folders: exit and entry, confirmed and unreadable
    import sqlite3
    conn = sqlite3.connect(str(BACKEND_ROOT / "data" / "gate.db"))
    c = conn.cursor()

    c.execute("""
        SELECT e.id, e.visit_id, e.event_type, e.status, e.container_number
        FROM gate_events e
        ORDER BY e.timestamp DESC
    """)
    db_events = c.fetchall()

    folder_map = {}
    for eid, vid, etype, status, cnum in db_events:
        target_dir = vid if vid else eid
        if target_dir not in folder_map:
            folder_map[target_dir] = {
                'event_id': eid,
                'event_type': etype,
                'status': status,
                'db_container': cnum
            }

    selected_folders = [d for d in visit_folders if d.name in folder_map][:limit]

    # Analytics counters
    stats = {
        'total_events_studied': 0,
        'total_images_tested': 0,
        'by_camera': Counter(),
        'by_event_type': Counter(),
        'by_db_status': Counter(),
        'ocr_success_check_digit': 0,
        'ocr_format_valid_digit_invalid': 0,
        'ocr_no_candidate': 0,
        'iso_code_found': 0,
        'iso_code_missing': 0,
        'feet_breakdown': Counter(),
        'twin_detected_in_visit': 0,
        'single_detected_in_visit': 0,
        'empty_or_no_container_detected': 0,
        'gate_out_rear_door_images': 0,
        'gate_out_front_cabin_images': 0,
        'gate_out_empty_yard_images': 0,
        'character_confusions': Counter(),
        'issues_found': defaultdict(list)
    }

    detailed_results = []

    for idx, folder in enumerate(selected_folders, 1):
        info = folder_map.get(folder.name, {})
        etype = info.get('event_type', 'UNKNOWN')
        status = info.get('status', 'UNKNOWN')
        db_cnum = info.get('db_container')
        
        stats['total_events_studied'] += 1
        stats['by_event_type'][etype] += 1
        stats['by_db_status'][status] += 1

        # Collect proof images in folder
        proof_images = list(folder.glob("*_proof.jpg")) + list(folder.glob("*_proof.png"))
        if not proof_images:
            proof_images = list(folder.glob("*_context_*.jpg"))

        event_cnums = set()
        event_isos = set()
        event_feet = set()

        folder_report = {
            'folder': folder.name,
            'event_type': etype,
            'db_status': status,
            'db_cnum': db_cnum,
            'images': []
        }

        for img_path in proof_images:
            cam_id = img_path.name.split('_')[0]
            stats['by_camera'][cam_id] += 1
            stats['total_images_tested'] += 1

            res = test_single_image(img_path, detector, ocr_engine, validator)
            
            # Analyze image content
            is_gate_out_1 = 'gate-out-1' in cam_id
            is_gate_out_2 = 'gate-out-2' in cam_id

            if is_gate_out_1:
                if res.get('container_numbers') or 'container' in res.get('box_summary', {}):
                    stats['gate_out_rear_door_images'] += 1
            elif is_gate_out_2:
                if 'truck' in res.get('box_summary', {}) or 'container' in res.get('box_summary', {}):
                    stats['gate_out_front_cabin_images'] += 1
                elif not res.get('boxes_detected'):
                    stats['gate_out_empty_yard_images'] += 1

            # Aggregate OCR findings
            cnums = res.get('container_numbers', [])
            if cnums:
                for cn in cnums:
                    if cn['valid_check_digit']:
                        stats['ocr_success_check_digit'] += 1
                        event_cnums.add(cn['normalized_text'])
                    elif cn['valid_format']:
                        stats['ocr_format_valid_digit_invalid'] += 1
                        try:
                            expected = check_digit(cn['normalized_text'][:10])
                            actual = int(cn['normalized_text'][10])
                            stats['character_confusions'][f"exp_{expected}_got_{actual}"] += 1
                            stats['issues_found']['check_digit_failures'].append({
                                'image': img_path.name,
                                'read': cn['normalized_text'],
                                'expected_digit': expected,
                                'actual_digit': actual
                            })
                        except Exception:
                            pass
                    else:
                        stats['ocr_no_candidate'] += 1
            else:
                stats['ocr_no_candidate'] += 1

            # Sizing findings
            isos = res.get('iso_codes', [])
            if isos:
                stats['iso_code_found'] += 1
                for it in isos:
                    if it.get('iso_code'):
                        event_isos.add(it['iso_code'])
            else:
                stats['iso_code_missing'] += 1

            if res.get('derived_feet'):
                event_feet.add(res['derived_feet'])

            folder_report['images'].append({
                'file': img_path.name,
                'camera': cam_id,
                'diagnostics': res
            })

        # Check visit-level twin condition
        if len(event_cnums) >= 2:
            stats['twin_detected_in_visit'] += 1
            folder_report['visit_classification'] = 'TWIN_20FT'
        elif len(event_cnums) == 1:
            stats['single_detected_in_visit'] += 1
            folder_report['visit_classification'] = 'SINGLE_CONTAINER'
        else:
            stats['empty_or_no_container_detected'] += 1
            folder_report['visit_classification'] = 'NO_VALID_CONTAINER'

        for f in event_feet:
            stats['feet_breakdown'][f] += 1

        detailed_results.append(folder_report)

        if idx % 10 == 0 or idx == len(selected_folders):
            print(f"Processed {idx}/{len(selected_folders)} events ({stats['total_images_tested']} images tested)...")

    # Generate Markdown Summary
    report_file = BACKEND_ROOT / "data" / "image_test_study_report.json"
    with open(report_file, "w") as fp:
        json.dump({'stats': {k: dict(v) if isinstance(v, (Counter, defaultdict)) else v for k, v in stats.items()},
                   'events': detailed_results}, fp, indent=2)

    print(f"\n=======================================================")
    print(f"           IMAGE TESTING & STUDY COMPLETED              ")
    print(f"=======================================================")
    print(f"Events Studied     : {stats['total_events_studied']}")
    print(f"Total Images Tested: {stats['total_images_tested']}")
    print(f"Check Digit Passed : {stats['ocr_success_check_digit']}")
    print(f"Check Digit Failed : {stats['ocr_format_valid_digit_invalid']}")
    print(f"ISO Codes Found    : {stats['iso_code_found']}")
    print(f"Feet Breakdown     : {dict(stats['feet_breakdown'])}")
    print(f"Twins in Visits    : {stats['twin_detected_in_visit']}")
    print(f"Singles in Visits  : {stats['single_detected_in_visit']}")
    print(f"Empty/Non-container: {stats['empty_or_no_container_detected']}")
    print(f"Gate OUT Rear Doors: {stats['gate_out_rear_door_images']}")
    print(f"Gate OUT Cabins    : {stats['gate_out_front_cabin_images']}")
    print(f"Gate OUT Empty Yard: {stats['gate_out_empty_yard_images']}")
    print(f"Saved full JSON to : {report_file}")
    print(f"=======================================================\n")

if __name__ == '__main__':
    run_study(limit=80)
