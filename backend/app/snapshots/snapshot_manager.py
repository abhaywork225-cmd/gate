from __future__ import annotations
from pathlib import Path
import os


class SnapshotManager:
    CATEGORIES = ('waiting', 'detected_in', 'detected_out')

    def __init__(self, root, categorized: bool = True):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.categorized = bool(categorized)
        if self.categorized:
            for cat in self.CATEGORIES:
                (self.root / cat).mkdir(parents=True, exist_ok=True)

    def category_for(self, direction: str | None = None, is_waiting: bool = False, gate_id: str | None = None) -> str:
        if not self.categorized:
            return ''
        if is_waiting:
            return 'waiting'
        d = str(direction or '').upper()
        if d in ('EXIT', 'OUT', 'GATE_OUT'):
            return 'detected_out'
        if d in ('ENTRY', 'IN', 'GATE_IN'):
            return 'detected_in'
        g = str(gate_id or '').lower()
        if any(k in g for k in ('out', 'exit')):
            return 'detected_out'
        return 'detected_in'

    def path(self, relative):
        if not relative:
            raise ValueError('Invalid snapshot reference')
        rel_str = str(relative).replace('\\', '/').strip('/')
        direct = (self.root / rel_str).resolve()
        if direct.is_relative_to(self.root) and direct.exists():
            return direct
        if self.categorized and not any(rel_str.startswith(f'{c}/') for c in self.CATEGORIES):
            for cat in self.CATEGORIES:
                candidate = (self.root / cat / rel_str).resolve()
                if candidate.is_relative_to(self.root) and candidate.exists():
                    return candidate
        if not direct.is_relative_to(self.root) or direct == self.root:
            raise ValueError('Invalid snapshot reference')
        return direct

    def ensure_event_dir(self, event_id: str, category: str = 'detected_in') -> Path:
        clean_id = str(event_id).replace('\\', '/').strip('/')
        if not self.categorized:
            event_dir = (self.root / clean_id).resolve()
        elif any(clean_id.startswith(f'{c}/') for c in self.CATEGORIES):
            event_dir = (self.root / clean_id).resolve()
        else:
            cat = category if category in self.CATEGORIES else 'detected_in'
            event_dir = (self.root / cat / clean_id).resolve()
        event_dir.mkdir(parents=True, exist_ok=True)
        return event_dir

    def promote_from_waiting(self, visit_id: str, direction: str | None = None) -> str:
        """Promote files from snapshots/waiting/<visit_id> or legacy root/<visit_id>
        to snapshots/detected_in/<visit_id> or snapshots/detected_out/<visit_id>."""
        if not self.categorized:
            return ''
        import shutil
        target_category = self.category_for(direction)
        vid = str(visit_id).replace('\\', '/').strip('/')
        for c in self.CATEGORIES:
            if vid.startswith(f'{c}/'):
                vid = vid[len(c)+1:]
                break

        target_dir = self.root / target_category / vid
        target_dir.mkdir(parents=True, exist_ok=True)

        # 1. Promote from waiting/<vid>
        waiting_dir = self.root / 'waiting' / vid
        if waiting_dir.is_dir() and waiting_dir != target_dir:
            for item in list(waiting_dir.iterdir()):
                target_file = target_dir / item.name
                if item.is_file():
                    shutil.move(str(item), str(target_file))
                elif item.is_dir():
                    if target_file.exists():
                        shutil.rmtree(str(target_file))
                    shutil.move(str(item), str(target_file))
            try:
                waiting_dir.rmdir()
            except OSError:
                pass

        # 2. Promote from legacy root/<vid>
        legacy_dir = self.root / vid
        if legacy_dir.is_dir() and legacy_dir != target_dir and vid not in self.CATEGORIES:
            for item in list(legacy_dir.iterdir()):
                target_file = target_dir / item.name
                if item.is_file():
                    shutil.move(str(item), str(target_file))
                elif item.is_dir():
                    if target_file.exists():
                        shutil.rmtree(str(target_file))
                    shutil.move(str(item), str(target_file))
            try:
                legacy_dir.rmdir()
            except OSError:
                pass

        return target_category

    @staticmethod
    def compute_quality(image) -> float:
        import cv2, math
        if image is None or image.size == 0:
            return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        exposure = float(((gray > 15) & (gray < 240)).mean())
        variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return math.log1p(variance) * exposure

    @staticmethod
    def _capture_has_visible_container(snapshot: dict, event_record: dict, image) -> bool:
        """Require a meaningful YOLO container-number box for proof images.

        Whole-frame sharpness is not sufficient proof quality: a sharp frame can
        contain only a tiny sliver of the truck/container.  Match detections by
        camera and capture time and reject boxes that are too small or clipped.
        """
        import datetime as _dt
        try:
            h, w = image.shape[:2]
            if h <= 0 or w <= 0:
                return False
            snap_ts = snapshot.get('timestamp')
            def parse(value):
                if not value:
                    return None
                text = str(value).replace('Z', '+00:00')
                try:
                    return _dt.datetime.fromisoformat(text)
                except ValueError:
                    return None
            snap_dt = parse(snap_ts)
            camera = snapshot.get('camera_id')
            candidates = []
            for detection in event_record.get('detections') or []:
                if detection.get('camera_id') != camera:
                    continue
                if detection.get('class_name') not in ('container', 'container_number', 'iso_code', 'truck', 'trailer', 'vehicle'):
                    continue
                det_dt = parse(detection.get('timestamp'))
                if snap_dt and det_dt:
                    if abs((snap_dt - det_dt).total_seconds()) > 6.0:
                        continue
                bbox = detection.get('bbox') or []
                if len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = [float(v) for v in bbox]
                # Detection coordinates are normally in source-frame pixels;
                # normalize against the actual saved frame dimensions.
                bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
                area_ratio = (bw * bh) / float(w * h)
                # OCR text regions must be reasonably close and legible at gate scale.
                # Reject distant background slivers (< 28px tall or area_ratio < 0.0010)
                # where the vehicle is already 50+ meters down the yard.
                if bw < 50.0 or bh < 28.0 or area_ratio < 0.0010:
                    continue
                # A clipped text box is not reliable proof, even if sharp.
                if x1 <= 1.0 or y1 <= 1.0 or x2 >= w - 1.0 or y2 >= h - 1.0:
                    continue
                candidates.append((area_ratio, bw, bh))
            if candidates:
                return True

            # Companion camera frames are captured at the same instant as the
            # OCR observation and are stored as {event_id}/{camera_id}.jpg.
            # They intentionally do not have a second OCR detection row.  The
            # old rule discarded these valid views even though the image was
            # present in the event folder, causing cloud payloads to contain
            # only the primary camera image.
            image_path = str(snapshot.get('image_path') or '').replace('\\', '/')
            filename = image_path.rsplit('/', 1)[-1]
            return filename.startswith(f'{camera}_') or filename == f'{camera}.jpg'
        except Exception:
            return False

    def build_evidence_manifest(self, event_record: dict, min_quality: float = 2.5) -> dict:
        """Select one usable camera capture per event/camera for UI and cloud."""
        snapshots = event_record.get('snapshots') or []
        by_camera = {}
        for snap in snapshots:
            camera_id, image_path = snap.get('camera_id'), snap.get('image_path')
            if not camera_id or not image_path:
                continue
            try:
                import cv2, numpy as np
                data = self.path(image_path).read_bytes()
                decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if decoded is None or decoded.size == 0 or decoded.shape[0] < 240 or decoded.shape[1] < 320:
                    continue
                gray = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
                quality = self.compute_quality(decoded)
                if float(gray.std()) < 8.0 or quality < min_quality:
                    continue
                if not self._capture_has_visible_container(snap, event_record, decoded):
                    continue
                by_camera.setdefault(camera_id, []).append({**snap, 'quality_score': round(float(quality), 3)})
            except Exception:
                continue
        # The event writer may have already selected a frame because it produced
        # the accepted OCR result.  That frame is the proof of record and must
        # not be replaced by a different (possibly sharper but text-invisible)
        # frame from the same camera.  Otherwise the local OCR result and the
        # cloud proof image describe different moments.
        preferred_ids = {
            value for value in (
                event_record.get('primary_snapshot_id'),
                event_record.get('rear_snapshot_id'),
            ) if value
        }
        def snap_score(item):
            score = float(item.get('quality_score') or 0.0)
            if item.get('id') in preferred_ids:
                score += 500.0
            try:
                from app.events.cloud_dispatcher import snapshot_has_identity
                c1 = event_record.get('container_number')
                c2 = event_record.get('container_number_2') if event_record.get('is_twin') else None
                if c1 and snapshot_has_identity(item, event_record, c1, 0.0):
                    score += 2000.0
                if c2 and snapshot_has_identity(item, event_record, c2, 0.0):
                    score += 2000.0
            except Exception:
                pass
            return score

        captures = []
        for items in by_camera.values():
            captures.append(max(
                items, key=lambda item: (snap_score(item), item.get('timestamp') or '')
            ))

        # Ensure all explicitly designated proof snapshots are included
        existing_ids = {c['id'] for c in captures}
        for snap in snapshots:
            if snap.get('id') in preferred_ids and snap.get('id') not in existing_ids:
                captures.append(snap)
                existing_ids.add(snap['id'])

        captures.sort(key=lambda item: (item.get('camera_id') or '', item.get('timestamp') or ''))
        ids = {item['id'] for item in captures}

        # Resolve primary (front) proof snapshot
        front_cands = [c for c in captures if not (c.get('image_path') or '').endswith('_rear.jpg')]
        if event_record.get('primary_snapshot_id') in ids:
            primary = event_record.get('primary_snapshot_id')
        elif front_cands:
            primary = front_cands[0]['id']
        elif captures:
            primary = captures[0]['id']
        else:
            primary = None

        # Resolve rear proof snapshot
        is_twin = bool(event_record.get('is_twin'))
        rear_cands = [c for c in captures if (c.get('image_path') or '').endswith('_rear.jpg') or any(k in (c.get('camera_id') or '').lower() for k in ('-3', 'rear'))]
        if event_record.get('rear_snapshot_id') in ids:
            rear = event_record.get('rear_snapshot_id')
        elif rear_cands:
            rear = rear_cands[0]['id']
        elif is_twin and len(captures) > 1:
            rear = next((c['id'] for c in captures if c['id'] != primary), None)
        else:
            rear = None

        # Ensure twin events have distinct snapshots for front and rear when multiple captures exist
        if is_twin and primary and rear and primary == rear and len(captures) > 1:
            diff_cand = next((c['id'] for c in captures if c['id'] != primary), None)
            if diff_cand:
                rear = diff_cand

        return {'version': 1, 'event_id': event_record.get('id'), 'captures': captures,
                'selected_snapshot_ids': [item['id'] for item in captures],
                'primary_snapshot_id': primary, 'rear_snapshot_id': rear}

    @staticmethod
    def compute_similarity(img1, img2) -> float:
        """
        Fast zero-mean normalized cross-correlation (NCC) of two images (0.0 to 1.0).
        Accepts numpy arrays, file paths (str/Path), or image bytes.
        Downsamples to 64x64 grayscale for fast sub-3ms comparison.
        """
        import cv2, numpy as np
        if isinstance(img1, (str, Path)):
            p = str(img1)
            if not os.path.exists(p):
                return 0.0
            img1 = cv2.imread(p)
        elif isinstance(img1, bytes):
            img1 = cv2.imdecode(np.frombuffer(img1, dtype=np.uint8), cv2.IMREAD_COLOR)

        if isinstance(img2, (str, Path)):
            p = str(img2)
            if not os.path.exists(p):
                return 0.0
            img2 = cv2.imread(p)
        elif isinstance(img2, bytes):
            img2 = cv2.imdecode(np.frombuffer(img2, dtype=np.uint8), cv2.IMREAD_COLOR)

        if img1 is None or img2 is None or getattr(img1, 'size', 0) == 0 or getattr(img2, 'size', 0) == 0:
            return 0.0

        try:
            s1 = cv2.resize(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), (64, 64)).astype(np.float32)
            s2 = cv2.resize(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), (64, 64)).astype(np.float32)
            s1 -= s1.mean()
            s2 -= s2.mean()
            n1 = float(np.linalg.norm(s1))
            n2 = float(np.linalg.norm(s2))
            if n1 <= 1e-6 or n2 <= 1e-6:
                return 0.0
            return float(np.clip((s1 * s2).sum() / (n1 * n2), 0.0, 1.0))
        except Exception:
            return 0.0

    def save_side_snapshot(self, event_id: str, camera_id: str, image, direction: str = '', text: str = '', force: bool = False, category: str | None = None) -> str:
        """Saves or upgrades the compulsory side snapshot for camera_id under snapshots/{category}/{event_id}/{camera_id}.jpg."""
        import cv2
        clean_folder = str(event_id).replace('\\', '/').strip('/')
        if not self.categorized or any(clean_folder.startswith(f'{c}/') for c in self.CATEGORIES):
            relative_dir = clean_folder
        else:
            cat = category or self.category_for(direction)
            relative_dir = f'{cat}/{clean_folder}' if cat else clean_folder

        self.ensure_event_dir(relative_dir)
        relative = f'{relative_dir}/{camera_id}.jpg'
        final = self.path(relative)
        final.parent.mkdir(parents=True, exist_ok=True)
        meta_quality_path = final.with_suffix('.quality')

        new_quality = self.compute_quality(image)
        if not force and final.exists() and meta_quality_path.exists():
            try:
                old_quality = float(meta_quality_path.read_text().strip())
                if new_quality <= old_quality:
                    return relative  # Existing snapshot is sharper/clearer
            except Exception:
                pass

        ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise OSError('Snapshot encoding failed')

        temp_img = final.with_suffix('.tmp')
        try:
            with temp_img.open('wb') as f:
                f.write(encoded.tobytes())
            os.replace(temp_img, final)
            meta_quality_path.write_text(f'{new_quality:.2f}', encoding='utf-8')
        finally:
            temp_img.unlink(missing_ok=True)
        return relative

    def save_best_crop(self, event_id: str, slot_pos: str, crop, text: str = '', force: bool = False, direction: str = '', category: str | None = None) -> str:
        """Saves or upgrades the highest-clarity OCR proof crop under snapshots/{category}/{event_id}/crop_{slot}_best.png."""
        import cv2
        clean_folder = str(event_id).replace('\\', '/').strip('/')
        if not self.categorized or any(clean_folder.startswith(f'{c}/') for c in self.CATEGORIES):
            relative_dir = clean_folder
        else:
            cat = category or self.category_for(direction)
            relative_dir = f'{cat}/{clean_folder}' if cat else clean_folder

        self.ensure_event_dir(relative_dir)
        slot_clean = slot_pos.lower().replace(' ', '_')
        relative = f'{relative_dir}/crop_{slot_clean}_best.png'
        final = self.path(relative)
        final.parent.mkdir(parents=True, exist_ok=True)
        meta_quality_path = final.with_suffix('.quality')

        new_quality = self.compute_quality(crop)
        if not force and final.exists() and meta_quality_path.exists():
            try:
                old_quality = float(meta_quality_path.read_text().strip())
                if new_quality <= old_quality:
                    return relative
            except Exception:
                pass

        crop_img = crop.copy()
        if text:
            cv2.putText(crop_img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 0), 2)
        ok, encoded = cv2.imencode('.jpg', crop_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise OSError('Crop encoding failed')

        temp_crop = final.with_suffix('.tmp')
        try:
            with temp_crop.open('wb') as f:
                f.write(encoded.tobytes())
            os.replace(temp_crop, final)
            meta_quality_path.write_text(f'{new_quality:.2f}', encoding='utf-8')
        finally:
            temp_crop.unlink(missing_ok=True)
        return relative

    def save(self, snapshot_id, observation, event_id=None, direction=None, category=None):
        import cv2
        image = observation.image.copy()

        is_rear = getattr(observation, 'slot_position', 'FRONT') == 'REAR' or observation.origin_key.endswith(':rear')
        slot_name = "REAR (SLOT 2)" if is_rear else "FRONT (SLOT 1)"
        container_num = observation.ocr.normalized_text if (observation.ocr and observation.ocr.normalized_text) else ""
        box_color = (0, 200, 255) if is_rear else (40, 190, 80)

        # Save clean raw camera photo without drawing boxes over text
        ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise OSError('Snapshot encoding failed')
        is_rear = getattr(observation, 'slot_position', 'FRONT') == 'REAR' or observation.origin_key.endswith(':rear')
        slot_suffix = '_rear' if is_rear else ''
        if event_id:
            clean_folder = str(event_id).replace('\\', '/').strip('/')
            if not self.categorized or any(clean_folder.startswith(f'{c}/') for c in self.CATEGORIES):
                relative_dir = clean_folder
            else:
                cat = category or self.category_for(direction or getattr(observation, 'direction', None), gate_id=getattr(observation, 'gate_id', None))
                relative_dir = f'{cat}/{clean_folder}' if cat else clean_folder
            relative = f'{relative_dir}/{observation.detection.camera_id}_{snapshot_id}{slot_suffix}.jpg'
        else:
            relative = f'{snapshot_id}.jpg'
        final = self.path(relative)
        final.parent.mkdir(parents=True, exist_ok=True)
        quality_path = final.with_suffix('.quality')
        new_quality = self.compute_quality(image)
        if final.exists() and quality_path.exists():
            try:
                if new_quality <= float(quality_path.read_text(encoding='utf-8')):
                    return relative
            except (OSError, ValueError):
                pass
        temporary = final.with_suffix('.tmp')
        try:
            with temporary.open('wb') as file:
                file.write(encoded.tobytes())
            os.replace(temporary, final)
            quality_path.write_text(f'{new_quality:.2f}', encoding='utf-8')
        finally:
            temporary.unlink(missing_ok=True)
        return relative

    def save_camera_frame(self, snapshot_id, event_id, camera_id, image, direction='', text=''):
        return self.save_side_snapshot(event_id, camera_id, image, direction=direction, text=text)

    def save_event_json(self, event_id: str, event_data: dict, direction: str | None = None) -> Path:
        import json
        visit_id = event_data.get('visit_id')
        raw_vid = str(visit_id or event_id).replace('\\', '/').strip('/')
        if not self.categorized or any(raw_vid.startswith(f'{c}/') for c in self.CATEGORIES):
            vid = raw_vid
        else:
            cat = self.category_for(direction or event_data.get('event_type'))
            vid = f'{cat}/{raw_vid}' if cat else raw_vid

        event_dir = self.path(vid)
        event_dir.mkdir(parents=True, exist_ok=True)
        if visit_id:
            visit = {'visit_id':visit_id,'gate_id':event_data.get('gate_id'),
                     'direction':event_data.get('event_type'),
                     'container_records':sorted({event_id,*(p.stem[len('event_'):] for p in event_dir.glob('event_*.json'))}),
                     'context_capture_files':sorted(p.name for p in event_dir.glob('*_context_*.jpg'))}
            manifest = event_dir / 'visit.json'
            temporary = manifest.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(visit,indent=2))
            os.replace(temporary,manifest)
        final_path = event_dir / (f'event_{event_id}.json' if visit_id else 'event.json')
        temp_path = final_path.with_suffix('.json.tmp')
        try:
            with temp_path.open('w', encoding='utf-8') as f:
                json.dump(event_data, f, indent=2, default=str)
            os.replace(temp_path, final_path)
        finally:
            temp_path.unlink(missing_ok=True)
        manifest = self.build_evidence_manifest(event_data)
        manifest_path = event_dir / (f'evidence_{event_id}.json' if visit_id else 'evidence.json')
        manifest_temp = manifest_path.with_suffix('.json.tmp')
        try:
            with manifest_temp.open('w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, default=str)
            os.replace(manifest_temp, manifest_path)
        finally:
            manifest_temp.unlink(missing_ok=True)
        return final_path

    def remove(self, relative):
        path = self.path(relative)
        path.unlink(missing_ok=True)
        if path.parent != self.root:
            try:
                path.parent.rmdir()
            except OSError:
                pass
