from __future__ import annotations
from pathlib import Path
import os


class SnapshotManager:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root) or path == self.root:
            raise ValueError('Invalid snapshot reference')
        return path

    def ensure_event_dir(self, event_id: str) -> Path:
        event_dir = self.root / str(event_id)
        event_dir.mkdir(parents=True, exist_ok=True)
        return event_dir

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
                # OCR text regions can be narrow on the fixed gate views.  Keep
                # a modest floor for readable text while still rejecting the
                # tiny 13-pixel slivers that previously became proof images.
                if bw < 35.0 or bh < 10.0 or area_ratio < 0.0005:
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
            return image_path.rsplit('/', 1)[-1] == f'{camera}.jpg'
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
        captures = []
        for items in by_camera.values():
            preferred = [item for item in items if item.get('id') in preferred_ids]
            if preferred:
                captures.extend(preferred)
            else:
                captures.append(max(
                    items, key=lambda item: (item['quality_score'], item.get('timestamp') or '')
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

    def save_side_snapshot(self, event_id: str, camera_id: str, image, direction: str = '', text: str = '', force: bool = False) -> str:
        """Saves or upgrades the compulsory side snapshot for camera_id under snapshots/{event_id}/{camera_id}.jpg."""
        import cv2
        self.ensure_event_dir(event_id)
        relative = f'{event_id}/{camera_id}.jpg'
        final = self.path(relative)
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

    def save_best_crop(self, event_id: str, slot_pos: str, crop, text: str = '', force: bool = False) -> str:
        """Saves or upgrades the highest-clarity OCR proof crop under snapshots/{event_id}/crop_{slot}_best.jpg."""
        import cv2
        self.ensure_event_dir(event_id)
        slot_clean = slot_pos.lower().replace(' ', '_')
        relative = f'{event_id}/crop_{slot_clean}_best.png'
        final = self.path(relative)
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

    def save(self, snapshot_id, observation, event_id=None):
        import cv2
        image = observation.image.copy()

        is_rear = getattr(observation, 'slot_position', 'FRONT') == 'REAR' or observation.origin_key.endswith(':rear')
        slot_name = "REAR (SLOT 2)" if is_rear else "FRONT (SLOT 1)"
        container_num = observation.ocr.normalized_text if (observation.ocr and observation.ocr.normalized_text) else ""
        box_color = (0, 200, 255) if is_rear else (40, 190, 80)

        # Save clean raw camera photo without drawing boxes over text
        # (Overlays over the numbers obscure human verification on the portal)
        ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise OSError('Snapshot encoding failed')
        is_rear = getattr(observation, 'slot_position', 'FRONT') == 'REAR' or observation.origin_key.endswith(':rear')
        slot_suffix = '_rear' if is_rear else ''
        relative = f'{event_id}/{observation.detection.camera_id}_{snapshot_id}{slot_suffix}.jpg' if event_id else f'{snapshot_id}.jpg'
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

    def save_event_json(self, event_id: str, event_data: dict) -> Path:
        import json
        visit_id = event_data.get('visit_id')
        event_dir = self.path(str(visit_id)) if visit_id else self.path(str(event_id))
        event_dir.mkdir(parents=True, exist_ok=True)
        if visit_id:
            visit = {'visit_id':visit_id,'gate_id':event_data.get('gate_id'),
                     'direction':event_data.get('event_type'),
                     'container_records':sorted({event_id,*(p.stem[len('event_'):] for p in event_dir.glob('event_*.json'))}),
                     'context_capture_files':sorted(p.name for p in event_dir.glob('*_context_*.jpg'))}
            manifest = self.path(f'{visit_id}/visit.json')
            temporary = manifest.with_suffix('.json.tmp')
            import json
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
