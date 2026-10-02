from __future__ import annotations
"""Validated uploaded videos and durable per-slot assignments, independent of SQL.

Original clips are retained when switching sources; no event evidence is deleted.
"""
import json
import os
from pathlib import Path
import threading


class VideoLibrary:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.manifest = self.root / 'assignments.json'
        self.assignments = json.loads(self.manifest.read_text(encoding='utf-8')) if self.manifest.exists() else {}

    def info(self, camera_id):
        with self.lock:
            value = self.assignments.get(camera_id)
            return dict(value) if value else None

    def path(self, filename):
        path = (self.root / filename).resolve()
        if path.parent != self.root:
            raise ValueError('Invalid video reference')
        return path

    def assign(self, camera_id, value):
        with self.lock:
            updated = {**self.assignments, camera_id: value}
            temporary = self.manifest.with_suffix('.tmp')
            temporary.write_text(json.dumps(updated), encoding='utf-8')
            os.replace(temporary, self.manifest)
            self.assignments = updated

    def validate(self, path):
        import cv2
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise ValueError('The uploaded file is not a readable video')
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ValueError('The video contains no readable frames')
            fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
            return {'width': int(frame.shape[1]), 'height': int(frame.shape[0]),
                    'duration_seconds': frames / fps if fps > 0 and frames > 0 else None}
        finally:
            cap.release()
