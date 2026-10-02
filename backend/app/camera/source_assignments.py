from __future__ import annotations
"""Durable camera-source selections. RTSP URLs remain in environment variables."""
import json
import os
from pathlib import Path
import threading


class SourceAssignments:
    def __init__(self, root, filename='camera-selections.json'):
        self.path = Path(root).resolve() / filename
        self.lock = threading.Lock()
        try:
            value = json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else {}
            self.selections = value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            self.selections = {}

    def get(self, slot_id):
        with self.lock:
            return self.selections.get(slot_id)

    def set(self, slot_id, source_id):
        with self.lock:
            updated = {**self.selections, slot_id: source_id}
            temporary = self.path.with_suffix('.tmp')
            temporary.write_text(json.dumps(updated), encoding='utf-8')
            os.replace(temporary, self.path)
            self.selections = updated
