from __future__ import annotations
"""Independent Gate Module service. Run from backend/ to avoid legacy app imports."""
from pathlib import Path

if not hasattr(Path, 'is_relative_to'):
    def _is_relative_to(self, other):
        try:
            self.relative_to(other)
            return True
        except ValueError:
            return False
    Path.is_relative_to = _is_relative_to
