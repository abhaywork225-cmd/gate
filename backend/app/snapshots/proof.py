from __future__ import annotations
"""Derived inspection image; never change the original captured evidence."""
import re

import cv2
import numpy as np


def proof_jpeg(data: bytes, snapshot: dict, event: dict) -> bytes:
    """
    Return clean, high-resolution camera photo without artificial black canvases,
    side panel crops, or debug text overlays.
    """
    return data

