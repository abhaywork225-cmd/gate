from __future__ import annotations
"""Derived inspection image; never change the original captured evidence."""
import cv2
import numpy as np


def proof_jpeg(data: bytes, snapshot: dict = None, event: dict = None) -> bytes:
    """
    Return clean, high-resolution camera photo without artificial black canvases,
    side panel crops, or debug text overlays.
    """
    if data.startswith(b'\xff\xd8'):
        return data
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Proof image cannot be decoded')
    ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError('Proof image cannot be encoded as JPEG')
    return encoded.tobytes()
