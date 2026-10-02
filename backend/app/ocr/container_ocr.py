from __future__ import annotations
import math


def crop_identification(image, bbox, roi=(0, 0, 1, 1)):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError('Empty detection crop')
    rx1, ry1, rx2, ry2 = roi
    left, top = math.floor(x1 + (x2 - x1) * rx1), math.floor(y1 + (y2 - y1) * ry1)
    right, bottom = math.ceil(x1 + (x2 - x1) * rx2), math.ceil(y1 + (y2 - y1) * ry2)
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        raise ValueError('Empty identification region')
    return crop.copy()
