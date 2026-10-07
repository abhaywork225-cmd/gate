from __future__ import annotations
import math

def prepare_ocr_crop(image):
    """Upscale and improve contrast for small/angled container markings."""
    import cv2
    if image is None or getattr(image, 'size', 0) == 0:
        return image
    h, w = image.shape[:2]
    scale = 3.0 if min(h, w) < 160 else 2.0
    enlarged = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(enlarged, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


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


def inside_gate(bbox, image_shape, zone):
    """Require a detection's center and most of its area inside the gate ROI."""
    h, w = image_shape[:2]
    x1, y1, x2, y2 = bbox
    zx1, zy1, zx2, zy2 = (zone[0]*w, zone[1]*h, zone[2]*w, zone[3]*h)
    area = max(0, x2-x1) * max(0, y2-y1)
    overlap = max(0, min(x2,zx2)-max(x1,zx1)) * max(0,min(y2,zy2)-max(y1,zy1))
    return (area > 0 and zx1 <= (x1+x2)/2 <= zx2 and
            zy1 <= (y1+y2)/2 <= zy2 and overlap/area >= .8)
