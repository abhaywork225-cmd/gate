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


def inside_gate(bbox, image_shape, zone, allow_headroom=True):
    """Require a detection's center and most of its area inside the gate ROI."""
    h, w = image_shape[:2]
    x1, y1, x2, y2 = bbox
    zx1, zy1, zx2, zy2 = (zone[0]*w, zone[1]*h, zone[2]*w, zone[3]*h)
    box_w = max(0, x2 - x1)
    if box_w == 0:
        return False
    x_center = (x1 + x2) / 2
    if not (zx1 <= x_center <= zx2):
        return False
    x_overlap = max(0, min(x2, zx2) - max(x1, zx1))
    if x_overlap / box_w < 0.7:
        return False

    box_h = max(0, y2 - y1)
    if box_h == 0:
        return False
    y_center = (y1 + y2) / 2
    effective_zy1 = 0.0 if allow_headroom else zy1
    if not (effective_zy1 <= y_center <= zy2):
        return False
    y_overlap = max(0, min(y2, zy2) - max(y1, effective_zy1))
    if y_overlap / box_h < 0.7:
        return False

    return True


def expand_number_crop(image, bbox, gate_roi):
    """Recover glyphs clipped by a text box without leaving the calibrated gate."""
    h, w = image.shape[:2]
    x1,y1,x2,y2 = bbox
    dx,dy = max(10, (x2-x1)*.15), max(5, (y2-y1)*.3)
    left=max(0,int(gate_roi[0]*w),int(x1-dx))
    top=max(0,int(y1-dy))
    right=min(w,int(gate_roi[2]*w),int(x2+dx))
    bottom=min(h,int(gate_roi[3]*h),int(y2+dy))
    if right<=left or bottom<=top:
        raise ValueError('Identification crop is outside the gate ROI')
    return image[top:bottom,left:right].copy()


def rectify_printed_digit_box(image):
    """Remove a measured quadrilateral border, without inventing a glyph.

    A closed border must enclose a substantial hole. This prevents treating an
    unboxed digit, a full-image threshold region, or arbitrary strokes as a box.
    The input and saved proof are never modified.
    """
    import cv2
    import numpy as np
    if image is None or image.size == 0:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape
    if min(h, w) < 8:
        return None
    for mode in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        _, mask = cv2.threshold(gray, 0, 255, mode | cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        for i in sorted(range(len(contours)), key=lambda k: cv2.contourArea(contours[k]), reverse=True):
            contour = contours[i]
            area = cv2.contourArea(contour)
            child = hierarchy[0][i][2]
            if not .25*w*h < area < .95*w*h or child < 0:
                continue
            if not .3 < cv2.contourArea(contours[child])/area < .95:
                continue
            quad = cv2.approxPolyDP(contour, .03*cv2.arcLength(contour, True), True)
            if len(quad) != 4 or not cv2.isContourConvex(quad):
                continue
            points = quad.reshape(4, 2).astype(np.float32)
            # Sum/difference ordering selects the same corner twice on a
            # narrow, steeply slanted printed box. These are upright digit
            # boxes: order the top and bottom pairs explicitly instead.
            by_y = points[np.argsort(points[:, 1])]
            top = by_y[:2][np.argsort(by_y[:2, 0])]
            bottom = by_y[2:][np.argsort(by_y[2:, 0])]
            corners = np.float32([top[0], top[1], bottom[1], bottom[0]])
            if len(np.unique(corners, axis=0)) != 4:
                continue
            width = max(np.linalg.norm(corners[1]-corners[0]), np.linalg.norm(corners[2]-corners[3]))
            height = max(np.linalg.norm(corners[3]-corners[0]), np.linalg.norm(corners[2]-corners[1]))
            if not .2 < width/max(height, 1) < 1.2:
                continue
            target_h = max(64, int(round(height)))
            target_w = max(24, int(round(target_h*width/height)))
            target = np.float32([[0, 0], [target_w-1, 0],
                                 [target_w-1, target_h-1], [0, target_h-1]])
            rectified = cv2.warpPerspective(image, cv2.getPerspectiveTransform(corners, target),
                                           (target_w, target_h), flags=cv2.INTER_CUBIC)
            dx, dy = max(2, round(target_w*.15)), max(2, round(target_h*.08))
            return rectified[dy:-dy, dx:-dx].copy()
    return None


def vertical_glyph_column(boxes, height):
    """Select one observed 11-glyph slanted column, excluding adjacent ISO text.

    Require a unique full-height column. Do not fill missing glyphs or use the
    checksum to choose a column. All coordinates refer to measured components.
    """
    if len(boxes) < 11 or len(boxes) > 80:
        return None
    candidates = set()
    for a in boxes:
        ax, ay = a[0]+a[2]/2, a[1]+a[3]/2
        for b in boxes:
            bx, by = b[0]+b[2]/2, b[1]+b[3]/2
            if by-ay < .6*height:
                continue
            slope = (bx-ax)/(by-ay)
            if abs(slope) > .5:
                continue
            column = sorted((z for z in boxes if abs(z[0]+z[2]/2 -
                             (ax+slope*(z[1]+z[3]/2-ay))) < max(4, z[2]*.5)), key=lambda z:z[1])
            if len(column) != 11:
                continue
            if column[-1][1]+column[-1][3]-column[0][1] < .7*height:
                continue
            if any(b[1] < a[1]+a[3]*.7 for a,b in zip(column, column[1:])):
                continue
            candidates.add(tuple(column))
    return list(next(iter(candidates))) if len(candidates) == 1 else None


def horizontal_number_layout(boxes, shape):
    """Find all 11 printed components on one baseline, excluding a door bar.

    Only a tall component spanning the entire crop can be excluded as a bar.
    Missing letters/digits and multiple text rows are never reconstructed.
    """
    import numpy as np
    h, w = shape[:2]
    if not boxes:
        return None
    median = float(np.median([b[3] for b in boxes]))
    glyphs = sorted(b for b in boxes if b[3] >= .65*median and not
                    (b[1] == 0 and b[1]+b[3] >= h and b[3] > 1.3*median))
    if len(glyphs) != 11 or glyphs[-1][0]+glyphs[-1][2]-glyphs[0][0] < .6*w:
        return None
    centers = np.array([(x+bw/2,y+bh/2) for x,y,bw,bh in glyphs])
    if np.ptp(centers[:,0]) < 1:
        return None
    slope, intercept = np.polyfit(centers[:,0],centers[:,1],1)
    if abs(slope) > .5 or np.max(np.abs(centers[:,1]-slope*centers[:,0]-intercept)) > .5*median:
        return None
    if any(b[0] < a[0]+a[2]*.7 for a,b in zip(glyphs,glyphs[1:])):
        return None
    owner_width = glyphs[3][0]+glyphs[3][2]-glyphs[0][0]
    if not .15*w <= owner_width <= .55*w:
        return None
    return glyphs[:4], glyphs[4:10], glyphs[10:]
