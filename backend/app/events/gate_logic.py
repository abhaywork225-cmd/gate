from __future__ import annotations
def update_direction(camera, track, image_shape):
    """No camera-name guessing. Crossing geometry is calibrated per viewpoint."""
    if camera.line_axis is None:
        track.direction = camera.direction
        return track.direction
    if track.direction != 'UNKNOWN':
        return track.direction
    x1, y1, x2, y2 = track.detection.bbox
    height, width = image_shape[:2]
    position = (x1 + x2) / (2 * width) if camera.line_axis == 'x' else (y1 + y2) / (2 * height)
    delta = position - camera.line_position
    side = 1 if delta > camera.line_deadband else -1 if delta < -camera.line_deadband else 0
    if side and track.line_side and side != track.line_side:
        positive = track.line_side == -1 and side == 1
        track.direction = camera.positive_crossing if positive else ('EXIT' if camera.positive_crossing == 'ENTRY' else 'ENTRY')
    if side:
        track.line_side = side
    return track.direction
