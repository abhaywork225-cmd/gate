from __future__ import annotations
from fastapi import APIRouter, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from app.config.settings import LaneRegion
from typing import Literal

router = APIRouter(prefix='/api/cameras', tags=['cameras'])


def live_jpeg(runtime, camera_id):
    cam = runtime.manager.workers.get(camera_id)
    if cam is None:
        return None
    if hasattr(cam, 'get_jpeg'):
        return cam.get_jpeg()
    frame = cam.latest()
    if frame is None or frame.image is None:
        return None
    import cv2
    img = frame.image
    if img.shape[1] > 640:
        img = cv2.resize(img, (640, round(img.shape[0] * 640 / img.shape[1])))
    ok, jpeg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return jpeg.tobytes() if ok else None


@router.websocket('/{camera_id}/live')
async def live(socket: WebSocket, camera_id: str):
    import asyncio
    runtime = socket.app.state.runtime
    origin = socket.headers.get('origin')
    own_origin = ('https' if socket.url.scheme == 'wss' else 'http') + '://' + socket.url.netloc
    if origin not in runtime.settings.cors_origins + [own_origin] or camera_id not in runtime.manager.workers:
        await socket.close(code=1008)
        return
    await socket.accept()
    try:
        while not runtime._stop.is_set():
            if await socket.receive_text() != 'next':
                await socket.close(code=1008)
                return
            # Request/response backpressure: no per-client queue of old video frames.
            jpeg = await asyncio.to_thread(live_jpeg, runtime, camera_id)
            if jpeg is None:
                await socket.send_text('waiting')
            else:
                await socket.send_bytes(jpeg)
            await asyncio.sleep(.01)
    except WebSocketDisconnect:
        pass


def worker(request, camera_id):
    result = request.app.state.runtime.manager.workers.get(camera_id)
    if result is None:
        raise HTTPException(404, 'Camera not found')
    return result


@router.get('')
def cameras(request: Request):
    return request.app.state.runtime.camera_status()


@router.get('/sources/available')
def sources(request: Request):
    return request.app.state.runtime.available_camera_sources()


class SourceSelection(BaseModel):
    source_id: str


class LaneCalibration(BaseModel):
    lanes: list[LaneRegion] = Field(default_factory=list, max_length=2)
    occupancy_roi: list[float]
    entry_line: float = 0.10
    exit_line: float = 0.95
    gate_exit_axis: Literal['x', 'y'] | None = None
    gate_exit_positive: bool = True


@router.get('/calibration/all')
def all_calibration(request: Request):
    runtime = request.app.state.runtime
    return {camera.id: runtime.calibration(camera.id) for camera in runtime.cameras}


@router.get('/{camera_id}/calibration')
def camera_calibration(camera_id: str, request: Request):
    worker(request, camera_id)
    try:
        return request.app.state.runtime.calibration(camera_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from None


@router.post('/{camera_id}/calibration')
def save_camera_calibration(camera_id: str, body: LaneCalibration, request: Request):
    worker(request, camera_id)
    try:
        return request.app.state.runtime.save_calibration(camera_id, body.model_dump())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None


@router.get('/discovery/status')
def discovery_status(request: Request):
    return request.app.state.runtime.discovery_status


@router.post('/discovery/rescan')
def rescan(request: Request):
    runtime = request.app.state.runtime
    runtime._discovery_wake.set()
    return runtime.discovery_status


class CameraRole(BaseModel):
    role: Literal['UNASSIGNED', 'FRONT_TOP', 'LEFT', 'RIGHT', 'REAR']
    direction: Literal['UNKNOWN', 'ENTRY', 'EXIT']


@router.post('/{camera_id}/role')
def role(camera_id: str, body: CameraRole, request: Request):
    worker(request, camera_id)
    request.app.state.runtime.configure_role(camera_id, body.role, body.direction)
    return {'camera_id': camera_id, 'role': body.role, 'direction': body.direction}


@router.post('/{camera_id}/source')
def select_source(camera_id: str, body: SourceSelection, request: Request):
    worker(request, camera_id)
    try:
        request.app.state.runtime.select_camera_source(camera_id, body.source_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from None
    return {'camera_id': camera_id, 'status': worker(request, camera_id).status()}


@router.get('/{camera_id}/status')
def status(camera_id: str, request: Request):
    return worker(request, camera_id).status()


@router.post('/{camera_id}/test')
def test_camera(camera_id: str, request: Request):
    return worker(request, camera_id).test()


@router.get('/{camera_id}/frame')
def preview(camera_id: str, request: Request):
    frame = worker(request, camera_id).latest()
    if frame is None:
        raise HTTPException(404, 'No frame available')
    import cv2
    image = frame.image
    if image.shape[1] > 640:
        image = cv2.resize(image, (640, round(image.shape[0] * 640 / image.shape[1])))
    ok, jpeg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        raise HTTPException(503, 'Preview unavailable')
    return Response(jpeg.tobytes(), media_type='image/jpeg', headers={'Cache-Control': 'no-store'})


@router.get('/{camera_id}/stream')
def stream_camera(camera_id: str, request: Request):
    worker(request, camera_id)
    runtime = request.app.state.runtime

    def frame_generator():
        import cv2, time
        while True:
            if runtime._stop.is_set():
                return
            cam = worker(request, camera_id)
            frame = cam.latest()
            if frame is not None and frame.image is not None:
                img = frame.image
                if img.shape[1] > 640:
                    img = cv2.resize(img, (640, round(img.shape[0] * 640 / img.shape[1])))
                ok, jpeg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.04)  # ~25 FPS smooth playback

    return StreamingResponse(frame_generator(), media_type='multipart/x-mixed-replace; boundary=frame')


class LensZoomAction(BaseModel):
    action: Literal['in', 'out', 'stop', 'step_in', 'step_out'] = 'step_in'


class LensFocusAction(BaseModel):
    action: Literal['far', 'near', 'stop', 'step_far', 'step_near'] = 'step_far'


@router.get('/{camera_id}/lens/status')
def lens_status(camera_id: str, request: Request):
    from app.camera.lens_control import get_lens_client_for_camera
    client = get_lens_client_for_camera(request.app.state.runtime, camera_id)
    if not client:
        return {'supported': False, 'message': 'Not an RTSP/IP camera'}
    return client.get_status()


@router.post('/{camera_id}/lens/zoom')
def lens_zoom(camera_id: str, body: LensZoomAction, request: Request):
    from app.camera.lens_control import get_lens_client_for_camera
    client = get_lens_client_for_camera(request.app.state.runtime, camera_id)
    if not client:
        raise HTTPException(400, 'Motorized lens control not supported for this camera')
    return client.zoom(body.action)


@router.post('/{camera_id}/lens/focus')
def lens_focus(camera_id: str, body: LensFocusAction, request: Request):
    from app.camera.lens_control import get_lens_client_for_camera
    client = get_lens_client_for_camera(request.app.state.runtime, camera_id)
    if not client:
        raise HTTPException(400, 'Motorized lens control not supported for this camera')
    return client.focus(body.action)


@router.post('/{camera_id}/lens/autofocus')
def lens_autofocus(camera_id: str, request: Request):
    from app.camera.lens_control import get_lens_client_for_camera
    client = get_lens_client_for_camera(request.app.state.runtime, camera_id)
    if not client:
        raise HTTPException(400, 'Motorized lens control not supported for this camera')
    return client.autofocus()


class ExposureSetting(BaseModel):
    shutter: Literal['1/1000s', '1/500s', '1/250s', '1/100s', '1/50s', '1/25s', 'auto'] = '1/500s'
    mode: Literal['auto', 'manual'] = 'auto'


@router.get('/{camera_id}/exposure')
def camera_exposure(camera_id: str, request: Request):
    from app.camera.lens_control import get_lens_client_for_camera
    client = get_lens_client_for_camera(request.app.state.runtime, camera_id)
    if not client:
        return {'supported': False, 'message': 'Not an IP camera with exposure API'}
    return client.get_exposure()


@router.post('/{camera_id}/exposure')
def set_camera_exposure(camera_id: str, body: ExposureSetting, request: Request):
    from app.camera.lens_control import get_lens_client_for_camera
    client = get_lens_client_for_camera(request.app.state.runtime, camera_id)
    if not client:
        raise HTTPException(400, 'Exposure control not supported for this camera')
    return client.set_shutter_speed(body.shutter, mode=body.mode)
