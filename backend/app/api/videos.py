from __future__ import annotations
import asyncio
from pathlib import Path
from typing import Literal
from urllib.parse import unquote
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from app.api.cameras import worker

router = APIRouter(prefix='/api', tags=['video inputs'])
EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}


@router.post('/cameras/{camera_id}/video')
async def upload_video(camera_id: str, request: Request, sync_lane: bool = False):
    worker(request, camera_id)
    runtime = request.app.state.runtime
    cam_cfg = next((c for c in runtime.cameras if c.id == camera_id), None)
    filename = unquote(request.headers.get('x-filename', 'video.mp4')).replace('\\', '/').split('/')[-1][:200]
    extension = Path(filename).suffix.lower()
    if extension not in EXTENSIONS:
        raise HTTPException(415, 'Choose an MP4, AVI, MOV, MKV, WEBM or M4V video')
    maximum = runtime.settings.max_upload_mb * 1024 * 1024
    try:
        declared = int(request.headers.get('content-length', '0'))
    except ValueError:
        raise HTTPException(400, 'Invalid upload length') from None
    if declared > maximum:
        raise HTTPException(413, f'Video exceeds the {runtime.settings.max_upload_mb} MB upload limit')
    stored_name = uuid4().hex + extension
    path = runtime.videos.path(stored_name)
    total, installed = 0, False
    try:
        with path.open('xb') as file:
            async for chunk in request.stream():
                total += len(chunk)
                if total > maximum:
                    raise HTTPException(413, f'Video exceeds the {runtime.settings.max_upload_mb} MB upload limit')
                await asyncio.to_thread(file.write, chunk)
        if total == 0:
            raise HTTPException(400, 'The uploaded file is empty')
        try:
            metadata = await asyncio.to_thread(runtime.videos.validate, path)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        info = {'filename': filename, 'stored_name': stored_name, 'size_bytes': total, **metadata, 'active': True}
        try:
            if sync_lane and cam_cfg:
                await asyncio.to_thread(runtime.activate_lane_video, cam_cfg.gate_id, info)
            else:
                await asyncio.to_thread(runtime.activate_video, camera_id, info)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from None
        installed = True
        return {'camera_id': camera_id, 'video': {k: v for k, v in info.items() if k != 'stored_name'},
                'sync_lane': sync_lane,
                'message': 'Video ready. Press Play to start.'}
    finally:
        if not installed:
            path.unlink(missing_ok=True)


class Playback(BaseModel):
    action: Literal['play', 'pause', 'restart', 'restore-camera']


@router.post('/cameras/{camera_id}/playback')
def playback(camera_id: str, body: Playback, request: Request):
    worker(request, camera_id)
    try:
        request.app.state.runtime.control_video(camera_id, body.action)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from None
    return {'camera_id': camera_id, 'status': worker(request, camera_id).status()}


@router.post('/lanes/{gate_id}/playback')
def lane_playback(gate_id: str, body: Playback, request: Request):
    runtime = request.app.state.runtime
    try:
        runtime.control_lane_video(gate_id, body.action)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from None
    statuses = [w.status() for w in runtime.manager.workers.values() if w.config.gate_id == gate_id]
    return {'gate_id': gate_id, 'action': body.action, 'statuses': statuses}


@router.post('/videos/play-all')
def play_all(request: Request):
    runtime = request.app.state.runtime
    results = []
    for camera_id, item in list(runtime.manager.workers.items()):
        if item.config.source_type == 'file':
            try:
                runtime.control_video(camera_id, 'play')
                results.append({'camera_id': camera_id, 'success': True})
            except (ValueError, RuntimeError):
                results.append({'camera_id': camera_id, 'success': False})
    return {'results': results}


@router.post('/processing/start')
def start_processing(request: Request):
    runtime = request.app.state.runtime
    try:
        request.app.state.database.check()
        runtime.repository.sync_cameras(runtime.cameras)
    except Exception:
        raise HTTPException(503, 'Configure and initialize MySQL before enabling event processing. Video playback is available.') from None
    runtime.start_processing()
    request.app.state.startup_error = None
    return {'models': runtime.model_status}


@router.post('/data/clear')
def clear_all_data(request: Request):
    import shutil
    from sqlalchemy import text
    runtime = request.app.state.runtime
    for cam in runtime.cameras:
        try:
            runtime.control_video(cam.id, 'restore-camera')
        except Exception:
            pass
    try:
        with runtime.repository.database.sessions.begin() as session:
            for t in ['system_logs', 'ocr_results', 'snapshots', 'detections', 'gate_events']:
                try:
                    session.execute(text(f'DELETE FROM {t}'))
                except Exception:
                    pass
        runtime.repository.sync_cameras(runtime.cameras)
    except Exception:
        pass
    snap_dir = Path(runtime.settings.snapshot_directory)
    if snap_dir.exists():
        for f in snap_dir.glob('*'):
            if f.is_file() and f.name != '.gitkeep':
                f.unlink(missing_ok=True)
            elif f.is_dir():
                shutil.rmtree(f, ignore_errors=True)
        (snap_dir / '.gitkeep').touch(exist_ok=True)
    upload_dir = Path(runtime.settings.upload_directory)
    if upload_dir.exists():
        for f in upload_dir.glob('*'):
            if f.is_file() and f.name != 'assignments.json':
                f.unlink(missing_ok=True)
    assign_file = upload_dir / 'assignments.json'
    if assign_file.exists():
        assign_file.write_text('{}', encoding='utf-8')
    spool_dir = upload_dir / 'ocr-spool'
    if spool_dir.exists():
        shutil.rmtree(spool_dir, ignore_errors=True)
        spool_dir.mkdir(parents=True, exist_ok=True)
    return {'status': 'cleared', 'message': 'All events, snapshots, spool, and uploaded videos have been removed.'}

