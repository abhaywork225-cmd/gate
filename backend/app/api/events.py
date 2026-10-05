from __future__ import annotations
from typing import Literal
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from app.snapshots.proof import proof_jpeg

router = APIRouter(prefix='/api', tags=['events'])


@router.get('/gate-events')
def events(request: Request, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0),
           container: str | None = Query(None, max_length=11), camera_id: str | None = None,
           event_type: Literal['ENTRY', 'EXIT', 'UNKNOWN'] | None = None,
           status: Literal['CONFIRMED', 'NEEDS_REVIEW'] | None = None):
    return request.app.state.repository.events(limit, offset, container, camera_id, event_type, status)


@router.get('/gate-events/{event_id}')
def event(event_id: str, request: Request):
    result = request.app.state.repository.event(event_id)
    if result is None:
        raise HTTPException(404, 'Event not found')
    result['evidence_manifest'] = request.app.state.snapshots.build_evidence_manifest(
        result, min_quality=getattr(request.app.state.runtime.settings, 'cloud_min_proof_quality', 2.5)
    )
    result['snapshots'] = result['evidence_manifest']['captures']
    result['ocr_crops'] = request.app.state.runtime.ocr_store.event_crops(event_id)
    return result


@router.get('/ocr-crops/{job_id}')
def ocr_crop(job_id: int, request: Request):
    path = request.app.state.runtime.ocr_store.crop_path(job_id)
    if path is None:
        raise HTTPException(404, 'OCR crop unavailable')
    return FileResponse(path, media_type='image/png', headers={'Cache-Control': 'no-cache'})


@router.get('/gate-events/{event_id}/json')
def event_json(event_id: str, request: Request):
    result = request.app.state.repository.event(event_id)
    if result is None:
        raise HTTPException(404, 'Event not found')
    json_path = request.app.state.snapshots.root / event_id / 'event.json'
    if not json_path.is_file():
        json_path = request.app.state.snapshots.save_event_json(event_id, result)
    return FileResponse(json_path, media_type='application/json', filename=f'event_{event_id}.json')



@router.get('/ocr-results')
def ocr_results(request: Request, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    return request.app.state.repository.ocr_results(limit, offset)


@router.get('/snapshots/{snapshot_id}')
def snapshot(snapshot_id: str, request: Request, proof: bool = False):
    row = request.app.state.repository.snapshot(snapshot_id)
    if row is None:
        raise HTTPException(404, 'Snapshot not found')
    path = request.app.state.snapshots.path(row['image_path'])
    if not path.is_file():
        raise HTTPException(404, 'Snapshot file unavailable')
    if proof:
        event_record = request.app.state.repository.event(row['event_id'])
        data = proof_jpeg(path.read_bytes(), row, event_record or {})
        return Response(data, media_type='image/jpeg', headers={'Cache-Control': 'no-cache'})
    return FileResponse(path, media_type='image/jpeg')
