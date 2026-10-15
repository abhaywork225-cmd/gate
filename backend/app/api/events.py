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
           status: Literal['CONFIRMED', 'NEEDS_REVIEW', 'CAPTURED', 'UNREADABLE'] | None = None):
    return request.app.state.repository.events(limit, offset, container, camera_id, event_type, status)


@router.get('/gate-events/{event_id}')
def event(event_id: str, request: Request):
    result = request.app.state.repository.event(event_id)
    if result is None:
        raise HTTPException(404, 'Event not found')
    retained_proofs = list(result.get('snapshots',[]))
    result['evidence_manifest'] = request.app.state.snapshots.build_evidence_manifest(
        result, min_quality=getattr(request.app.state.runtime.settings, 'cloud_min_proof_quality', 2.5)
    )
    result['snapshots'] = result['evidence_manifest']['captures']
    result['primary_snapshot_id'] = result['evidence_manifest']['primary_snapshot_id']
    result['rear_snapshot_id'] = result['evidence_manifest']['rear_snapshot_id']
    if result.get('capture_policy') in ('FINAL_CAPTURE','UNREADABLE_CAPTURE'):
        # Final captures already own their source selection. Unreadable best
        # proof must not disappear through a readable-number quality filter.
        result['snapshots'] = retained_proofs
    result['ocr_crops'] = request.app.state.runtime.ocr_store.event_crops(event_id)
    dispatcher = getattr(request.app.state.runtime, 'dispatcher', None)
    result['cloud_auto_dispatch'] = getattr(dispatcher, 'auto_dispatch', False) if dispatcher else False
    # Preview uses the upload builder itself: no queueing, login or network calls.
    result['cloud_proof_images'] = []
    result['cloud_proof_error'] = None
    result['cloud_proof_coverage'] = []
    result['cloud_proof_complete'] = False
    if dispatcher:
        try:
            payload = dispatcher.build_proof_preview(event_id)
            for container in payload.get('Containers', []):
                cameras = sorted({image.get('CameraId') for image in container.get('Images',[]) if image.get('CameraId')})
                required = request.app.state.runtime.settings.minimum_number_proof_cameras if container.get('ContainerNumber') else 0
                result['cloud_proof_coverage'].append({'container_number':container.get('ContainerNumber'),
                    'cameras':cameras,'required':required,'complete':len(cameras)>=required})
                for image in container.get('Images', []):
                    result['cloud_proof_images'].append({
                        'container_number': container.get('ContainerNumber'),
                        'position': container.get('Position'),
                        'camera_id': image.get('CameraId'),
                        'image_type': image.get('ImageType'),
                        'captured_at': image.get('CapturedAt'),
                        'image': image.get('Image'),
                    })
            result['cloud_proof_complete'] = bool(result['cloud_proof_coverage']) and all(c['complete'] for c in result['cloud_proof_coverage'])
        except (ValueError, OSError) as exc:
            result['cloud_proof_error'] = str(exc)
    else:
        result['cloud_proof_error'] = 'Cloud proof preview is unavailable.'
    return result


@router.post('/gate-events/{event_id}/dispatch-cloud')
def dispatch_cloud_event(event_id: str, request: Request):
    runtime = request.app.state.runtime
    if not hasattr(runtime, 'dispatcher') or not runtime.dispatcher:
        raise HTTPException(503, 'Cloud dispatcher is not initialized')
    result = runtime.dispatcher.dispatch_event(event_id)
    if not result.get('success'):
        raise HTTPException(400, detail=result.get('error', 'Cloud dispatch failed'))
    return result


@router.get('/gate-events/{event_id}/cloud-logs')
def event_cloud_logs(event_id: str, request: Request):
    repo = request.app.state.repository
    if hasattr(repo, 'get_cloud_dispatch_logs'):
        return repo.get_cloud_dispatch_logs(event_id=event_id)
    return []


@router.get('/cloud-dispatch-logs')
def cloud_dispatch_logs(request: Request, event_id: str | None = None, limit: int = Query(50, ge=1, le=200)):
    repo = request.app.state.repository
    if hasattr(repo, 'get_cloud_dispatch_logs'):
        return repo.get_cloud_dispatch_logs(event_id=event_id, limit=limit)
    return []


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
    visit_id = result.get('visit_id')
    relative = f'{visit_id}/event_{event_id}.json' if visit_id else f'{event_id}/event.json'
    json_path = request.app.state.snapshots.path(relative)
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
