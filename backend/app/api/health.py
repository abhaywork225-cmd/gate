from __future__ import annotations
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix='/api', tags=['system'])


@router.get('/health')
def health(request: Request):
    runtime = request.app.state.runtime
    try:
        request.app.state.database.check()
        database = 'READY'
    except Exception:
        database = 'UNAVAILABLE'
    cameras = runtime.camera_status()
    enabled = [c for c in cameras if c['enabled']]
    active = [c for c in enabled if c.get('active_source_id') or c.get('video') or c.get('status') == 'ONLINE']
    spool = runtime.ocr_store.status()
    ready = database == 'READY' and not request.app.state.startup_error and not runtime.last_processing_error and not runtime.ocr_spool_error and not spool['failed']
    if runtime.settings.pipeline_enabled:
        ready = ready and runtime.model_status['detector'] == 'READY' and runtime.model_status['ocr'] == 'READY'
        if active:
            ready = ready and all(c['status'] == 'ONLINE' or c['playback_status'] in ('READY', 'COMPLETED', 'PAUSED') for c in active)
    server_alive = database == 'READY' and not request.app.state.startup_error
    cloud_sync = runtime.dispatcher.status() if (hasattr(runtime, 'dispatcher') and runtime.dispatcher) else {
        'enabled': False, 'is_online': False, 'pending_count': 0, 'queue_depth': 0, 'last_sync_time': None, 'last_error': None
    }
    return JSONResponse({'status': 'READY' if ready else 'DEGRADED', 'database': database,
                         'pipeline_enabled': runtime.settings.pipeline_enabled,
                         'site': None, 'admin': None,
                         'max_upload_mb': runtime.settings.max_upload_mb,
                         'online_cameras': sum(c['status'] == 'ONLINE' for c in enabled),
                         'enabled_cameras': len(enabled), 'models': runtime.model_status,
                         'processing_error': runtime.last_processing_error,
                         'ocr_spool': spool, 'ocr_spool_error': runtime.ocr_spool_error,
                         'cloud_sync': cloud_sync,
                         'startup_error': request.app.state.startup_error}, status_code=200 if server_alive else 503)


@router.get('/cloud-sync/status')
def cloud_sync_status(request: Request):
    runtime = request.app.state.runtime
    if hasattr(runtime, 'dispatcher') and runtime.dispatcher:
        return runtime.dispatcher.status()
    return {'enabled': False, 'is_online': False, 'pending_count': 0, 'queue_depth': 0, 'last_sync_time': None, 'last_error': None}


@router.post('/cloud-sync/retry')
def retry_cloud_sync(request: Request):
    runtime = request.app.state.runtime
    if hasattr(runtime, 'dispatcher') and runtime.dispatcher:
        pending = runtime.dispatcher.trigger_retry()
        return {'triggered': True, 'pending': pending}
    return {'triggered': False, 'pending': 0}


@router.get('/models')
def models(request: Request):
    runtime = request.app.state.runtime
    return {**runtime.model_status, 'backend': runtime.settings.model_backend,
            'model_file': runtime.settings.model_path.name, 'ocr_engine': runtime.settings.ocr_engine,
            'target_classes': runtime.settings.target_classes}


@router.get('/metrics')
def metrics(request: Request):
    runtime = request.app.state.runtime
    return {**runtime.metrics.snapshot(), 'cameras': [w.status() for w in runtime.manager.workers.values()],
            'ocr_queue_depth': runtime.ocr_store.status()['pending'],
            'ocr_save_queue_depth': runtime.queue.qsize(), 'ocr_spool': runtime.ocr_store.status()}
