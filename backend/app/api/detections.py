from __future__ import annotations
from fastapi import APIRouter, Query, Request

router = APIRouter(prefix='/api/detections', tags=['detections'])


@router.get('')
def detections(request: Request, limit: int = Query(50, ge=1, le=200), camera_id: str | None = None):
    return request.app.state.runtime.recent_detections(limit, camera_id)
