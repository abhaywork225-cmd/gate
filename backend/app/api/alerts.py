from __future__ import annotations
import time
import threading
from uuid import uuid4
from datetime import datetime, timezone
from typing import Optional, List
from pydantic import BaseModel
from fastapi import APIRouter, Request

router = APIRouter(prefix='/api/alerts', tags=['alerts'])


def trigger_hardware_buzzer(pattern: str = 'alert'):
    """Trigger audible alert buzzer via host hardware speaker (Windows winsound)."""
    def _beep():
        try:
            import winsound
            if pattern in ('gate_out', 'exit'):
                winsound.Beep(950, 220)
                time.sleep(0.08)
                winsound.Beep(1200, 320)
            elif pattern in ('cloud_error', 'conflict'):
                winsound.Beep(600, 300)
                time.sleep(0.08)
                winsound.Beep(600, 300)
            else:
                winsound.Beep(1000, 250)
        except Exception:
            pass

    threading.Thread(target=_beep, daemon=True).start()


class AlertModel(BaseModel):
    id: str
    alert_type: str
    level: str
    title: str
    message: str
    container_number: Optional[str] = None
    event_id: Optional[str] = None
    timestamp: str


@router.get('', response_model=List[AlertModel])
def get_alerts(request: Request):
    """Return active recent alerts for dashboard notification and buzzer."""
    runtime = getattr(request.app.state, 'runtime', None)
    if not runtime or not hasattr(runtime, 'alerts'):
        return []
    with runtime._state_lock:
        return list(runtime.alerts)


@router.post('/test')
def test_buzzer(pattern: str = 'gate_out'):
    """Trigger a test hardware buzzer sound on the host machine."""
    trigger_hardware_buzzer(pattern)
    return {'status': 'triggered', 'pattern': pattern}


@router.post('/clear')
def clear_alerts(request: Request):
    """Clear all active alerts."""
    runtime = getattr(request.app.state, 'runtime', None)
    if runtime and hasattr(runtime, 'alerts'):
        with runtime._state_lock:
            runtime.alerts.clear()
    return {'status': 'cleared'}
