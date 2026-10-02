from __future__ import annotations
"""Motorized lens controller for Dahua IP cameras with motorized varifocal lenses.

Supports optical zoom (Tele/Wide), focus adjustment (Far/Near), and one-click AutoFocus.
Communicates directly with the camera's secure HTTPS JSON-RPC endpoint (devVideoInput).
"""
import hashlib
from http.cookiejar import CookieJar
import json
import logging
import ssl
import threading
import time
import urllib.parse
import urllib.request

log = logging.getLogger('gate')


class DahuaLensClient:
    def __init__(self, host: str, username: str = "admin", password: str = "Prosper@123"):
        self.host = host
        self.username = username
        self.password = password
        self.base_url = f"https://{host}"
        self.session = None
        self.vi_object = None
        self._lock = threading.RLock()
        self.ctx = ssl._create_unverified_context()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=self.ctx),
            urllib.request.HTTPCookieProcessor(CookieJar())
        )
        self._req_id = 1

    def _rpc(self, path: str, payload: dict, timeout: float = 3.0) -> dict:
        body = json.dumps(payload)
        etag = hashlib.sha256(body.encode('utf-8')).hexdigest()
        headers = {
            'Content-Type': 'application/json',
            'ETag': etag,
            'X-Requested-With': 'XMLHttpRequest'
        }
        if self.session:
            headers['Cookie'] = f'DWebClientSessionID={self.session}; WebClientSessionID={self.session}'
        req = urllib.request.Request(f"{self.base_url}{path}", data=body.encode('utf-8'), headers=headers)
        with self.opener.open(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode('utf-8'))

    def login(self) -> bool:
        with self._lock:
            try:
                # Step 1: Challenge
                self._req_id += 1
                res1 = self._rpc('/cpapi2_Login', {
                    "method": "user.signin",
                    "params": {"userName": self.username, "password": "", "clientType": "Web3.0"},
                    "id": self._req_id
                })
                challenge_sess = res1.get('session')
                realm = res1.get('params', {}).get('realm', '')
                random = res1.get('params', {}).get('random', '')
                if not realm or not random:
                    log.warning("dahua_lens_login_failed", extra={"reason": "missing_realm_or_random"})
                    return False

                # Step 2: Digest auth
                ha1 = hashlib.md5(f"{self.username}:{realm}:{self.password}".encode()).hexdigest().upper()
                pwd_hash = hashlib.md5(f"{self.username}:{random}:{ha1}".encode()).hexdigest().upper()
                self._req_id += 1
                res2 = self._rpc('/cpapi2_Login', {
                    "method": "user.signin",
                    "params": {"userName": self.username, "password": pwd_hash, "clientType": "Web3.0"},
                    "id": self._req_id,
                    "session": challenge_sess
                })
                if not res2.get('result'):
                    log.warning("dahua_lens_login_rejected", extra={"host": self.host})
                    return False
                self.session = res2.get('session', challenge_sess)

                # Step 3: devVideoInput instance
                self._req_id += 1
                vi = self._rpc('/cpapi2', {
                    "method": "devVideoInput.factory.instance",
                    "params": {"channel": 0},
                    "id": self._req_id,
                    "session": self.session
                })
                self.vi_object = vi.get('result')
                return self.vi_object is not None
            except Exception as e:
                log.error("dahua_lens_connection_error", extra={"error": str(e), "host": self.host})
                return False

    def _ensure_connected(self) -> bool:
        if self.session and self.vi_object is not None:
            return True
        return self.login()

    def get_status(self) -> dict:
        """Fetch current motorized lens position, zoom, and focus steps."""
        with self._lock:
            if not self._ensure_connected():
                return {"supported": False, "error": "Camera offline or credentials rejected"}
            try:
                self._req_id += 1
                res = self._rpc('/cpapi2', {
                    "method": "devVideoInput.getFocusStatus",
                    "params": None,
                    "id": self._req_id,
                    "session": self.session,
                    "object": self.vi_object
                })
                if not res.get('result'):
                    # Session may have expired, retry once
                    self.session = None
                    if not self.login():
                        return {"supported": False, "error": "Session expired"}
                    self._req_id += 1
                    res = self._rpc('/cpapi2', {
                        "method": "devVideoInput.getFocusStatus",
                        "params": None,
                        "id": self._req_id,
                        "session": self.session,
                        "object": self.vi_object
                    })
                status = res.get('params', {}).get('status', {})
                return {
                    "supported": True,
                    "status": status.get('Status', 'Normal'),
                    "zoom": status.get('Zoom', 0.0),
                    "focus": status.get('Focus', 0.0),
                    "zoom_steps": status.get('ZoomMotorSteps', 0),
                    "focus_steps": status.get('FocusMotorSteps', 0),
                }
            except Exception as e:
                return {"supported": False, "error": str(e)}

    def autofocus(self) -> dict:
        """Trigger camera optical auto-focus routine."""
        with self._lock:
            if not self._ensure_connected():
                return {"success": False, "error": "Camera offline"}
            try:
                self._req_id += 1
                res = self._rpc('/cpapi2', {
                    "method": "devVideoInput.autoFocus",
                    "params": None,
                    "id": self._req_id,
                    "session": self.session,
                    "object": self.vi_object
                })
                return {"success": bool(res.get('result'))}
            except Exception as e:
                return {"success": False, "error": str(e)}

    def stop_autofocus(self) -> dict:
        """Stop auto-focus routine."""
        with self._lock:
            if not self._ensure_connected():
                return {"success": False, "error": "Camera offline"}
            try:
                self._req_id += 1
                res = self._rpc('/cpapi2', {
                    "method": "devVideoInput.stopAutoFocus",
                    "params": None,
                    "id": self._req_id,
                    "session": self.session,
                    "object": self.vi_object
                })
                return {"success": bool(res.get('result'))}
            except Exception as e:
                return {"success": False, "error": str(e)}

    def zoom(self, action: str) -> dict:
        """
        action: 'in' | 'out' | 'stop' | 'step_in' | 'step_out'
        """
        with self._lock:
            if not self._ensure_connected():
                return {"success": False, "error": "Camera offline"}
            try:
                if action == 'step_in':
                    self._req_id += 1
                    self._rpc('/cpapi2', {
                        "method": "devVideoInput.adjustFocusContinuously",
                        "params": {"focus": -1, "zoom": 0.05},
                        "id": self._req_id, "session": self.session, "object": self.vi_object
                    })
                    time.sleep(0.35)
                    self._req_id += 1
                    res = self._rpc('/cpapi2', {
                        "method": "devVideoInput.adjustFocusContinuously",
                        "params": {"focus": -1, "zoom": 0},
                        "id": self._req_id, "session": self.session, "object": self.vi_object
                    })
                    return {"success": bool(res.get('result')), "action": action}
                elif action == 'step_out':
                    self._req_id += 1
                    self._rpc('/cpapi2', {
                        "method": "devVideoInput.adjustFocusContinuously",
                        "params": {"focus": -1, "zoom": -0.05},
                        "id": self._req_id, "session": self.session, "object": self.vi_object
                    })
                    time.sleep(0.35)
                    self._req_id += 1
                    res = self._rpc('/cpapi2', {
                        "method": "devVideoInput.adjustFocusContinuously",
                        "params": {"focus": -1, "zoom": 0},
                        "id": self._req_id, "session": self.session, "object": self.vi_object
                    })
                    return {"success": bool(res.get('result')), "action": action}
                elif action == 'in':
                    zoom_val = 0.05
                elif action == 'out':
                    zoom_val = -0.05
                else:  # 'stop'
                    zoom_val = 0

                self._req_id += 1
                res = self._rpc('/cpapi2', {
                    "method": "devVideoInput.adjustFocusContinuously",
                    "params": {"focus": -1, "zoom": zoom_val},
                    "id": self._req_id,
                    "session": self.session,
                    "object": self.vi_object
                })
                return {"success": bool(res.get('result')), "action": action}
            except Exception as e:
                return {"success": False, "error": str(e)}

    def focus(self, action: str) -> dict:
        """
        action: 'far' | 'near' | 'stop' | 'step_far' | 'step_near'
        """
        with self._lock:
            if not self._ensure_connected():
                return {"success": False, "error": "Camera offline"}
            try:
                if action == 'step_far':
                    self._req_id += 1
                    self._rpc('/cpapi2', {
                        "method": "devVideoInput.adjustFocusContinuously",
                        "params": {"focus": 0.03, "zoom": -1},
                        "id": self._req_id, "session": self.session, "object": self.vi_object
                    })
                    time.sleep(0.3)
                    self._req_id += 1
                    res = self._rpc('/cpapi2', {
                        "method": "devVideoInput.adjustFocusContinuously",
                        "params": {"focus": 0, "zoom": -1},
                        "id": self._req_id, "session": self.session, "object": self.vi_object
                    })
                    return {"success": bool(res.get('result')), "action": action}
                elif action == 'step_near':
                    self._req_id += 1
                    self._rpc('/cpapi2', {
                        "method": "devVideoInput.adjustFocusContinuously",
                        "params": {"focus": -0.03, "zoom": -1},
                        "id": self._req_id, "session": self.session, "object": self.vi_object
                    })
                    time.sleep(0.3)
                    self._req_id += 1
                    res = self._rpc('/cpapi2', {
                        "method": "devVideoInput.adjustFocusContinuously",
                        "params": {"focus": 0, "zoom": -1},
                        "id": self._req_id, "session": self.session, "object": self.vi_object
                    })
                    return {"success": bool(res.get('result')), "action": action}
                elif action == 'far':
                    focus_val = 0.03
                elif action == 'near':
                    focus_val = -0.03
                else:  # 'stop'
                    focus_val = 0

                self._req_id += 1
                res = self._rpc('/cpapi2', {
                    "method": "devVideoInput.adjustFocusContinuously",
                    "params": {"focus": focus_val, "zoom": -1},
                    "id": self._req_id,
                    "session": self.session,
                    "object": self.vi_object
                })
                return {"success": bool(res.get('result')), "action": action}
            except Exception as e:
                return {"success": False, "error": str(e)}

    SHUTTER_MAP = {
        '1/1000s': 1,
        '1/1000': 1,
        '1/500s': 2,
        '1/500': 2,
        '1/250s': 4,
        '1/250': 4,
        '1/100s': 10,
        '1/100': 10,
        '1/50s': 20,
        '1/50': 20,
        '1/25s': 40,
        '1/25': 40,
        'auto': 40,
    }
    INV_SHUTTER_MAP = {
        1: '1/1000s',
        2: '1/500s',
        4: '1/250s',
        10: '1/100s',
        20: '1/50s',
        40: '1/25s',
    }

    def get_exposure(self) -> dict:
        """Fetch current exposure mode and shutter speed configuration."""
        with self._lock:
            if not self._ensure_connected():
                return {"supported": False, "error": "Camera offline"}
            try:
                self._req_id += 1
                res = self._rpc('/cpapi2', {
                    "method": "configManager.getConfig",
                    "params": {"name": "VideoInExposure[0]"},
                    "id": self._req_id,
                    "session": self.session
                })
                if not res.get('result'):
                    return {"supported": False, "error": "Failed to get exposure config"}
                table = res.get('params', {}).get('table', [])
                if not table:
                    return {"supported": False, "error": "Empty exposure config"}
                first = table[0]
                val2 = first.get('Value2', 40)
                mode_num = first.get('Mode', 0)
                shutter_label = self.INV_SHUTTER_MAP.get(val2, f"{val2}ms")
                return {
                    "supported": True,
                    "mode": "manual" if mode_num == 1 else "auto",
                    "shutter_speed": shutter_label,
                    "shutter_ms": val2,
                    "gain_max": first.get('GainMax', 50),
                    "compensation": first.get('Compensation', 50)
                }
            except Exception as e:
                return {"supported": False, "error": str(e)}

    def set_shutter_speed(self, shutter: str, mode: str = "auto") -> dict:
        """
        Configure electronic shutter speed and exposure mode.
        shutter: '1/1000s' | '1/500s' | '1/250s' | '1/100s' | '1/50s' | '1/25s' | 'auto'
        mode: 'auto' (auto-exposure with shutter speed ceiling) | 'manual' (fixed shutter speed)
        """
        ms = self.SHUTTER_MAP.get(shutter.lower())
        if ms is None:
            return {"success": False, "error": f"Invalid shutter speed '{shutter}'. Supported: {list(self.INV_SHUTTER_MAP.values())}"}
        mode_val = 1 if mode == "manual" else 0
        with self._lock:
            if not self._ensure_connected():
                return {"success": False, "error": "Camera offline"}
            try:
                self._req_id += 1
                res = self._rpc('/cpapi2', {
                    "method": "configManager.getConfig",
                    "params": {"name": "VideoInExposure[0]"},
                    "id": self._req_id,
                    "session": self.session
                })
                table = res.get('params', {}).get('table', [])
                if not table:
                    return {"success": False, "error": "Could not read exposure table"}
                for profile in table:
                    profile['Mode'] = mode_val
                    profile['Value2'] = ms
                    if mode_val == 1:
                        profile['Value1'] = ms
                    else:
                        profile['Value1'] = 0
                self._req_id += 1
                set_res = self._rpc('/cpapi2', {
                    "method": "configManager.setConfig",
                    "params": {"name": "VideoInExposure[0]", "table": table},
                    "id": self._req_id,
                    "session": self.session
                })
                ok = bool(set_res.get('result'))
                return {
                    "success": ok,
                    "shutter_speed": self.INV_SHUTTER_MAP.get(ms, f"{ms}ms"),
                    "shutter_ms": ms,
                    "mode": mode
                }
            except Exception as e:
                return {"success": False, "error": str(e)}


# Global cache of DahuaLensClients per host
_lens_clients: dict[str, DahuaLensClient] = {}
_cache_lock = threading.Lock()


def get_lens_client_for_camera(runtime, camera_id: str) -> DahuaLensClient | None:
    """Resolve camera IP/credentials and retrieve or initialize a DahuaLensClient."""
    worker = runtime.manager.workers.get(camera_id)
    if not worker or not worker._source:
        return None
    source = worker._source
    parsed = urllib.parse.urlsplit(source)
    host = parsed.hostname
    if not host:
        return None

    username = urllib.parse.unquote(parsed.username) if parsed.username else (runtime.settings.camera_onvif_username or "admin")
    password = urllib.parse.unquote(parsed.password) if parsed.password else None
    if not password:
        pw_secret = runtime.settings.camera_onvif_password.get_secret_value()
        password = pw_secret or ("Prosper@123" if "203" in host else "Admin@123")

    with _cache_lock:
        client = _lens_clients.get(host)
        if client is None or client.password != password:
            client = DahuaLensClient(host, username=username, password=password)
            _lens_clients[host] = client
        return client
