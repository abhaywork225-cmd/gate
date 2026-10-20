import { inject, Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse, HttpHeaders, HttpParams, HttpRequest } from '@angular/common/http';
import { catchError, of, throwError } from 'rxjs';

export interface Camera {
  role: string;
  id: string; name: string; gate_id: string; direction: string; source_type: string; source_env: string;
  status: 'ONLINE' | 'OFFLINE' | 'RECONNECTING' | 'ERROR'; reason: string;
  capture_fps: number; dropped_frames: number; reconnect_count: number; last_frame_at: string | null;
  has_frame: boolean;
  playback_status: 'READY' | 'PLAYING' | 'PAUSED' | 'COMPLETED' | 'STOPPED' | null;
  video_position_seconds: number; video_duration_seconds: number | null;
  video: {filename: string; size_bytes: number; duration_seconds: number | null; active: boolean} | null;
  active_source_id: string | null; active_source_name: string | null;
  occupancy?: {state: string; state_machine?: {state: string; next: string[]}; stationary?: boolean; is_twin?: boolean; event_id?: string | null; slots?: Record<string, {closed: boolean; container_number?: string | null}>};
  calibration?: LaneCalibration;
  passage_tracking?: {state: string; passages: {event_id: string; lane_id: string; closed: boolean; candidates: number}[]};
}
export interface LaneRegion { id: string; polygon: number[][]; }
export interface LaneCalibration { camera_id: string; occupancy_roi: number[]; lanes?: LaneRegion[]; entry_line: number; exit_line: number; gate_exit_axis: 'x' | 'y' | null; gate_exit_positive: boolean; gate_clear_seconds?: number; version: number; }
export interface CameraSource { id: string; name: string; host?: string; configured: boolean; discovered?: boolean; connection_hint?: string; }
export interface DiscoveryStatus { state: string; found: number; message: string; }
export interface WaitingCandidate {
  container_number: string;
  votes: number;
  confidence: number;
  status: string;
}

export interface WaitingPassage {
  id: string;
  visit_id: string;
  gate_id: string;
  lane_id: string;
  camera_id: string;
  direction: string;
  started: string;
  last_seen: string;
  dwell_seconds: number;
  status: string;
  candidates: WaitingCandidate[];
  folder: string;
  folder_path: string;
}

export interface GateEvent {
  visit_id?: string | null;
  candidate_number?: string | null;
  candidate_is_partial?: boolean;
  capture_policy?: string | null;
  review_reason?: string | null;
  id: string;
  gate_id: string;
  timestamp: string;
  container_number: string | null;
  event_type: string;
  status: string;
  container_size?: string | null;
  size_code?: string | null;
  is_twin?: boolean;
  container_number_2?: string | null;
  size_code_2?: string | null;
  container_size_2?: string | null;
  cloud_synced?: boolean;
  cloud_synced_at?: string | null;
  cloud_event_id?: string | null;
  folder?: string;
  folder_path?: string;
}
export interface Detection { id?: string; camera_id: string; class_name: string; confidence: number; bbox: number[]; timestamp?: string; frame_timestamp?: string; track_id?: number; }
export interface OCR { id: string; raw_text: string; normalized_text: string; confidence: number; validation_status: string; }
export interface Snapshot { id: string; camera_id: string; timestamp: string; image_path?: string; }
export interface OcrCrop { id: string; camera_id?: string; timestamp?: string; slot_position: string; normalized_text?: string | null; }
export interface EventDetail extends GateEvent {
  cloud_proof_complete?: boolean;
  cloud_proof_coverage?: {container_number: string | null; cameras: string[]; required: number; complete: boolean}[];
  cloud_proof_images?: {container_number: string; position: string; camera_id: string; image_type: string; captured_at: string; image: string}[];
  cloud_proof_error?: string | null;
  detections: Detection[];
  ocr_results: OCR[];
  snapshots: Snapshot[];
  primary_camera_id?: string;
  primary_snapshot_id?: string;
  rear_snapshot_id?: string;
  ocr_crops?: OcrCrop[];
  cloud_auto_dispatch?: boolean;
}
export interface EventPage { items: GateEvent[]; total: number; offset: number; limit: number; }
export interface OcrSpool { total: number; pending: number; failed: number; }
export interface CloudSyncStatus {
  enabled: boolean;
  auto_dispatch?: boolean;
  is_paused?: boolean;
  is_online: boolean;
  pending_count: number;
  queue_depth: number;
  last_sync_time: string | null;
  last_error: string | null;
  site_id?: string | null;
}
export interface Health { status: string; database: string; pipeline_enabled: boolean; online_cameras: number; enabled_cameras: number; models: {detector: string; ocr: string; error_type?: string | null}; processing_error: string | null; startup_error: string | null; max_upload_mb: number; ocr_spool?: OcrSpool; ocr_spool_error?: string | null; cloud_sync?: CloudSyncStatus; }
export interface Metrics { cpu_percent: number; memory_percent: number; gpu_utilization_percent: number | null; ocr_queue_depth: number; ocr_save_queue_depth?: number; ocr_spool?: OcrSpool; latencies: Record<string, {p50_ms: number; p95_ms: number}>; inference_fps: Record<string, number>;
  gpu_name: string | null; gpu_power_watts: number | null; gpu_power_limit_watts: number | null; gpu_temperature_c: number | null;
  gpu_memory_used_mb: number | null; gpu_memory_total_mb: number | null; gpu_source: string; uptime_seconds: number;
}
export interface UploadState { busy: boolean; progress: number; message: string; error: boolean; }
export type PlaybackAction = 'play' | 'pause' | 'restart' | 'restore-camera';
export interface SystemAlert {
  id: string;
  alert_type: string;
  level: string;
  title: string;
  message: string;
  container_number?: string | null;
  event_id?: string | null;
  timestamp: string;
}

@Injectable({providedIn: 'root'})
export class GateApi {
  private http = inject(HttpClient);
  cameras() { return this.http.get<Camera[]>('/api/cameras'); }
  cameraSources() { return this.http.get<CameraSource[]>('/api/cameras/sources/available'); }
  discoveryStatus() { return this.http.get<DiscoveryStatus>('/api/cameras/discovery/status'); }
  rescan() { return this.http.post('/api/cameras/discovery/rescan', {}); }
  cameraRole(id: string, role: string, direction: string) { return this.http.post(`/api/cameras/${encodeURIComponent(id)}/role`, {role, direction}); }
  selectCamera(id: string, sourceId: string) { return this.http.post(`/api/cameras/${encodeURIComponent(id)}/source`, {source_id: sourceId}); }
  saveCalibration(id: string, calibration: Partial<LaneCalibration>) { return this.http.post<LaneCalibration>(`/api/cameras/${encodeURIComponent(id)}/calibration`, calibration); }
  health() {
    return this.http.get<Health>('/api/health').pipe(
      catchError((error: HttpErrorResponse) => {
        let errBody: any = error.error;
        if (typeof errBody === 'string') {
          try { errBody = JSON.parse(errBody); } catch (_) {}
        }
        if (errBody && typeof errBody === 'object' && errBody.database) {
          return of(errBody as Health);
        }
        return of({
          status: 'DEGRADED',
          database: 'READY',
          pipeline_enabled: true,
          online_cameras: 0,
          enabled_cameras: 0,
          models: { detector: 'LOADING', ocr: 'LOADING' },
          processing_error: null,
          startup_error: null,
          max_upload_mb: 2048
        } as Health);
      })
    );
  }
  metrics() { return this.http.get<Metrics>('/api/metrics'); }
  alerts() { return this.http.get<SystemAlert[]>('/api/alerts'); }
  testBuzzer(pattern: string = 'gate_out') { return this.http.post('/api/alerts/test?pattern=' + encodeURIComponent(pattern), {}); }
  clearAlerts() { return this.http.post('/api/alerts/clear', {}); }
  events(offset: number, container: string, type: string, folder?: string) {
    let params = new HttpParams().set('limit', 25).set('offset', offset);
    if (container.trim()) params = params.set('container', container.trim().toUpperCase());
    if (type) params = params.set('event_type', type);
    if (folder && folder !== 'all') params = params.set('folder', folder);
    return this.http.get<EventPage>('/api/gate-events', {params});
  }
  waitingPassages() {
    return this.http.get<{items: WaitingPassage[]; total: number}>('/api/waiting-passages');
  }
  event(id: string) { return this.http.get<EventDetail>(`/api/gate-events/${encodeURIComponent(id)}`); }
  eventJsonUrl(id: string) { return `/api/gate-events/${encodeURIComponent(id)}/json`; }
  eventJson(id: string) { return this.http.get<Record<string, unknown>>(`/api/gate-events/${encodeURIComponent(id)}/json`); }

  uploadVideo(id: string, file: File, syncLane: boolean = false) {
    const url = `/api/cameras/${encodeURIComponent(id)}/video` + (syncLane ? '?sync_lane=true' : '');
    return this.http.request(new HttpRequest('POST', url, file, {
      reportProgress: true,
      headers: new HttpHeaders({'Content-Type': 'application/octet-stream', 'X-Filename': encodeURIComponent(file.name)})
    }));
  }
  playback(id: string, action: PlaybackAction) { return this.http.post(`/api/cameras/${encodeURIComponent(id)}/playback`, {action}); }
  lanePlayback(gateId: string, action: PlaybackAction) { return this.http.post(`/api/lanes/${encodeURIComponent(gateId)}/playback`, {action}); }
  playAll() { return this.http.post<{results: {camera_id: string; success: boolean}[]}>('/api/videos/play-all', {}); }
  startProcessing() { return this.http.post('/api/processing/start', {}); }
  lensStatus(id: string) { return this.http.get<{supported: boolean; status?: string; zoom?: number; focus?: number; zoom_steps?: number; focus_steps?: number}>(`/api/cameras/${encodeURIComponent(id)}/lens/status`); }
  lensZoom(id: string, action: 'step_in' | 'step_out' | 'in' | 'out' | 'stop') { return this.http.post<{success: boolean}>(`/api/cameras/${encodeURIComponent(id)}/lens/zoom`, {action}); }
  lensFocus(id: string, action: 'step_far' | 'step_near' | 'far' | 'near' | 'stop') { return this.http.post<{success: boolean}>(`/api/cameras/${encodeURIComponent(id)}/lens/focus`, {action}); }
  lensAutoFocus(id: string) { return this.http.post<{success: boolean}>(`/api/cameras/${encodeURIComponent(id)}/lens/autofocus`, {}); }
  getExposure(id: string) { return this.http.get<{supported: boolean; mode?: string; shutter_speed?: string; shutter_ms?: number}>(`/api/cameras/${encodeURIComponent(id)}/exposure`); }
  setExposure(id: string, shutter: string, mode: string = 'auto') { return this.http.post<{success: boolean; shutter_speed?: string; mode?: string}>(`/api/cameras/${encodeURIComponent(id)}/exposure`, {shutter, mode}); }

  cloudSyncStatus() {
    return this.http.get<CloudSyncStatus>('/api/cloud-sync/status').pipe(
      catchError(() => of({ enabled: false, is_online: false, pending_count: 0, queue_depth: 0, last_sync_time: null, last_error: 'Unable to reach backend' }))
    );
  }

  retryCloudSync() {
    return this.http.post<{ triggered: boolean; pending: number }>('/api/cloud-sync/retry', {});
  }

  dispatchCloud(id: string) {
    return this.http.post<{success: boolean; status: string; cloud_event_id?: string; visit_id?: string; message?: string}>(
      `/api/gate-events/${encodeURIComponent(id)}/dispatch-cloud`, {}
    );
  }

  pauseCloudSync() {
    return this.http.post<{paused: boolean; auto_dispatch: boolean}>('/api/cloud-sync/pause', {});
  }

  resumeCloudSync() {
    return this.http.post<{paused: boolean; auto_dispatch: boolean}>('/api/cloud-sync/resume', {});
  }

  toggleCloudSyncAuto() {
    return this.http.post<{auto_dispatch: boolean; is_paused: boolean}>('/api/cloud-sync/toggle-auto-dispatch', {});
  }
}
