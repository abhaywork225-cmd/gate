import { Component, inject, input, output } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { LivePreviewComponent } from './live-preview.component';
import { Camera, CameraSource, GateApi, PlaybackAction, UploadState } from './gate-api.service';

@Component({
  selector: 'gate-camera-grid', standalone: true, imports: [DecimalPipe, LivePreviewComponent],
  template: `
  <div class="camera-grid">
    @for (camera of cameras(); track camera.id; let index = $index) {
      <article class="camera-card" [attr.aria-label]="camera.name">
        <div class="card-heading"><span class="camera-number">0{{ index + 1 }}</span><div class="camera-name"><strong>{{ camera.name }}</strong><small>{{ camera.gate_id }} · {{ camera.direction }}</small></div>
          <button type="button" class="btn-virtual-line" (click)="openCalibration.emit(camera.id)" title="Configure Virtual Laser Line and Detection Area">⚡ Virtual Line & ROI</button>
          <span class="badge" [class.good]="camera.status === 'ONLINE'" [class.warn]="camera.status === 'RECONNECTING'">{{ camera.source_type === 'file' ? camera.playback_status : camera.status }}</span></div>
        <label class="source-selector"><span>Camera source</span><select [disabled]="!!busy()[camera.id] || stale()" (change)="selectSource($event, camera.id)">
          <option value="" disabled [selected]="!camera.active_source_id">Select a camera</option>
          @for (source of cameraSources(); track source.id) {<option [value]="source.id" [selected]="source.id === camera.active_source_id">{{ source.name }}{{ source.host ? ' — ' + source.host : '' }}{{ source.configured ? '' : source.discovered ? ' — needs setup' : ' — not configured' }}</option>}
        </select></label>
        <label class="source-selector"><span>Position</span><select [disabled]="!!busy()[camera.id] || stale()" (change)="roleChange.emit({id: camera.id, role: $any($event.target).value, direction: camera.direction})">
          @for (role of roles; track role.value) {<option [value]="role.value" [selected]="role.value === camera.role">{{ role.label }}</option>}
        </select></label>
        <label class="source-selector"><span>Gate direction</span><select [disabled]="!!busy()[camera.id] || stale()" (change)="roleChange.emit({id: camera.id, role: camera.role, direction: $any($event.target).value})">
          <option value="UNKNOWN" [selected]="camera.direction === 'UNKNOWN'">Unassigned</option>
          <option value="ENTRY" [selected]="camera.direction === 'ENTRY'">Gate IN</option>
          <option value="EXIT" [selected]="camera.direction === 'EXIT'">Gate OUT</option>
        </select></label>
        <div class="camera-frame" (dragover)="$event.preventDefault()" (drop)="drop($event, camera.id)">
          <span class="source-tag">{{ camera.source_type === 'file' ? 'VIDEO FILE' : 'CAMERA INPUT' }}</span>
          @if (camera.has_frame && !stale()) {
            <gate-live-preview [cameraId]="camera.id" [name]="camera.name" />
          } @else {
            <div class="offline-frame"><svg width="34" height="28" viewBox="0 0 34 28" fill="none" aria-hidden="true"><rect x="2" y="6" width="23" height="18" rx="3" stroke="currentColor" stroke-width="1.5"/><path d="m25 12 7-4v14l-7-4M9 6V3h9v3" stroke="currentColor" stroke-width="1.5"/></svg><strong>{{ stale() ? 'Backend disconnected' : 'No camera connected' }}</strong><small>Drop a recorded video here to test this view</small></div>
          }
        </div>
        @if (camera.source_type !== 'file') {
          <div class="lens-control-bar">
            <span class="lens-title">LENS</span>
            <div class="lens-btn-group">
              <button type="button" class="lens-btn" [disabled]="!!lensBusy[camera.id] || stale()" (click)="zoomStep(camera.id, 'step_out')" title="Zoom Out (Wide)">🔍−</button>
              <button type="button" class="lens-btn" [disabled]="!!lensBusy[camera.id] || stale()" (click)="zoomStep(camera.id, 'step_in')" title="Zoom In (Tele)">🔍+</button>
              <button type="button" class="lens-btn af-btn" [disabled]="!!lensBusy[camera.id] || stale()" (click)="autoFocus(camera.id)" title="One-Click Auto-Focus">🎯 Auto-Focus</button>
              <button type="button" class="lens-btn" [disabled]="!!lensBusy[camera.id] || stale()" (click)="focusStep(camera.id, 'step_near')" title="Focus Near">F−</button>
              <button type="button" class="lens-btn" [disabled]="!!lensBusy[camera.id] || stale()" (click)="focusStep(camera.id, 'step_far')" title="Focus Far">F+</button>
            </div>
            <div class="shutter-group">
              <span class="shutter-label" title="Electronic Shutter Speed for Motion Blur Reduction">⚡ Shutter</span>
              <select class="shutter-select" [disabled]="!!lensBusy[camera.id] || stale()" (change)="changeShutter($event, camera.id)" [title]="'Set camera shutter speed for ' + camera.name">
                <option value="1/500s" selected>1/500s (OCR)</option>
                <option value="1/1000s">1/1000s (Fast)</option>
                <option value="1/250s">1/250s (Dusk)</option>
                <option value="1/100s">1/100s</option>
                <option value="1/50s">1/50s</option>
                <option value="auto">Auto (1/25s)</option>
              </select>
            </div>
            @if (lensBusy[camera.id]) {
              <span class="lens-busy-pill">{{ lensBusy[camera.id] }}</span>
            }
          </div>
        }
        <div class="video-input">
          <div class="file-row"><span class="file-name" [title]="camera.video?.filename || ''">{{ camera.source_type === 'file' ? (camera.video?.filename || 'Configured video file') : 'No video selected' }}</span>
            <div class="upload-btn-group">
              <label class="upload-button" [class.disabled]="uploads()[camera.id]?.busy" [title]="'Upload video for ' + camera.name"><input type="file" [attr.aria-label]="'Add video for ' + camera.name" accept=".mp4,.avi,.mov,.mkv,.webm,.m4v" [disabled]="uploads()[camera.id]?.busy || stale()" (change)="choose($event, camera.id, false)">{{ uploads()[camera.id]?.busy ? 'Uploading…' : camera.video?.active ? 'Replace' : '+ Video' }}</label>
              <label class="upload-button sync-btn" [class.disabled]="uploads()[camera.id]?.busy" [title]="'Upload and sync to all 4 cameras in ' + camera.gate_id"><input type="file" [attr.aria-label]="'Sync video to ' + camera.gate_id" accept=".mp4,.avi,.mov,.mkv,.webm,.m4v" [disabled]="uploads()[camera.id]?.busy || stale()" (change)="choose($event, camera.id, true)">⚡ Sync All</label>
            </div>
          </div>
          @if (uploads()[camera.id]; as state) {
            <div class="upload-status" [class.error-text]="state.error" role="status">{{ state.message }}</div>
            @if (state.busy) {<progress [value]="state.progress" max="100" aria-label="Upload progress"></progress>}
          }
          @if (camera.source_type === 'file') {
            <div class="playback-timeline"><progress [value]="camera.video_position_seconds" [max]="camera.video_duration_seconds || 1" aria-label="Video playback progress"></progress><small>{{ camera.video_position_seconds | number:'1.0-0' }}s / {{ camera.video_duration_seconds === null ? '—' : (camera.video_duration_seconds | number:'1.0-0') + 's' }}</small></div>
            <div class="playback-buttons"><button class="primary" (click)="control.emit({id: camera.id, action: camera.playback_status === 'PLAYING' ? 'pause' : 'play'})" [disabled]="!!busy()[camera.id] || stale()">{{ camera.playback_status === 'PLAYING' ? 'Ⅱ Pause' : '▶ Play' }}</button><button (click)="control.emit({id: camera.id, action: 'restart'})" [disabled]="!!busy()[camera.id] || stale()">Restart</button><button class="text-button" (click)="control.emit({id: camera.id, action: 'restore-camera'})" [disabled]="!!busy()[camera.id] || stale()">Use camera</button></div>
          }
        </div>
        <dl class="camera-stats"><div><dt>CAPTURE</dt><dd>{{ camera.capture_fps | number:'1.1-1' }} <span>fps</span></dd></div><div><dt>INFERENCE</dt><dd>{{ inferenceFps()[camera.id] ?? 0 | number:'1.1-1' }} <span>fps</span></dd></div><div><dt>DROPPED</dt><dd>{{ camera.dropped_frames }}</dd></div></dl>
      </article>
    } @empty { <p class="empty">Loading camera views…</p> }
  </div>`
})
export class CameraGridComponent {
  gateApi = inject(GateApi);
  lensBusy: Record<string, string> = {};

  zoomStep(id: string, action: 'step_in' | 'step_out') {
    this.lensBusy[id] = action === 'step_in' ? 'Zooming in…' : 'Zooming out…';
    this.gateApi.lensZoom(id, action).subscribe({
      next: () => setTimeout(() => delete this.lensBusy[id], 900),
      error: (err) => {
        this.lensBusy[id] = err?.error?.detail || 'Lens error';
        setTimeout(() => delete this.lensBusy[id], 1500);
      }
    });
  }

  focusStep(id: string, action: 'step_far' | 'step_near') {
    this.lensBusy[id] = action === 'step_far' ? 'Focus Far…' : 'Focus Near…';
    this.gateApi.lensFocus(id, action).subscribe({
      next: () => setTimeout(() => delete this.lensBusy[id], 900),
      error: (err) => {
        this.lensBusy[id] = err?.error?.detail || 'Lens error';
        setTimeout(() => delete this.lensBusy[id], 1500);
      }
    });
  }

  autoFocus(id: string) {
    this.lensBusy[id] = 'Auto-focusing…';
    this.gateApi.lensAutoFocus(id).subscribe({
      next: () => setTimeout(() => delete this.lensBusy[id], 2000),
      error: (err) => {
        this.lensBusy[id] = err?.error?.detail || 'Lens error';
        setTimeout(() => delete this.lensBusy[id], 1500);
      }
    });
  }

  changeShutter(event: Event, id: string) {
    const val = (event.target as HTMLSelectElement).value;
    this.lensBusy[id] = `Shutter ${val}…`;
    this.gateApi.setExposure(id, val, 'auto').subscribe({
      next: (res) => {
        this.lensBusy[id] = `Shutter: ${res.shutter_speed || val}`;
        setTimeout(() => delete this.lensBusy[id], 1500);
      },
      error: (err) => {
        this.lensBusy[id] = err?.error?.detail || 'Shutter error';
        setTimeout(() => delete this.lensBusy[id], 1500);
      }
    });
  }

  roles = [{value: 'UNASSIGNED', label: 'Unassigned'}, {value: 'FRONT_TOP', label: 'Front / Top'}, {value: 'LEFT', label: 'Left'}, {value: 'RIGHT', label: 'Right'}, {value: 'REAR', label: 'Rear'}];
  roleChange = output<{id: string; role: string; direction: string}>();
  openCalibration = output<string>();
  cameras = input<Camera[]>([]);
  cameraSources = input<CameraSource[]>([]);
  inferenceFps = input<Partial<Record<string, number>>>({});
  refreshToken = input(0);
  stale = input(false);
  uploads = input<Partial<Record<string, UploadState>>>({});
  busy = input<Record<string, boolean>>({});
  upload = output<{id: string; file: File; syncLane?: boolean}>();
  control = output<{id: string; action: PlaybackAction}>();
  laneControl = output<{gateId: string; action: PlaybackAction}>();
  sourceChange = output<{id: string; sourceId: string}>();
  selectSource(event: Event, id: string) {
    this.sourceChange.emit({id, sourceId: (event.target as HTMLSelectElement).value});
  }
  choose(event: Event, id: string, syncLane: boolean = false) {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (file) this.upload.emit({id, file, syncLane});
    input.value = '';
  }
  drop(event: DragEvent, id: string) {
    event.preventDefault();
    if (this.stale() || this.uploads()[id]?.busy) return;
    const file = event.dataTransfer?.files[0];
    if (file) this.upload.emit({id, file, syncLane: false});
  }
}
