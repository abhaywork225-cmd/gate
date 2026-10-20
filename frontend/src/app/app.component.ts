import { Component, computed, DestroyRef, inject, signal } from '@angular/core';
import { DatePipe, DecimalPipe } from '@angular/common';
import { HttpErrorResponse, HttpEventType } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { catchError, exhaustMap, forkJoin, merge, of, Subject, timer, timeout } from 'rxjs';
import { GateApi, Camera, CameraSource, EventPage, EventDetail, Health, Metrics, UploadState, PlaybackAction, SystemAlert, CloudSyncStatus, WaitingPassage } from './gate-api.service';
import { CameraGridComponent } from './camera-grid.component';
import { EventTableComponent } from './event-table.component';
import { EventDetailComponent } from './event-detail.component';
import { SystemHealthComponent } from './system-health.component';
import { RoiEditorComponent } from './roi-editor.component';

@Component({
  selector: 'gate-app', standalone: true,
  imports: [FormsModule, DatePipe, DecimalPipe, CameraGridComponent, EventTableComponent, EventDetailComponent, SystemHealthComponent, RoiEditorComponent],
  templateUrl: './app.component.html'
})
export class AppComponent {
  private api = inject(GateApi);
  private destroy = inject(DestroyRef);
  private refresh = new Subject<void>();
  activeTab = signal<'overview' | 'cameras' | 'calibration' | 'events' | 'detail'>('cameras');
  folderFilter = signal<'all' | 'waiting' | 'detected_in' | 'detected_out'>('all');
  waitingPassages = signal<WaitingPassage[]>([]);
  waitingCount = computed(() => this.waitingPassages().length);
  cameras = signal<Camera[]>([]);
  cameraSources = signal<CameraSource[]>([]);
  discovery = signal({state: 'WAITING', found: 0, message: 'Waiting for camera discovery'});
  page = signal<EventPage>({items: [], total: 0, limit: 25, offset: 0});
  health = signal<Health | null>(null);
  metrics = signal<Metrics | null>(null);
  detail = signal<EventDetail | null>(null);
  cloudSync = signal<CloudSyncStatus | null>(null);
  syncBusy = signal(false);
  error = signal('');
  eventsError = signal('');
  notice = signal('');
  lastUpdated = signal<Date | null>(null);
  refreshToken = signal(0);
  uploads = signal<Partial<Record<string, UploadState>>>({});
  controlsBusy = signal<Record<string, boolean>>({});
  loading = signal(true);
  buzzerEnabled = signal(true);
  alerts = signal<SystemAlert[]>([]);
  activeAlert = computed(() => this.alerts()[0] || null);
  private lastAlertId = '';
  private audioCtx: any = null;
  videoCount = computed(() => this.cameras().filter(c => c.source_type === 'file').length);
  playingCount = computed(() => this.cameras().filter(c => c.playback_status === 'PLAYING').length);
  selectedCalibrationCameraId = signal('');
  calibrationCamera = computed(() => this.cameras().find(c => c.id === this.selectedCalibrationCameraId()) || this.cameras()[0] || null);
  containerFilter = '';
  typeFilter = '';
  offset = 0;

  constructor() {
    this.syncFromHash();
    window.addEventListener('hashchange', () => this.syncFromHash());

    // SQL failures must not hide camera slots, upload controls or machine health.
    merge(timer(0, 2500), this.refresh).pipe(
      exhaustMap(() => forkJoin({
        cameras: this.api.cameras(),
        sources: this.api.cameraSources(),
        discovery: this.api.discoveryStatus(),
        health: this.api.health(),
        metrics: this.api.metrics(),
        alerts: this.api.alerts().pipe(catchError(() => of([]))),
        waiting: this.api.waitingPassages().pipe(catchError(() => of({items: [], total: 0}))),
        events: this.api.events(this.offset, this.containerFilter, this.typeFilter, this.folderFilter()).pipe(catchError(() => of(null))),
        cloudSync: this.api.cloudSyncStatus(),
        detail: this.activeTab() === 'detail' && this.detail()?.id
          ? this.api.event(this.detail()!.id).pipe(catchError(() => of(null)))
          : of(null)
      }).pipe(timeout(15000), catchError(() => of(null)))),
      takeUntilDestroyed(this.destroy)
    ).subscribe(data => {
      this.loading.set(false);
      if (!data) { this.error.set('Backend disconnected. Start the Gate service to upload and play videos.'); return; }
      this.error.set(''); this.cameras.set(data.cameras); this.cameraSources.set(data.sources); this.health.set(data.health); this.metrics.set(data.metrics);
      if (!this.selectedCalibrationCameraId() && data.cameras.length) this.selectedCalibrationCameraId.set(data.cameras[0].id);
      this.discovery.set(data.discovery);
      this.cloudSync.set(data.cloudSync || data.health?.cloud_sync || null);
      if (data.waiting) {
        this.waitingPassages.set(data.waiting.items || []);
      }
      if (data.detail && this.activeTab() === 'detail') {
        this.detail.set(data.detail);
      }
      if (data.alerts && data.alerts.length > 0) {
        this.alerts.set(data.alerts);
        const top = data.alerts[0];
        if (top.id !== this.lastAlertId) {
          this.lastAlertId = top.id;
          this.playBuzzerSound();
        }
      } else {
        this.alerts.set([]);
      }
      if (data.events) { this.page.set(data.events); this.eventsError.set(''); }
      else this.eventsError.set('Event storage is unavailable. You can still upload and preview videos.');
      this.lastUpdated.set(new Date());
    });
    timer(0, 1000).pipe(takeUntilDestroyed(this.destroy)).subscribe(() => this.refreshToken.update(n => n + 1));
  }

  private syncFromHash() {
    const hash = window.location.hash.replace(/^#/, '');
    if (hash === 'events' || hash === 'events-all') {
      this.activeTab.set('events');
      this.folderFilter.set('all');
    } else if (hash === 'waiting' || hash === 'events-waiting') {
      this.activeTab.set('events');
      this.folderFilter.set('waiting');
    } else if (hash === 'detected_in' || hash === 'events-in') {
      this.activeTab.set('events');
      this.folderFilter.set('detected_in');
    } else if (hash === 'detected_out' || hash === 'events-out') {
      this.activeTab.set('events');
      this.folderFilter.set('detected_out');
    } else if (hash === 'cameras') {
      this.activeTab.set('cameras');
    } else if (hash === 'calibration') {
      this.activeTab.set('calibration');
    } else if (hash.startsWith('event-')) {
      const eventId = hash.slice(6);
      if (eventId && (!this.detail() || this.detail()?.id !== eventId)) {
        this.select(eventId);
      } else {
        this.activeTab.set('detail');
      }
    } else if (hash === 'overview') {
      this.activeTab.set('cameras');
    }
  }

  setFolder(folder: 'all' | 'waiting' | 'detected_in' | 'detected_out') {
    this.folderFilter.set(folder);
    this.offset = 0;
    this.refresh.next();
  }

  openCalibration(cameraId?: string) {
    if (cameraId) {
      this.selectedCalibrationCameraId.set(cameraId);
    }
    this.setTab('calibration');
  }

  setTab(tab: 'overview' | 'cameras' | 'calibration' | 'events' | 'detail') {
    this.activeTab.set(tab);
    if (tab === 'overview') {
      window.location.hash = 'overview';
    } else if (tab === 'cameras') {
      window.location.hash = 'cameras';
    } else if (tab === 'calibration') {
      window.location.hash = 'calibration';
    } else if (tab === 'events') {
      window.location.hash = 'events';
    } else if (tab === 'detail' && this.detail()) {
      window.location.hash = `event-${this.detail()!.id}`;
    }
  }

  filter() { this.offset = 0; this.refresh.next(); }
  movePage(delta: number) { this.offset = Math.max(0, this.offset + delta); this.refresh.next(); }
  
  select(id: string) {
    this.api.event(id).pipe(takeUntilDestroyed(this.destroy)).subscribe({
      next: d => {
        this.detail.set(d);
        this.activeTab.set('detail');
        window.location.hash = `event-${id}`;
      },
      error: () => this.notice.set('Event details could not be loaded.')
    });
  }

  backToEvents() {
    this.activeTab.set('events');
    window.location.hash = 'events';
  }

  uploadVideo({id, file, syncLane}: {id: string; file: File; syncLane?: boolean}) {
    if (this.uploads()[id]?.busy) return;
    const maximum = this.health()?.max_upload_mb ?? 2048;
    if (!/\.(mp4|avi|mov|mkv|webm|m4v)$/i.test(file.name) || file.size > maximum * 1024 * 1024) {
      this.setUpload(id, {busy: false, progress: 0, error: true, message: `Choose a supported video smaller than ${maximum} MB.`}); return;
    }
    this.setUpload(id, {busy: true, progress: 0, error: false, message: syncLane ? 'Syncing video to all cameras in lane…' : 'Uploading video…'});
    this.api.uploadVideo(id, file, !!syncLane).pipe(takeUntilDestroyed(this.destroy)).subscribe({
      next: event => {
        if (event.type === HttpEventType.UploadProgress) {
          const progress = Math.round(100 * event.loaded / (event.total || file.size));
          this.setUpload(id, {busy: true, progress, error: false, message: progress === 100 ? 'Configuring cameras…' : `Uploading ${progress}%`});
        } else if (event.type === HttpEventType.Response) {
          this.setUpload(id, {busy: false, progress: 100, error: false, message: syncLane ? 'All lane cameras configured! Press Play to start.' : 'Ready. Press Play to begin.'});
          this.notice.set(syncLane ? 'Video synchronized to all lane cameras.' : 'Video uploaded.');
          this.refresh.next();
        }
      },
      error: (error: HttpErrorResponse) => this.setUpload(id, {busy: false, progress: 0, error: true, message: this.errorMessage(error, 'Upload failed. Please try again.')})
    });
  }
  playback({id, action}: {id: string; action: PlaybackAction}) {
    if (this.controlsBusy()[id]) return;
    this.controlsBusy.update(value => ({...value, [id]: true}));
    this.api.playback(id, action).pipe(takeUntilDestroyed(this.destroy)).subscribe({
      next: () => { this.controlsBusy.update(value => ({...value, [id]: false})); this.refresh.next(); },
      error: (error: HttpErrorResponse) => { this.controlsBusy.update(value => ({...value, [id]: false})); this.notice.set(this.errorMessage(error, 'Playback control failed.')); }
    });
  }
  lanePlayback(gateId: string, action: PlaybackAction) {
    this.api.lanePlayback(gateId, action).pipe(takeUntilDestroyed(this.destroy)).subscribe({
      next: () => { this.notice.set(`Gate ${gateId} cameras updated: ${action}.`); this.refresh.next(); },
      error: (error: HttpErrorResponse) => this.notice.set(this.errorMessage(error, 'Lane playback failed.'))
    });
  }
  selectCamera({id, sourceId}: {id: string; sourceId: string}) {
    if (!sourceId || this.controlsBusy()[id]) return;
    this.controlsBusy.update(value => ({...value, [id]: true}));
    this.api.selectCamera(id, sourceId).pipe(takeUntilDestroyed(this.destroy)).subscribe({
      next: () => { this.controlsBusy.update(value => ({...value, [id]: false})); this.notice.set('Camera source switched.'); this.refresh.next(); },
      error: (error: HttpErrorResponse) => { this.controlsBusy.update(value => ({...value, [id]: false})); this.notice.set(this.errorMessage(error, 'Camera could not be selected.')); this.refresh.next(); }
    });
  }

  saveCalibration(value: any) {
    const camera = this.calibrationCamera();
    if (!camera) return;
    this.api.saveCalibration(camera.id, value).pipe(takeUntilDestroyed(this.destroy)).subscribe({
      next: saved => { this.cameras.update(items => items.map(item => item.id === camera.id ? {...item, calibration: saved, occupancy: item.occupancy} : item)); this.notice.set(`Lane calibration saved for ${camera.name}.`); this.refresh.next(); },
      error: () => this.notice.set('Lane calibration could not be saved.')
    });
  }
  playAll() {
    this.api.playAll().pipe(takeUntilDestroyed(this.destroy)).subscribe({next: value => {
      this.notice.set(`Started ${value.results.filter(r => r.success).length} video inputs.`); this.refresh.next();
    }, error: () => this.notice.set('Could not start the video inputs.')});
  }
  rescan() {
    this.api.rescan().pipe(takeUntilDestroyed(this.destroy)).subscribe({
      next: () => { this.notice.set('Camera search requested.'); this.refresh.next(); },
      error: () => this.notice.set('Could not request camera search.')
    });
  }
  setCameraRole({id, role, direction}: {id: string; role: string; direction: string}) {
    if (this.controlsBusy()[id]) return;
    this.controlsBusy.update(v => ({...v, [id]: true}));
    this.api.cameraRole(id, role, direction).pipe(takeUntilDestroyed(this.destroy)).subscribe({
      next: () => { this.controlsBusy.update(v => ({...v, [id]: false})); this.refresh.next(); },
      error: (e: HttpErrorResponse) => { this.controlsBusy.update(v => ({...v, [id]: false})); this.notice.set(this.errorMessage(e, 'Could not save camera role.')); }
    });
  }
  startProcessing() {
    this.api.startProcessing().pipe(takeUntilDestroyed(this.destroy)).subscribe({next: () => {
      this.notice.set('Loading the configured detection and OCR models…'); this.refresh.next();
    }, error: (error: HttpErrorResponse) => this.notice.set(this.errorMessage(error, 'AI processing could not start.'))});
  }

  playBuzzerSound() {
    if (!this.buzzerEnabled()) return;
    try {
      const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      if (!AudioCtx) return;
      if (!this.audioCtx) {
        this.audioCtx = new AudioCtx();
      }
      if (this.audioCtx.state === 'suspended') {
        this.audioCtx.resume();
      }
      const t = this.audioCtx.currentTime;

      // Pulse 1: Industrial alarm buzz
      const osc1 = this.audioCtx.createOscillator();
      const gain1 = this.audioCtx.createGain();
      osc1.type = 'sawtooth';
      osc1.frequency.setValueAtTime(850, t);
      osc1.frequency.exponentialRampToValueAtTime(580, t + 0.22);
      gain1.gain.setValueAtTime(0.35, t);
      gain1.gain.exponentialRampToValueAtTime(0.01, t + 0.22);
      osc1.connect(gain1);
      gain1.connect(this.audioCtx.destination);
      osc1.start(t);
      osc1.stop(t + 0.23);

      // Pulse 2: Higher pitch warning buzz
      const osc2 = this.audioCtx.createOscillator();
      const gain2 = this.audioCtx.createGain();
      osc2.type = 'sawtooth';
      osc2.frequency.setValueAtTime(1050, t + 0.28);
      osc2.frequency.exponentialRampToValueAtTime(680, t + 0.52);
      gain2.gain.setValueAtTime(0.38, t + 0.28);
      gain2.gain.exponentialRampToValueAtTime(0.01, t + 0.52);
      osc2.connect(gain2);
      gain2.connect(this.audioCtx.destination);
      osc2.start(t + 0.28);
      osc2.stop(t + 0.53);
    } catch {
      // Browser audio autoplay policy
    }
  }

  testBuzzer() {
    this.playBuzzerSound();
    this.api.testBuzzer('gate_out').pipe(takeUntilDestroyed(this.destroy)).subscribe();
    this.notice.set('Audible buzzer sound triggered (Browser & Host System).');
  }

  dismissAlert(id?: string) {
    this.api.clearAlerts().pipe(takeUntilDestroyed(this.destroy)).subscribe(() => {
      this.alerts.set([]);
      this.notice.set('Alert dismissed.');
    });
  }

  private setUpload(id: string, state: UploadState) { this.uploads.update(value => ({...value, [id]: state})); }
  private errorMessage(error: HttpErrorResponse, fallback: string) { return typeof error.error?.detail === 'string' ? error.error.detail : fallback; }

  retrySync() {
    this.syncBusy.set(true);
    this.api.retryCloudSync().subscribe({
      next: (res) => {
        this.syncBusy.set(false);
        this.notice.set('Cloud retry sweep triggered: ' + res.pending + ' events pending.');
        this.refresh.next();
        setTimeout(() => this.notice.set(''), 4000);
      },
      error: () => {
        this.syncBusy.set(false);
        this.error.set('Failed to trigger cloud sync retry.');
        setTimeout(() => this.error.set(''), 4000);
      }
    });
  }
}
