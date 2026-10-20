import { Component, inject, input, output, signal } from '@angular/core';
import { DatePipe, PercentPipe } from '@angular/common';
import { EventDetail, GateApi, OcrCrop } from './gate-api.service';

@Component({
  selector: 'gate-event-detail',
  standalone: true,
  imports: [DatePipe, PercentPipe],
  template: `
    @if (event(); as item) {
      <div class="detail-page" aria-label="Event details page">
        <div class="detail-header-bar">
          <div class="detail-nav-actions">
            <button class="back-button" (click)="back.emit()">
              <span class="arrow">←</span> Back to Recent gate events
            </button>
            <div class="detail-breadcrumbs">
              <span>Gate events</span>
              <span class="sep">/</span>
              <span class="current">{{ item.container_number || item.id.slice(0, 8) }}</span>
            </div>
          </div>
          <div class="detail-top-actions">
            <a class="action-btn download-btn" [href]="'/api/gate-events/' + item.id + '/json'" [download]="'event_' + (item.container_number || item.id) + '.json'">
              <span class="icon">📥</span> Download event.json
            </a>
            <button class="action-btn json-toggle-btn" [class.active]="showJson()" (click)="toggleJson()">
              <span class="icon">{{ showJson() ? '✕' : '{ }' }}</span> {{ showJson() ? 'Hide raw JSON' : 'View event.json' }}
            </button>
          </div>
        </div>

        <div class="event-hero-card" [class.twin-hero]="item.is_twin">
          <div class="hero-main">
            <div class="hero-label">
              @if (item.is_twin) {
                <span class="twin-header-pill">{{ item.status === 'CONFIRMED' && item.container_number_2 ? 'TWO CONTAINER IDENTITIES' : 'POSSIBLE TWIN · REVIEW REQUIRED' }}</span>
              } @else {
                CONTAINER IDENTIFICATION & EVIDENCE
              }
            </div>

            @if (item.is_twin) {
              <div class="twin-hero-boxes">
                <!-- Front Container -->
                <div class="twin-box-card">
                  <div class="twin-box-header">
                    <span class="slot-tag front">FRONT CONTAINER (SLOT 1)</span>
                    <span class="size-pill feet">📏 {{ item.container_size || 'Size unconfirmed' }} {{ item.size_code ? '(' + item.size_code + ')' : '' }}</span>
                  </div>
                  <div class="hero-container-title">
                    <span class="container-badge mono">{{ item.container_number || 'UNCONFIRMED' }}</span>
                  </div>
                </div>

                <!-- Rear Container -->
                <div class="twin-box-card">
                  <div class="twin-box-header">
                    <span class="slot-tag rear">REAR CONTAINER (SLOT 2)</span>
                    <span class="size-pill feet">📏 {{ item.container_size_2 || 'Size unconfirmed' }} {{ item.size_code_2 ? '(' + item.size_code_2 + ')' : '' }}</span>
                  </div>
                  <div class="hero-container-title">
                    <span class="container-badge mono">{{ item.container_number_2 || 'UNCONFIRMED' }}</span>
                  </div>
                </div>
              </div>

              <div class="hero-meta-row twin-meta-row">
                <span class="status-pill" [class.confirmed]="item.status === 'CONFIRMED'" [class.review]="(item.status !== 'CONFIRMED' && item.capture_policy !== 'BEST_AVAILABLE')">{{ item.status.replaceAll('_', ' ') }}</span>
                <span class="movement-pill">{{ item.event_type }}</span>
                <span class="meta-item"><strong>Lane:</strong> {{ item.gate_id }}</span>
                <span class="meta-dot">·</span>
                <span class="meta-item"><strong>Time:</strong> {{ item.timestamp | date:'medium' }}</span>
                <span class="meta-dot">·</span>
                <span class="meta-item mono id-meta"><strong>ID:</strong> {{ item.id }}</span>
              </div>
            } @else {
              <h1 class="hero-container-title">
                <span class="container-badge mono">{{ item.container_number || item.candidate_number || 'NUMBER NOT READ' }}</span>
                <span class="status-pill" [class.confirmed]="item.status === 'CONFIRMED'" [class.review]="(item.status !== 'CONFIRMED' && item.capture_policy !== 'BEST_AVAILABLE')">
                  {{ item.status.replaceAll('_', ' ') }}
                </span>
                <span class="movement-pill">{{ item.event_type }}</span>
                @if (item.container_size) {
                  <span class="size-pill feet">📏 {{ item.container_size }}{{ item.size_code ? ' (' + item.size_code + ')' : '' }}</span>
                }
                @if (hasFeetDetection(item)) {
                  <span class="feet-pill">🦶 FEET DETECTED</span>
                }
              </h1>

              <div class="hero-meta-row">
                <span class="meta-item"><strong>Lane:</strong> {{ item.gate_id }}</span>
                <span class="meta-dot">·</span>
                <span class="meta-item"><strong>Time:</strong> {{ item.timestamp | date:'medium' }}</span>
                <span class="meta-dot">·</span>
                <span class="meta-item"><strong>Snapshots:</strong> {{ item.snapshots.length }} viewpoints</span>
                <span class="meta-dot">·</span>
                <span class="meta-item mono id-meta"><strong>ID:</strong> {{ item.id }}</span>
              </div>
            }
          </div>
        </div>

        @if (item.review_reason) {
          <div class="review-explanation" role="status"><strong>Review required</strong><p>{{ item.review_reason }}</p></div>
        }
        @if (!item.container_number && item.candidate_number) {
          <p>This OCR number is unverified and has not been sent to cloud.</p>
        }
        @if (item.capture_policy === 'UNREADABLE_CAPTURE') {
          <div class="review-explanation" role="status"><strong>Container number unreadable</strong><p>The visit is captured with its best available proof. It can be sent with the number explicitly marked unreadable.</p></div>
        }

        <!-- Cloud Dispatch Review & Action Banner -->
        <div class="cloud-dispatch-banner" [class.synced]="item.cloud_synced" [class.paused]="!item.cloud_synced">
          <div class="cloud-banner-left">
            @if (item.cloud_synced) {
              <div class="cloud-status-badge synced">
                <span class="badge-icon">✓</span>
                <span>SENT TO SITE</span>
              </div>
              <div class="cloud-banner-text">
                <strong>Event published to Prosper Smart Yard cloud site</strong>
                @if (item.cloud_event_id) {
                  <span class="cloud-id mono">Cloud ID: {{ item.cloud_event_id }}</span>
                }
                @if (item.cloud_synced_at) {
                  <span class="cloud-synced-time">Synced: {{ item.cloud_synced_at | date:'HH:mm:ss · dd MMM' }}</span>
                }
              </div>
            } @else {
              <div class="cloud-status-badge paused">
                <span class="badge-icon">⏸</span>
                <span>{{ cloudEnabled() ? 'SENDING PAUSED' : 'LOCAL ONLY' }}</span>
              </div>
              <div class="cloud-banner-text">
                <strong>{{ cloudEnabled() ? 'Review before sending' : 'Cloud sending is disabled' }}</strong>
                <span>{{ cloudEnabled() ? 'Review the number and proof before sending this event.' : 'Events and proof images stay on this device while you review.' }}</span>
              </div>
            }
          </div>
          <div class="cloud-banner-right">
            <button class="send-to-site-btn" [class.resend]="item.cloud_synced" [disabled]="dispatching() || !cloudEnabled() || item.cloud_proof_complete === false || (item.status !== 'CONFIRMED' && !['BEST_AVAILABLE', 'FINAL_CAPTURE', 'UNREADABLE_CAPTURE'].includes(item.capture_policy || ''))" (click)="dispatchToSite(item)">
              @if (dispatching()) {
                <span class="spinner">⏳</span> Sending to Site...
              } @else if (item.cloud_synced) {
                <span>🔄 Re-send to Site</span>
              } @else {
                <span>📤 Send to Site</span>
              }
            </button>
          </div>
        </div>

        @if (dispatchMessage()) {
          <div class="dispatch-message-bar" [class.success]="dispatchSuccess()" [class.error]="!dispatchSuccess()">
            <span>{{ dispatchMessage() }}</span>
            <button class="close-msg-btn" (click)="clearDispatchMessage()">✕</button>
          </div>
        }

        @if (showJson()) {
          <div class="json-viewer-panel">
            <div class="json-header">
              <div>
                <strong>event.json</strong>
                <small class="json-path">snapshots/{{ item.visit_id || item.id }}/{{ item.visit_id ? 'event_' + item.id + '.json' : 'event.json' }}</small>
              </div>
              <div class="json-actions">
                <button class="copy-btn" (click)="copyJson(item)">
                  {{ copied() ? '✓ Copied!' : 'Copy JSON' }}
                </button>
              </div>
            </div>
            <pre class="json-body"><code>{{ formatJson(item) }}</code></pre>
          </div>
        }

        <div class="evidence-columns">
          <div class="evidence-left">
            <div class="panel-card">
              <div class="panel-card-title">
                <h3>OCR Recognition Evidence</h3>
                <span class="badge">{{ item.ocr_results.length }} readings</span>
              </div>
              @if (item.ocr_results.length) {
                <div class="ocr-cards-list">
                  @for (ocr of item.ocr_results; track ocr.id) {
                    <div class="ocr-card" [class.ocr-valid]="ocr.validation_status === 'VALID' || ocr.validation_status === 'VALID_SIZE_CODE'">
                      <div class="ocr-card-header">
                        <span class="mono ocr-text">{{ ocr.normalized_text || 'No text' }}</span>
                        <span class="ocr-badge" [class.good]="ocr.validation_status === 'VALID' || ocr.validation_status === 'VALID_SIZE_CODE'">
                          {{ ocr.validation_status === 'VALID_SIZE_CODE' ? 'VALID SIZE CODE' : ocr.validation_status }}
                        </span>
                      </div>
                      <div class="ocr-card-details">
                        <div class="detail-row">
                          <span>Confidence:</span>
                          <strong>{{ ocr.confidence | percent:'1.0-1' }}</strong>
                        </div>
                        <div class="detail-row">
                          <span>Raw reading:</span>
                          <span class="mono">{{ ocr.raw_text || '(empty)' }}</span>
                        </div>
                      </div>
                    </div>
                  }
                </div>
              } @else {
                <p class="empty-hint">No OCR readings captured for this event.</p>
              }
            </div>

            <div class="panel-card" style="margin-top: 20px;">
              <div class="panel-card-title">
                <h3>Visual Detection Tracks</h3>
                <span class="badge">{{ item.detections.length }} objects</span>
              </div>
              @if (item.detections.length) {
                <div class="detection-list">
                  @for (d of item.detections; track d.id) {
                    <div class="detection-item" [class.feet-detection]="d.class_name.toLowerCase() === 'feet'">
                      <div class="detection-top">
                        <strong [class.feet-strong]="d.class_name.toLowerCase() === 'feet'">
                          {{ d.class_name.toUpperCase() }}
                          @if (d.class_name.toLowerCase() === 'feet') {
                            <span class="feet-badge-mini">TWISTLOCK / SIZE</span>
                          }
                        </strong>
                        <span class="mono">Camera: {{ d.camera_id }}</span>
                      </div>
                      <div class="detection-sub">
                        <span>Track #{{ d.track_id }}</span>
                        <span>Confidence: {{ d.confidence | percent:'1.0-1' }}</span>
                      </div>
                    </div>
                  }
                </div>
              } @else {
                <p class="empty-hint">No detection records found.</p>
              }
            </div>
          </div>

          <div class="evidence-right">
            <div class="panel-card">
              <div class="panel-card-title">
                <h3>Cloud upload proof</h3>
                <span class="badge">{{ item.cloud_proof_images?.length || 0 }} images</span>
              </div>
              <p class="image-refresh-note">Exact images selected by the cloud upload builder. Preview only; no data is sent.</p>
              @for (coverage of item.cloud_proof_coverage || []; track $index) {
                @if (coverage.required > 0) {
                  <p class="image-refresh-note" [class.review-explanation]="!coverage.complete">
                    {{ coverage.container_number }}: number proof from {{ coverage.cameras.length }} of {{ coverage.required }} required cameras.
                    @if (!coverage.complete) { Sending is held until a second camera provides matching number proof. }
                  </p>
                }
              }
              @for (proof of item.cloud_proof_images || []; track $index) {
                <figure class="primary-proof-card">
                  <div class="proof-header">
                    <span class="proof-badge">{{ proof.position }} · {{ proof.container_number }}</span>
                    <span class="proof-cam">{{ proof.camera_id }} · {{ proof.image_type }}</span>
                  </div>
                  <div class="proof-img-wrap"><img [src]="proof.image" [alt]="'Cloud upload proof for ' + proof.container_number + ' from ' + proof.camera_id"></div>
                  <figcaption class="proof-caption">{{ proof.captured_at | date:'HH:mm:ss · dd MMM yyyy' }}</figcaption>
                </figure>
              } @empty {
                <p class="review-explanation" role="status">{{ item.cloud_proof_error || 'No usable cloud proof images are available.' }}</p>
              }
            </div>
          </div>
        </div>
      </div>
    }
  `
})
export class EventDetailComponent {
  private gateApi = inject(GateApi);

  event = input<EventDetail | null>(null);
  back = output<void>();
  showJson = signal(false);
  copied = signal(false);
  imageLoaded = signal<Record<string, boolean>>({});
  imageErrors = signal<Record<string, boolean>>({});

  cloudEnabled = signal(false);

  constructor() {
    this.gateApi.cloudSyncStatus().subscribe({next: status => this.cloudEnabled.set(status.enabled), error: () => this.cloudEnabled.set(false)});
  }

  dispatching = signal(false);
  dispatchSuccess = signal(false);
  dispatchMessage = signal<string | null>(null);

  snapshotUrl(id: string): string {
    return `/api/snapshots/${id}?proof=true`;
  }

  rawSnapshotUrl(id: string): string {
    return `/api/snapshots/${encodeURIComponent(id)}`;
  }

  markImageLoaded(id: string) {
    this.imageLoaded.update(state => ({ ...state, [id]: true }));
  }

  markImageError(id: string) {
    this.imageErrors.update(state => ({ ...state, [id]: true }));
  }

  toggleJson() {
    this.showJson.update(v => !v);
  }

  dispatchToSite(item: EventDetail) {
    if (!item?.id || this.dispatching() || !this.cloudEnabled() || (item.status !== 'CONFIRMED' && !['BEST_AVAILABLE', 'FINAL_CAPTURE', 'UNREADABLE_CAPTURE'].includes(item.capture_policy || ''))) return;
    this.dispatching.set(true);
    this.dispatchMessage.set(null);

    this.gateApi.dispatchCloud(item.id).subscribe({
      next: (res) => {
        this.dispatching.set(false);
        this.dispatchSuccess.set(true);
        this.dispatchMessage.set(res.message || '✓ Event successfully dispatched to cloud site!');
        item.cloud_synced = true;
        if (res.cloud_event_id) {
          item.cloud_event_id = res.cloud_event_id;
        }
        setTimeout(() => {
          if (this.dispatchSuccess()) {
            this.dispatchMessage.set(null);
          }
        }, 5000);
      },
      error: (err) => {
        this.dispatching.set(false);
        this.dispatchSuccess.set(false);
        const detail = err?.error?.detail || err?.error?.message || err?.message || 'Failed to dispatch event to site';
        this.dispatchMessage.set('⚠ Error: ' + detail);
        setTimeout(() => {
          if (this.dispatchMessage()?.startsWith('⚠ Error:')) {
            this.dispatchMessage.set(null);
          }
        }, 8000);
      }
    });
  }

  clearDispatchMessage() {
    this.dispatchMessage.set(null);
  }

  getCropForContainer(item: EventDetail, slot: 1 | 2): OcrCrop | null {
    if (!item?.ocr_crops?.length) return null;
    const targetNumber = slot === 1 ? item.container_number : item.container_number_2;
    if (targetNumber) {
      const match = item.ocr_crops.find(c => c.normalized_text === targetNumber);
      if (match) return match;
    }
    const slotStr = slot === 1 ? 'FRONT' : 'REAR';
    const slotMatch = item.ocr_crops.find(c => c.slot_position?.toUpperCase() === slotStr);
    if (slotMatch) return slotMatch;
    return slot === 1 ? item.ocr_crops[0] : (item.ocr_crops.length > 1 ? item.ocr_crops[1] : null);
  }

  getPrimarySnapshot(item: EventDetail) {
    if (!item?.snapshots?.length) return null;
    if (item.primary_snapshot_id) {
      const found = item.snapshots.find(s => s.id === item.primary_snapshot_id);
      if (found) return found;
    }
    return item.snapshots[0];
  }

  getFrontSnapshot(item: EventDetail) {
    if (!item?.snapshots?.length) return null;
    if (item.primary_snapshot_id) {
      const found = item.snapshots.find(s => s.id === item.primary_snapshot_id);
      if (found) return found;
    }
    const nonRear = item.snapshots.find(s => !s.image_path?.endsWith('_rear.jpg'));
    return nonRear || item.snapshots[0];
  }

  getRearSnapshot(item: EventDetail) {
    if (!item?.snapshots?.length) return null;
    if (item.rear_snapshot_id) {
      const found = item.snapshots.find(s => s.id === item.rear_snapshot_id);
      if (found) return found;
    }
    const rear = item.snapshots.find(s => s.image_path?.endsWith('_rear.jpg'));
    if (rear) return rear;
    const frontSnap = this.getFrontSnapshot(item);
    const diff = item.snapshots.find(s => s.id !== frontSnap?.id);
    return diff || (item.snapshots.length > 1 ? item.snapshots[1] : null);
  }

  hasFeetDetection(item: EventDetail | null): boolean {
    if (!item?.detections) return false;
    return item.detections.some(d => d.class_name?.toLowerCase() === 'feet');
  }

  formatJson(data: EventDetail): string {
    return JSON.stringify(data, null, 2);
  }

  copyJson(data: EventDetail) {
    const text = JSON.stringify(data, null, 2);
    navigator.clipboard.writeText(text).then(() => {
      this.copied.set(true);
      setTimeout(() => this.copied.set(false), 2000);
    });
  }
}
