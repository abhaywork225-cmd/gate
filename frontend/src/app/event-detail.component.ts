import { Component, input, output, signal } from '@angular/core';
import { DatePipe, PercentPipe } from '@angular/common';
import { EventDetail } from './gate-api.service';

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
                <span class="status-pill" [class.confirmed]="item.status === 'CONFIRMED'" [class.review]="item.status !== 'CONFIRMED'">{{ item.status.replaceAll('_', ' ') }}</span>
                <span class="movement-pill">{{ item.event_type }}</span>
                <span class="meta-item"><strong>Lane:</strong> {{ item.gate_id }}</span>
                <span class="meta-dot">·</span>
                <span class="meta-item"><strong>Time:</strong> {{ item.timestamp | date:'medium' }}</span>
                <span class="meta-dot">·</span>
                <span class="meta-item mono id-meta"><strong>ID:</strong> {{ item.id }}</span>
              </div>
            } @else {
              <h1 class="hero-container-title">
                <span class="container-badge mono">{{ item.container_number || 'UNCONFIRMED / REVIEW NEEDED' }}</span>
                <span class="status-pill" [class.confirmed]="item.status === 'CONFIRMED'" [class.review]="item.status !== 'CONFIRMED'">
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

        @if (showJson()) {
          <div class="json-viewer-panel">
            <div class="json-header">
              <div>
                <strong>event.json</strong>
                <small class="json-path">snapshots/{{ item.id }}/event.json</small>
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
                <h3>Multi-Camera Snapshots & Proof</h3>
                <span class="badge">{{ item.snapshots.length }} viewpoints</span>
              </div>

              @if (item.is_twin) {
                <div class="twin-proof-grid">
                  <!-- Front Container Proof Card -->
                  @if (getFrontSnapshot(item); as frontSnap) {
                    <div class="primary-proof-card twin-proof-card front-proof">
                      <div class="proof-header front-header">
                        <span class="proof-badge">FRONT (SLOT 1): <strong class="mono">{{ item.container_number || 'PENDING OCR' }}</strong></span>
                        <span class="proof-cam mono">{{ frontSnap.camera_id }}</span>
                      </div>
                      <div class="proof-img-wrap">
                        <img [src]="snapshotUrl(frontSnap.id)" [alt]="'Front container proof from ' + frontSnap.camera_id" (error)="markImageError(frontSnap.id)">
                      </div>
                      <div class="proof-caption">
                        <span><strong class="mono">{{ item.container_number || 'Front Container' }}</strong> · {{ frontSnap.timestamp | date:'HH:mm:ss · dd MMM yyyy' }}</span>
                        <a [href]="rawSnapshotUrl(frontSnap.id)" target="_blank" class="view-raw-link">Original frame ↗</a>
                      </div>
                    </div>
                  }

                  <!-- Rear Container Proof Card -->
                  @if (getRearSnapshot(item); as rearSnap) {
                    <div class="primary-proof-card twin-proof-card rear-proof">
                      <div class="proof-header rear-header">
                        <span class="proof-badge">REAR (SLOT 2): <strong class="mono">{{ item.container_number_2 || 'PENDING OCR' }}</strong></span>
                        <span class="proof-cam mono">{{ rearSnap.camera_id }}</span>
                      </div>
                      <div class="proof-img-wrap">
                        <img [src]="snapshotUrl(rearSnap.id)" [alt]="'Rear container proof from ' + rearSnap.camera_id" (error)="markImageError(rearSnap.id)">
                      </div>
                      <div class="proof-caption">
                        <span><strong class="mono">{{ item.container_number_2 || 'Rear Container' }}</strong> · {{ rearSnap.timestamp | date:'HH:mm:ss · dd MMM yyyy' }}</span>
                        <a [href]="rawSnapshotUrl(rearSnap.id)" target="_blank" class="view-raw-link">Original frame ↗</a>
                      </div>
                    </div>
                  }
                </div>
              } @else {
                @if (getPrimarySnapshot(item); as primary) {
                  <div class="primary-proof-card">
                    <div class="proof-header">
                      <span class="proof-badge">PRIMARY CONTAINER PROOF</span>
                      <span class="proof-cam mono">{{ primary.camera_id }}</span>
                    </div>
                    <div class="proof-img-wrap">
                      <img [src]="snapshotUrl(primary.id)" [alt]="'Container proof from ' + primary.camera_id" (error)="markImageError(primary.id)">
                    </div>
                    <div class="proof-caption">
                      <span><strong>{{ item.container_number || 'Detected Container' }}</strong> · {{ primary.timestamp | date:'HH:mm:ss · dd MMM yyyy' }}</span>
                      <a [href]="rawSnapshotUrl(primary.id)" target="_blank" class="view-raw-link">Original frame ↗</a>
                    </div>
                  </div>
                }
              }

              <div class="multi-angle-heading">
                <h4 class="multi-angle-title">All captured camera images ({{ item.snapshots.length }})</h4>
                <span class="image-refresh-note">Matched proof includes the number crop and raw OCR. Full resolution opens the unmodified capture.</span>
              </div>
              <div class="snapshots-grid">
                @for (s of item.snapshots; track s.id) {
                  <figure class="snapshot-card" [class.is-primary]="s.id === item.primary_snapshot_id || s.id === item.rear_snapshot_id">
                    <div class="snapshot-img-wrap">
                      @if (imageErrors()[s.id]) {
                        <div class="snapshot-error" role="status">
                          <span class="snapshot-error-icon">⚠</span>
                          <strong>Image unavailable</strong>
                          <small>{{ s.camera_id }} · refresh the event</small>
                        </div>
                      } @else {
                        <img [src]="snapshotUrl(s.id)" [alt]="'Snapshot from ' + s.camera_id" (error)="markImageError(s.id)" (load)="markImageLoaded(s.id)">
                        @if (!imageLoaded()[s.id]) {
                          <span class="snapshot-loading">Loading proof view…</span>
                        }
                      }
                      <span class="camera-tag">{{ s.camera_id }}</span>
                      @if (s.id === item.primary_snapshot_id) {
                        <span class="primary-tag">FRONT PROOF</span>
                      } @else if (s.id === item.rear_snapshot_id) {
                        <span class="primary-tag rear-tag">REAR PROOF</span>
                      }
                    </div>
                    <figcaption>
                      <span class="snap-time">{{ s.timestamp | date:'HH:mm:ss · dd MMM yyyy' }}</span>
                      <a [href]="'/api/snapshots/' + s.id" target="_blank" class="view-raw-link">Full resolution ↗</a>
                    </figcaption>
                  </figure>
                } @empty {
                  <p class="empty-hint">No camera snapshots recorded for this event.</p>
                }
              </div>
            </div>
          </div>
        </div>
      </div>
    }
  `
})
export class EventDetailComponent {
  event = input<EventDetail | null>(null);
  back = output<void>();
  showJson = signal(false);
  copied = signal(false);
  imageLoaded = signal<Record<string, boolean>>({});
  imageErrors = signal<Record<string, boolean>>({});

  snapshotUrl(id: string): string {
    // The proof view is the clean original camera capture. OCR crops are
    // internal processing artifacts and are never shown as user proof.
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
    return item.snapshots.length > 1 ? item.snapshots[1] : null;
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
