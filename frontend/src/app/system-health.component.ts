import { Component, input } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { Health, Metrics } from './gate-api.service';

@Component({
  selector: 'gate-system-health', standalone: true, imports: [DecimalPipe],
  template: `<aside class="system-panel" aria-label="System health and GPU telemetry">
    <div class="system-heading"><h2>System health</h2><span class="status-dot" [class.online]="!stale() && health()?.status === 'READY'"></span></div>
    <p class="system-caption">{{ stale() ? 'Telemetry connection lost' : 'Edge computer · current readings' }}</p>
    <div class="health-line"><span>Database</span><strong [class.error-text]="health()?.database !== 'READY'">{{ health()?.database || 'CHECKING' }}</strong></div>
    <div class="health-line"><span>Detector</span><strong>{{ health()?.models?.detector || '—' }}</strong></div>
    <div class="health-line"><span>OCR engine</span><strong [class.error-text]="health()?.models?.ocr === 'ERROR'">{{ health()?.models?.ocr || '—' }}</strong></div>
    <section class="ocr-panel" aria-label="OCR workload"><div class="ocr-panel-heading"><strong>OCR workload</strong><span class="workload-state" [class.workload-active]="ocrActive()" [class.workload-error]="ocrError()">{{ ocrState() }}</span></div>
      <div class="ocr-counters"><div><small>WAITING</small><strong>{{ pending() }}</strong></div><div><small>SAVED</small><strong>{{ spool()?.total ?? 0 }}</strong></div><div><small>FAILED</small><strong [class.error-text]="(spool()?.failed ?? 0) > 0">{{ spool()?.failed ?? 0 }}</strong></div></div>
      <div class="health-line compact"><span>Capture → OCR queue</span><strong>{{ metrics()?.ocr_save_queue_depth ?? 0 }} / {{ metrics()?.ocr_queue_depth ?? 0 }}</strong></div>
      <div class="health-line compact"><span>Detection rate</span><strong>{{ inferenceRate() | number:'1.1-1' }} fps</strong></div>
      @if (health()?.processing_error || health()?.ocr_spool_error) {<p class="error-text workload-error-text">{{ health()?.processing_error || health()?.ocr_spool_error }}</p>}
    </section>
    <div class="resource"><div><span>CPU utilization</span><strong>{{ stale() || metrics()?.cpu_percent == null ? '—' : (metrics()?.cpu_percent | number:'1.0-0') + '%' }}</strong></div><progress [value]="stale() ? 0 : metrics()?.cpu_percent || 0" max="100" aria-label="CPU utilization"></progress></div>
    <div class="resource"><div><span>System memory</span><strong>{{ stale() || metrics()?.memory_percent == null ? '—' : (metrics()?.memory_percent | number:'1.0-0') + '%' }}</strong></div><progress [value]="stale() ? 0 : metrics()?.memory_percent || 0" max="100" aria-label="System memory utilization"></progress></div>
    <div class="resource"><div><span>GPU utilization</span><strong>{{ stale() || metrics()?.gpu_utilization_percent == null ? '—' : (metrics()?.gpu_utilization_percent | number:'1.0-0') + '%' }}</strong></div><progress [value]="stale() ? 0 : metrics()?.gpu_utilization_percent || 0" max="100" aria-label="GPU utilization"></progress></div>
    <section class="gpu-panel" aria-label="GPU accelerator"><div class="gpu-heading"><span class="chip-icon">GPU</span><small>ACCELERATOR</small></div><h3>{{ metrics()?.gpu_name || 'GPU telemetry unavailable' }}</h3>
      <small>GPU LOAD</small><div class="gpu-power">{{ stale() || metrics()?.gpu_utilization_percent == null ? '—' : (metrics()?.gpu_utilization_percent | number:'1.0-0') }} <span>%</span></div>
      <div class="gpu-values"><div><small>TEMPERATURE</small><strong>{{ stale() || metrics()?.gpu_temperature_c == null ? '—' : (metrics()?.gpu_temperature_c | number:'1.0-0') + ' °C' }}</strong></div>@if (metrics()?.gpu_power_watts != null && !stale()) {<div><small>POWER</small><strong>{{ metrics()?.gpu_power_watts | number:'1.1-1' }} W</strong></div>} @else {<div><small>ENGINE</small><strong>{{ metrics()?.gpu_source || 'tegrastats' }}</strong></div>}</div>
      <div class="gpu-memory"><small>GPU UNIFIED MEMORY</small><span>{{ stale() || metrics()?.gpu_memory_used_mb == null ? 'Unavailable' : (metrics()?.gpu_memory_used_mb | number:'1.0-0') + ' / ' + (metrics()?.gpu_memory_total_mb | number:'1.0-0') + ' MB' }}</span></div>
    </section>
    <div class="context-block">
      <small>CLOUD DISPATCH MODE</small>
      @if (health()?.cloud_sync?.enabled) {
        @if (health()?.cloud_sync?.is_paused) {
          <strong style="color: #f59e0b;">PAUSED (MANUAL REVIEW)</strong>
          <span>Held locally until operator click</span>
        } @else {
          <strong style="color: #10b981;">AUTO DISPATCH ACTIVE</strong>
          <span>Auto-publishing to Smart Yard</span>
        }
      } @else {
        <strong>DISABLED</strong>
        <span>Local edge storage only</span>
      }
    </div>
    <div class="context-block"><small>ADMIN SESSION</small><strong>Not signed in</strong><span>Local testing · login setup deferred</span></div>
  </aside>`
})
export class SystemHealthComponent {
  health = input<Health | null>(null);
  metrics = input<Metrics | null>(null);
  stale = input(false);
  spool() { return this.metrics()?.ocr_spool || this.health()?.ocr_spool; }
  pending() { return this.spool()?.pending ?? 0; }
  ocrError() { return this.health()?.models?.ocr === 'ERROR' || !!this.health()?.ocr_spool_error || !!this.health()?.processing_error; }
  ocrActive() { return !this.stale() && !this.ocrError() && (this.health()?.models?.ocr === 'READY' || this.pending() > 0); }
  ocrState() { if (this.stale()) return 'OFFLINE'; if (this.ocrError()) return 'ERROR'; if (this.pending() > 0 || (this.metrics()?.ocr_save_queue_depth ?? 0) > 0) return 'PROCESSING'; if (this.health()?.models?.ocr === 'READY') return 'READY'; return 'STARTING'; }
  inferenceRate() { return Object.values(this.metrics()?.inference_fps || {}).reduce((sum, value) => sum + Number(value || 0), 0); }
}
