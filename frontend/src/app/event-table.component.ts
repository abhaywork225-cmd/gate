import { Component, input, output } from '@angular/core';
import { DatePipe } from '@angular/common';
import { GateEvent } from './gate-api.service';

@Component({
  selector: 'gate-event-table', standalone: true, imports: [DatePipe],
  template: `<div class="table-scroll"><table><thead><tr>
    <th>Time</th>
    <th>Container number</th>
    <th>Size / Feet</th>
    <th>Lane</th>
    <th>Movement</th>
    <th>Status</th>
    <th>Site Sync</th>
    <th>Evidence & Actions</th>
  </tr></thead>
  <tbody>@for (event of visibleEvents(); track event.id) {<tr>
    <td>{{ event.timestamp | date:'dd MMM, HH:mm:ss' }}</td>
    <td class="mono font-bold">
      @if (event.is_twin) {
        <div class="twin-container-cell">
          <span class="twin-badge-pill">{{ event.status === 'CONFIRMED' && event.container_number_2 ? 'TWO CONTAINERS' : 'POSSIBLE TWIN' }}</span>
          <div class="twin-numbers">
            <div class="twin-row"><span class="slot-tag front">F</span> {{ event.container_number || 'Unread' }}</div>
            <div class="twin-row"><span class="slot-tag rear">R</span> {{ event.container_number_2 || 'Unread' }}</div>
          </div>
        </div>
      } @else {
        {{ event.container_number || event.candidate_number || (event.status === 'UNREADABLE' ? 'Unreadable · proof retained' : 'Unread / unconfirmed') }}
        @if (!event.container_number && event.candidate_number) { <small>{{ event.candidate_is_partial ? 'Incomplete · check digit unread' : 'Unverified' }}</small> }
      }
    </td>
    <td>
      @if (event.is_twin) {
        <div class="twin-sizes">
          <span class="size-pill feet">📏 {{ event.container_size || 'Size unconfirmed' }} {{ event.size_code ? '(' + event.size_code + ')' : '' }}</span>
          <span class="size-pill feet">📏 {{ event.container_size_2 || 'Size unconfirmed' }} {{ event.size_code_2 ? '(' + event.size_code_2 + ')' : '' }}</span>
        </div>
      } @else if (event.container_size) {
        <span class="size-pill" [class.feet]="event.container_size.includes('FT')">
          📏 {{ event.container_size }}{{ event.size_code ? ' (' + event.size_code + ')' : '' }}
        </span>
      } @else {
        <span class="subtle-dash">—</span>
      }
    </td>
    <td>{{ event.gate_id }}</td>
    <td>
      <span class="badge movement-badge">{{ event.event_type }}</span>
      <div class="table-folder-subtext" [class.folder-in]="event.folder === 'detected_in'" [class.folder-out]="event.folder === 'detected_out'">
        {{ event.folder === 'detected_out' ? '📤 detected_out' : '📥 detected_in' }}
      </div>
    </td>
    <td><span class="badge" [class.good]="event.status === 'CONFIRMED'" [class.warn]="event.status === 'NEEDS_REVIEW'">{{ event.status.replaceAll('_', ' ') }}</span></td>
    <td>
      @if (event.cloud_synced) {
        <span class="badge good cloud-badge">✓ Synced</span>
      } @else if (event.status === 'CONFIRMED') {
        <span class="badge paused cloud-badge">⏸ Paused</span>
      } @else {
        <span class="subtle-dash">—</span>
      }
    </td>
    <td>
      <div class="action-cell">
        <button class="view-event-btn" (click)="selected.emit(event.id)">View event →</button>
        <a class="json-badge-link" [href]="'/api/gate-events/' + event.id + '/json'" [download]="'event_' + (event.container_number || event.id) + '.json'" title="Download event.json">
          event.json
        </a>
      </div>
    </td>
  </tr>} @empty {<tr><td colspan="8" class="empty">No events match this view. Events appear after sufficient detection and OCR evidence.</td></tr>}</tbody></table></div>`
})
export class EventTableComponent {
  events = input<GateEvent[]>([]);
  selected = output<string>();

  visibleEvents(): GateEvent[] {
    return this.events();
  }
}
