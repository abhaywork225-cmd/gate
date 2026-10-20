import { Component, effect, input, output, signal, untracked } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { Camera, LaneCalibration } from './gate-api.service';

@Component({
  selector: 'gate-roi-editor', standalone: true, imports: [DecimalPipe],
  template: `
    <div class="roi-editor" aria-label="Camera detection area and virtual laser line editor">
      <div class="roi-toolbar">
        <div class="roi-title">
          <span class="roi-icon">⚡</span>
          <div>
            <strong>1. Draw Detection Area (ROI) & Virtual Laser Lines</strong>
            <small>Configure the detection zone and virtual laser tripwire sensor for optical clearance.</small>
          </div>
        </div>
        <div class="roi-context">
          <span class="calibration-pill">{{ camera().name }}</span>
          <span class="calibration-pill">{{ camera().direction === 'ENTRY' ? 'Gate IN' : 'Gate OUT' }}</span>
        </div>
      </div>

      <div class="roi-actions">
        <button type="button" [class.selected]="tool() === 'draw'" (click)="tool.set('draw')">Draw new area</button>
        <button type="button" [class.selected]="tool() === 'adjust'" (click)="tool.set('adjust')">Move / resize</button>
        <button type="button" (click)="fullFrame()">Use full frame</button>
        <span>{{ tool() === 'draw' ? 'Drag from one corner to the opposite corner.' : 'Drag inside to move. Drag a corner to resize.' }}</span>
      </div>

      <div class="roi-canvas" [style.aspect-ratio]="aspect()" (pointerdown)="start($event)" (pointermove)="move($event)" (pointerup)="end($event)" (pointercancel)="cancel()">
        <svg viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true" style="position:absolute;inset:0;width:100%;height:100%;z-index:3;pointer-events:none">
          @for (lane of config().lanes || []; track $index) {
            <polygon [attr.points]="lanePoints(lane.polygon)" [attr.fill]="$index === 0 ? '#22c55e33' : '#38bdf833'" [attr.stroke]="$index === 0 ? '#22c55e' : '#38bdf8'" stroke-width="0.003"/>
            @for (p of lane.polygon; track $index) { <circle [attr.cx]="p[0]" [attr.cy]="p[1]" r="0.006" fill="white"/> }
          }
        </svg>

        <img [src]="'/api/cameras/' + camera().id + '/frame?calibration=' + refresh()" [alt]="camera().name + ' calibration frame'" draggable="false" (load)="frameLoaded($event)" (error)="ready.set(false)">

        @if (ready()) {
          <svg viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true">
            <defs>
              <filter id="laserGlowRed" x="-20%" y="-20%" width="140%" height="140%">
                <feDropShadow dx="0" dy="0" stdDeviation="0.008" flood-color="#ef4444" flood-opacity="0.9"/>
              </filter>
              <filter id="laserGlowCyan" x="-20%" y="-20%" width="140%" height="140%">
                <feDropShadow dx="0" dy="0" stdDeviation="0.006" flood-color="#06b6d4" flood-opacity="0.85"/>
              </filter>
            </defs>
            <path [attr.d]="shade()" fill-rule="evenodd" fill="rgba(0,0,0,.55)"/>
            <rect [attr.x]="roi()[0]" [attr.y]="roi()[1]" [attr.width]="roi()[2]-roi()[0]" [attr.height]="roi()[3]-roi()[1]" class="roi-box"/>

            <!-- Virtual Laser Line: Entry Trigger Sensor (Cyan) -->
            @if (config().gate_exit_axis === 'x') {
              <line [attr.x1]="config().entry_line" y1="0" [attr.x2]="config().entry_line" y2="1" stroke="#06b6d4" stroke-width="0.004" stroke-dasharray="0.015 0.008" filter="url(#laserGlowCyan)"/>
            }
            @if (config().gate_exit_axis === 'y') {
              <line x1="0" [attr.y1]="config().entry_line" x2="1" [attr.y2]="config().entry_line" stroke="#06b6d4" stroke-width="0.004" stroke-dasharray="0.015 0.008" filter="url(#laserGlowCyan)"/>
            }

            <!-- Virtual Laser Line: Exit Clearance Sensor (Red/Crimson) -->
            @if (config().gate_exit_axis === 'x') {
              <line [attr.x1]="config().exit_line" y1="0" [attr.x2]="config().exit_line" y2="1" stroke="#ef4444" stroke-width="0.006" stroke-dasharray="0.02 0.01" filter="url(#laserGlowRed)"/>
            }
            @if (config().gate_exit_axis === 'y') {
              <line x1="0" [attr.y1]="config().exit_line" x2="1" [attr.y2]="config().exit_line" stroke="#ef4444" stroke-width="0.006" stroke-dasharray="0.02 0.01" filter="url(#laserGlowRed)"/>
            }
          </svg>

          @for (corner of corners; track corner[0]) {
            <span class="roi-corner" [style.left.%]="roi()[corner[1]] * 100" [style.top.%]="roi()[corner[2]] * 100"></span>
          }

          <div class="roi-frame-badge">GREEN = DETECTION ZONE</div>

          @if (config().gate_exit_axis !== null) {
            <div class="roi-clear-badge">⚡ VIRTUAL LASER ACTIVE · {{ directionArrow() }}</div>

            <!-- Entry Trigger Laser Label -->
            <span class="laser-badge-entry"
                  [style.left.%]="config().gate_exit_axis === 'x' ? config().entry_line * 100 : 8"
                  [style.top.%]="config().gate_exit_axis === 'y' ? config().entry_line * 100 : 8">
              ⚡ Laser Trigger ({{ config().entry_line * 100 | number:'1.0-0' }}%)
            </span>

            <!-- Exit Clearance Laser Label -->
            <span class="laser-badge-exit"
                  [style.left.%]="config().gate_exit_axis === 'x' ? config().exit_line * 100 : 60"
                  [style.top.%]="config().gate_exit_axis === 'y' ? config().exit_line * 100 : 85">
              ⚡ Laser Clearance {{ directionArrow() }} ({{ config().exit_line * 100 | number:'1.0-0' }}%)
            </span>
          }
        } @else {
          <div class="roi-unavailable">Waiting for a camera frame. Connect the camera or start its video to draw.</div>
        }
      </div>

      <div class="roi-readout">
        <span>Selected: {{ ((roi()[2]-roi()[0]) * (roi()[3]-roi()[1]) * 100) | number:'1.0-0' }}% of frame</span>
        <span>{{ dirty() ? 'Unsaved changes' : 'Saved settings loaded' }}</span>
      </div>

      <!-- Section 2: Virtual Laser Line Configuration -->
      <div class="roi-clearance">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
          <strong style="color:#38bdf8;font-size:13px;">⚡ 2. Virtual Laser Line (Tripwire Sensor) & Clearance</strong>
          <span style="font-size:10px;padding:2px 6px;border-radius:4px;background:#0c4a6e;color:#7dd3fc;">OpenCV Virtual Sensor</span>
        </div>
        <p>Acts like a physical laser sensor across the gate. Vehicles tripping the laser line trigger immediate high-res capture and departure clearance. Stationary vehicles dwelling in the zone will never trigger false duplicate events.</p>

        <label style="margin-bottom:10px;">
          <span>Operation Mode:</span>
          <select [value]="config().gate_exit_axis === null ? 'timer' : 'line'" (change)="setMethod($event)">
            <option value="line">⚡ Virtual Laser Line (Tripwire Sensor — Recommended)</option>
            <option value="timer">⏱ Area Dwell Timer (Legacy Clearance Fallback)</option>
          </select>
        </label>

        @if (config().gate_exit_axis === null) {
          <p class="roi-help">
            ⏱ Dwell timer mode: Gate clears when no container is detected inside the green area for {{ calibration().gate_clear_seconds || 7 }} seconds.
          </p>
        } @else {
          <div class="roi-boundary-controls" style="background:#071522;padding:12px;border-radius:8px;border:1px solid #164e63;margin-top:8px;">
            <div style="display:flex;align-items:center;gap:12px;width:100%;flex-wrap:wrap;">
              <label>
                <span>Laser Sensor Orientation:</span>
                <select [value]="config().gate_exit_axis + (config().gate_exit_positive ? '+' : '-')" (change)="setDirection($event)">
                  <option value="y+">Horizontal Line · Truck crosses Downward ↓ (Default Entry/Exit)</option>
                  <option value="y-">Horizontal Line · Truck crosses Upward ↑</option>
                  <option value="x+">Vertical Line · Truck crosses Rightward →</option>
                  <option value="x-">Vertical Line · Truck crosses Leftward ←</option>
                </select>
              </label>
            </div>

            <!-- Exit Clearance Laser Line Controls -->
            <div style="margin-top:12px;width:100%;">
              <div style="display:flex;align-items:center;gap:10px;justify-content:space-between;">
                <label style="color:#ef4444;font-weight:600;">
                  🔴 Laser Clearance Line (Exit): {{ config().exit_line * 100 | number:'1.0-0' }}%
                </label>
                <div style="display:flex;gap:4px;">
                  <button type="button" class="btn-preset" (click)="setPresetLine(0.50)">50% Center</button>
                  <button type="button" class="btn-preset" (click)="setPresetLine(0.75)">75%</button>
                  <button type="button" class="btn-preset" (click)="setPresetLine(0.85)">85% Barrier</button>
                  <button type="button" class="btn-preset" (click)="setPresetLine(0.95)">95% Exit</button>
                </div>
              </div>
              <input type="range" min="0.05" max="0.98" step="0.01" style="width:100%;accent-color:#ef4444;margin-top:6px;" [value]="config().exit_line" (input)="setLine($event)">
            </div>

            <!-- Entry Trigger Laser Line Controls -->
            <div style="margin-top:12px;width:100%;">
              <div style="display:flex;align-items:center;gap:10px;justify-content:space-between;">
                <label style="color:#06b6d4;font-weight:600;">
                  🔵 Laser Trigger Line (Entry): {{ config().entry_line * 100 | number:'1.0-0' }}%
                </label>
                <div style="display:flex;gap:4px;">
                  <button type="button" class="btn-preset" (click)="setPresetEntryLine(0.10)">10% Top</button>
                  <button type="button" class="btn-preset" (click)="setPresetEntryLine(0.20)">20% Entry</button>
                  <button type="button" class="btn-preset" (click)="setPresetEntryLine(0.35)">35% In-Gate</button>
                </div>
              </div>
              <input type="range" min="0.02" max="0.90" step="0.01" style="width:100%;accent-color:#06b6d4;margin-top:6px;" [value]="config().entry_line" (input)="setEntryLine($event)">
            </div>

            <p style="color:#38bdf8;font-size:10px;margin-top:10px;line-height:1.4;">
              💡 <strong>Optical Tripwire Operation:</strong> Moving vehicles trigger evidence freeze at the cyan laser line and complete clearance when crossing the red laser line.
            </p>
          </div>
        }
      </div>

      <div class="roi-controls">
        <button type="button" class="roi-reset" [disabled]="!dirty()" (click)="restore()">Discard changes</button>
        <button type="button" class="primary" [disabled]="!dirty() || !ready() || drawing" (click)="submit()">Save Virtual Line & ROI</button>
      </div>
    </div>`
})
export class RoiEditorComponent {
  camera = input.required<Camera>(); calibration = input.required<LaneCalibration>(); refresh = input(0); save = output<LaneCalibration>();
  roi = signal<number[]>([0, 0, 1, 1]); config = signal<LaneCalibration>({camera_id:'', occupancy_roi:[0,0,1,1], entry_line:.1, exit_line:.85, gate_exit_axis:'y', gate_exit_positive:true, version:1});
  dirty = signal(false); ready = signal(false); aspect = signal(16/9); tool = signal<'draw'|'adjust'>('draw');
  laneIndex = signal(-1);
  lanePoints(points:number[][]){return points.map(p=>p.join(',')).join(' ');}
  addLane(){this.config.update(c=>({...c,lanes:[...(c.lanes || []),{id:'lane-'+((c.lanes?.length || 0)+1),polygon:[]}]}));this.laneIndex.set((this.config().lanes?.length || 1)-1);this.dirty.set(true);}
  clearLane(){this.config.update(c=>({...c,lanes:c.lanes?.map((l,i)=>i===this.laneIndex()?{...l,polygon:[]}:l)}));this.dirty.set(true);}
  removeLane(){this.config.update(c=>({...c,lanes:c.lanes?.filter((_,i)=>i!==this.laneIndex())}));this.laneIndex.set(-1);this.dirty.set(true);}
  renameLane(e:Event){const id=(e.target as HTMLInputElement).value.trim();this.config.update(c=>({...c,lanes:c.lanes?.map((l,i)=>i===this.laneIndex()?{...l,id}:l)}));this.dirty.set(true);}
  corners = [['tl',0,1],['tr',2,1],['bl',0,3],['br',2,3]] as const; drawing=false; private cameraId=''; private anchor=[0,0]; private before=[0,0,1,1]; private mode='draw';
  constructor() { effect(() => { const value=this.calibration(), id=this.camera().id; untracked(() => { const changed=id !== this.cameraId; if (changed) { this.cameraId=id; this.ready.set(false); this.drawing=false; } if (!this.dirty() || changed) { this.config.set({...value, gate_exit_axis: value.gate_exit_axis || 'y', exit_line: value.exit_line || 0.85, entry_line: value.entry_line || 0.15}); this.roi.set([...value.occupancy_roi]); this.dirty.set(false); } }); }); }
  frameLoaded(e: Event) { const i=e.target as HTMLImageElement; if(i.naturalWidth&&i.naturalHeight){this.aspect.set(i.naturalWidth/i.naturalHeight);this.ready.set(true);} }
  shade(){const[a,b,c,d]=this.roi();return `M0 0H1V1H0Z M${a} ${b}V${d}H${c}V${b}Z`;}
  directionArrow(){const c=this.config();return c.gate_exit_axis==='x'?(c.gate_exit_positive?'→':'←'):(c.gate_exit_positive?'↓':'↑');}
  start(e:PointerEvent){if(!this.ready()||e.button!==0)return;const p=this.point(e);if(this.laneIndex()>=0){this.config.update(c=>({...c,lanes:c.lanes?.map((l,i)=>i===this.laneIndex()&&l.polygon.length<12?{...l,polygon:[...l.polygon,p]}:l)}));this.dirty.set(true);return;}const r=(e.currentTarget as HTMLElement).getBoundingClientRect();this.before=[...this.roi()];this.anchor=p;this.mode='draw';if(this.tool()==='adjust'){const q=this.corners.find(c=>Math.abs(p[0]-this.before[c[1]])*r.width<18&&Math.abs(p[1]-this.before[c[2]])*r.height<18);if(q)this.mode=q[0];else if(p[0]>=this.before[0]&&p[0]<=this.before[2]&&p[1]>=this.before[1]&&p[1]<=this.before[3])this.mode='move';else return;}this.drawing=true;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);}
  move(e:PointerEvent){if(!this.drawing)return;const[x,y]=this.point(e),b=this.before;if(this.mode==='draw')this.roi.set([Math.min(this.anchor[0],x),Math.min(this.anchor[1],y),Math.max(this.anchor[0],x),Math.max(this.anchor[1],y)]);else if(this.mode==='move'){const dx=Math.max(-b[0],Math.min(1-b[2],x-this.anchor[0])),dy=Math.max(-b[1],Math.min(1-b[3],y-this.anchor[1]));this.roi.set([b[0]+dx,b[1]+dy,b[2]+dx,b[3]+dy]);}else{const n=[...b];if(this.mode.endsWith('l'))n[0]=Math.min(x,b[2]-.02);else n[2]=Math.max(x,b[0]+.02);if(this.mode.startsWith('t'))n[1]=Math.min(y,b[3]-.02);else n[3]=Math.max(y,b[1]+.02);this.roi.set(n);}}
  end(e:PointerEvent){if(!this.drawing)return;this.move(e);this.drawing=false;const b=this.roi();if(b[2]-b[0]<.02||b[3]-b[1]<.02)this.roi.set(this.before);else if(JSON.stringify(b)!==JSON.stringify(this.before)){this.dirty.set(true);this.tool.set('adjust');}}
  cancel(){if(this.drawing)this.roi.set(this.before);this.drawing=false;} fullFrame(){this.roi.set([0,0,1,1]);this.dirty.set(true);} restore(){const c=this.calibration();this.config.set({...c});this.roi.set([...c.occupancy_roi]);this.dirty.set(false);}
  setMethod(e:Event){const on=(e.target as HTMLSelectElement).value==='line';this.config.update(c=>({...c,gate_exit_axis:on?'y':null}));this.dirty.set(true);}
  setDirection(e:Event){const v=(e.target as HTMLSelectElement).value;this.config.update(c=>({...c,gate_exit_axis:v[0] as 'x'|'y',gate_exit_positive:v[1]==='+'}));this.dirty.set(true);}
  setLine(e:Event){this.config.update(c=>({...c,exit_line:Number((e.target as HTMLInputElement).value)}));this.dirty.set(true);}
  setPresetLine(val:number){this.config.update(c=>({...c,exit_line:val}));this.dirty.set(true);}
  setEntryLine(e:Event){this.config.update(c=>({...c,entry_line:Number((e.target as HTMLInputElement).value)}));this.dirty.set(true);}
  setPresetEntryLine(val:number){this.config.update(c=>({...c,entry_line:val}));this.dirty.set(true);}
  submit(){this.save.emit({...this.config(),occupancy_roi:[...this.roi()]});this.dirty.set(false);}
  private point(e:PointerEvent){const r=(e.currentTarget as HTMLElement).getBoundingClientRect();return[Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))];}
}

