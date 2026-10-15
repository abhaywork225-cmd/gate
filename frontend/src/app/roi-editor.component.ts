import { Component, effect, input, output, signal, untracked } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { Camera, LaneCalibration } from './gate-api.service';

@Component({
  selector: 'gate-roi-editor', standalone: true, imports: [DecimalPipe],
  template: `
    <div class="roi-editor" aria-label="Camera detection area editor">
      <div class="roi-toolbar"><div class="roi-title"><span class="roi-icon">⌖</span><div><strong>1. Draw the detection area</strong><small>Include where containers pass or wait. Exclude the road, buildings and other lanes.</small></div></div><div class="roi-context"><span class="calibration-pill">{{ camera().name }}</span><span class="calibration-pill">{{ camera().direction === 'ENTRY' ? 'Gate IN' : 'Gate OUT' }}</span></div></div>
      <div class="roi-clearance">
        <strong>Physical lanes visible from this camera</strong>
        @if (config().lanes?.length) {
          <p>Tracking: {{ camera().passage_tracking?.state || 'Waiting for camera' }}</p>
          @for (passage of camera().passage_tracking?.passages || []; track passage.event_id) {
            <p>{{ passage.lane_id }} · {{ passage.event_id }} · {{passage.candidates}} text regions · {{passage.closed ? 'Finishing OCR' : 'Collecting evidence'}}</p>
          }
        }
        <p>Use the same lane name on every camera that sees that road lane. Click the road corners in order. Truck/trailer detection is required for separate truck events.</p>
        <div class="roi-actions">
          @for (lane of config().lanes || []; track $index) {
            <button type="button" [class.selected]="laneIndex() === $index" (click)="laneIndex.set($index)">{{lane.id}}</button>
          }
          <button type="button" [disabled]="(config().lanes?.length || 0) >= 2" (click)="addLane()">Add lane</button>
          <button type="button" (click)="laneIndex.set(-1)">Edit general ROI</button>
        </div>
        @if (laneIndex() >= 0) {
          <label>Lane name <input [value]="config().lanes?.[laneIndex()]?.id" (change)="renameLane($event)"></label>
          <p>Click 3–12 corners around the lane. Overlapping lane areas are treated as ambiguous.</p>
          <button type="button" (click)="clearLane()">Redraw lane</button>
          <button type="button" (click)="removeLane()">Remove lane</button>
        }
      </div>
      <div class="roi-actions"><button type="button" [class.selected]="tool() === 'draw'" (click)="tool.set('draw')">Draw new area</button><button type="button" [class.selected]="tool() === 'adjust'" (click)="tool.set('adjust')">Move / resize</button><button type="button" (click)="fullFrame()">Use full frame</button><span>{{ tool() === 'draw' ? 'Drag from one corner to the opposite corner.' : 'Drag inside to move. Drag a corner to resize.' }}</span></div>
      <div class="roi-canvas" [style.aspect-ratio]="aspect()" (pointerdown)="start($event)" (pointermove)="move($event)" (pointerup)="end($event)" (pointercancel)="cancel()">
        <svg viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true" style="position:absolute;inset:0;width:100%;height:100%;z-index:3;pointer-events:none">
          @for (lane of config().lanes || []; track $index) {
            <polygon [attr.points]="lanePoints(lane.polygon)" [attr.fill]="$index === 0 ? '#22c55e33' : '#38bdf833'" [attr.stroke]="$index === 0 ? '#22c55e' : '#38bdf8'" stroke-width="0.003"/>
            @for (p of lane.polygon; track $index) { <circle [attr.cx]="p[0]" [attr.cy]="p[1]" r="0.006" fill="white"/> }
          }
        </svg>
        <img [src]="'/api/cameras/' + camera().id + '/frame?calibration=' + refresh()" [alt]="camera().name + ' calibration frame'" draggable="false" (load)="frameLoaded($event)" (error)="ready.set(false)">
        @if (ready()) {<svg viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true"><path [attr.d]="shade()" fill-rule="evenodd" fill="rgba(0,0,0,.55)"/><rect [attr.x]="roi()[0]" [attr.y]="roi()[1]" [attr.width]="roi()[2]-roi()[0]" [attr.height]="roi()[3]-roi()[1]" class="roi-box"/>@if (config().gate_exit_axis === 'x') {<line [attr.x1]="config().exit_line" y1="0" [attr.x2]="config().exit_line" y2="1" class="exit-line"/>}@if (config().gate_exit_axis === 'y') {<line x1="0" [attr.y1]="config().exit_line" x2="1" [attr.y2]="config().exit_line" class="exit-line"/>}</svg>@for (corner of corners; track corner[0]) {<span class="roi-corner" [style.left.%]="roi()[corner[1]] * 100" [style.top.%]="roi()[corner[2]] * 100"></span>}<div class="roi-frame-badge">GREEN = DETECTION AREA</div>@if (config().gate_exit_axis !== null) {<div class="roi-clear-badge">Clear beyond red line {{ directionArrow() }}</div><span class="roi-boundary-marker" [style.left.%]="config().gate_exit_axis === 'x' ? config().exit_line * 100 : 50" [style.top.%]="config().gate_exit_axis === 'y' ? config().exit_line * 100 : 50">Clearance line {{ directionArrow() }} · {{ config().exit_line * 100 | number:'1.0-0' }}%</span>}} @else {<div class="roi-unavailable">Waiting for a camera frame. Connect the camera or start its video to draw.</div>}
      </div>
      <div class="roi-readout"><span>Selected: {{ ((roi()[2]-roi()[0]) * (roi()[3]-roi()[1]) * 100) | number:'1.0-0' }}% of the frame</span><span>{{ dirty() ? 'Unsaved changes' : 'Saved settings loaded' }}</span></div>
      <div class="roi-clearance"><strong>2. When is this view clear?</strong><p>Gate IN / OUT describes the truck’s site movement. You do not need both an entry and an exit line on each camera.</p><label>Clearance method <select [value]="config().gate_exit_axis === null ? 'timer' : 'line'" (change)="setMethod($event)"><option value="timer">Empty detection area (recommended)</option><option value="line">Container passes a visible boundary</option></select></label>@if (config().gate_exit_axis === null) {<p class="roi-help">Ready for the next truck after no container is detected in the green area for {{ calibration().gate_clear_seconds || 7 }} seconds. Use this when the departure point is outside the camera view.</p>} @else {<div class="roi-boundary-controls"><label>Truck leaves toward <select [value]="config().gate_exit_axis + (config().gate_exit_positive ? '+' : '-')" (change)="setDirection($event)"><option value="x+">Right →</option><option value="x-">Left ←</option><option value="y+">Bottom ↓</option><option value="y-">Top ↑</option></select></label><label>Red boundary position <input type="range" min="0.01" max="0.99" step="0.01" [value]="config().exit_line" (input)="setLine($event)"><output>{{ config().exit_line * 100 | number:'1.0-0' }}%</output></label></div><p class="roi-help">The entire container box must pass the red line. The {{ calibration().gate_clear_seconds || 7 }}-second empty-area timer also remains active.</p>}</div>
      <div class="roi-controls"><button type="button" class="roi-reset" [disabled]="!dirty()" (click)="restore()">Discard changes</button><button type="button" class="primary" [disabled]="!dirty() || !ready() || drawing" (click)="submit()">Save detection area</button></div>
    </div>`
})
export class RoiEditorComponent {
  camera = input.required<Camera>(); calibration = input.required<LaneCalibration>(); refresh = input(0); save = output<LaneCalibration>();
  roi = signal<number[]>([0, 0, 1, 1]); config = signal<LaneCalibration>({camera_id:'', occupancy_roi:[0,0,1,1], entry_line:.1, exit_line:.95, gate_exit_axis:null, gate_exit_positive:true, version:1});
  dirty = signal(false); ready = signal(false); aspect = signal(16/9); tool = signal<'draw'|'adjust'>('draw');
  laneIndex = signal(-1);
  lanePoints(points:number[][]){return points.map(p=>p.join(',')).join(' ');}
  addLane(){this.config.update(c=>({...c,lanes:[...(c.lanes || []),{id:'lane-'+((c.lanes?.length || 0)+1),polygon:[]}]}));this.laneIndex.set((this.config().lanes?.length || 1)-1);this.dirty.set(true);}
  clearLane(){this.config.update(c=>({...c,lanes:c.lanes?.map((l,i)=>i===this.laneIndex()?{...l,polygon:[]}:l)}));this.dirty.set(true);}
  removeLane(){this.config.update(c=>({...c,lanes:c.lanes?.filter((_,i)=>i!==this.laneIndex())}));this.laneIndex.set(-1);this.dirty.set(true);}
  renameLane(e:Event){const id=(e.target as HTMLInputElement).value.trim();this.config.update(c=>({...c,lanes:c.lanes?.map((l,i)=>i===this.laneIndex()?{...l,id}:l)}));this.dirty.set(true);}
  corners = [['tl',0,1],['tr',2,1],['bl',0,3],['br',2,3]] as const; drawing=false; private cameraId=''; private anchor=[0,0]; private before=[0,0,1,1]; private mode='draw';
  constructor() { effect(() => { const value=this.calibration(), id=this.camera().id; untracked(() => { const changed=id !== this.cameraId; if (changed) { this.cameraId=id; this.ready.set(false); this.drawing=false; } if (!this.dirty() || changed) { this.config.set({...value}); this.roi.set([...value.occupancy_roi]); this.dirty.set(false); } }); }); }
  frameLoaded(e: Event) { const i=e.target as HTMLImageElement; if(i.naturalWidth&&i.naturalHeight){this.aspect.set(i.naturalWidth/i.naturalHeight);this.ready.set(true);} }
  shade(){const[a,b,c,d]=this.roi();return `M0 0H1V1H0Z M${a} ${b}V${d}H${c}V${b}Z`;}
  directionArrow(){const c=this.config();return c.gate_exit_axis==='x'?(c.gate_exit_positive?'→':'←'):(c.gate_exit_positive?'↓':'↑');}
  start(e:PointerEvent){if(!this.ready()||e.button!==0)return;const p=this.point(e);if(this.laneIndex()>=0){this.config.update(c=>({...c,lanes:c.lanes?.map((l,i)=>i===this.laneIndex()&&l.polygon.length<12?{...l,polygon:[...l.polygon,p]}:l)}));this.dirty.set(true);return;}const r=(e.currentTarget as HTMLElement).getBoundingClientRect();this.before=[...this.roi()];this.anchor=p;this.mode='draw';if(this.tool()==='adjust'){const q=this.corners.find(c=>Math.abs(p[0]-this.before[c[1]])*r.width<18&&Math.abs(p[1]-this.before[c[2]])*r.height<18);if(q)this.mode=q[0];else if(p[0]>=this.before[0]&&p[0]<=this.before[2]&&p[1]>=this.before[1]&&p[1]<=this.before[3])this.mode='move';else return;}this.drawing=true;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);}
  move(e:PointerEvent){if(!this.drawing)return;const[x,y]=this.point(e),b=this.before;if(this.mode==='draw')this.roi.set([Math.min(this.anchor[0],x),Math.min(this.anchor[1],y),Math.max(this.anchor[0],x),Math.max(this.anchor[1],y)]);else if(this.mode==='move'){const dx=Math.max(-b[0],Math.min(1-b[2],x-this.anchor[0])),dy=Math.max(-b[1],Math.min(1-b[3],y-this.anchor[1]));this.roi.set([b[0]+dx,b[1]+dy,b[2]+dx,b[3]+dy]);}else{const n=[...b];if(this.mode.endsWith('l'))n[0]=Math.min(x,b[2]-.02);else n[2]=Math.max(x,b[0]+.02);if(this.mode.startsWith('t'))n[1]=Math.min(y,b[3]-.02);else n[3]=Math.max(y,b[1]+.02);this.roi.set(n);}}
  end(e:PointerEvent){if(!this.drawing)return;this.move(e);this.drawing=false;const b=this.roi();if(b[2]-b[0]<.02||b[3]-b[1]<.02)this.roi.set(this.before);else if(JSON.stringify(b)!==JSON.stringify(this.before)){this.dirty.set(true);this.tool.set('adjust');}}
  cancel(){if(this.drawing)this.roi.set(this.before);this.drawing=false;} fullFrame(){this.roi.set([0,0,1,1]);this.dirty.set(true);} restore(){const c=this.calibration();this.config.set({...c});this.roi.set([...c.occupancy_roi]);this.dirty.set(false);}
  setMethod(e:Event){const on=(e.target as HTMLSelectElement).value==='line';this.config.update(c=>({...c,gate_exit_axis:on?'x':null}));this.dirty.set(true);} setDirection(e:Event){const v=(e.target as HTMLSelectElement).value;this.config.update(c=>({...c,gate_exit_axis:v[0] as 'x'|'y',gate_exit_positive:v[1]==='+'}));this.dirty.set(true);} setLine(e:Event){this.config.update(c=>({...c,exit_line:Number((e.target as HTMLInputElement).value)}));this.dirty.set(true);} submit(){this.save.emit({...this.config(),occupancy_roi:[...this.roi()]});this.dirty.set(false);}
  private point(e:PointerEvent){const r=(e.currentTarget as HTMLElement).getBoundingClientRect();return[Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))];}
}
