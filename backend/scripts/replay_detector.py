"""Read-only GPU detector replay for every saved visit folder."""
import json,sqlite3,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
from dataclasses import asdict
from datetime import datetime
from app.domain import Frame
from app.config.settings import Settings
from app.detection.yolo_detector import YoloDetector
from app.ocr.container_ocr import inside_gate

def main():
 root=Path('snapshots');out=Path('data/capture-audit/detector-replay.jsonl');cal=json.loads(Path('lane_calibration.json').read_text());c=sqlite3.connect('file:uploads/ocr-spool/jobs.sqlite3?mode=ro',uri=True);choices={}
 for metadata,image in c.execute('select metadata,image from jobs'):
  m=json.loads(metadata);folder=image.split('/')[0]
  if folder not in choices or m.get('proof_quality',0)>choices[folder][0]:choices[folder]=(m.get('proof_quality',0),m,image)
 settings=Settings(prosper_enabled=False);settings.model_device='0';detector=YoloDetector(settings);done=set()
 if out.exists():done={json.loads(line)['folder'] for line in out.read_text().splitlines()}
 for i,folder in enumerate(sorted(x for x in root.iterdir() if x.is_dir()),1):
  if folder.name in done:continue
  selected=choices.get(folder.name)
  if not selected:continue
  _,m,path=selected;im=cv2.imread(str(root/path));cam=m['detection']['camera_id'];frame=Frame(cam,datetime.fromisoformat(m['frame']['timestamp']),m['frame']['sequence'],im,m['frame']['source_id'])
  start=time.monotonic();ds=detector.detect(frame);zone=cal.get(cam,{}).get('occupancy_roi',[0,0,1,1])
  result={'folder':folder.name,'file':str(root/path),'seconds':round(time.monotonic()-start,3),'detections':[{'class':d.class_name,'confidence':d.confidence,'bbox':d.bbox,'inside_roi':inside_gate(d.bbox,im.shape,zone)} for d in ds]}
  with out.open('a') as f:f.write(json.dumps(result)+'\n')
  print(i,folder.name,len(ds),flush=True)
 Path('data/capture-audit/detector-model.json').write_text(json.dumps({'names':detector.model.names,'model':str(settings.model_path),'device':settings.model_device},indent=2))
 print('complete',flush=True)
if __name__=='__main__':main()
