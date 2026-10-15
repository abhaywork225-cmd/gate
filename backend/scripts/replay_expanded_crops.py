"""Recheck every folder with bounded full-frame context for failed OCR crops."""
import sys,json,sqlite3,time,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
from app.config.settings import Settings
from app.ocr.ocr_engine import EasyOCREngine
from app.ocr.validator import ContainerValidator
from app.ocr.container_ocr import expand_number_crop,prepare_ocr_crop

def main():
 out=Path('data/capture-audit');root=Path('snapshots');cal=json.loads(Path('lane_calibration.json').read_text());source=[json.loads(s) for s in (out/'replay-final.jsonl').read_text().splitlines()];target=out/'replay-context.jsonl';done=set()
 if target.exists():done={json.loads(s)['folder'] for s in target.read_text().splitlines()}
 c=sqlite3.connect('file:uploads/ocr-spool/jobs.sqlite3?mode=ro',uri=True);c.row_factory=sqlite3.Row
 e=EasyOCREngine(Settings(ocr_gpu=True,ocr_download_enabled=False));v=ContainerValidator();cv2.setNumThreads(1)
 for i,item in enumerate(source,1):
  if item['folder'] in done:continue
  results=[]
  for r in item['results']:
   row=c.execute('select * from jobs where id=?',(r['job_id'],)).fetchone();m=json.loads(row['metadata']);result={**r,'stationary':m.get('is_stationary',False)}
   if not r['validation']['valid_check_digit'] and r['inside_gate_roi'] and tuple(m['roi'])==(0,0,1,1):
    image=cv2.imread(str(root/row['image']));zone=cal[m['detection']['camera_id']]['occupancy_roi'];crop=expand_number_crop(image,m['detection']['bbox'],zone)
    read=e.read(crop);value=v.validate(read)
    if not value.valid_check_digit:
     er=e.read(prepare_ocr_crop(crop));ev=v.validate(er)
     if ev.valid_check_digit:read,value=er,ev
    if value.valid_check_digit:
     result={**result,'variant':'expanded_inside_gate_roi','read':read.__dict__,'validation':value.__dict__}
   results.append(result)
  votes=collections.Counter(r['validation']['normalized_text'] for r in results if r['stationary'] and r['inside_gate_roi'] and r['validation']['valid_check_digit'] and r['validation']['confidence']>=.6)
  updated={**item,'results':results,'stopped_valid_votes':dict(votes)}
  with target.open('a') as f:f.write(json.dumps(updated)+'\n')
  print(i,'/',len(source),item['folder'],dict(votes),flush=True)
 print('complete',flush=True)
if __name__=='__main__':main()
