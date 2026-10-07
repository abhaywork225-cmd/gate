"""GPU replay of up to three best independent number crops in EVERY visit folder.
Read-only: no event writes or cloud calls. Resume from the JSONL report.
"""
from pathlib import Path
import sys,json,sqlite3,collections,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,torch
from PIL import Image
from app.domain import OCRRead
from app.ocr.validator import ContainerValidator
from app.config.settings import Settings
from app.ocr.ocr_engine import EasyOCREngine
from app.ocr.container_ocr import prepare_ocr_crop,inside_gate

def main():
 root=Path('snapshots');out=Path('data/capture-audit');out.mkdir(exist_ok=True)
 calibration=json.loads(Path('lane_calibration.json').read_text())
 db=sqlite3.connect('file:uploads/ocr-spool/jobs.sqlite3?mode=ro',uri=True);db.row_factory=sqlite3.Row
 grouped=collections.defaultdict(list)
 for row in db.execute('select * from jobs'):
  meta=json.loads(row['metadata']);grouped[row['crop'].split('/')[0]].append((dict(row),meta))
 target=out/'replay-final.jsonl';done=set()
 if target.exists():
  for line in target.read_text().splitlines():done.add(json.loads(line)['folder'])
 events=sqlite3.connect('file:data/gate.db?mode=ro',uri=True)
 review={r[0] for r in events.execute("select id from gate_events where status='NEEDS_REVIEW'")}
 folders=sorted((p for p in root.iterdir() if p.is_dir()),key=lambda p:(p.name!='e0ac0b97-2ab3-4fa8-b79b-4bf1e04e9d6b',p.name not in review,p.name))
 if not torch.cuda.is_available():raise RuntimeError('GPU is required')
 engine=EasyOCREngine(Settings(ocr_gpu=True,ocr_download_enabled=False,prosper_enabled=False));v=ContainerValidator()
 cv2.setNumThreads(1)
 for i,folder in enumerate(folders,1):
  if folder.name in done:continue
  attempts=sorted(grouped[folder.name],key=lambda p:(p[1]['detection']['class_name']=='container_number',p[1].get('proof_quality',0)),reverse=True)
  selected=attempts[:3];results=[]
  for row,meta in selected:
   path=root/row['crop'];im=cv2.imread(str(path))
   if im is None:continue
   cam=meta['detection']['camera_id'];zone=calibration.get(cam,{}).get('occupancy_roi',[0,0,1,1])
   with Image.open(root/row['image']) as image:w,h=image.size
   eligible=inside_gate(meta['detection']['bbox'],(h,w,3),zone)
   started=time.monotonic();r=engine.read(im);valid=v.validate(r);variant='original'
   if not valid.valid_check_digit:
    er=engine.read(prepare_ocr_crop(im));ev=v.validate(er)
    if (ev.valid_check_digit,ev.valid_format,ev.confidence)>(valid.valid_check_digit,valid.valid_format,valid.confidence):r,valid,variant=er,ev,'enhanced'
   results.append({'job_id':row['id'],'file':str(path),'camera':cam,'inside_gate_roi':eligible,'variant':variant,'read':r.__dict__,'validation':valid.__dict__,'seconds':round(time.monotonic()-started,3)})
  votes=collections.Counter(x['validation']['normalized_text'] for x in results if x['inside_gate_roi'] and x['validation']['valid_check_digit'] and x['validation']['confidence']>=.6)
  item={'folder':folder.name,'results':results,'valid_votes':dict(votes),'review_status':folder.name in review}
  with target.open('a') as f:f.write(json.dumps(item)+'\n')
  print('replay',i,'/',len(folders),folder.name,dict(votes),flush=True)
 summary={'folders_replayed':len(folders),'representative_crops_per_folder':3,'report':str(target),'complete':True}
 (out/'replay-final-summary.json').write_text(json.dumps(summary,indent=2));print('complete',flush=True)
if __name__=='__main__':main()
