"""Read-only saved-evidence audit; no event writes or cloud calls."""
from pathlib import Path
import sys,json,sqlite3,collections,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
from app.domain import OCRRead
from app.ocr.validator import ContainerValidator

def main():
 out=Path('data/capture-audit');out.mkdir(exist_ok=True)
 root=Path('snapshots');folders=sorted(x for x in root.iterdir() if x.is_dir())
 db=sqlite3.connect('file:data/gate.db?mode=ro',uri=True);db.row_factory=sqlite3.Row
 jobs=sqlite3.connect('file:uploads/ocr-spool/jobs.sqlite3?mode=ro',uri=True);jobs.row_factory=sqlite3.Row
 rows=list(jobs.execute('select * from jobs'));validator=ContainerValidator();historical=[];grouped=collections.defaultdict(list)
 for row in rows:
  meta=json.loads(row['metadata']);old=json.loads(row['validation']) if row['validation'] else {}
  value=validator.validate(OCRRead(old.get('raw_text',''),old.get('confidence',0)))
  folder=row['crop'].split('/')[0];grouped[folder].append((dict(row),meta))
  historical.append({'job_id':row['id'],'folder':folder,'old':old,'strict':value.__dict__})
 (out/'historical.json').write_text(json.dumps(historical,indent=2))
 events={x['id']:dict(x) for x in db.execute('select * from gate_events')};inventory=[]
 for folder in folders:
  paths=[x for x in folder.iterdir() if x.suffix.lower() in ('.png','.jpg','.jpeg')];bad=[]
  for path in paths:
   im=cv2.imread(str(path))
   if im is None or not im.size:bad.append(path.name)
  inventory.append({'folder':folder.name,'event':events.get(folder.name),'images':len(paths),'unreadable_images':bad,'jobs':len(grouped[folder.name])})
  if len(inventory)%25==0:print('inventory',len(inventory),'/',len(folders),flush=True)
 (out/'inventory.json').write_text(json.dumps(inventory,indent=2))
 summary={'folders':len(folders),'images':sum(x['images'] for x in inventory),'corrupt_images':sum(len(x['unreadable_images']) for x in inventory),'jobs':len(rows),'database_statuses':dict(collections.Counter(x['status'] for x in events.values())),'old_valid_now_rejected':sum(bool(x['old'].get('valid_check_digit')) and not x['strict']['valid_check_digit'] for x in historical)}
 (out/'summary.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)

if __name__=='__main__':main()
