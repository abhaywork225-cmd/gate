"""Apply only a replay-supported historical event; never dispatch to cloud."""
import json,sys,collections
from pathlib import Path
from datetime import datetime
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
from app.config.settings import Settings
from app.database.database import Database
from app.database.repositories.gate import GateRepository
from app.domain import Detection,Observation,OCRRead
from app.ocr.validator import ContainerValidator
from app.snapshots.snapshot_manager import SnapshotManager
import sqlite3

def main():
 event='e0ac0b97-2ab3-4fa8-b79b-4bf1e04e9d6b'
 settings=Settings(prosper_enabled=False);root=settings.snapshot_directory
 report=json.loads(Path('data/capture-audit/requested-event-full-replay.json').read_text())
 accepted=[x for x in report if x['stationary'] and x['inside_roi'] and x['validation']['valid_check_digit'] and x['validation']['confidence']>=settings.ocr_min_confidence]
 votes=collections.Counter(x['validation']['normalized_text'] for x in accepted)
 if len(votes)!=1 or max(votes.values(),default=0)<settings.ocr_confirmations:raise RuntimeError('Insufficient unambiguous independent stopped-ROI evidence')
 best=max(accepted,key=lambda x:x['validation']['confidence'])
 spool=sqlite3.connect('file:uploads/ocr-spool/jobs.sqlite3?mode=ro',uri=True);spool.row_factory=sqlite3.Row
 rows=[spool.execute('select * from jobs where id=?',(x['job_id'],)).fetchone() for x in accepted]
 if any(r['event_id']!=event for r in rows):raise RuntimeError('Evidence belongs to another event')
 if len({(r['origin'],r['sequence']) for r in rows})<settings.ocr_confirmations:raise RuntimeError('Repeated frames are not independent votes')
 row=next(r for r in rows if r['id']==best['job_id']);m=json.loads(row['metadata'])
 value=ContainerValidator().validate(OCRRead(best['validation']['raw_text'],best['validation']['confidence']))
 det=m['detection'];detection=Detection(**{**det,'frame_timestamp':datetime.fromisoformat(det['frame_timestamp']),'bbox':tuple(det['bbox'])})
 image=cv2.imread(str(root/row['image']))
 if image is None:raise RuntimeError('Source proof is missing')
 db=Database(settings.database_url.get_secret_value());repo=GateRepository(db);before=repo.event(event)
 if not before:raise RuntimeError('Event is missing')
 if before['container_number'] and before['container_number']!=value.normalized_text:raise RuntimeError('Conflicting existing number requires operator review')
 observation=Observation(f'revalidated:{event}:{row["id"]}',m['gate_id'],m['track_id'],detection,value,m['direction'],image,datetime.fromisoformat(m['started_at']),True,all_detections=[Detection(**{**d,'frame_timestamp':datetime.fromisoformat(d['frame_timestamp']),'bbox':tuple(d['bbox'])}) for d in m['all_detections']],event_id=event)
 result=repo.save_observation(observation,SnapshotManager(root),settings.dedup_seconds,ongoing_event_id=event)
 after=repo.event(event)
 Path('data/capture-audit/requested-event-correction.json').write_text(json.dumps({'event_id':event,'independent_frames':len(accepted),'source_job_ids':[r['id'] for r in rows],'before':before,'after':after},indent=2,default=str))
 print({'event_id':event,'container_number':after['container_number'],'status':after['status'],'cloud_synced':after['cloud_synced'],'independent_frames':len(accepted)})
 db.close()
if __name__=='__main__':main()
