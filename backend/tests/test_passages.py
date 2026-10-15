from dataclasses import replace
from datetime import timedelta
from collections import Counter
import numpy as np
from app.config.settings import CameraConfig, LaneRegion
from app.domain import Detection, Frame, Validation
from app.events.passages import PassageCollector, Passage, Region, lane_for_box


def setup_view(settings):
    lanes = [LaneRegion(id='one', polygon=[(0,0),(.49,0),(.49,1),(0,1)]),
             LaneRegion(id='two', polygon=[(.51,0),(1,0),(1,1),(.51,1)])]
    return PassageCollector(settings), CameraConfig(id='camera-1',name='view',gate_id='lane-1',
        source_env='CAMERA_1_RTSP', direction='ENTRY',lanes=lanes)


def test_two_trucks_get_separate_events_and_lane_survives_movement(settings, make_job):
    collector, camera = setup_view(settings)
    base = make_job().frame
    frame = replace(base,image=np.random.default_rng(0).integers(0,255,(100,200,3),dtype=np.uint8))
    body1=Detection(camera.id,frame.timestamp,'truck',.9,(10,10,90,95))
    body2=replace(body1,bbox=(110,10,190,95))
    texts=[replace(body1,class_name='container_number',bbox=(20,30,60,45)),
           replace(body2,class_name='container_number',bbox=(120,30,160,45))]
    collector.update(camera,frame,[body1,body2,*texts])
    assert len(collector.passages)==2
    assert {p.lane_id for p in collector.passages.values()}=={'one','two'}
    assert all(len(p.regions)==1 for p in collector.passages.values())
    first=next(p for p in collector.passages.values() if p.lane_id=='one')
    collector.update(camera,replace(frame,timestamp=frame.timestamp+timedelta(seconds=1)),
                     [replace(body1,bbox=(30,10,110,95)),body2])
    assert first.lane_id=='one' and not first.closed


def test_text_only_does_not_invent_trucks_and_buffer_is_bounded(settings, make_job):
    collector,camera=setup_view(settings)
    frame=replace(make_job().frame,image=np.random.default_rng(1).integers(0,255,(100,200,3),dtype=np.uint8))
    text=Detection(camera.id,frame.timestamp,'container_number',.9,(20,30,60,45))
    collector.update(camera,frame,[text])
    assert not collector.passages
    body=replace(text,class_name='truck',bbox=(10,10,90,95))
    for i in range(20):
        collector.update(camera,replace(frame,sequence=i,timestamp=frame.timestamp+timedelta(seconds=i/5)),[body,text])
    region=next(iter(collector.passages.values())).regions[0]
    assert len(region.candidates)==3


def test_camera_name_does_not_assign_lane(settings, make_job):
    _,camera=setup_view(settings)
    assert lane_for_box((110,10,190,95),(100,200,3),camera.lanes)=='two'
    assert lane_for_box((90,10,110,95),(100,200,3),camera.lanes) is None


def test_multicamera_requires_shared_identity_and_never_groups_unaligned_files(settings, make_job):
    collector,_=setup_view(settings)
    ts=make_job().frame.timestamp
    a=Passage('a','camera-1','s','gate','one','ENTRY',ts,ts,1,(0,0,100,100))
    a.regions=[Region(votes=Counter({'CSQU3054383':2}))]
    b=replace(a,id='b',camera_id='camera-2',regions=[Region(votes=Counter({'CSQU3054383':2}))])
    collector.passages={'a':a,'b':b}
    assert len(collector.groups())==1
    b.lane_id='two'
    assert len(collector.groups())==2


def test_multicamera_distinct_numbers_remain_separate(settings, make_job):
    collector, _ = setup_view(settings)
    ts = make_job().frame.timestamp
    a = Passage('a', 'camera-1', 's1', 'gate', 'one', 'ENTRY', ts, ts + timedelta(seconds=2), 1, (0,0,100,100))
    a.regions = [Region(votes=Counter({'CSQU3054383':2}))]
    b = Passage('b', 'camera-2', 's2', 'gate', 'one', 'ENTRY', ts + timedelta(seconds=1), ts + timedelta(seconds=3), 2, (0,0,100,100))
    b.regions = [Region(votes=Counter({'MSCU6639870':2}))]
    collector.passages = {'a': a, 'b': b}
    groups = collector.groups()
    assert len(groups) == 2
    assert not a.ambiguous and not b.ambiguous
    b.lane_id='one';b.source_type='file'
    assert len(collector.groups())==2


def test_passage_twin_and_conflicting_iso_stays_review(repository,snapshots,make_job):
    ts=make_job().frame.timestamp
    p=Passage('passage-one','camera-1','s','lane-1','one','ENTRY',ts,ts,1,(0,0,150,100),closed=True)
    for i,number in enumerate(['CSQU3054383','MSCU6639870']):
        val=Validation(number,number,.95,True,True,'VALID','22G1')
        job=make_job(origin=f'passage:one:{i}')
        p.regions.append(Region(votes=Counter({number:2}),reads={number:(val,job)}))
    repository.save_passage(p,snapshots,2)
    event=repository.event(p.id)
    assert event['is_twin'] and event['status']=='CONFIRMED'
    assert event['lane_id']=='one' and len(event['snapshots'])==4  # two proofs and raw context
    assert event['rear_snapshot_id']
    p.id='passage-two'
    for i,region in enumerate(p.regions):
        number=next(iter(region.reads));val,job=region.reads[number]
        region.reads[number]=(replace(val,size_type='45G1' if i else '22G1'),replace(job,origin_key=f'passage:two:{i}'))
    repository.save_passage(p,snapshots,2)
    assert repository.event(p.id)['status']=='NEEDS_REVIEW'


def test_cloud_ack_is_persisted(repository,snapshots,make_job):
    from app.domain import Observation
    job=make_job()
    val=Validation('CSQU3054383','CSQU3054383',.95,True,True,'VALID','22G1')
    obs=Observation(job.origin_key,job.gate_id,job.track_id,job.detection,val,
                    job.direction,job.frame.image,job.started_at,True)
    identity,_=repository.save_observation(obs,snapshots,30)
    repository.mark_cloud_synced(identity,'remote-id')
    event=repository.event(identity)
    assert event['cloud_synced'] and event['cloud_synced_at'] and event['cloud_event_id']=='remote-id'


def test_runtime_keeps_passage_open_until_clear_and_confirms_two_slots(settings,repository,snapshots,make_job):
    from app.runtime import GateRuntime
    settings.ocr_confirmations=2
    camera=CameraConfig(id='camera-1',name='view',gate_id='lane-1',source_env='CAMERA_1_RTSP',
        direction='ENTRY',lanes=[LaneRegion(id='one',polygon=[(0,0),(1,0),(1,1),(0,1)])])
    runtime=GateRuntime(settings,[camera],repository,snapshots)
    class Detector:
        empty=False
        def detect(self,frame):
            if self.empty: return []
            return [Detection('camera-1',frame.timestamp,'truck',.95,(5,5,195,99)),
                Detection('camera-1',frame.timestamp,'container_number',.9,(20,30,65,45)),
                Detection('camera-1',frame.timestamp,'container_number',.9,(120,30,165,45))]
    runtime.detector=Detector()
    runtime.model_status['detector'] = 'READY'
    frame=replace(make_job().frame,image=np.random.default_rng(2).integers(0,255,(100,200,3),dtype=np.uint8))
    for sequence in (1,2):
        runtime.process_frame(camera,replace(frame,sequence=sequence,timestamp=frame.timestamp+timedelta(seconds=sequence)))
        assert runtime.queue.qsize()==2
        for _ in range(2):
            job=runtime.queue.get_nowait()
            # Number candidates must remain isolated even on the same image.
            assert len([d for d in job.all_detections if d.class_name=='container_number'])==1
            identity='CSQU3054383' if job.detection.bbox[0]<100 else 'MSCU6639870'
            index=runtime.ocr_store.save(job,job.frame.image[30:45,20:65])
            runtime.ocr_store.save_validation(index,Validation(identity,identity,.95,True,True,'VALID','22G1'))
            runtime._process_saved_ocr(index)
            runtime.queue.task_done()
        runtime._finalize_passages()
        assert repository.events()['total']==0
    runtime.detector.empty=True
    for second in range(3,11):
        runtime.process_frame(camera,replace(frame,sequence=second,timestamp=frame.timestamp+timedelta(seconds=second)))
    runtime._finalize_passages()
    event=repository.events()['items'][0]
    assert event['is_twin'] and event['status']=='CONFIRMED' and event['lane_id']=='one'
    assert not runtime.passages.passages


def test_completed_ocr_recovers_after_restart_as_review(settings,repository,snapshots,make_job):
    from app.runtime import GateRuntime
    camera=CameraConfig(id='camera-1',name='view',gate_id='lane-1',source_env='CAMERA_1_RTSP')
    runtime=GateRuntime(settings,[camera],repository,snapshots)
    val=Validation('CSQU3054383','CSQU3054383',.95,True,True,'VALID','22G1')
    for seq in (1,2):
        job=replace(make_job(seq,origin='passage:recovered:region'),passage_id='recovered',
            region_id='region',passage_metadata={'lane_id':'one','bbox':[0,0,150,100],'source_type':'file'})
        idx=runtime.ocr_store.save(job,job.frame.image)
        runtime.ocr_store.save_validation(idx,val)
        runtime.ocr_store.complete(idx,'recovered')
    restarted=GateRuntime(settings,[camera],repository,snapshots)
    restarted._recover_passages()
    restarted._finalize_passages()
    event=repository.event('recovered')
    assert event['container_number']=='CSQU3054383'
    assert event['status']=='NEEDS_REVIEW'
