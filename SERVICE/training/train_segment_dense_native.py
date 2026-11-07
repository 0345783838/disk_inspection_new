"""Bounded reviewed-union / hard-example ablation; immutable labels, no activation."""
import os,sys,json,shutil,subprocess,copy,argparse,traceback,collections,math
from pathlib import Path
from datetime import datetime,timezone
import cv2,numpy as np,torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import unwrap_model
from segment_supplementary_core import *
from segment_mask_head_loss import freeze_except_masks
from segment_dense_native import NativeYOLODataset as DenseYOLODataset, NativeUnionLoss as DenseUnionLoss
import train_segment_mask_head as previous

PROJECT=Path(os.environ.get('DISK_PROJECT_ROOT',str(Path.cwd()))).resolve()
PREV=PROJECT/'.work/segmentation_supplementary_20261023/training_mask_head'
RUN=Path(os.environ.get('SEGMENT_SUPPLEMENT_ROOT',str(PREV.parent/'training_dense_native_20261024'))).resolve()
ARCHIVE=read_json(PROJECT/'.work/SEGMENT_ARCHIVE_20261024.json')
BASE_PT=Path(ARCHIVE['artifacts']['pt']['path']);BASE_ONNX=Path(ARCHIVE['artifacts']['onnx']['path'])
previous.RUN=RUN;previous.BASE_PT=BASE_PT;previous.BASE_ONNX=BASE_ONNX

def now():return datetime.now(timezone.utc).isoformat()
def status(**kw):
 p=RUN/'status.json';d=read_json(p) if p.exists() else {};d.update(kw,updated_utc=now());tmp=p.with_suffix('.tmp');write_json(tmp,d);os.replace(tmp,p)
def rows(split='val'):return [r for r in read_json(RUN/'dataset/manifest.json') if r['split']==split]
def integrity():
 p=read_json(RUN/'protocol.json')
 for path,h in {**p['frozen_hashes'],**p['protected_runtime_hashes']}.items():assert sha(path)==h,path
 for r in read_json(RUN/'dataset/manifest.json'):
  assert sha(r['image'])==r['image_sha256'] and sha(r['label'])==r['label_sha256'],r['id']
 assert sha(RUN.parent/'review/annotations.json')==p['annotations_sha256']
 return True
def strict_fitness(m):
 s=m['summary'];new=np.mean([s['new_'+c]['pixel_f1'] for c in ['Cam1','Cam2']]);ha=np.mean([s['old_ha_'+c]['f2'] for c in ['Cam1','Cam2']]);demo=s['old_demo']['f2'];specificity=1-s['old_demo']['background_fp_images']/max(1,s['old_demo']['background_images'])
 return float(.8*new+.1*ha+.05*demo+.05*specificity)
def recommendation(m,installed,archived):
 checks=gates(m,installed)['tests'];a=m['summary'];b=archived['summary']
 checks.update(archive_strict_pixel_f1_improves=a['new']['pixel_f1']>b['new']['pixel_f1']+1e-6,archive_both_camera_pixel_f1_retained=all(a['new_'+c]['pixel_f1']>=b['new_'+c]['pixel_f1'] for c in ['Cam1','Cam2']),archive_pixel_precision_not_reduced=a['new']['tolerant_pixel_precision']>=b['new']['tolerant_pixel_precision'],archive_pixel_recall_within_0_1pp=a['new']['tolerant_pixel_recall']>=b['new']['tolerant_pixel_recall']-.001,archive_uncovered_regions_not_increased=a['new']['region_fn']<=b['new']['region_fn'])
 return {'passed':all(checks.values()),'tests':checks}

def prepare():
 if RUN.exists():raise FileExistsError(RUN)
 assert read_json(PREV/'status.json')['complete']
 assert sha(BASE_PT)==ARCHIVE['artifacts']['pt']['sha256'] and sha(BASE_ONNX)==ARCHIVE['artifacts']['onnx']['sha256']
 RUN.mkdir();os.environ['SEGMENT_SUPPLEMENT_ROOT']=str(RUN)
 for name in ['input','masks']:shutil.copytree(PREV/name,RUN/name)
 (RUN/'crops').mkdir();shutil.copy2(PREV/'crops/manifest.json',RUN/'crops/manifest.json')
 (RUN/'dataset').mkdir();shutil.copy2(PREV/'dataset/manifest.json',RUN/'dataset/manifest.json');shutil.copy2(PREV/'split_report.json',RUN/'split_report.json')
 for name in ['background.html','background_crops.csv','DATASET.md']:shutil.copy2(PREV/name,RUN/name)
 data=read_json(RUN/'dataset/manifest.json');train=[r for r in data if r['split']=='train'];val=[r for r in data if r['split']=='val']
 assert len(train)==2364 and len(val)==261
 train_sources={r['source_group'] for r in train};val_sources={r['source_group'] for r in val};assert not train_sources&val_sources
 assert not {r['image_sha256'] for r in train}&{r['image_sha256'] for r in val}
 originals=[r for r in train if r['origin']=='new_reviewed' and r['repeat']==0];assert len(originals)==228
 scores=[]
 for r,p,s,out in predict_onnx(BASE_ONNX,originals):
  m=compare_record(r,p);f1=1. if m['target_pixels']==m['predicted_pixels']==0 else 2*m['intersection_pixels']/max(1,m['target_pixels']+m['predicted_pixels'])
  scores.append({'id':r['id'],'base_crop_id':r['base_crop_id'],'image':r['image'],'frame_id':r['frame_id'],'split':'train','strict_pixel_error':1-f1,'false_pixels':m['predicted_pixels']-m['intersection_pixels'],'missed_pixels':m['target_pixels']-m['intersection_pixels']})
 hard=sorted(scores,key=lambda r:(-r['strict_pixel_error'],r['id']))[:math.ceil(.25*len(scores))]
 write_json(RUN/'hard_sampling.json',{'model_sha256':sha(BASE_ONNX),'source':'228 original new training crops only; no validation predictions or labels used','rule':'Top 25% by strict pixel F1 error (empty correct mask scores zero); 4 additional copies of each original crop filepath, no pixel or label edits','all_training_scores':scores,'selected':hard,'extra_image_references':4*len(hard)})
 lists={'train':[r['image'] for r in train],'train_hard':[r['image'] for r in train]+[r['image'] for r in hard for _ in range(4)],'val':[r['image'] for r in val]}
 for name,paths in lists.items():(RUN/'dataset'/f'{name}.txt').write_text('\n'.join(paths)+'\n',encoding='utf-8')
 for name,trainlist in [('plain','train'),('hard','train_hard')]:
  (RUN/'dataset'/f'{name}.yaml').write_text(f'path: {RUN.as_posix()}/dataset\ntrain: {trainlist}.txt\nval: val.txt\nnames:\n  0: disk\n',encoding='utf-8')
 shutil.copy2(PREV/'baseline.json',RUN/'installed_reference.json');shutil.copy2(PREV/'selected_validation.json',RUN/'archived_reference.json')
 shutil.copytree(PREV/'predictions/selected',RUN/'predictions/baseline')
 code=RUN/'code';code.mkdir()
 common=['train_segment_mask_head.py','segment_supplementary_core.py','train_segmentor_thin_boxes.py','replay_segment_supplementary.py','smoke_local_white.py','segment_exact_mask_dataset.py','segment_center_mask_dataset.py','segment_mask_head_loss.py']
 for name in common:shutil.copy2(PREV/'code'/name,code/name)
 for name in ['train_segment_dense_native.py','segment_dense_native.py']:shutil.copy2(PROJECT/'SERVICE/training'/name,code/name)
 shutil.copytree(PREV/'code/SERVICE',code/'SERVICE')
 protected=read_json(PREV/'protocol.json')['protected_runtime_hashes'];protected.update({str(BASE_PT):sha(BASE_PT),str(BASE_ONNX):sha(BASE_ONNX),ARCHIVE['manifest']:sha(ARCHIVE['manifest'])})
 frozen=list(code.rglob('*.py'))+list((RUN/'input').glob('*.json'))+list((RUN/'dataset').glob('*'))+[RUN/'hard_sampling.json',RUN/'split_report.json',RUN/'installed_reference.json',RUN/'archived_reference.json']+list((RUN/'masks').rglob('*.png'))
 frozen+=list({Path(r['target_masks']) for r in data if r.get('target_masks')})
 protocol={'created_utc':now(),'explicit_user_request':'Save just-trained version to SERVICE models dated today, then experiment with the proposed dense-mask supervision and hard examples. User confirms reviewed masks are authoritative.','annotations_sha256':sha(RUN/'input/annotations.json'),'dataset_manifest_sha256':sha(RUN/'dataset/manifest.json'),'init_pt':str(BASE_PT),'init_onnx':str(BASE_ONNX),'methods':[{'name':'native_union','data':'plain.yaml','lr0':.0001},{'name':'native_union_hard','data':'hard.yaml','lr0':.0001}],'max_fits':2,'epochs':40,'patience':10,'batch':8,'seed':1024,'dense_loss':{'weight':.5,'objective':'Calibrated pixel BCE within proposal/GT ROI + .1 Dice on binary 512x512 reviewed union after native bilinear logit upsampling; only new reviewed images, including empty ones','proposal_selection':'Detached runtime confidence .5, NMS .8; top300 before NMS, top64 after NMS; masks cropped to detached boxes after logit upsampling; max probability union matches native binary OR','historical_labels':'Original assigned instance loss + existing .01 empty-image mask loss unchanged; historical union not supervised'},'trainable':'Only proto/cv4 mask branches; all BN statistics and non-mask weights fixed to saved 24 October parent.','sampling':{'plain':len(lists['train']),'hard':len(lists['train_hard']),'new_hard_originals':len(hard),'validation':len(val),'no_new_labels_or_geometry':True},'checkpoint_fitness':'.80 mean new-camera strict pixel F1 + .10 mean HA-camera instance F2 + .05 demo instance F2 + .05 old-background specificity; validation only','candidate_grid':[1.,.25,.5,.75],'candidate_selection':'Among passing candidates maximize mean new-camera strict pixel F1, then fewer new uncovered regions, then strict fitness. If none pass, keep archived parent as recommendation.','recommendation':'All 9 previous installed-parent gates, plus strict new pixel F1 improves over archived, both new-camera strict F1 retained, tolerant precision retained, tolerant recall within .1pp, uncovered regions not increased vs archived.','limitations':['Same fixed same-session development validation, repeatedly reused for model selection.','All 8 new empty crops remain training only.','5 inspected files are train diagnostics, 4 unique frames; never fitness.','Old masks remain partly reviewed/model-assisted.'],'frozen_hashes':{str(p):sha(p) for p in frozen},'protected_runtime_hashes':protected,'no_activation':True,'superseded_diagnostic':{'root':str(RUN.parent/'training_dense_20261024'),'reason':'Balanced soft prototype target objective favored 8px over 5px stripe; earlier exploratory fit was technically stopped, preserved unchanged.'}}
 write_json(RUN/'protocol.json',protocol)
 write_json(RUN/'authorization.json',{'explicit_request':protocol['explicit_user_request'],'protocol_sha256':sha(RUN/'protocol.json'),'annotations_sha256':protocol['annotations_sha256'],'dataset_manifest_sha256':protocol['dataset_manifest_sha256'],'archive':ARCHIVE,'no_activation':True})
 status(phase='prepared',complete=False,root=str(RUN),dashboard='http://127.0.0.1:8802/',training_started=False)
 write_json(PROJECT/'.work/ACTIVE_SEGMENT_TRAINING.json',{'root':str(RUN),'status':'prepared','entrypoint':str(code/'train_segment_dense_native.py'),'protocol_sha256':sha(RUN/'protocol.json'),'dashboard':'http://127.0.0.1:8802/','no_activation':True})
 integrity();print('PREPARED',str(RUN),flush=True)

class Trainer(previous.CampaignTrainer):
 def _setup_train(self,*a,**kw):
  # Parent attaches the old loss only after EMA creation; replace it on the
  # train model alone, keeping exported/EMA checkpoints standard.
  super()._setup_train(*a,**kw);self.model.criterion=DenseUnionLoss(self.model,weight=.5)
  assert not isinstance(getattr(self.ema.ema,'criterion',None),DenseUnionLoss)
 def build_dataset(self,img_path,mode='train',batch=None):
  a=self.args;gs=max(int(unwrap_model(self.model).stride.max()),32)
  return DenseYOLODataset(img_path=img_path,imgsz=a.imgsz,batch_size=batch,augment=mode=='train',hyp=a,rect=a.rect or mode=='val',cache=a.cache or None,single_cls=a.single_cls or False,stride=gs,pad=0 if mode=='train' else .5,task=a.task,classes=a.classes,data=self.data,fraction=a.fraction if mode=='train' else 1.)
 def validate(self):
  metrics=self.validator(self);metrics.pop('fitness',None);m=evaluate(predict_torch(self.ema.ema if self.ema else self.model,rows(),batch=8));fitness=strict_fitness(m);m['selection_fitness']=fitness
  name=os.environ['SEGMENT_FIT_NAME'];write_json(RUN/'fits'/name/f'fixed_epoch_{self.epoch+1:03d}.json',m)
  metrics.update(strict_new_f1=m['summary']['new']['pixel_f1'],strict_selection_fitness=fitness)
  if not self.best_fitness or fitness>self.best_fitness:self.best_fitness=fitness
  status(phase='training',fit=name,epoch=self.epoch+1,max_epochs=self.epochs,best_fitness=self.best_fitness,fitness=fitness,new_pixel_f1=m['summary']['new']['pixel_f1'],new_uncovered_regions=m['summary']['new']['region_fn'],dense_loss=getattr(self.model.criterion,'last',{}))
  return metrics,fitness

def fit(name):
 p=read_json(RUN/'protocol.json');method=next(x for x in p['methods'] if x['name']==name);integrity()
 assert not (RUN/'STOP_REQUESTED').exists()
 folder=RUN/'fits'/name
 if folder.exists():raise FileExistsError(folder)
 os.environ['SEGMENT_FIT_NAME']=name;os.environ['SEGMENT_FREEZE_BN']='1';torch.set_num_threads(4)
 model=YOLO(str(BASE_PT));model.add_callback('on_fit_epoch_end',lambda trainer:setattr(trainer,'stop',True) if (RUN/'STOP_REQUESTED').exists() else None)
 status(phase='training',fit=name,epoch=0,max_epochs=p['epochs'],training_started=True)
 model.train(trainer=Trainer,data=str(RUN/'dataset'/method['data']),epochs=p['epochs'],patience=p['patience'],imgsz=512,batch=p['batch'],workers=2,device=0,seed=p['seed'],deterministic=True,optimizer='AdamW',lr0=method['lr0'],lrf=.05,weight_decay=.0005,warmup_epochs=2,warmup_bias_lr=method['lr0'],amp=True,overlap_mask=False,mask_ratio=4,mosaic=0,mixup=0,copy_paste=0,degrees=0,translate=0,scale=0,shear=0,perspective=0,flipud=0,fliplr=0,hsv_h=.005,hsv_s=.1,hsv_v=.12,bgr=0,project=str(RUN/'fits'),name=name,exist_ok=False,plots=True,verbose=False,save_period=-1)
 assert not (RUN/'STOP_REQUESTED').exists()
 a=YOLO(str(BASE_PT)).model.state_dict();best=YOLO(str(folder/'weights/best.pt')).model;b=best.state_dict();prefix=('model.23.proto.','model.23.cv4.')
 assert all(torch.equal(a[k].cpu(),b[k].cpu()) for k in a if not k.startswith(prefix));assert not isinstance(getattr(best,'criterion',None),DenseUnionLoss)
 write_json(folder/'frozen_detector_verification.json',{'non_mask_state_exact':True,'ema_has_no_custom_criterion':True});integrity()
 write_json(folder/'fit_complete.json',{'completed_utc':now(),'best':str(folder/'weights/best.pt'),'sha256':sha(folder/'weights/best.pt')})

def campaign(resume=False):
 if read_json(RUN/'status.json').get('complete'):raise RuntimeError('Completed campaign cannot restart')
 if (RUN/'STOP_REQUESTED').exists():raise RuntimeError('STOP_REQUESTED')
 lock=RUN/'campaign.lock'
 with lock.open('x') as f:json.dump({'pid':os.getpid(),'created_utc':now()},f)
 try:
  integrity();p=read_json(RUN/'protocol.json');status(parent_pid=os.getpid());completed=[]
  for method in p['methods']:
   name=method['name'];folder=RUN/'fits'/name
   if (folder/'fit_complete.json').exists():completed.append(name);continue
   if folder.exists():raise RuntimeError('Unfinished fit: inspect before recovery '+name)
   assert not (RUN/'STOP_REQUESTED').exists();env=os.environ.copy();env.update(DISK_PROJECT_ROOT=str(PROJECT),SEGMENT_SUPPLEMENT_ROOT=str(RUN))
   with (RUN/f'{name}.stdout.log').open('wb') as out,(RUN/f'{name}.stderr.log').open('wb') as err:
    child=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__).resolve()),'--phase','fit','--name',name],cwd=PROJECT,env=env,stdout=out,stderr=err);write_json(RUN/'active_child.json',{'pid':child.pid,'parent_pid':os.getpid(),'fit':name});rc=child.wait()
   write_json(RUN/'active_child.json',{'pid':None,'fit':name,'exit_code':rc})
   if rc:raise RuntimeError('Fit failed: '+name)
   completed.append(name)
  status(phase='candidate_evaluation',completed_fits=completed);installed=read_json(RUN/'installed_reference.json');archived=read_json(RUN/'archived_reference.json');candidates=[]
  for name in completed:
   for alpha in p['candidate_grid']:
    identifier=name+('_best' if alpha==1 else f'_blend{int(alpha*100)}');cache=RUN/'evaluations'/f'{identifier}.json'
    if resume and cache.exists():
     c=read_json(cache)
     if sha(c['pt'])==c['pt_sha256'] and sha(c['onnx'])==c['onnx_sha256']:candidates.append(c);continue
    pt=RUN/'fits'/name/'weights/best.pt'
    if alpha!=1:pt=RUN/'candidates'/identifier/'model.pt';previous.blend(BASE_PT,RUN/'fits'/name/'weights/best.pt',alpha,pt)
    onnx=previous.export(pt);m=evaluate(predict_onnx(onnx,rows()));c={'id':identifier,'fit':name,'alpha':alpha,'pt':str(pt),'onnx':str(onnx),'pt_sha256':sha(pt),'onnx_sha256':sha(onnx),'metrics':m,'selection_fitness':strict_fitness(m),'gates':recommendation(m,installed,archived)}
    write_json(cache,c);candidates.append(c);status(candidate=identifier,candidates_finished=len(candidates));print('CANDIDATE',identifier,m['summary']['new']['pixel_f1'],c['gates'],flush=True)
  passing=[c for c in candidates if c['gates']['passed']]
  key=lambda c:(np.mean([c['metrics']['summary']['new_'+cam]['pixel_f1'] for cam in ['Cam1','Cam2']]),-c['metrics']['summary']['new']['region_fn'],c['selection_fitness'])
  selected=max(passing or candidates,key=key);write_json(RUN/'selected.json',selected);write_json(RUN/'candidate_summary.json',{'candidates':candidates,'passing_candidates':[c['id'] for c in passing],'selected':selected['id'],'recommend_improved':bool(passing),'no_activation':True})
  status(phase='parity_and_review',selected=selected['id'],recommendation_gates_passed=bool(passing));m=evaluate(predict_onnx(selected['onnx'],rows()),RUN/'predictions/selected');write_json(RUN/'selected_validation.json',m)
  parity=previous.parity(selected['pt'],selected['onnx'],read_json(RUN/'crops/manifest.json')+[r for r in rows() if r['origin']!='new_reviewed'])
  previous.band_gallery(selected['onnx']);previous.error_gallery(archived,m)
  status(phase='white_replay');env=os.environ.copy();env.update(DISK_PROJECT_ROOT=str(PROJECT),SEGMENT_SUPPLEMENT_ROOT=str(RUN))
  with (RUN/'replay.stdout.log').open('wb') as out,(RUN/'replay.stderr.log').open('wb') as err:
   rc=subprocess.run([str(PROJECT/'SERVICE/.venv/Scripts/python.exe'),'-u','-B',str(Path(__file__).parent/'replay_segment_supplementary.py'),'--model',selected['onnx'],'--tag','candidate'],cwd=PROJECT,env=env,stdout=out,stderr=err).returncode
  assert rc==0
  baseline=read_json(PREV/'replay/candidate.json');after=read_json(RUN/'replay/candidate.json');assert len(baseline['frames'])==len(after['frames'])==28;changes=[]
  for b,a in zip(baseline['frames'],after['frames']):
   assert b['source_sha256']==a['source_sha256'] and b['geometry']==a['geometry'] and b['classifier']==a['classifier']
   if b['counts']!=a['counts']:changes.append({'source':a['source'],'archived_counts':b['counts'],'candidate_counts':a['counts']})
  write_json(RUN/'replay/comparison.json',{'frames':28,'geometry_classifier_unchanged':True,'production_debug_parity':after['production_debug_parity'],'count_changes_vs_archived':changes,'hardware_calls':False})
  integrity();write_json(RUN/'completion_verification.json',{'completed_utc':now(),'completed_fits':completed,'candidates':len(candidates),'passing_candidates':[c['id'] for c in passing],'selected':selected['id'],'pt_onnx_images':parity['images'],'pt_onnx_allclose':parity['all_raw_outputs_close'],'replay_frames':28,'protected_hashes_unchanged':True,'annotations_unchanged':True,'no_activation':True})
  table='| Group | Installed pixel F1 | Archived pixel F1 | Candidate pixel F1 |\n|---|---:|---:|---:|\n'
  for g in ['new_Cam1','new_Cam2','new','old_ha','old_demo']:table+=f'| {g} | {installed["summary"][g]["pixel_f1"]:.6f} | {archived["summary"][g]["pixel_f1"]:.6f} | {m["summary"][g]["pixel_f1"]:.6f} |\n'
  report=f'# Dense-mask experiments — 24 October 2026\n\nCOMPLETE. Candidate {selected["id"]}; recommendation gates {"PASSED" if passing else "FAILED; archived parent retained"}. No activation.\n\n{table}\nTwo fits compare dense union supervision with/without train-only hard sampling. Exact reviewed masks and split unchanged. All candidates and gate failures are in candidate_summary.json. Train-only five-file diagnostics never enter fitness.\n\n489-crop PT/ONNX parity and 28 White hardware-off replay completed. Count changes are reported without asserting independently confirmed disk counts. Same-session development validation is repeatedly reused; no new-background independent holdout. Historical labels retain partial-review limitations.\n'
  (RUN/'RESULTS.md').write_text(report,encoding='utf-8');status(phase='complete',complete=True,selected=selected['id'],recommendation_gates_passed=bool(passing),completed_utc=now())
  write_json(PROJECT/'.work/ACTIVE_SEGMENT_TRAINING.json',{'root':str(RUN),'status':'complete','selected':selected['id'],'recommendation_gates_passed':bool(passing),'no_activation':True,'dashboard':'http://127.0.0.1:8802/'})
 except Exception as e:status(phase='failed',error=repr(e),traceback=traceback.format_exc());raise
 finally:
  if lock.exists():lock.unlink()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prepare','fit','campaign'],required=True);p.add_argument('--name');p.add_argument('--resume',action='store_true');a=p.parse_args()
 os.environ['SEGMENT_SUPPLEMENT_ROOT']=str(RUN)
 if a.phase=='prepare':prepare()
 elif a.phase=='fit':fit(a.name)
 else:campaign(a.resume)
