"""One fresh factory segmentation supplement: native mask union + train-only hard sampling."""
import argparse,collections,copy,hashlib,html,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
PROJECT=Path(os.environ.get('DISK_PROJECT_ROOT',str(Path.cwd()))).resolve()
RUN=Path(os.environ.get('SEGMENT_SUPPLEMENT_ROOT',str(PROJECT/'.work/segmentation_factory_train_20261117'))).resolve()
os.environ.setdefault('YOLO_CONFIG_DIR',str(RUN/'ultralytics_settings'))
Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True,exist_ok=True)
(Path(os.environ['YOLO_CONFIG_DIR'])/'Ultralytics').mkdir(exist_ok=True)
os.environ['SEGMENT_SUPPLEMENT_ROOT']=str(RUN)
import cv2,numpy as np,torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import unwrap_model
import segment_supplementary_core as core
from segment_supplementary_core import sha,read_json,write_json,predict_onnx,predict_torch,compare_record
import train_segment_mask_head as previous
from segment_dense_native import NativeYOLODataset,NativeUnionLoss

BASE_PT=PROJECT/'.work/segmentation_supplementary_20261023/training_dense_native_20261024/fits/native_union_hard/weights/best.pt'
BASE_ONNX=PROJECT/'SERVICE/config/models/s_p_edge_white_native_hard_20261024_102450_old.onnx'
previous.RUN=RUN;previous.BASE_PT=BASE_PT;previous.BASE_ONNX=BASE_ONNX
def now():return datetime.now(timezone.utc).isoformat()
def status(**kw):
 p=RUN/'status.json';value=read_json(p) if p.exists() else {};value.update(kw,updated_utc=now());tmp=p.with_suffix('.tmp');write_json(tmp,value);os.replace(tmp,p)
def rows(split=None):return [r for r in read_json(RUN/'dataset/manifest.json') if split is None or r['split']==split]
def group(r):
 if r.get('source_epoch')=='factory_nov2026':return 'new_'+r['camera']
 if r['origin']=='old_reviewed':return 'old_oct_'+r['camera']
 if r['origin']=='factory':return 'old_ha_'+r['camera']
 return 'old_demo'
core.group_for=group
def evaluate(predictions,save_masks=None):
 result=core.evaluate(predictions,save_masks)
 counter=collections.Counter()
 for r in result['records']:
  if r['group'].startswith('old_oct_'):
   counter.update({k:v for k,v in r.items() if isinstance(v,(int,float))})
 if counter:result['summary']['old_oct']=core.finish(counter)
 result['selection_fitness']=fitness(result)
 return result
def fitness(m):
 s=m['summary'];new=np.mean([s['new_'+c]['pixel_f1'] for c in ['Cam1','Cam2']]);ha=np.mean([s['old_ha_'+c]['f2'] for c in ['Cam1','Cam2']]);old=s['old_oct']['pixel_f1'];bg=s['old_demo'];specificity=1-bg['background_fp_images']/max(1,bg['background_images'])
 return float(.8*new+.1*ha+.05*old+.025*bg['f2']+.025*specificity)
def gates(m,b):
 checks=core.gates(m,b)['tests'];a=m['summary'];ref=b['summary']
 checks.update(parent_strict_new_f1_improves=a['new']['pixel_f1']>ref['new']['pixel_f1']+1e-6,
  parent_both_new_camera_f1_retained=all(a['new_'+c]['pixel_f1']>=ref['new_'+c]['pixel_f1'] for c in ['Cam1','Cam2']),
  parent_tolerant_precision_retained=a['new']['tolerant_pixel_precision']>=ref['new']['tolerant_pixel_precision'],
  parent_tolerant_recall_within_0_1pp=a['new']['tolerant_pixel_recall']>=ref['new']['tolerant_pixel_recall']-.001,
  parent_uncovered_regions_not_increased=a['new']['region_fn']<=ref['new']['region_fn'],
  old_oct_both_camera_f1_within_half_pp=all(a['old_oct_'+c]['pixel_f1']>=ref['old_oct_'+c]['pixel_f1']-.005 for c in ['Cam1','Cam2']),
  new_empty_false_crops_not_increased=a['new']['background_fp_images']<=ref['new']['background_fp_images'])
 return {'passed':all(checks.values()),'tests':checks}
def integrity():
 p=read_json(RUN/'protocol.json')
 for path,h in {**p['frozen_hashes'],**p['protected_runtime_hashes']}.items():assert sha(path)==h,path
 for r in rows():
  assert sha(r['image'])==r['image_sha256'] and sha(r['label'])==r['label_sha256'],r['id']
  if r.get('target_masks_sha256'):assert sha(r['target_masks'])==r['target_masks_sha256'],r['id']
 return True
def prepare():
 assert not (RUN/'protocol.json').exists()
 assert sha(BASE_PT)=='10bcc17c0c68be3d54e5049ca651f38ff60348d115bf9deb5bd72501ff1e8d29'
 assert sha(BASE_ONNX)=='133945894b62f6dd9bc46baf80070947c4854dd488e64b181c28d035e8ead2c0'
 assert read_json(RUN/'input/authorization.json')['training_authorized']
 val=rows('val');newtrain=[r for r in rows('train') if r.get('source_epoch')=='factory_nov2026']
 assert len(read_json(RUN/'crops/manifest.json'))==2064
 status(phase='train_only_hard_sampling',training_started=False)
 scores=[]
 for n,(r,p,s,raw) in enumerate(predict_onnx(BASE_ONNX,newtrain)):
  metric=compare_record(r,p);f1=1 if metric['target_pixels']==metric['predicted_pixels']==0 else 2*metric['intersection_pixels']/max(1,metric['target_pixels']+metric['predicted_pixels'])
  scores.append({'id':r['id'],'image':r['image'],'camera':r['camera'],'split':'train','strict_pixel_error':1-f1})
  if n%200==0:print('HARD SCORING',n,'/',len(newtrain),flush=True)
 old_count=sum(r.get('source_epoch')!='factory_nov2026' for r in rows('train'));extra_count=old_count-len(newtrain);assert extra_count>=0 and extra_count%2==0
 hard=[];extras=[]
 for camera in ['Cam1','Cam2']:
  ranked=sorted([r for r in scores if r['camera']==camera],key=lambda r:(-r['strict_pixel_error'],r['id']));selected=ranked[:int(np.ceil(len(ranked)*.25))];hard+=selected;extras.extend(selected[n%len(selected)]['image'] for n in range(extra_count//2))
 write_json(RUN/'hard_sampling.json',{'rule':'Every old/new train tile once; top25% November train-only hard pool within each camera supplies extra draws until old/new exposure is exactly50/50. Equal extra draws for both new cameras. No validation mining.','model_sha256':sha(BASE_ONNX),'old_references':old_count,'new_references':len(newtrain)+len(extras),'extra_per_new_camera':extra_count//2,'all_scores':scores,'selected':hard})
 train=[r['image'] for r in rows('train')]+extras
 (RUN/'dataset/train_hard.txt').write_text('\n'.join(train)+'\n',encoding='utf-8');(RUN/'dataset/val.txt').write_text('\n'.join(r['image'] for r in val)+'\n',encoding='utf-8')
 (RUN/'dataset/all.yaml').write_text(f'path: {RUN.as_posix()}/dataset\ntrain: train_hard.txt\nval: val.txt\nnames:\n  0: disk\n',encoding='utf-8')
 names=['train_segment_factory.py','segment_supplementary_core.py','segment_center_mask_dataset.py','segment_exact_mask_dataset.py','segment_mask_head_loss.py','segment_dense_native.py','train_segment_mask_head.py','train_segmentor_thin_boxes.py','smoke_local_white.py','replay_factory_segment.py']
 for name in names:shutil.copy2(PROJECT/'SERVICE/training'/name,RUN/'code'/name)
 shutil.copytree(PROJECT/'SERVICE/src',RUN/'code/SERVICE/src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 protected=[PROJECT/'SERVICE/config/config.env',PROJECT/'DATASET_CLASSIFIER/manifest.json',PROJECT/'DATASET_CLASSIFIER/labels.csv',PROJECT/'DATASET_CLASSIFIER/catalog.json',PROJECT/'.work/ACTIVE_CLASSIFIER_DATASET.json',PROJECT/'APP/DiskInspection/bin/Debug/DiskInspection.exe.config',PROJECT/'APP/DiskInspection/bin/Debug/plugin/config/config.env',BASE_PT]
 protected+=list((PROJECT/'SERVICE/config/models').glob('*'));protected+=[p for p in (PROJECT/'SERVICE/src').rglob('*.py')]
 frozen=list((RUN/'input').glob('*'))+list((RUN/'code').rglob('*.py'))+[RUN/'dataset/manifest.json',RUN/'dataset/train_hard.txt',RUN/'dataset/val.txt',RUN/'dataset/all.yaml',RUN/'hard_sampling.json',RUN/'split_report.json',RUN/'crops/manifest.json']+list((RUN/'masks').glob('*.npz'))+[Path(r['target_masks']) for r in rows() if r.get('target_masks')]
 protocol={'created_utc':now(),'scope':'One fresh segmentation fit only, no classifier changes or deployment. All 2064 new reviewed tiles included. Object overlap unrestricted; union OR is pixel authority.',
  'baseline_pt':str(BASE_PT),'baseline_onnx':str(BASE_ONNX),'max_fits':1,'epochs':40,'patience':10,'batch':8,'imgsz':512,'seed':1117,'lr0':.0001,
  'trainable':'Only proto/cv4 mask branches; all backbone/bbox/confidence parameters and all BatchNorm running statistics frozen.',
  'loss':'Same calibrated native 512-grid binary union BCE + 0.1 Dice (weight0.5) as previous native hard fit, plus retained instance-mask/background loss. Mask instances may overlap; overlap does not create exclusive target classes. Exact NPZ masks, never proxy polygons, define training pixels.',
  'sampling':{'old_train':old_count,'new_train':len(newtrain),'hard_extra':len(extras),'old_epoch_references':old_count,'new_epoch_references':len(newtrain)+len(extras),'extra_per_new_camera':extra_count//2,'epoch_references':len(train),'validation':len(val),'policy':'Every retained/new train tile once; camera-balanced train-only hard extras make old/new exposure exactly50/50. Online bounded HSV only, no geometry/flip/mosaic.'},
  'fitness':'.80 mean November-camera union strict pixel F1 +.10 mean HA-camera instance F2 +.05 October reviewed union pixel F1 +.025 demo F2 +.025 demo-background specificity. Validation only.',
  'candidate_grid':[1.,.25,.5,.75],'gates':'Original nine gates + previous native-parent retention gates, both October-camera union F1 within0.5pp and new-empty false crops not increased. Frozen before baseline metrics. If no candidate passes retain installed parent.',
  'split':read_json(RUN/'split_report.json'),'annotations_sha256':sha(RUN/'input/annotations.json'),'no_activation':True,'whole_software_latency_limit':2.,
  'limitations':['Development validation, same factory capture sessions and heuristic tray grouping, not independent physical-tray acceptance testing.','Old retained validation has parent exposure/partial historical label review.','Object/fragment counts are not physical disk counts.'],
  'frozen_hashes':{str(p):sha(p) for p in set(frozen) if p.is_file()},'protected_runtime_hashes':{str(p):sha(p) for p in protected if p.is_file()}}
 write_json(RUN/'protocol.json',protocol);write_json(RUN/'authorization.json',{'user_receipt':read_json(RUN/'input/authorization.json'),'union_instruction':'Object masks may overlap freely and merge into one whole-image mask, as previous workflow.','protocol_sha256':sha(RUN/'protocol.json'),'training_authorized':True,'no_deployment':True})
 integrity();write_json(PROJECT/'.work/ACTIVE_SEGMENT_TRAINING.json',{'root':str(RUN),'status':'prepared','entrypoint':str(RUN/'code/train_segment_factory.py'),'protocol_sha256':sha(RUN/'protocol.json'),'dashboard':'http://127.0.0.1:8811/','no_activation':True})
 status(phase='prepared',complete=False,training_started=False,dashboard='http://127.0.0.1:8811/')
 print('PROTOCOL FROZEN',sha(RUN/'protocol.json'),flush=True)

class Trainer(previous.CampaignTrainer):
 def _setup_train(self,*a,**kw):
  super()._setup_train(*a,**kw);self.model.criterion=NativeUnionLoss(self.model,weight=.5)
  assert not isinstance(getattr(self.ema.ema,'criterion',None),NativeUnionLoss)
 def build_dataset(self,img_path,mode='train',batch=None):
  a=self.args;gs=max(int(unwrap_model(self.model).stride.max()),32)
  return NativeYOLODataset(img_path=img_path,imgsz=a.imgsz,batch_size=batch,augment=mode=='train',hyp=a,rect=a.rect or mode=='val',cache=a.cache or None,single_cls=a.single_cls or False,stride=gs,pad=0 if mode=='train' else .5,task=a.task,classes=a.classes,data=self.data,fraction=a.fraction if mode=='train' else 1.)
 def validate(self):
  metrics=self.validator(self);metrics.pop('fitness',None);m=evaluate(predict_torch(self.ema.ema if self.ema else self.model,rows('val'),batch=8));f=m['selection_fitness']
  write_json(RUN/'fits/native_hard'/f'fixed_epoch_{self.epoch+1:03d}.json',m)
  if not self.best_fitness or f>self.best_fitness:self.best_fitness=f
  metrics.update(strict_new_union_f1=m['summary']['new']['pixel_f1'],selection_fitness=f)
  status(phase='training',fit='native_hard',epoch=self.epoch+1,max_epochs=self.epochs,fitness=f,best_fitness=self.best_fitness,new_pixel_f1=m['summary']['new']['pixel_f1'],new_uncovered_regions=m['summary']['new']['region_fn'])
  return metrics,f
def fit():
 p=read_json(RUN/'protocol.json');integrity();assert not (RUN/'STOP_REQUESTED').exists();assert not (RUN/'fits/native_hard').exists()
 os.environ['SEGMENT_FREEZE_BN']='1';os.environ['SEGMENT_FIT_NAME']='native_hard';torch.set_num_threads(4)
 model=YOLO(str(BASE_PT));model.add_callback('on_fit_epoch_end',lambda t:setattr(t,'stop',True) if (RUN/'STOP_REQUESTED').exists() else None)
 status(phase='training',fit='native_hard',epoch=0,max_epochs=p['epochs'],training_started=True)
 model.train(trainer=Trainer,data=str(RUN/'dataset/all.yaml'),epochs=p['epochs'],patience=p['patience'],imgsz=512,batch=p['batch'],workers=2,device=0,seed=p['seed'],deterministic=True,optimizer='AdamW',lr0=p['lr0'],lrf=.05,weight_decay=.0005,warmup_epochs=2,warmup_bias_lr=p['lr0'],amp=True,overlap_mask=False,mask_ratio=4,mosaic=0,mixup=0,copy_paste=0,degrees=0,translate=0,scale=0,shear=0,perspective=0,flipud=0,fliplr=0,hsv_h=.005,hsv_s=.1,hsv_v=.12,bgr=0,project=str(RUN/'fits'),name='native_hard',exist_ok=False,plots=True,verbose=False,save_period=-1)
 assert not (RUN/'STOP_REQUESTED').exists()
 parent=YOLO(str(BASE_PT)).model;best=YOLO(str(RUN/'fits/native_hard/weights/best.pt')).model;a=parent.state_dict();b=best.state_dict();idx=len(best.model)-1;prefix=(f'model.{idx}.proto.',f'model.{idx}.cv4.')
 assert all(torch.equal(a[k].cpu(),b[k].cpu()) for k in a if not k.startswith(prefix))
 assert all(torch.equal(a[k].cpu(),b[k].cpu()) for k in a if any(x in k for x in ['running_mean','running_var','num_batches_tracked']))
 write_json(RUN/'fits/native_hard/fit_complete.json',{'completed_utc':now(),'best_sha256':sha(RUN/'fits/native_hard/weights/best.pt'),'non_mask_and_all_bn_state_exact':True});integrity()
def campaign():
 if read_json(RUN/'status.json').get('complete'):raise RuntimeError('Completed run cannot resume')
 assert not (RUN/'STOP_REQUESTED').exists()
 lock=RUN/'campaign.lock'
 with lock.open('x') as stream:json.dump({'pid':os.getpid(),'started_utc':now()},stream)
 try:
  integrity();status(parent_pid=os.getpid(),phase='baseline_evaluation')
  base=evaluate(predict_onnx(BASE_ONNX,rows('val')),RUN/'predictions/baseline');write_json(RUN/'baseline.json',base);write_json(RUN/'baseline_summary.json',base['summary'])
  # Parent identity before fitting: official retained PT vs actually installed ONNX.
  smoke=previous.parity(BASE_PT,BASE_ONNX,rows('val')[:6]);shutil.move(RUN/'parity.json',RUN/'parent_parity_smoke.json')
  env=os.environ.copy();env.update(DISK_PROJECT_ROOT=str(PROJECT),SEGMENT_SUPPLEMENT_ROOT=str(RUN),YOLO_CONFIG_DIR=str(RUN/'ultralytics_settings'))
  if not (RUN/'fits/native_hard/fit_complete.json').exists():
   if (RUN/'fits/native_hard').exists():raise RuntimeError('Unfinished fit must be inspected, never silently restarted')
   with (RUN/'fit.stdout.log').open('wb') as out,(RUN/'fit.stderr.log').open('wb') as err:
    child=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__).resolve()),'--phase','fit'],cwd=PROJECT,env=env,stdout=out,stderr=err);write_json(RUN/'active_child.json',{'pid':child.pid,'parent_pid':os.getpid()});rc=child.wait()
   write_json(RUN/'active_child.json',{'pid':None,'exit_code':rc});assert rc==0,'Fit failed; inspect logs'
  assert not (RUN/'STOP_REQUESTED').exists();status(phase='candidate_evaluation');candidates=[]
  for alpha in read_json(RUN/'protocol.json')['candidate_grid']:
   identifier='native_hard_best' if alpha==1 else f'native_hard_blend{int(alpha*100)}';pt=RUN/'fits/native_hard/weights/best.pt'
   if alpha!=1:pt=RUN/'candidates'/identifier/'model.pt';previous.blend(BASE_PT,RUN/'fits/native_hard/weights/best.pt',alpha,pt)
   onnx=previous.export(pt);m=evaluate(predict_onnx(onnx,rows('val')));record={'id':identifier,'alpha':alpha,'pt':str(pt),'onnx':str(onnx),'pt_sha256':sha(pt),'onnx_sha256':sha(onnx),'metrics':m,'gates':gates(m,base)}
   write_json(RUN/'evaluations'/f'{identifier}.json',record);candidates.append(record);status(candidate=identifier,candidates_finished=len(candidates))
  passing=[r for r in candidates if r['gates']['passed']];rank=lambda r:(r['metrics']['selection_fitness'],-r['metrics']['summary']['new']['region_fn'])
  diagnostic=max(candidates,key=rank);selected=max(passing,key=rank) if passing else {'id':'installed_parent_retained','alpha':0.,'pt':str(BASE_PT),'onnx':str(BASE_ONNX),'pt_sha256':sha(BASE_PT),'onnx_sha256':sha(BASE_ONNX),'metrics':base,'gates':{'passed':False}}
  write_json(RUN/'candidate_summary.json',{'candidates':candidates,'passing':[r['id'] for r in passing],'selected':selected['id'],'best_learned_diagnostic':diagnostic['id'],'no_deployment':True});write_json(RUN/'selected.json',selected)
  release=RUN/'release';release.mkdir(exist_ok=True)
  for kind in ['pt','onnx']:shutil.copy2(selected[kind],release/f's_p_factory_20261117.{kind}')
  status(phase='full_parity',selected=selected['id'],gates_passed=bool(passing));allnew=read_json(RUN/'crops/manifest.json');oldval=[r for r in rows('val') if r.get('source_epoch')!='factory_nov2026'];parity=previous.parity(selected['pt'],selected['onnx'],allnew+oldval)
  chosen=evaluate(predict_onnx(selected['onnx'],rows('val')),RUN/'predictions/selected');write_json(RUN/'selected_validation.json',chosen)
  error_gallery(base,chosen)
  status(phase='hardware_off_white_replay')
  with (RUN/'replay.stdout.log').open('wb') as out,(RUN/'replay.stderr.log').open('wb') as err:
   rc=subprocess.run([str(PROJECT/'SERVICE/.venv/Scripts/python.exe'),'-u','-B',str(Path(__file__).parent/'replay_factory_segment.py'),'--root',str(RUN),'--candidate',selected['onnx']],cwd=PROJECT,env=env,stdout=out,stderr=err).returncode
  assert rc==0,'Replay failed; inspect logs'
  replay=read_json(RUN/'replay/comparison.json');integrity()
  decision={'selected':selected['id'],'recommend_improved':bool(passing) and replay['all_paired_ratios_le_2x'],'candidate_gates_passed':bool(passing),'no_activation':True,'validation_is_development':True,'pt_sha256':selected['pt_sha256'],'onnx_sha256':selected['onnx_sha256']};write_json(release/'release_manifest.json',decision)
  verification={'complete':True,'completed_utc':now(),'one_fit_only':True,'selected':selected['id'],'passing_candidates':[r['id'] for r in passing],'full_parity_images':parity['images'],'pt_onnx_allclose':parity['all_raw_outputs_close'],'replay_frames':replay['frames'],'both_camera_debug_parity':replay['both_camera_debug_parity'],'geometry_classifier_unchanged':True,'runtime_frozen_data_code_hashes_unchanged':True,'annotations_unchanged':sha(PROJECT/'.work/segmentation_factory_20261117/review/annotations.json')==sha(RUN/'input/annotations.json'),'no_deployment':True}
  write_json(RUN/'completion_verification.json',verification);report(base,chosen,decision,replay)
  status(phase='complete',complete=True,selected=selected['id'],recommend_improved=decision['recommend_improved'],completed_utc=now())
  pointer=read_json(PROJECT/'.work/ACTIVE_SEGMENT_TRAINING.json');pointer.update(status='complete',completed_utc=now());write_json(PROJECT/'.work/ACTIVE_SEGMENT_TRAINING.json',pointer)
 except BaseException as error:status(phase='failed',error=repr(error),traceback=traceback.format_exc(),complete=False);raise
 finally:lock.unlink(missing_ok=True)
def error_gallery(base,candidate):
 by={r['id']:r for r in rows('val')};bm={r['id']:r for r in base['records']};folder=RUN/'error_review';folder.mkdir(exist_ok=True);cards=[]
 for result in candidate['records']:
  r=by[result['id']];b=bm[r['id']];f=lambda x:2*x['intersection_pixels']/max(1,x['target_pixels']+x['predicted_pixels'])
  if f(result)>.98 and f(b)>.98:continue
  raw=core.image(r);target=core.targets(r)[1];panels=[raw];gt=raw.copy();gt[target]=(30,230,100);panels.append(cv2.addWeighted(raw,.5,gt,.5,0))
  for tag in ['baseline','selected']:
   with np.load(RUN/'predictions'/tag/f'{r["id"]}.npz') as data:m=data['masks'];union=m.any(0) if len(m) else np.zeros_like(target)
   im=raw.copy();im[union]=(0,190,255) if tag=='baseline' else (220,180,30);panels.append(cv2.addWeighted(raw,.5,im,.5,0))
  path=folder/f'{r["id"]}.jpg';cv2.imwrite(str(path),np.hstack(panels));cards.append(f'<article><p>{html.escape(r["id"])} · {result["group"]} · union F1 {f(b):.4f} → {f(result):.4f}</p><img loading="lazy" src="error_review/{path.name}"></article>')
 (RUN/'error_review.html').write_text('<meta charset="utf-8"><title>Segmentation validation review</title><style>body{font:14px system-ui;background:#142135;color:white;padding:20px}img{width:100%}article{border-bottom:1px solid #678;padding:12px}</style><h1>Ảnh gốc / union nhãn / baseline / model được chọn</h1><p>Development validation; fragment counts are not physical disk counts.</p>'+''.join(cards),encoding='utf-8')
def report(base,selected,decision,replay):
 table='| Validation group | Parent union F1 | Selected union F1 | Uncovered parent → selected |\n|---|---:|---:|---:|\n'
 for group in ['new_Cam1','new_Cam2','new','old_oct','old_ha','old_demo']:
  a=base['summary'][group];b=selected['summary'][group];table+=f'| {group} | {a["pixel_f1"]:.6f} | {b["pixel_f1"]:.6f} | {a["region_fn"]} → {b["region_fn"]} |\n'
 (RUN/'RESULTS.md').write_text(f'# Factory segmentation supplement —17November2026\n\nCOMPLETE. Selected: {decision["selected"]}. Recommend improvement: {decision["recommend_improved"]}. No deployment.\n\n{table}\nAll2064 new tiles retained; labels exported with shared Canvas geometry including62 erasures and unrestricted mask OR overlap. Grouped70/30 split; old labels/splits preserved. One mask-branch-only native-hard fit, max40/patience10, all BN and non-mask state frozen. All four fixed candidates and failures/gates are in candidate_summary.json.\n\nFull PT/ONNX parity, {replay["frames"]} paired White hardware-off replays and both-camera debug checks completed; classifier/detector/geometry and protected runtime hashes retained. Whole software ratio median {replay["median_ratio"]:.3f}x, maximum {replay["max_ratio"]:.3f}x. Count changes are diagnostics without independent physical count truth.\n\nValidation is development data, not independent factory acceptance or a production guarantee; older validation has prior parent exposure and partial historical review. New reviewed union pixels are frozen; object IDs are not disk count labels. No classifier training or runtime activation.\n',encoding='utf-8')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['prepare','campaign','fit'],required=True);a=ap.parse_args()
 {'prepare':prepare,'campaign':campaign,'fit':fit}[a.phase]()
if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');main()
