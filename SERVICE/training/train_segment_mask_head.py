"""Bounded supplementary segmentation training/evaluation, never deployment."""
import os,sys,json,time,copy,shutil,argparse,subprocess,hashlib,collections,csv,html,traceback
from pathlib import Path
from datetime import datetime,timezone
import cv2,numpy as np,torch
from ultralytics import YOLO
from segment_supplementary_core import *
from train_segmentor_thin_boxes import ThinBoxTrainer
from segment_center_mask_dataset import CenterYOLODataset as ExactYOLODataset
from segment_mask_head_loss import MaskBackgroundLoss,freeze_except_masks
from ultralytics.utils.torch_utils import unwrap_model
ROOT=Path(os.environ.get('DISK_PROJECT_ROOT',str(Path.cwd()))).resolve()
RUN=Path(os.environ.get('SEGMENT_SUPPLEMENT_ROOT',str(ROOT/'.work/segmentation_supplementary_20261023/training_mask_head'))).resolve()
BASE_PT=ROOT/'.work/segmentation_faint_20261001/runs/thinbox512_v3/weights/best.pt'
BASE_ONNX=ROOT/'APP/DiskInspection/bin/BroSight_Disinspector/plugin/config/models/s_p_ha_thinbox512_20261001.onnx'
def now():return datetime.now(timezone.utc).isoformat()
def status(**kw):
 p=RUN/'status.json';old=read_json(p) if p.exists() else {};old.update(kw);old['updated_utc']=now();tmp=p.with_suffix('.tmp');write_json(tmp,old);os.replace(tmp,p)
def val_records():return [r for r in read_json(RUN/'dataset/manifest.json') if r['split']=='val']
def integrity():
 protocol=read_json(RUN/'protocol.json')
 for p,h in protocol['frozen_hashes'].items():
  if sha(p)!=h:raise ValueError('Frozen input/code changed: '+p)
 for r in read_json(RUN/'dataset/manifest.json'):
  if sha(r['image'])!=r['image_sha256'] or sha(r['label'])!=r['label_sha256']:raise ValueError('Dataset file changed: '+r['id'])
 for row in read_json(RUN.parent/'dataset_inventory.json'):
  if sha(row['image'])!=row['image_sha256'] or sha(row['label'])!=row['label_sha256']:raise ValueError('Original old dataset changed: '+row['image'])
 for p,h in protocol['protected_runtime_hashes'].items():
  if sha(p)!=h:raise ValueError('Protected runtime changed: '+p)
 return True

class CampaignTrainer(ThinBoxTrainer):
 def build_optimizer(self,model,*a,**kw):
  names,prefixes=freeze_except_masks(model)
  write_json(RUN/'mask_trainable_parameters.json',{'trainable':names,'prefixes':prefixes,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'frozen_parameters':sum(p.numel() for p in model.parameters() if not p.requires_grad)})
  return super().build_optimizer(model,*a,**kw)
 def _setup_train(self,*a,**kw):
  super()._setup_train(*a,**kw)
  self.model.criterion=MaskBackgroundLoss(self.model,weight=.01)
  assert not isinstance(getattr(self.ema.ema,'criterion',None),MaskBackgroundLoss)

 def build_dataset(self,img_path,mode='train',batch=None):
  gs=max(int(unwrap_model(self.model).stride.max() if self.model else 0),32)
  a=self.args
  return ExactYOLODataset(img_path=img_path,imgsz=a.imgsz,batch_size=batch,augment=mode=='train',hyp=a,rect=a.rect or mode=='val',cache=a.cache or None,single_cls=a.single_cls or False,stride=gs,pad=0. if mode=='train' else .5,prefix=f'{mode}: ',task=a.task,classes=a.classes,data=self.data,fraction=a.fraction if mode=='train' else 1.)
 def _model_train(self):
  super()._model_train()
  if os.environ.get('SEGMENT_FREEZE_BN')=='1':
   for m in self.model.modules():
    if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):m.eval()
 def validate(self):
  metrics=self.validator(self);metrics.pop('fitness',None)
  model=self.ema.ema if self.ema else self.model
  fixed=evaluate(predict_torch(model,val_records(),batch=8));fitness=fixed['fitness']
  metrics['fixed/new_f2']=fixed['summary']['new']['tolerant_pixel_f2'];metrics['fixed/old_ha_f2']=fixed['summary']['old_ha']['f2'];metrics['fixed/background_fp_crops']=fixed['summary']['old_demo']['background_fp_images']
  if not self.best_fitness or self.best_fitness<fitness:self.best_fitness=fitness
  name=os.environ['SEGMENT_FIT_NAME'];folder=RUN/'fits'/name
  write_json(folder/f'fixed_epoch_{self.epoch+1:03d}.json',fixed)
  status(phase='training',fit=name,epoch=self.epoch+1,max_epochs=self.epochs,fitness=fitness,best_fitness=self.best_fitness,new_uncovered_regions=fixed['summary']['new']['region_fn'],new_pixel_recall=fixed['summary']['new']['tolerant_pixel_recall'],new_pixel_precision=fixed['summary']['new']['tolerant_pixel_precision'],old_ha_fn=fixed['summary']['old_ha']['fn'])
  return metrics,fitness

def setup():
 if (RUN/'protocol.json').exists():raise FileExistsError('Protocol already frozen')
 input_receipt=read_json(RUN/'input/export_receipt.json');split=read_json(RUN/'split_report.json')
 assert sha(RUN.parent/'review/annotations.json')==input_receipt['review_annotations_sha256']
 assert split['frame_counts']=={'val':8,'train':19}
 code=RUN/'code';code.mkdir()
 names=['train_segment_mask_head.py','segment_supplementary_core.py','train_segmentor_thin_boxes.py','export_reviewed_segment_crops.py','build_segment_supplementary_dataset.py','replay_segment_supplementary.py','smoke_local_white.py','segment_exact_mask_dataset.py','repair_segment_multipart_export.py','segment_center_mask_dataset.py','segment_mask_head_loss.py']
 for name in names:shutil.copy2(ROOT/'SERVICE/training'/name,code/name)
 shutil.copytree(ROOT/'SERVICE/src',code/'SERVICE/src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 runtime=ROOT/'APP/DiskInspection/bin/BroSight_Disinspector/plugin'
 protected=[BASE_ONNX,BASE_PT,ROOT/'SERVICE/config/config.env',runtime/'config/config.env',runtime/'main.exe',runtime.parent/'DiskInspection.exe',runtime.parent/'DiskInspection.exe.config',ROOT/'DATASET_CLASSIFIER/manifest.json',ROOT/'DATASET_CLASSIFIER/labels.csv']
 protected+=list((runtime/'config/models').glob('*.onnx'))
 extra_targets=list({Path(r['target_masks']) for r in read_json(RUN/'dataset/manifest.json') if r.get('target_masks')})+list((RUN/'masks').rglob('*.png'))
 frozen=[RUN/'repair_receipt.json',RUN/'exact_mask_preflight_verification.json']+extra_targets+[RUN/'input/items.json',RUN/'input/annotations.json',RUN/'input/export_receipt.json',RUN/'dataset/manifest.json',RUN/'split_report.json',RUN/'dataset/all.yaml']+list(code.rglob('*.py'))
 protocol={'created_utc':now(),'authorization':input_receipt['user_authorization'],'dataset':str(RUN/'dataset'),'split':split,'baseline_pt':str(BASE_PT),'baseline_onnx':str(BASE_ONNX),'baseline_onnx_sha256':sha(BASE_ONNX),'methods':[{'name':'mask_head','lr0':.0001,'freeze_bn':True}],'max_successful_fits':1,'epochs':30,'patience':8,'batch':8,'imgsz':512,'seed':1023,'initialization':'One targeted fit from installed HA thinbox512_v3 PT. Only model.23.proto and model.23.cv4 train; backbone, bbox and confidence parameters plus all BN running statistics remain fixed. New supervised empty-image mask loss weight .01. Raw masks use center-aligned nearest resize and soft area occupancy directly on the 128x128 prototype grid, mask_ratio4; no second nearest target downsample.','mask_representation':'New GT masks come from exact NPZ raw multipart masks, including gaps; txt polygon support paths are bbox proxies only, never loss-mask authority. All 324 original crop unions unchanged by repair. New fragment IDs are not physical disks; pixel/region coverage drives new fitness. No live geometric augmentation.', 'augmentation':'Keep all 1420 old train images. New train positives 4 total photometric variants; 8 true empty train crops 8 total variants. No val augmentation, no flips/mosaic, no new AI geometry.','fitness':'0.55 * mean new camera tolerant pixel F2 + 0.25 * mean HA camera instance F2 + 0.10 * old-demo instance F2 + 0.05 * old-background crop specificity + 0.05 * new region full-length coverage, at runtime conf=.5, NMS=.8, mask match IoU=.3. Training rows never enter fitness.','recommendation_gates':['New tolerant pixel recall strictly improves','New tolerant pixel precision drops no more than .5pp; pixel F1 and region coverage do not worsen','Old HA recall and precision each drop no more than 1pp','Old-demo F1 drops no more than 1pp','Old background false-positive crops do not increase'],'interpolation_fractions':[.25,.5,.75],'interpolation_rule':'For each fit best, a single network with baseline + alpha*(fit-best - baseline) for all floating state, including BN statistics; retain baseline integer buffers. All candidates, including failures, reported.','selection':'Among passing candidates minimize new uncovered regions, then maximize new tolerant pixel F2, then minimize HA FN, then maximize fixed fitness; if none pass retain installed baseline and show highest-fitness candidate as non-recommended.','limitations':['Old HA labels remain partly model-assisted.','New validation from the same acquisition session, not independent-session production testing.','Validation repeatedly selects checkpoints/candidates and is development data.','All 8 new empty crops are train-only; empty-background holdout is 7 old-demo crops.','Instance labels may contain reviewed tile/core fragments; instance counts are not physical disk counts.'],'frozen_hashes':{str(p):sha(p) for p in frozen},'protected_runtime_hashes':{str(p):sha(p) for p in protected},'targeted_followup':{'prior_campaign':str(ROOT/'.work/segmentation_supplementary_20261023/training_v2'),'reason':'Two whole-network fits and 8 candidates failed unchanged gates; independent 5px stripe fixtures demonstrate legacy nearest mask resize mean center error .475 input px versus zero with center-aligned resize. Preserve detector outputs while correcting mask loss grid.','same_raw_labels_and_split':True,'same_evaluation_core_and_gates':True,'max_additional_fits':1},'export_repair':read_json(RUN/'repair_receipt.json') if (RUN/'repair_receipt.json').exists() else None,'no_deployment':True}
 write_json(RUN/'protocol.json',protocol);write_json(RUN/'input/training_authorization.json',{'created_utc':now(),'review_annotations_sha256':input_receipt['review_annotations_sha256'],'dataset_manifest_sha256':sha(RUN/'dataset/manifest.json'),'protocol_sha256':sha(RUN/'protocol.json'),'explicit_user_request':'crop all including background, combine old dataset, train and evaluate, improve over current model; 70/30 clusters; user delegated conservative grouping after clarification','training_authorized':True})
 status(phase='prepared',complete=False,training_started=False,root=str(RUN),dashboard='http://127.0.0.1:8799/')
 write_json(ROOT/'.work/ACTIVE_SEGMENT_TRAINING.json',{'root':str(RUN),'protocol_sha256':sha(RUN/'protocol.json'),'status':'prepared','dashboard':'http://127.0.0.1:8799/','runtime_model_unchanged':True})
 print('Protocol frozen',sha(RUN/'protocol.json'),flush=True)

def fit(name):
 protocol=read_json(RUN/'protocol.json');method=next(x for x in protocol['methods'] if x['name']==name);integrity()
 if (RUN/'STOP_REQUESTED').exists():raise RuntimeError('STOP_REQUESTED exists')
 folder=RUN/'fits'/name
 if folder.exists():raise FileExistsError('Never overwrite an existing fit: '+str(folder))
 os.environ['SEGMENT_FIT_NAME']=name;os.environ['SEGMENT_FREEZE_BN']='1' if method['freeze_bn'] else '0'
 torch.set_num_threads(4)
 model=YOLO(str(BASE_PT))
 def stop_callback(trainer):
  if (RUN/'STOP_REQUESTED').exists():trainer.stop=True
 model.add_callback('on_fit_epoch_end',stop_callback)
 status(phase='training',fit=name,epoch=0,max_epochs=protocol['epochs'],training_started=True)
 model.train(trainer=CampaignTrainer,data=str(RUN/'dataset/all.yaml'),epochs=protocol['epochs'],patience=protocol['patience'],imgsz=512,batch=protocol['batch'],workers=2,device=0,seed=protocol['seed'],deterministic=True,optimizer='AdamW',lr0=method['lr0'],lrf=.05,weight_decay=.0005,warmup_epochs=2,warmup_bias_lr=method['lr0'],amp=True,overlap_mask=False,mask_ratio=4,mosaic=0,mixup=0,copy_paste=0,degrees=0,translate=0,scale=0,shear=0,perspective=0,flipud=0,fliplr=0,hsv_h=.005,hsv_s=.1,hsv_v=.12,bgr=0,project=str(RUN/'fits'),name=name,exist_ok=False,plots=True,verbose=False,save_period=-1)
 if (RUN/'STOP_REQUESTED').exists():raise RuntimeError('Stopped by user request; no next fit/evaluation')
 base_state=torch.load(BASE_PT,map_location='cpu',weights_only=False)['model'].state_dict();best_model=torch.load(folder/'weights/best.pt',map_location='cpu',weights_only=False)['model'];trained_state=best_model.state_dict();prefixes=(f'model.{len(best_model.model)-1}.proto.',f'model.{len(best_model.model)-1}.cv4.');changed=[k for k in base_state if not k.startswith(prefixes) and not torch.equal(base_state[k].cpu(),trained_state[k].cpu())]
 assert not changed,('Non-mask state changed',changed)
 assert not isinstance(getattr(best_model,'criterion',None),MaskBackgroundLoss)
 write_json(folder/'frozen_detector_verification.json',{'backbone_bbox_confidence_state_exactly_equal_to_installed_parent':True,'only_mask_prefixes_may_change':prefixes,'export_checkpoint_has_no_custom_loss_object':True})
 integrity();write_json(folder/'fit_complete.json',{'completed_utc':now(),'best':str(folder/'weights/best.pt'),'best_sha256':sha(folder/'weights/best.pt')})
 print('FIT COMPLETE',name,flush=True)

def export(pt):
 pt=Path(pt);model=YOLO(str(pt));result=model.export(format='onnx',imgsz=512,batch=1,dynamic=False,simplify=False,opset=12,half=False,device='cpu');return Path(result)
def blend(base,fit,alpha,path):
 first=torch.load(base,map_location='cpu',weights_only=False);second=torch.load(fit,map_location='cpu',weights_only=False);parent=(first.get('ema') or first['model']).float();trained=(second.get('ema') or second['model']).float();s=parent.state_dict();t=trained.state_dict();assert s.keys()==t.keys()
 for k in s:
  if torch.is_floating_point(s[k]):s[k]=(1-alpha)*s[k]+alpha*t[k].float()
 parent.load_state_dict(s);first.update(model=parent.half(),ema=None,optimizer=None,epoch=-1,best_fitness=None);first['supplementary_interpolation']={'alpha':alpha,'base':str(base),'fit':str(fit)};Path(path).parent.mkdir(parents=True,exist_ok=True);torch.save(first,path)
def parity(pt,onnx,records):
 import onnxruntime as ort
 torch.set_num_threads(4)
 network=YOLO(str(pt)).model.float().fuse(verbose=False).eval().to('cpu');d=decoder();opt=ort.SessionOptions();opt.intra_op_num_threads=4;opt.inter_op_num_threads=1;session=ort.InferenceSession(str(onnx),sess_options=opt,providers=['CPUExecutionProvider']);rows=[]
 for n,r in enumerate(records):
  x,h,w=d.prepare_input(image(r))
  with torch.inference_mode():y=network(torch.from_numpy(x).to('cpu'))
  expected=[y[0].cpu().numpy(),y[1][-1].cpu().numpy()];actual=session.run(None,{'images':x});checks=[];masks=[]
  for p,z in zip(expected,actual):checks.append({'max_abs_error':float(np.max(np.abs(p-z))),'allclose':bool(np.allclose(p,z,atol=.001,rtol=.0002))})
  for raw in [expected,actual]:
   b,s,c,co=d.process_box_output_debug(raw[0],h,w,.5,.8);m=d.process_mask_output(co,raw[1],b,h,w) if len(co) else [];u=np.maximum.reduce(m)>0 if len(m) else np.zeros((h,w),bool);masks.append(u)
  differing=int(np.count_nonzero(masks[0]!=masks[1]));union=int(np.count_nonzero(masks[0]|masks[1]));rows.append({'id':r['id'],'raw_outputs':checks,'different_mask_pixels':differing,'union_mask_iou':int(np.count_nonzero(masks[0]&masks[1]))/union if union else 1.})
  if n%50==0:print('PARITY',n,'/',len(records),flush=True)
 del network;torch.cuda.empty_cache()
 result={'images':len(rows),'all_raw_outputs_close':all(all(v['allclose'] for v in r['raw_outputs']) for r in rows),'exact_union_masks':sum(r['different_mask_pixels']==0 for r in rows),'minimum_union_mask_iou':min(r['union_mask_iou'] for r in rows),'records':rows};write_json(RUN/'parity.json',result);assert result['all_raw_outputs_close'];return result

def band_gallery(selected_onnx):
 import onnxruntime as ort
 payload=read_json(RUN/'input/items.json');new=read_json(RUN/'crops/manifest.json');splits={r['frame_id']:r['split'] for r in new};folder=RUN/'band_review';folder.mkdir(exist_ok=True);cards=[];rows=[]
 models={}
 for name,path in [('baseline',BASE_ONNX),('candidate',selected_onnx)]:
  d=decoder();opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1;s=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider']);d.inference=lambda tensor,s=s:s.run(None,{'images':tensor});models[name]=d
 for item in payload['items']:
  raw=cv2.imread(str(RUN.parent/'review'/item['image']));target=np.squeeze(cv2.imread(str(RUN/'masks'/item['id']/'union.png'),0))>0;values={};images=[('raw',raw)]
  gt=raw.copy();gt[target]=(30,230,100);images.append(('label',cv2.addWeighted(raw,.5,gt,.5,0)))
  for name,model in models.items():
   mask=model.segment_large_image_debug(raw,.5,.8)[0]>0;union=int(np.count_nonzero(mask|target));inter=int(np.count_nonzero(mask&target));values[name]={'pixel_iou':inter/union if union else 1.,'pixel_precision':inter/max(1,int(mask.sum())),'pixel_recall':inter/max(1,int(target.sum())),'mask_pixels':int(mask.sum())}
   out=raw.copy();out[mask]=(0,190,255) if name=='baseline' else (230,190,20);images.append((name,cv2.addWeighted(raw,.5,out,.5,0)))
  links=[]
  for name,im in images:
   file=folder/f'{item["id"]}_{name}.jpg';cv2.imwrite(str(file),im);links.append(f'<div><p>{name}</p><a href="band_review/{file.name}" target="_blank"><img src="band_review/{file.name}"></a></div>')
  row={'id':item['id'],'camera':item['camera'],'band':item['band'],'source':item['sources'][0],'split':splits[item['frame_id']],'values':values};rows.append(row);cards.append(f'<article><h3>{item["camera"]} {item["filename"]} {item["band"]} · {row["split"]}</h3><section>'+''.join(links)+'</section></article>')
 write_json(RUN/'full_band_comparison.json',{'bands':rows,'limit':'Full-band pixel comparison, not physical disk counts; train rows are diagnostics only.'});(RUN/'comparison.html').write_text('<meta charset="utf-8"><title>Segment before / after</title><style>body{font:14px system-ui;background:#111827;color:#ddd;padding:20px}section{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}img{width:100%}article{border-bottom:1px solid #445;padding:12px 0}h3{font-size:14px}</style><h1>Cạnh đĩa: ảnh gốc / nhãn / baseline / candidate</h1><p>54 band gồm cả train và val, mỗi hàng có đánh dấu split. Bấm ảnh để xem nguyên kích thước. Mask màu không phải số đĩa thật.</p>'+''.join(cards),encoding='utf-8')

def error_gallery(baseline,candidate):
 base={r['id']:r for r in baseline['records']};cand={r['id']:r for r in candidate['records']};records={r['id']:r for r in val_records()};folder=RUN/'errors';folder.mkdir(exist_ok=True);cards=[]
 for identifier,r in cand.items():
  if not (r['fn'] or r['fp'] or base[identifier]['fn'] or base[identifier]['fp']):continue
  row=records[identifier];raw=image(row);target=targets(row)[1];panels=[raw]
  gt=raw.copy();gt[target]=(50,220,100);panels.append(cv2.addWeighted(raw,.5,gt,.5,0))
  for tag in ['baseline','selected']:
   with np.load(RUN/'predictions'/tag/(identifier+'.npz')) as v:m=v['masks'];union=np.maximum.reduce(m)>0 if len(m) else np.zeros(target.shape,bool)
   im=raw.copy();im[union]=(0,170,255) if tag=='baseline' else (255,200,30);panels.append(cv2.addWeighted(raw,.5,im,.5,0))
  file=folder/(identifier+'.jpg');cv2.imwrite(str(file),np.hstack(panels));b=base[identifier];cards.append(f'<article><p>{html.escape(identifier)} · {r["group"]} · FN {b["fn"]}→{r["fn"]}, FP {b["fp"]}→{r["fp"]}</p><a href="errors/{file.name}"><img src="errors/{file.name}"></a></article>')
 (RUN/'errors.html').write_text('<meta charset="utf-8"><title>Validation segment errors</title><style>body{font:14px system-ui;background:#111827;color:#ddd;padding:20px}img{width:100%}article{border-bottom:1px solid #445;padding:12px}</style><h1>Lỗi validation</h1><p>Trái → phải: ảnh gốc / nhãn / baseline / candidate. Đây là lỗi đối chiếu nhãn; nhãn cũ vẫn chỉ review một phần.</p>'+''.join(cards),encoding='utf-8')

def campaign(resume=False):
 if read_json(RUN/'status.json').get('complete'):raise RuntimeError('Completed campaign must not be restarted')
 if (RUN/'STOP_REQUESTED').exists():raise RuntimeError('STOP_REQUESTED exists')
 lock=RUN/'campaign.lock'
 if lock.exists():raise RuntimeError('Campaign lock exists; inspect actual parent/child state before recovery')
 with lock.open('x') as f:json.dump({'pid':os.getpid(),'started_utc':now()},f)
 try:
  integrity();protocol=read_json(RUN/'protocol.json');rows=val_records();status(phase='baseline_evaluation',parent_pid=os.getpid())
  baseline_path=RUN/'baseline.json'
  if baseline_path.exists():baseline=read_json(baseline_path)
  else:
   baseline=evaluate(predict_onnx(BASE_ONNX,rows),RUN/'predictions/baseline');write_json(baseline_path,baseline)
  print('BASELINE',json.dumps(baseline['summary']),flush=True)
  completed=[];failures=[]
  for method in protocol['methods']:
   name=method['name'];folder=RUN/'fits'/name
   if (folder/'fit_complete.json').exists():completed.append(name);continue
   if folder.exists():raise RuntimeError('Unfinished fit exists; inspect logs/checkpoint before any recovery: '+name)
   if (RUN/'STOP_REQUESTED').exists():raise RuntimeError('STOP_REQUESTED exists')
   status(phase='fit_starting',fit=name,parent_pid=os.getpid());env=os.environ.copy();env.update(DISK_PROJECT_ROOT=str(ROOT),SEGMENT_SUPPLEMENT_ROOT=str(RUN))
   with (RUN/f'{name}.stdout.log').open('wb') as out,(RUN/f'{name}.stderr.log').open('wb') as err:
    child=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__).resolve()),'--phase','fit','--name',name],cwd=ROOT,env=env,stdout=out,stderr=err)
    write_json(RUN/'active_child.json',{'pid':child.pid,'parent_pid':os.getpid(),'fit':name,'started_utc':now()})
    code=child.wait()
   write_json(RUN/'active_child.json',{'pid':None,'fit':name,'exit_code':code,'finished_utc':now()})
   if (RUN/'STOP_REQUESTED').exists():raise RuntimeError('STOP_REQUESTED exists')
   if code:
    failure={'fit':name,'exit_code':code,'stderr':str(RUN/f'{name}.stderr.log')};failures.append(failure);write_json(RUN/'fit_failures.json',failures);print('FIT FAILED',failure,flush=True)
   else:completed.append(name)
  if not completed:raise RuntimeError('No fit completed; inspect all recorded failures')
  integrity();status(phase='candidate_evaluation',completed_fits=completed,fit_failures=failures)
  candidates=[]
  for name in completed:
   fitpt=RUN/'fits'/name/'weights/best.pt'
   for alpha in [1.,*protocol['interpolation_fractions']]:
    identifier=name+('_best' if alpha==1 else f'_blend{int(alpha*100):02d}');pt=fitpt
    cached_path=RUN/'evaluations'/f'{identifier}.json'
    if resume and cached_path.exists():
     cached=read_json(cached_path)
     if 'metrics' in cached and Path(cached['pt']).is_file() and Path(cached['onnx']).is_file() and sha(cached['pt'])==cached['pt_sha256'] and sha(cached['onnx'])==cached['onnx_sha256']:
      candidates.append(cached);print('VERIFIED CACHED CANDIDATE',identifier,flush=True);continue
    try:
     if alpha!=1:pt=RUN/'candidates'/identifier/'model.pt';blend(BASE_PT,fitpt,alpha,pt)
     onnx=export(pt);metrics=evaluate(predict_onnx(onnx,rows));result={'id':identifier,'fit':name,'alpha':alpha,'pt':str(pt),'onnx':str(onnx),'pt_sha256':sha(pt),'onnx_sha256':sha(onnx),'metrics':metrics,'gates':gates(metrics,baseline)};candidates.append(result);write_json(RUN/'evaluations'/f'{identifier}.json',result);print('CANDIDATE',identifier,'new',metrics['summary']['new'],'gates',result['gates'],flush=True)
    except Exception as e:
     result={'id':identifier,'fit':name,'alpha':alpha,'error':repr(e),'traceback':traceback.format_exc()};candidates.append(result);write_json(RUN/'evaluations'/f'{identifier}.json',result);print('CANDIDATE FAILED',identifier,repr(e),flush=True)
    status(phase='candidate_evaluation',candidate=identifier,candidates_finished=len(candidates))
  successful=[c for c in candidates if 'metrics' in c]
  if not successful:raise RuntimeError('No candidate could be evaluated')
  passed=[c for c in successful if c['gates']['passed']]
  selected=min(passed,key=lambda c:(c['metrics']['summary']['new']['region_fn'],-c['metrics']['summary']['new']['tolerant_pixel_f2'],c['metrics']['summary']['old_ha']['fn'],-c['metrics']['fitness'])) if passed else max(successful,key=lambda c:c['metrics']['fitness'])
  write_json(RUN/'candidate_summary.json',{'selected':selected['id'],'passing_candidates':[c['id'] for c in passed],'candidates':candidates,'installed_baseline_retained':True,'no_deployment':True});write_json(RUN/'selected.json',selected)
  selected_metrics=evaluate(predict_onnx(selected['onnx'],rows),RUN/'predictions/selected');write_json(RUN/'selected_validation.json',selected_metrics)
  status(phase='parity_and_error_review',selected=selected['id'],recommendation_gates_passed=bool(passed));allnew=read_json(RUN/'crops/manifest.json');oldval=[r for r in rows if r['origin']!='new_reviewed'];parity_result=parity(selected['pt'],selected['onnx'],allnew+oldval)
  band_gallery(selected['onnx']);error_gallery(baseline,selected_metrics)
  # Empty new training crops are diagnostics, never holdout fitness.
  bg=[r for r in allnew if r['background']];diagnostics={}
  for name,model in [('baseline',BASE_ONNX),('candidate',selected['onnx'])]:
   ds=[]
   for r,m,s,raw in predict_onnx(model,bg):ds.append({'id':r['id'],**compare_record(r,m)})
   diagnostics[name]=ds
  write_json(RUN/'new_background_training_diagnostics.json',{'scope':'8 user-reviewed train-only empty crops, not independent validation','models':diagnostics})
  status(phase='white_replay',selected=selected['id']);env=os.environ.copy();env.update(DISK_PROJECT_ROOT=str(ROOT),SEGMENT_SUPPLEMENT_ROOT=str(RUN));replay_code=Path(__file__).parent/'replay_segment_supplementary.py'
  for tag,path in [('baseline',BASE_ONNX),('candidate',Path(selected['onnx']))]:
   with (RUN/f'replay_{tag}.stdout.log').open('wb') as out,(RUN/f'replay_{tag}.stderr.log').open('wb') as err:
    result=subprocess.run([str(ROOT/'SERVICE/.venv/Scripts/python.exe'),'-u','-B',str(replay_code),'--model',str(path),'--tag',tag],cwd=ROOT,env=env,stdout=out,stderr=err)
   if result.returncode:raise RuntimeError('White replay failed: '+tag)
  b=read_json(RUN/'replay/baseline.json');a=read_json(RUN/'replay/candidate.json');assert len(a['frames'])==len(b['frames'])==28;changes=[]
  for before,after in zip(b['frames'],a['frames']):
   assert before['source_sha256']==after['source_sha256'] and before['geometry']==after['geometry'] and before['classifier']==after['classifier']
   if before['counts']!=after['counts']:changes.append({'source':before['source'],'baseline_counts':before['counts'],'candidate_counts':after['counts']})
  replay={'frames':28,'geometry_and_classification_unchanged':True,'production_debug_parity':a['production_debug_parity'],'count_changes':changes,'limit':'Segmentation changes may change caliper counts. These counts are not compared with independently confirmed physical disk counts.','hardware_calls':False};write_json(RUN/'replay/comparison.json',replay)
  integrity();verification={'completed_utc':now(),'completed_fits':completed,'fit_failures':failures,'successful_candidates':len(successful),'passing_candidates':[c['id'] for c in passed],'selected':selected['id'],'full_parity_images':parity_result['images'],'pt_onnx_raw_allclose':parity_result['all_raw_outputs_close'],'white_replay_frames':28,'geometry_and_classifier_retained':True,'protected_inputs_code_runtime_hashes_unchanged':True,'no_deployment':True};write_json(RUN/'completion_verification.json',verification)
  groups=['new_Cam1','new_Cam2','new','old_ha_Cam1','old_ha_Cam2','old_ha','old_demo'];table='| Validation group | Baseline TP/FP/FN | Candidate TP/FP/FN | Full length baseline → candidate |\n|---|---:|---:|---:|\n'
  for g in groups:
   x=baseline['summary'][g];y=selected_metrics['summary'][g];table+=f'| {g} | {x["tp"]}/{x["fp"]}/{x["fn"]} | {y["tp"]}/{y["fp"]}/{y["fn"]} | {x["full_length"]}/{x["gt"]} → {y["full_length"]}/{y["gt"]} |\n'
  pixel_table='| New validation | Tolerant precision baseline to candidate | Tolerant recall baseline to candidate | Strict pixel F1 baseline to candidate | Uncovered regions baseline to candidate |\n|---|---:|---:|---:|---:|\n'
  for g in ['new_Cam1','new_Cam2','new']:
   x=baseline['summary'][g];y=selected_metrics['summary'][g];pixel_table+=f'| {g} | {x["tolerant_pixel_precision"]:.4f} to {y["tolerant_pixel_precision"]:.4f} | {x["tolerant_pixel_recall"]:.4f} to {y["tolerant_pixel_recall"]:.4f} | {x["pixel_f1"]:.4f} to {y["pixel_f1"]:.4f} | {x["region_fn"]} to {y["region_fn"]} |\n'
  report=f'''# Supplementary White segment training — 23 October 2026

Status: COMPLETE. Selected evaluation candidate: {selected['id']}. Recommendation gates {'PASSED' if passed else 'FAILED; installed baseline retained'}. No deployment: the installed baseline is unchanged regardless of recommendation.

324 newly reviewed crops, including 8 empty crops, were exported. New source split: 19 train / 8 val frames, 228 train / 96 val original crops, with entire conservative 120-second capture clusters and decoded aliases kept together. Train has 2,364 images after combining all 1,420 old train images and 944 new photometric/repeated samples. Validation: 96 new + 48 old HA + 117 old-demo crops. All 8 new empty crops are train-only; the background holdout has 7 old-demo crops.

New-data primary comparison uses reviewed mask pixels because source IDs may be tile fragments, not physical disk instances.

{pixel_table}

The instance table below is an auxiliary fragment-level diagnostic, not a physical missed-disk count:

{table}

Instance matching uses runtime confidence .5, NMS .8, mask IoU >= .3; full length additionally needs >=90% target rows and both ends. These are clipped instances/fragments, not physical disk counts. Pixel metrics and per-camera results are in the evaluation JSONs. Old labels retain partial-review/model-assisted limitations. New validation is same-session development data repeatedly used for selection, not independent factory acceptance testing.

One additional mask-head-only fit was scheduled (30 epochs max, patience8), followed by fixed single-network interpolation fractions .25/.5/.75 for each completed fit. Every fit/candidate/failure is recorded in candidate_summary.json and fit_failures.json if present. Methods initialized from the hash-verified installed thinbox512_v3 PT. No new AI labels or mask extrapolation were introduced.

Full PT/ONNX raw-output parity passed on {parity_result['images']} crops; {parity_result['exact_union_masks']} union masks are pixel-exact. Minimum union IoU: {parity_result['minimum_union_mask_iou']:.8f}. Tiny mask-boundary differences can occur near zero logits.

28 hardware-off White frames passed geometry/classifier retention and both-camera production/debug segmentation parity. Caliper counts changed on {len(changes)} frames; this is reported in replay/comparison.json and is not claimed to prove correct disk counts. Input/code/runtime hashes passed integrity checks.

Files: background.html / background_crops.csv; comparison.html; errors.html; selected.json; candidate_summary.json; parity.json; replay/comparison.json; completion_verification.json. New-background training diagnostics are separated from holdout evaluation. The reviewed annotation snapshot and export receipt are under input/. Original reviewed labels, old datasets and installed runtime remain unchanged.
'''
  (RUN/'RESULTS.md').write_text(report,encoding='utf-8');status(phase='complete',complete=True,selected=selected['id'],recommendation_gates_passed=bool(passed),completed_utc=now(),results=str(RUN/'RESULTS.md'),installed_baseline_retained=True);write_json(ROOT/'.work/ACTIVE_SEGMENT_TRAINING.json',{'root':str(RUN),'status':'complete','selected':selected['id'],'recommendation_gates_passed':bool(passed),'no_deployment':True,'dashboard':'http://127.0.0.1:8799/'})
  print('CAMPAIGN COMPLETE',selected['id'],'gates',bool(passed),flush=True)
 except Exception as e:
  status(phase='stopped' if (RUN/'STOP_REQUESTED').exists() else 'failed',error=repr(e),traceback=traceback.format_exc(),complete=False);raise
 finally:
  if lock.exists():lock.unlink()

def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['setup','fit','campaign'],required=True);p.add_argument('--name');p.add_argument('--resume',action='store_true');a=p.parse_args()
 if a.phase=='setup':setup()
 elif a.phase=='fit':fit(a.name)
 else:campaign(a.resume)
if __name__=='__main__':main()
