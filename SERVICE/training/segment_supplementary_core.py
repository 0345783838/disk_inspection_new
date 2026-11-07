"""Native-decoder segmentation evaluation shared by the supplementary campaign."""
import sys,json,hashlib,collections,os
from pathlib import Path
import cv2,numpy as np,torch
from scipy.optimize import linear_sum_assignment
project=Path(os.environ.get('DISK_PROJECT_ROOT',str(Path.cwd())))
frozen_src=Path(__file__).resolve().parent/'SERVICE/src'
sys.path.insert(0,str(frozen_src if frozen_src.exists() else project/'SERVICE/src'))
from inference.yolo_segment.onnx_segmentor import OnnxSegmentor
cv2.setNumThreads(1)
try:
 from threadpoolctl import threadpool_limits
 threadpool_limits(limits=2)
except ImportError:pass
TARGET_CACHE={};IMAGE_CACHE={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)),encoding='utf-8')
def read_json(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def decoder():
 d=OnnxSegmentor.__new__(OnnxSegmentor);d.input_width=d.input_height=512;d.num_masks=32;d.num_splits=6;d.overlap_ratio=.05;d.preprocess_mode='letterbox';d.rgb_input=True;d.mask_decode='native';d.tile_merge='core';d.conf_threshold=.5;d.iou_threshold=.8;return d
def image(r):
 p=r['image']
 if p not in IMAGE_CACHE:
  x=cv2.imread(p)
  if x is None:raise ValueError(p)
  IMAGE_CACHE[p]=x
 return IMAGE_CACHE[p]
def targets(r):
 k=r['label']
 if k not in TARGET_CACHE:
  h,w=image(r).shape[:2]
  if r.get('target_masks'):
   with np.load(r['target_masks']) as v:parts=[m.astype(bool) for m in v['masks']];union=v['review_union'].astype(bool)
  else:
   parts=[]
   for line in Path(k).read_text().splitlines():
    p=np.asarray([float(v) for v in line.split()[1:]],np.float32).reshape(-1,2)*[w,h];mask=np.zeros((h,w),np.uint8);cv2.fillPoly(mask,[np.rint(p).astype(np.int32)],1);parts.append(mask.astype(bool))
   union=np.maximum.reduce(parts) if parts else np.zeros((h,w),bool)
  TARGET_CACHE[k]=(parts,union)
 return TARGET_CACHE[k]
def predict_torch(network,records,batch=8):
 d=decoder();param=next(network.parameters());device=param.device;dtype=param.dtype
 for start in range(0,len(records),batch):
  rs=records[start:start+batch];arrays=[d.prepare_input(image(r))[0] for r in rs];x=torch.from_numpy(np.concatenate(arrays)).to(device=device,dtype=dtype)
  with torch.inference_mode():out=network(x)
  boxes=out[0].detach().float().cpu().numpy();proto=out[1][-1].detach().float().cpu().numpy()
  for n,r in enumerate(rs):
   h,w=image(r).shape[:2];b,s,c,co=d.process_box_output_debug(boxes[n:n+1],h,w,.5,.8);m=d.process_mask_output(co,proto[n:n+1],b,h,w) if len(co) else []
   yield r,[np.asarray(a,bool) for a in m if np.any(a)],np.asarray(s),[boxes[n:n+1],proto[n:n+1]]
def predict_onnx(path,records):
 import onnxruntime as ort
 d=decoder();opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
 session=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])
 for r in records:
  x,h,w=d.prepare_input(image(r));out=session.run(None,{'images':x});b,s,c,co=d.process_box_output_debug(out[0],h,w,.5,.8);m=d.process_mask_output(co,out[1],b,h,w) if len(co) else []
  yield r,[np.asarray(a,bool) for a in m if np.any(a)],np.asarray(s),out

def compare_record(r,pred):
 truth,union=targets(r);h,w=union.shape;p_union=np.maximum.reduce(pred) if pred else np.zeros((h,w),bool)
 ious=np.zeros((len(truth),len(pred)),np.float32)
 for i,t in enumerate(truth):
  for j,p in enumerate(pred):ious[i,j]=np.count_nonzero(t&p)/max(1,np.count_nonzero(t|p))
 pairs=[]
 if ious.size:
  a,b=linear_sum_assignment(-(ious+100*(ious>=.3)));pairs=[(i,j) for i,j in zip(a,b) if ious[i,j]>=.3]
 full=0
 for i,j in pairs:
  rows=np.flatnonzero(truth[i].any(axis=1));hits=(truth[i]&pred[j]).any(axis=1)[rows]
  if len(hits) and hits.mean()>=.9 and hits[:12].any() and hits[-12:].any():full+=1
 kernel=np.ones((5,5),np.uint8)
 t_d=cv2.dilate(union.astype(np.uint8),kernel)>0;p_d=cv2.dilate(p_union.astype(np.uint8),kernel)>0
 region_fn=0;region_full=0
 for target in truth:
  rows=np.flatnonzero(target.any(axis=1));covered=(target&p_d).any(axis=1)[rows];coverage=np.count_nonzero(target&p_d)/max(1,np.count_nonzero(target));found=coverage>=.8
  region_fn+=int(not found)
  region_full+=int(found and len(covered)>0 and covered.mean()>=.9 and covered[:12].any() and covered[-12:].any())
 return {'region_fn':region_fn,'region_full':region_full,'tp':len(pairs),'fp':len(pred)-len(pairs),'fn':len(truth)-len(pairs),'gt':len(truth),'full_length':full,'images':1,'background_images':int(not union.any()),'background_fp_images':int(not union.any() and p_union.any()),'background_fp_pixels':int(p_union.sum()) if not union.any() else 0,'target_pixels':int(union.sum()),'predicted_pixels':int(p_union.sum()),'intersection_pixels':int((union&p_union).sum()),'tolerant_target_hit_pixels':int((union&p_d).sum()),'tolerant_pred_hit_pixels':int((p_union&t_d).sum())}
def finish(v):
 v=dict(v);tp,fp,fn=v['tp'],v['fp'],v['fn'];v.update(precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),f1=2*tp/max(1,2*tp+fp+fn),f2=5*tp/max(1,5*tp+fp+4*fn),full_length_recall=v['full_length']/max(1,v['gt']))
 v['region_full_recall']=v['region_full']/max(1,v['gt'])
 v['pixel_f1']=2*v['intersection_pixels']/max(1,v['target_pixels']+v['predicted_pixels'])
 pr=v['tolerant_pred_hit_pixels']/max(1,v['predicted_pixels']);re=v['tolerant_target_hit_pixels']/max(1,v['target_pixels']);v['tolerant_pixel_precision']=pr;v['tolerant_pixel_recall']=re;v['tolerant_pixel_f2']=5*pr*re/max(1e-12,4*pr+re)
 return v
def group_for(r):
 if r['origin']=='new_reviewed':return 'new_'+r['camera']
 if r['origin']=='factory':return 'old_ha_'+r['camera']
 return 'old_demo'
def evaluate(predictions,save_masks=None):
 grouped=collections.defaultdict(collections.Counter);rows=[]
 if save_masks:Path(save_masks).mkdir(parents=True,exist_ok=True)
 for r,p,s,raw in predictions:
  v=compare_record(r,p);g=group_for(r);grouped[g].update(v);grouped['all'].update(v)
  if g.startswith('new_'):grouped['new'].update(v)
  if g.startswith('old_ha_'):grouped['old_ha'].update(v)
  rows.append({'id':r['id'],'group':g,'camera':r.get('camera'),'source':r['source'],'image':r['image'],**v})
  if save_masks:
   h,w=image(r).shape[:2];np.savez_compressed(Path(save_masks)/(r['id']+'.npz'),masks=np.asarray(p,np.uint8) if p else np.empty((0,h,w),np.uint8))
 summary={g:finish(v) for g,v in grouped.items()}
 # Both cameras contribute equally; background validation is old-demo because the new background burst is train-only.
 new=np.mean([summary['new_'+c]['tolerant_pixel_f2'] for c in ['Cam1','Cam2'] if 'new_'+c in summary]);ha=np.mean([summary['old_ha_'+c]['f2'] for c in ['Cam1','Cam2'] if 'old_ha_'+c in summary]);old=summary.get('old_demo',{}).get('f2',0)
 bg=sum(v['background_images'] for k,v in summary.items() if k in ['new','old_ha','old_demo']);bgfp=sum(v['background_fp_images'] for k,v in summary.items() if k in ['new','old_ha','old_demo']);specificity=1-bgfp/max(1,bg)
 fitness=.55*new+.25*ha+.1*old+.05*specificity+.05*summary.get('new',{}).get('region_full_recall',0)
 return {'conf':.5,'nms_iou':.8,'mask_match_iou':.3,'decoder':'runtime letterbox RGB / native','fitness':float(fitness),'summary':summary,'records':rows}
def gates(candidate,baseline):
 a,b=candidate['summary'],baseline['summary'];tests={}
 tests['new_tolerant_pixel_recall_improves']=a['new']['tolerant_pixel_recall']>b['new']['tolerant_pixel_recall']+1e-6
 tests['new_tolerant_pixel_precision_within_half_pp']=a['new']['tolerant_pixel_precision']>=b['new']['tolerant_pixel_precision']-.005
 tests['new_pixel_f1_not_reduced']=a['new']['pixel_f1']>=b['new']['pixel_f1']
 tests['new_uncovered_regions_not_increased']=a['new']['region_fn']<=b['new']['region_fn']
 tests['new_region_full_length_not_reduced']=a['new']['region_full']>=b['new']['region_full']
 tests['old_ha_recall_within_1pp']=a['old_ha']['recall']>=b['old_ha']['recall']-.01
 tests['old_ha_precision_within_1pp']=a['old_ha']['precision']>=b['old_ha']['precision']-.01
 tests['old_demo_f1_within_1pp']=a['old_demo']['f1']>=b['old_demo']['f1']-.01
 tests['old_background_false_crops_not_increased']=a['old_demo']['background_fp_images']<=b['old_demo']['background_fp_images']
 return {'passed':all(tests.values()),'tests':tests}
