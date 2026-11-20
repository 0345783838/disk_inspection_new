"""Paired current-config White replay; no camera/PLC/light access."""
import argparse,ast,hashlib,json,os,sys,time
from pathlib import Path
import cv2,numpy as np,decouple
PROJECT=Path.cwd().resolve();sys.path[:0]=[str(PROJECT/'SERVICE'),str(PROJECT/'SERVICE/training')]
from smoke_local_white import normalize
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--candidate',type=Path,required=True);a=ap.parse_args();run=a.root.resolve()
 protocol=read(run/'protocol.json');before={p:sha(p) for p in protocol['protected_runtime_hashes']};assert before==protocol['protected_runtime_hashes']
 decouple.config=decouple.Config(decouple.RepositoryEnv(str(PROJECT/'SERVICE/config/config.env')))
 from src.communication.modbus_client import MbClient
 def forbidden(*a,**kw):raise AssertionError('Hardware access forbidden in replay')
 for name in ['connect','disconnect','check_connection','_MbClient__write_bit','_MbClient__read_bit']:setattr(MbClient,name,forbidden)
 os.chdir(PROJECT/'SERVICE')
 from src.service.check_disk_service_yolo import DiskCheckingService
 from src.inference.yolo_segment.onnx_segmentor import OnnxSegmentor
 service=DiskCheckingService();base=service.disk_segmentor_yolo
 candidate=OnnxSegmentor(str(a.candidate.resolve()),['disk'],conf_thres=base.conf_threshold,iou_thres=base.iou_threshold,preprocess_mode=base.preprocess_mode,rgb_input=base.rgb_input,mask_decode=base.mask_decode,tile_merge=base.tile_merge)
 models={'baseline':base,'candidate':candidate};capture={'bands':[],'classes':[]};original_classify=service.point_classification_model.predict_batch
 def classify(*a,**kw):
  value=original_classify(*a,**kw);capture['classes'].append({'labels':[str(x) for x in value[0]],'confidences':[float(x) for x in value[1]]});return value
 service.point_classification_model.predict_batch=classify
 segment_debug=[];checked=set()
 for tag,model in models.items():
  original=model.segment_large_image_debug
  def segment(image,*args,original=original,tag=tag,model=model,**kw):
   result=original(image,*args,**kw);capture['bands'].append({'image':image.copy(),'mask':result[0].copy()})
   key=(tag,capture['camera'])
   if key not in checked:
    normal=model.segment_large_image(image)[0];assert np.array_equal(normal,result[0]);checked.add(key);segment_debug.append({'model':tag,'camera':capture['camera'],'mask_exact':True})
   return result
  model.segment_large_image_debug=segment
 payload=read(run/'input/items.json');items={(i['frame_id'],i['band']):i for i in payload['items']};frame_by_source={s:f for f in payload['frames'] for s in f['sources']}
 sources=[(c,p,'old_28') for c in [1,2] for p in sorted((PROJECT/f'.work/NEW_IMAGES/ANH_18_10/white_cam{c}').glob('*.bmp'))]
 assert len(sources)==28
 sources += [(int(f['camera'][-1]),Path(f['sources'][0]),'new_172') for f in payload['frames']];assert len(sources)==200
 app_config=PROJECT/'APP/DiskInspection/bin/Debug/DiskInspection.exe.config';records=[];ratios=[];debug=[];fullbands=[];params=service._default_white_params();split={r['frame_id']:r['split'] for r in read(run/'crops/manifest.json')};gallery=run/'band_review';gallery.mkdir(exist_ok=True);cards=[]
 source_hashes={str(p):sha(p) for _,p,_ in sources}
 for index,(camera,source,scope) in enumerate(sources):
  raw=cv2.imread(str(source));image=normalize(raw,camera,app_config);ok,encoded=cv2.imencode('.jpg',image);assert ok;image=cv2.imdecode(encoded,cv2.IMREAD_COLOR);values={};captured={}
  for tag in (['baseline','candidate'] if index%2==0 else ['candidate','baseline']):
   service.disk_segmentor_yolo=models[tag];capture.update(bands=[],classes=[],camera=camera);started=time.perf_counter();response=service.check_disk_white(image.copy());duration=time.perf_counter()-started
   fields={k:getattr(response,k) for k in ['CropBox','UvBox1','UvBox2','PointGeometryStatus','InferredSlots']};counts=[len(ast.literal_eval(getattr(response,k))) if getattr(response,k) else None for k in ['Mid1','Mid2']]
   values[tag]={'geometry':fields,'classifier':capture['classes'],'counts':counts,'result':response.Result,'error':response.ErrorCode,'gap_checks':response.CaliperGapChecks,'seconds':duration};captured[tag]=capture['bands'].copy()
  assert values['baseline']['geometry']==values['candidate']['geometry'];assert values['baseline']['classifier']==values['candidate']['classifier'];ratio=values['candidate']['seconds']/values['baseline']['seconds'];ratios.append(ratio)
  if scope=='new_172':
   frame=frame_by_source[str(source)];assert source_hashes[str(source)]==frame['sha256'];assert len(captured['baseline'])==len(captured['candidate'])==2
   for bi in [1,2]:
    item=items[(frame['id'],f'band{bi}')]
    with np.load(run/'masks'/f'{item["id"]}.npz') as d:truth=d['review_union']>0
    panels=[captured['baseline'][bi-1]['image']];marked=panels[0].copy();marked[truth]=(30,230,100);panels.append(cv2.addWeighted(panels[0],.5,marked,.5,0));row={'id':item['id'],'camera':item['camera'],'band':item['band'],'split':split[frame['id']],'target_pixels':int(truth.sum())}
    for tag in ['baseline','candidate']:
     band=captured[tag][bi-1];assert band['mask'].shape==truth.shape;original_band=PROJECT/'.work/segmentation_factory_20261117/review'/item['image'];assert sha(original_band)==item['image_sha256'];assert np.array_equal(band['image'],cv2.imread(str(original_band))),'Band pixels/geometry changed'
     mask=band['mask']>0;intersection=int((mask&truth).sum());denom=int(mask.sum()+truth.sum());row[tag]={'pixel_f1':2*intersection/max(1,denom) if denom else 1.,'intersection':intersection,'predicted_pixels':int(mask.sum()),'missed_pixels':int((truth&~mask).sum()),'false_pixels':int((mask&~truth).sum())};out=band['image'].copy();out[mask]=(0,190,255) if tag=='baseline' else (220,180,30);panels.append(cv2.addWeighted(band['image'],.5,out,.5,0))
    path=gallery/f'{item["id"]}.jpg';stack=np.vstack(panels);cv2.imwrite(str(path),cv2.resize(stack,(1000,round(stack.shape[0]*1000/stack.shape[1]))));cards.append(f'<article><h3>{item["id"]} · {row["split"]}</h3><p>Union F1 {row["baseline"]["pixel_f1"]:.4f} → {row["candidate"]["pixel_f1"]:.4f}</p><img loading="lazy" src="band_review/{path.name}"></article>');fullbands.append(row)
  if (scope,camera) not in {(d['scope'],d['camera']) for d in debug}:
   service.disk_segmentor_yolo=candidate;capture.update(bands=[],classes=[],camera=camera);r=service.check_disk_debug(image.copy(),params)
   assert all(getattr(r,k)==values['candidate']['geometry'][k] for k in ['CropBox','UvBox1','UvBox2','PointGeometryStatus'])
   assert [len(ast.literal_eval(getattr(r,k))) if getattr(r,k) else None for k in ['Mid1','Mid2']]==values['candidate']['counts'];assert capture['classes']==values['candidate']['classifier'];debug.append({'scope':scope,'camera':camera,'geometry_counts_classifier_parity':True})
  records.append({'source':str(source),'source_sha256':source_hashes[str(source)],'camera':camera,'scope':scope,'models':values,'paired_ratio':ratio});print('WHITE',index+1,'/',len(sources),camera,source.name,f'{ratio:.3f}x',flush=True)
 after={p:sha(p) for p in before};assert after==before;assert all(sha(p)==h for p,h in source_hashes.items());assert checked=={('baseline',1),('baseline',2),('candidate',1),('candidate',2)}
 result={'frames':len(records),'both_camera_debug_parity':debug,'segment_production_debug_parity':segment_debug,'geometry_classifier_unchanged':True,'hardware_calls':False,'protected_runtime_unchanged':True,'median_ratio':float(np.median(ratios)),'max_ratio':float(max(ratios)),'all_paired_ratios_le_2x':all(r<=2 for r in ratios),'timing_scope':'Actual current-config whole White software pipeline, excluding camera/PLC/network/file loading/JPEG preparation; paired model order alternated','records':records}
 write(run/'replay/comparison.json',result);write(run/'full_band_comparison.json',{'bands':fullbands,'semantics':'Full band mask union; overlaps allowed. Train bands are fitted diagnostics, val is development. Not independent physical disk count truth.'})
 (run/'comparison.html').write_text('<meta charset="utf-8"><title>Full-band union comparison</title><style>body{font:14px system-ui;background:#142135;color:white;padding:20px}img{width:100%;max-width:1200px}article{padding:16px;border-bottom:1px solid #789}</style><h1>Ảnh gốc / union nhãn / baseline / selected</h1><p>344 band, ghi rõ train/val. Object được phép chồng lên nhau; so sánh union pixels.</p>'+''.join(cards),encoding='utf-8')
if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');main()
