"""Materialize the camera-balanced capture-cluster 70/30 segmentation split."""
import sys,os,shutil,hashlib,itertools,collections,csv,html
from pathlib import Path
from datetime import datetime
import cv2,numpy as np,yaml
from segment_supplementary_core import sha,read_json,write_json
ROOT=Path(__file__).resolve().parents[2];RUN=Path(os.environ.get('SEGMENT_SUPPLEMENT_ROOT',str(ROOT/'.work/segmentation_supplementary_20261023/training')))

def main():
 new=read_json(RUN/'crops/manifest.json');items=read_json(RUN/'input/items.json');old=read_json(ROOT/'.work/segmentation_faint_20261001/dataset_faint_v2/manifest.json');audit=read_json(RUN.parent/'model_dataset_audit.json')
 for p in audit['protected_files']:assert sha(p['path'])==p['sha256'],p['path']
 for r in read_json(RUN.parent/'dataset_inventory.json'):
  assert sha(r['image'])==r['image_sha256'];assert sha(r['label'])==r['label_sha256']
 frames=items['frames'];byid={f['id']:f for f in frames};ordered=sorted(frames,key=lambda f:Path(f['sources'][0]).stem)
 parent={f['id']:f['id'] for f in frames}
 def find(a):
  while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
  return a
 def join(a,b):parent[find(a)]=find(b)
 previous=None
 for f in ordered:
  stamp=datetime.strptime(Path(f['sources'][0]).stem.replace('Image_',''),'%Y%m%d%H%M%S%f')
  if previous and (stamp-previous[1]).total_seconds()<=120:join(f['id'],previous[0])
  previous=f['id'],stamp
 aliases={}
 for f in frames:
  raw=cv2.imread(f['sources'][0]);assert raw is not None
  digests=[hashlib.sha256(v.tobytes()).hexdigest() for v in [raw,cv2.flip(raw,0),cv2.flip(raw,1),cv2.flip(raw,-1)]]
  signature=(f['camera'],tuple(raw.shape),min(digests))
  if signature in aliases:join(f['id'],aliases[signature])
  else:aliases[signature]=f['id']
 groups=collections.defaultdict(list)
 for f in ordered:groups[find(f['id'])].append(f['id'])
 groups=dict(groups);keys=sorted(groups,key=lambda g:Path(byid[groups[g][0]]['sources'][0]).stem)
 background_frames={r['frame_id'] for r in new if r['background']};forced_train={find(f) for f in background_frames}
 best=None
 for bits in itertools.product([0,1],repeat=len(keys)):
  val={f for g,bit in zip(keys,bits) if bit for f in groups[g]}
  if any(g in forced_train and bit for g,bit in zip(keys,bits)):continue
  cams=collections.Counter(byid[f]['camera'] for f in val);allcams=collections.Counter(f['camera'] for f in frames)
  if any(cams[c]==0 or cams[c]==allcams[c] for c in allcams):continue
  score=sum((cams[c]/allcams[c]-.3)**2 for c in allcams)+2*(len(val)/len(frames)-.3)**2
  candidate=(score,tuple(sorted(val)))
  if best is None or candidate<best:best=candidate
 assert best
 val=set(best[1]);split_by_frame={f['id']:'val' if f['id'] in val else 'train' for f in frames}
 dataset=RUN/'dataset'
 if dataset.exists():raise FileExistsError(dataset)
 for split in ['train','val']:
  for kind in ['images','labels']:(dataset/kind/split).mkdir(parents=True)
 records=[]
 for r in old:
  nr={**r,'id':'old_'+Path(r['image']).stem,'old_image':r['image'],'old_label':r['label'],'repeat':r.get('repeat',0)}
  for key,kind in [('image','images'),('label','labels')]:
   target=dataset/kind/r['split']/('old_'+Path(r[key]).name);shutil.copy2(r[key],target);nr[key]=str(target)
  nr['label_sha256']=sha(nr['label']);nr['image_sha256']=sha(nr['image']);records.append(nr)
 for r in new:
  r['split']=split_by_frame[r['frame_id']];r['capture_group']=find(r['frame_id'])
  repeats=1 if r['split']=='val' else (8 if r['background'] else 4)
  original=cv2.imread(r['image']);assert original is not None
  for rep in range(repeats):
   nr={**r,'id':r['id']+f'_r{rep}','base_crop_id':r['id'],'repeat':rep,'augmentation':'original' if rep==0 else 'photometric_only','new_user_reviewed':True}
   im=dataset/'images'/r['split']/(nr['id']+'.png');label=dataset/'labels'/r['split']/(nr['id']+'.txt')
   if rep==0:shutil.copy2(r['image'],im)
   else:
    rng=np.random.default_rng(int(hashlib.sha256(nr['id'].encode()).hexdigest()[:8],16));x=original.astype(np.float32)/255
    gamma=[1,.94,1.08,1.02][rep%4];gain=[1,.98,.96,1.02][rep%4]
    x=np.power(np.clip(x,0,1),gamma)*gain
    if rep%4 in [2,3]:x+=rng.normal(0,.8/255,x.shape)
    assert cv2.imwrite(str(im),np.uint8(np.clip(x*255,0,255)))
   shutil.copy2(r['label'],label);nr['image']=str(im);nr['label']=str(label);nr['image_sha256']=sha(im);nr['label_sha256']=sha(label);records.append(nr)
 
 if (RUN/'repair_receipt.json').exists():
  prior=read_json(Path(read_json(RUN/'repair_receipt.json')['prior_run'])/'crops/manifest.json');assert {r['frame_id']:r['split'] for r in prior}=={r['frame_id']:r['split'] for r in new},'Repair must not change split'
 write_json(RUN/'crops/manifest.json',new);write_json(RUN/'dataset/manifest.json',records)
 for name,chosen in [('all',[r for r in records if r['split']=='val']),('new',[r for r in records if r['split']=='val' and r['origin']=='new_reviewed']),('old_ha',[r for r in records if r['split']=='val' and r['origin']=='factory']),('old_demo',[r for r in records if r['split']=='val' and r['origin']=='old'])]:
  listing=dataset/f'{name}_val.txt';listing.write_text('\n'.join(r['image'] for r in chosen)+'\n',encoding='utf-8')
  (dataset/f'{name}.yaml').write_text(yaml.safe_dump({'path':dataset.as_posix(),'train':'images/train','val':listing.as_posix(),'names':{0:'disk'}},sort_keys=False),encoding='utf-8')
 image_sides=collections.defaultdict(set);group_sides=collections.defaultdict(set)
 for r in records:
  image_sides[r['image_sha256']].add(r['split']);group_sides[(r['capture_group'] if r['origin']=='new_reviewed' else r['source_group'])].add(r['split'])
 assert all(len(s)==1 for s in image_sides.values());assert all(len(s)==1 for s in group_sides.values())
 background=[r for r in new if r['background']]
 with (RUN/'background_crops.csv').open('w',newline='',encoding='utf-8-sig') as f:
  writer=csv.DictWriter(f,fieldnames=['id','camera','source','band','tile','split','image','label']);writer.writeheader();writer.writerows({k:r[k] for k in writer.fieldnames} for r in background)
 counts=collections.Counter((r['origin'],r['split']) for r in records)
 report={'policy':'70/30 capture clusters; consecutive source timestamps <=120 seconds connect across both cameras; exact decoded/flip/180 aliases connected; background capture cluster forced to train by user-delegated choice after clarification','frame_counts':dict(collections.Counter(split_by_frame.values())),'camera_frame_counts':{c:dict(collections.Counter(s for f,s in split_by_frame.items() if byid[f]['camera']==c)) for c in ['Cam1','Cam2']},'capture_groups':[{'id':g,'split':split_by_frame[groups[g][0]],'frames':[{'id':f,'camera':byid[f]['camera'],'sources':byid[f]['sources']} for f in groups[g]]} for g in keys],'new_original_crop_counts':dict(collections.Counter(r['split'] for r in new)),'background_original_counts':dict(collections.Counter(r['split'] for r in background)),'materialized_counts':{f'{o}|{s}':n for (o,s),n in counts.items()},'materialized_background_train':sum(r.get('background',False) for r in records if r['split']=='train' and r['origin']=='new_reviewed'),'validation_limit':'New validation has no fully empty crop: all 8 new empty crops belong to the train-only capture cluster. Old validation backgrounds are reported separately. Capture-time clusters are a conservative heuristic, not verified tray IDs; all new images share one acquisition session.','leakage_checks':{'source_group':True,'image_sha256':True,'decoded_flip_aliases_connected':True},'training_started':False}
 write_json(RUN/'split_report.json',report)
 cards=''.join(f'<article><img src="crops/previews/{r["id"]}.jpg"><p>{html.escape(Path(r["source"]).name)}<br>{r["camera"]} · {r["band"]} · tile {r["tile"]} · {r["split"]}<br>Mask pixels: 0 · label file: 0 bytes</p></article>' for r in background)
 (RUN/'background.html').write_text('<meta charset="utf-8"><title>8 crop background đã duyệt</title><style>body{font:15px system-ui;background:#111827;color:#e5e7eb;padding:24px}main{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}article{background:#1f2937;padding:12px;border-radius:8px}img{width:100%}p{font-size:12px;overflow-wrap:anywhere}</style><h1>8 crop background đã duyệt</h1><p>Tất cả ở train. Đây là 8 crop khác nhau từ 4 ảnh nguồn; không phải 8 ảnh toàn khung trống. Nhãn rỗng được giữ. Chưa có background mới độc lập ở val.</p><main>'+cards+'</main>',encoding='utf-8')
 print(__import__('json').dumps(report,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
