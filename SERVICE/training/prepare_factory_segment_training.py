"""Freeze/export reviewed Canvas masks, preserve union pixels, build grouped 70/30 data."""
import argparse,base64,collections,hashlib,json,os,shutil,subprocess,sys,threading,time
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import cv2,numpy as np

PROJECT=Path.cwd().resolve()
sys.path.insert(0,str(PROJECT/'SERVICE/training'))
from repair_segment_multipart_export import contour_path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,p)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();run=a.root.resolve();review=PROJECT/'.work/segmentation_factory_20261117/review'
 payload=read(run/'input/items.json');state=read(run/'input/annotations.json');assert state['items_sha256']==sha(run/'input/items.json')
 assert len(payload['items'])==len(state['bands'])==344 and all(x['status']=='reviewed' and x['geometry_checked'] for x in state['bands'].values())
 if (run/'crops/manifest.json').exists():raise FileExistsError('Export already exists; do not overwrite frozen data')
 (run/'masks').mkdir();work=[{'id':i['id'],'width':i['width'],'height':i['height'],'instances':state['bands'][i['id']]['instances']} for i in payload['items']];expected={i['id']:i for i in work};done=threading.Event();errors=[];exported=set()
 html='''<!doctype html><meta charset="utf-8"><script src="/geometry.js"></script><body>Rendering reviewed union masks<script>
 (async()=>{const work=await(await fetch('/data')).json();for(const item of work){const c=document.createElement('canvas');c.width=item.width;c.height=item.height;const ctx=c.getContext('2d',{willReadFrequently:true}),masks=[];for(const ins of item.instances){ctx.clearRect(0,0,c.width,c.height);ctx.fillStyle=ctx.strokeStyle='#fff';SegmentMaskGeometry.drawMask(ctx,ins);const p=ctx.getImageData(0,0,c.width,c.height);for(let n=0;n<p.data.length;n+=4){const v=p.data[n+3]>=128?255:0;p.data[n]=p.data[n+1]=p.data[n+2]=v;p.data[n+3]=255}ctx.putImageData(p,0,0);masks.push({id:ins.id,png:c.toDataURL('image/png').split(',')[1]})}const r=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:item.id,masks})});if(!r.ok)throw Error(await r.text());document.body.firstChild.textContent=item.id}await fetch('/done',{method:'POST'})})().catch(async e=>{await fetch('/error',{method:'POST',body:String(e.stack)})});</script>'''
 class Handler(BaseHTTPRequestHandler):
  def do_GET(self):
   if self.path=='/data':b=json.dumps(work,ensure_ascii=False).encode('utf-8');kind='application/json'
   elif self.path=='/geometry.js':b=(run/'input/mask_geometry.js').read_bytes();kind='text/javascript'
   else:b=html.encode();kind='text/html'
   self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
  def do_POST(self):
   raw=self.rfile.read(int(self.headers.get('Content-Length',0)))
   try:
    if self.path=='/done':done.set()
    elif self.path=='/error':errors.append(raw.decode());done.set()
    elif self.path=='/save':
     value=json.loads(raw);item=expected[value['id']];assert item['id'] not in exported;assert [m['id'] for m in value['masks']]==[i['id'] for i in item['instances']]
     masks=[];ids=[]
     for m in value['masks']:
      mask=cv2.imdecode(np.frombuffer(base64.b64decode(m['png']),np.uint8),0);assert mask.shape==(item['height'],item['width']);assert set(np.unique(mask)).issubset({0,255})
      if mask.any():masks.append((mask>0).astype(np.uint8));ids.append(m['id'])
     union=np.maximum.reduce(masks) if masks else np.zeros((item['height'],item['width']),np.uint8)
     np.savez_compressed(run/'masks'/f'{item["id"]}.npz',masks=np.asarray(masks,np.uint8) if masks else np.empty((0,item['height'],item['width']),np.uint8),review_union=union,instance_ids=np.asarray(ids))
     exported.add(item['id']);write(run/'status.json',{'phase':'exporting_reviewed_masks','bands_exported':len(exported),'total_bands':len(work),'complete':False,'training_started':False})
     if len(exported)%20==0:print('CANVAS',len(exported),'/',len(work),flush=True)
    else:raise ValueError(self.path)
    self.send_response(200);self.end_headers();self.wfile.write(b'ok')
   except Exception as e:errors.append(repr(e));done.set();self.send_response(500);self.end_headers();self.wfile.write(repr(e).encode())
  def log_message(self,*args):pass
 server=ThreadingHTTPServer(('127.0.0.1',0),Handler);threading.Thread(target=server.serve_forever,daemon=True).start()
 edge=Path('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')
 with (run/'canvas_export.log').open('wb') as log:
  browser=subprocess.Popen([str(edge),'--headless','--disable-gpu','--no-first-run','--no-default-browser-check','--disable-background-networking',f'--user-data-dir={run/"canvas_profile"}','--dump-dom','--virtual-time-budget=1800000',f'http://127.0.0.1:{server.server_port}/'],stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
  if not done.wait(1800):raise TimeoutError('Canvas export did not finish')
  server.shutdown()
  if browser.poll() is None:browser.terminate()
 assert not errors,errors;assert len(exported)==344
 # Union truth is authoritative: overlap is OR, never exclusive ownership/count.
 records=[]
 for key in ['images','labels','targets','previews']:(run/'crops'/key).mkdir(parents=True)
 for item in payload['items']:
  source=review/item['image'];assert sha(source)==item['image_sha256'];band=cv2.imread(str(source))
  with np.load(run/'masks'/f'{item["id"]}.npz') as data:masks=data['masks'];union=data['review_union'];ids=data['instance_ids'].tolist()
  assert np.array_equal(masks.any(0) if len(masks) else np.zeros_like(union,bool),union>0)
  for ti,(x1,x2) in enumerate(item['tile_positions']):
   identifier=f'{item["id"]}_tile{ti}';im=band[:,x1:x2];h,w=im.shape[:2];parts=[];parent_ids=[];lines=[]
   for mask,ins_id in zip(masks,ids):
    tile=mask[:,x1:x2]
    if not tile.any():continue
    parts.append(tile);parent_ids.append(ins_id);points=contour_path(tile)/np.asarray([w,h]);lines.append('0 '+' '.join(f'{v:.8f}' for v in points.ravel()))
   truth=union[:,x1:x2];assert np.array_equal(np.asarray(parts).any(0) if parts else np.zeros_like(truth,bool),truth>0)
   image=run/'crops/images'/f'{identifier}.png';label=run/'crops/labels'/f'{identifier}.txt';target=run/'crops/targets'/f'{identifier}.npz'
   assert cv2.imwrite(str(image),im);label.write_text('\n'.join(lines)+('\n' if lines else ''),encoding='utf-8');np.savez_compressed(target,masks=np.asarray(parts,np.uint8) if parts else np.empty((0,h,w),np.uint8),review_union=truth)
   preview=im.copy();preview[truth>0]=(50,230,100);cv2.imwrite(str(run/'crops/previews'/f'{identifier}.jpg'),cv2.addWeighted(im,.5,preview,.5,0))
   variants=[im,cv2.flip(im,0),cv2.flip(im,1),cv2.flip(im,-1)];alias=min(hashlib.sha256(str(v.shape).encode()+v.tobytes()).hexdigest() for v in variants)
   records.append({'id':identifier,'base_crop_id':identifier,'origin':'new_reviewed','source_epoch':'factory_nov2026','camera':item['camera'],'band':item['band'],'tile':ti,'frame_id':item['frame_id'],'source':item['sources'][0],'sources':item['sources'],'source_group':item['frame_id'],'capture_group_draft':item['capture_group'],'image':str(image),'label':str(label),'target_masks':str(target),'shape_hw':[h,w],'objects':len(parts),'background':not bool(truth.any()),'positive_pixels':int(truth.sum()),'parent_instance_ids':parent_ids,'image_sha256':sha(image),'label_sha256':sha(label),'target_masks_sha256':sha(target),'alias_sha256':alias,'repeat':0,'training_mask_authority':'Exact Canvas raster NPZ; txt bbox proxies do not define loss pixels','label_status':'user_reviewed_20261117'})
 assert len(records)==2064
 # Connect all conservative near-duplicate candidates rather than risk leaking them.
 frames={f['id']:f for f in payload['frames']};parent={f:f for f in frames}
 def find(x):
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 def join(a,b):parent[find(b)]=find(a)
 group_seen={};source_frame={s:f['id'] for f in frames.values() for s in f['sources']}
 for f in frames.values():
  g=f['capture_group']
  if g in group_seen:join(f['id'],group_seen[g])
  else:group_seen[g]=f['id']
 draft=read(PROJECT/'.work/segmentation_factory_20261117/grouping_draft.json')
 for pair in draft['near_duplicate_candidates']:join(source_frame[pair['a']],source_frame[pair['b']])
 aliases={}
 for r in records:
  if r['alias_sha256'] in aliases:join(r['frame_id'],aliases[r['alias_sha256']])
  else:aliases[r['alias_sha256']]=r['frame_id']
 groups=collections.defaultdict(list)
 for f in frames:groups[find(f)].append(f)
 keys=sorted(groups);dates=sorted({f['timestamp'][:10] for f in frames.values()});metrics=[]
 metrics.append(np.asarray([len(groups[g]) for g in keys],float))
 for date in dates:metrics.append(np.asarray([sum(frames[f]['timestamp'].startswith(date) for f in groups[g]) for g in keys],float))
 metrics.append(np.asarray([sum(r['background'] for r in records if find(r['frame_id'])==g) for g in keys],float))
 # Deterministic MILP minimizes absolute distribution deviations; labels only.
 from scipy.optimize import milp,Bounds,LinearConstraint
 matrix=np.asarray(metrics);totals=matrix.sum(1);keep=totals>0;matrix=matrix[keep];totals=totals[keep];n=len(keys);k=len(totals)
 objective=np.r_[np.arange(n)*1e-10,1/np.maximum(totals,1)];constraints=np.zeros((2*k,n+k));lo=np.full(2*k,-np.inf);hi=np.zeros(2*k)
 for j in range(k):constraints[2*j,:n]=matrix[j];constraints[2*j,n+j]=-1;hi[2*j]=.3*totals[j];constraints[2*j+1,:n]=-matrix[j];constraints[2*j+1,n+j]=-1;hi[2*j+1]=-.3*totals[j]
 result=milp(objective,integrality=np.r_[np.ones(n),np.zeros(k)],bounds=Bounds(np.zeros(n+k),np.r_[np.ones(n),np.full(k,np.inf)]),constraints=LinearConstraint(constraints,lo,hi),options={'time_limit':30,'mip_rel_gap':0.0})
 assert result.x is not None,result;val_groups={g for g,v in zip(keys,result.x[:n]) if v>.5};assert 0<len(val_groups)<len(keys)
 for r in records:r['capture_group']=find(r['frame_id']);r['split']='val' if r['capture_group'] in val_groups else 'train'
 old=read(PROJECT/'.work/segmentation_supplementary_20261023/training_v2/dataset/manifest.json')
 dataset=run/'dataset'
 for kind in ['images','labels']:
  for split in ['train','val']:(dataset/kind/split).mkdir(parents=True)
 combined=[]
 for original in old+records:
  r=original.copy();is_new=r.get('source_epoch')=='factory_nov2026'
  if not is_new:r['id']='retained_'+r['id'];r['origin']='old_reviewed' if r['origin']=='new_reviewed' else r['origin']
  for field,kind in [('image','images'),('label','labels')]:
   source=Path(r[field]);target=dataset/kind/r['split']/(r['id']+source.suffix);assert sha(source)==r[field+'_sha256'];shutil.copy2(source,target);r[field]=str(target)
  combined.append(r)
 sides=collections.defaultdict(set);group_sides=collections.defaultdict(set)
 for r in combined:sides[r['image_sha256']].add(r['split']);group_sides[('new',r['capture_group']) if r.get('source_epoch')=='factory_nov2026' else ('old',r['source_group'])].add(r['split'])
 assert all(len(x)==1 for x in sides.values()),'Image hash crosses splits';assert all(len(x)==1 for x in group_sides.values())
 write(run/'crops/manifest.json',records);write(dataset/'manifest.json',combined)
 split_report={'policy':'70/30 by paired-camera/capture/<=120s/exact crop flip aliases; all 212 similar-appearance candidate pairs conservatively connected','new_frames':dict(collections.Counter('val' if find(f) in val_groups else 'train' for f in frames)),'new_crops':dict(collections.Counter(r['split'] for r in records)),'new_background_crops':dict(collections.Counter(r['split'] for r in records if r['background'])),'new_camera_crops':{c:dict(collections.Counter(r['split'] for r in records if r['camera']==c)) for c in ['Cam1','Cam2']},'groups':[{'id':g,'frames':fs,'split':'val' if g in val_groups else 'train'} for g,fs in groups.items()],'milp_status':int(result.status),'milp_gap':float(result.mip_gap),'combined_counts':{f'{o}|{s}':v for (o,s),v in collections.Counter((r['origin'],r['split']) for r in combined).items()},'hash_group_alias_isolation':True,'all_new_tiles_included':2064,'limitation':'Heuristic groups are not verified physical tray identities; same factory sessions, development validation; parent has historical exposure to retained old validation.'}
 write(run/'split_report.json',split_report)
 write(run/'input/export_receipt.json',{'bands':344,'crops':2064,'erasures_applied':62,'mask_authority':'Exact shared browser Canvas renderer, alpha>=128; all object masks OR into full-band union, overlaps unrestricted; no fixed object/disk count','input_annotations_sha256':sha(run/'input/annotations.json'),'input_items_sha256':sha(run/'input/items.json'),'all_sources_included_even_if_baseline_correct':True,'split_report':str(run/'split_report.json'),'training_started':False})
 assert sha(review/'annotations.json')==sha(run/'input/annotations.json'),'Live labels changed during export; frozen snapshot retained, inspect before training'
 write(run/'status.json',{'phase':'reviewed_dataset_ready','complete':False,'training_started':False,'new_crops':2064,'train':sum(r['split']=='train' for r in combined),'val':sum(r['split']=='val' for r in combined)})
 print(json.dumps(split_report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
 sys.stdout.reconfigure(encoding='utf-8');main()
