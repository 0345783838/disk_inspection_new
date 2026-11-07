"""Freeze reviewed vector labels and rasterize them with the editor's Canvas2D semantics."""
import sys,os,json,hashlib,shutil,threading,subprocess,time,base64,io
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from datetime import datetime,timezone
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'.work/segmentation_supplementary_20261023/training'
REVIEW=RUN.parent/'review'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 resume='--resume-crops' in sys.argv
 if RUN.exists() and not resume:raise FileExistsError(RUN)
 (RUN/'input').mkdir(parents=True,exist_ok=resume);(RUN/'masks').mkdir(exist_ok=resume);(RUN/'crops/images').mkdir(parents=True,exist_ok=resume);(RUN/'crops/labels').mkdir(exist_ok=resume);(RUN/'crops/targets').mkdir(exist_ok=resume);(RUN/'crops/previews').mkdir(exist_ok=resume)
 for name in ['items.json','annotations.json']:
  
  if not resume:shutil.copy2(REVIEW/name,RUN/'input'/name)
  else:assert sha(REVIEW/name)==sha(RUN/'input'/name)
 payload=json.loads((RUN/'input/items.json').read_text(encoding='utf-8'));state=json.loads((RUN/'input/annotations.json').read_text(encoding='utf-8'))
 assert state['items_sha256']==sha(RUN/'input/items.json')
 assert len(state['bands'])==len(payload['items'])==54
 assert all(state['bands'][i['id']]['status']=='reviewed' and state['bands'][i['id']]['geometry_checked'] for i in payload['items'])
 source_hashes={str(REVIEW/i['image']):i['image_sha256'] for i in payload['items']}
 for path,h in source_hashes.items():assert sha(path)==h
 work=[{'id':i['id'],'width':i['width'],'height':i['height'],'instances':state['bands'][i['id']]['instances']} for i in payload['items']]
 if not resume:
  done=threading.Event();fail=[];exported={};expected={i['id']:i for i in work}
  html='''<!doctype html><meta charset="utf-8"><body>Rendering reviewed labelsâ€¦<script>
 (async()=>{const data=await(await fetch('/data')).json();for(const item of data){const masks=[];const canvas=document.createElement('canvas');canvas.width=item.width;canvas.height=item.height;const ctx=canvas.getContext('2d',{willReadFrequently:true});for(const ins of item.instances){ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#fff';ctx.strokeStyle='#fff';ctx.lineJoin='round';ctx.lineCap='round';ctx.beginPath();if(ins.kind==='line'){ctx.lineWidth=ins.width;ins.points.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.stroke()}else{ins.parts.forEach(part=>{part.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.closePath()});ctx.fill()}const pixels=ctx.getImageData(0,0,canvas.width,canvas.height);for(let j=0;j<pixels.data.length;j+=4){const v=pixels.data[j+3]>=128?255:0;pixels.data[j]=pixels.data[j+1]=pixels.data[j+2]=v;pixels.data[j+3]=255}ctx.putImageData(pixels,0,0);masks.push({id:ins.id,png:canvas.toDataURL('image/png').split(',')[1]})}const r=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:item.id,masks})});if(!r.ok)throw Error(await r.text());document.body.textContent=item.id+' rendered'}await fetch('/done',{method:'POST'});document.body.textContent='ALL REVIEWED MASKS EXPORTED'} )().catch(async e=>{document.body.textContent=e.stack;await fetch('/error',{method:'POST',body:String(e.stack)})})
 </script>'''
  class Handler(BaseHTTPRequestHandler):
   def do_GET(self):
    b=json.dumps(work).encode() if self.path=='/data' else html.encode()
    self.send_response(200);self.send_header('Content-Type','application/json' if self.path=='/data' else 'text/html');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
   def do_POST(self):
    size=int(self.headers.get('Content-Length',0));raw=self.rfile.read(size)
    try:
     if self.path=='/done':done.set()
     elif self.path=='/error':fail.append(raw.decode());done.set()
     elif self.path=='/save':
      value=json.loads(raw);item=expected[value['id']];assert [m['id'] for m in value['masks']]==[i['id'] for i in item['instances']]
      folder=RUN/'masks'/item['id'];folder.mkdir(exist_ok=True);masks=[]
      for n,row in enumerate(value['masks']):
       b=base64.b64decode(row['png']);mask=cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_GRAYSCALE)
       assert mask.shape==(item['height'],item['width']) and set(np.unique(mask)).issubset({0,255})
       path=folder/f'{n:03d}.png';path.write_bytes(b);masks.append(str(path))
      exported[item['id']]=masks
     else:raise ValueError('unknown endpoint')
     self.send_response(200);self.end_headers();self.wfile.write(b'ok')
    except Exception as e:
     fail.append(repr(e));done.set();self.send_response(500);self.end_headers();self.wfile.write(repr(e).encode())
   def log_message(self,*args):pass
  server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
  edge=Path('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')
  args=[str(edge),'--headless','--disable-gpu','--no-first-run','--no-default-browser-check','--disable-background-networking',f'--user-data-dir={RUN/"raster_browser_profile"}','--dump-dom','--virtual-time-budget=120000',f'http://127.0.0.1:{server.server_port}/']
  with (RUN/'raster_browser.log').open('wb') as log:
   proc=subprocess.Popen(args,stdout=log,stderr=log)
   if not done.wait(180):raise TimeoutError('Canvas export did not finish')
   server.shutdown()
   if proc.poll() is None:proc.terminate()
  assert not fail,fail
  assert len(exported)==len(work)
 else:
  exported={i['id']:[str(p) for p in sorted((RUN/'masks'/i['id']).glob('[0-9]*.png'))] for i in work}
  assert all(len(exported[i['id']])==len(i['instances']) for i in work)
 records=[];negative=[];conversion=[]
 for item in payload['items']:
  band=cv2.imread(str(REVIEW/item['image']));masks=[cv2.imread(p,0)>0 for p in exported[item['id']]];union=np.maximum.reduce(masks) if masks else np.zeros(band.shape[:2],bool)
  cv2.imwrite(str(RUN/'masks'/item['id']/'union.png'),union.astype(np.uint8)*255)
  for ti,(x1,x2) in enumerate(item['tile_positions']):
   identifier=f'{item["id"]}_tile{ti}';impath=RUN/'crops/images'/f'{identifier}.png';label=RUN/'crops/labels'/f'{identifier}.txt';targetpath=RUN/'crops/targets'/f'{identifier}.npz';image=band[:,x1:x2];h,w=image.shape[:2]
   parts=[];parent_ids=[];lines=[];target_parts=[];lost=0
   for j,mask in enumerate(masks):
    tile=mask[:,x1:x2].astype(np.uint8);contours,_=cv2.findContours(tile,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
     if len(c)<3 or cv2.contourArea(c)<=0:
      component=np.zeros_like(tile);cv2.drawContours(component,[c],-1,1,1)
      expanded=cv2.resize(component,None,fx=8,fy=8,interpolation=cv2.INTER_NEAREST)
      expanded_contours,_=cv2.findContours(expanded,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
      q=expanded_contours[0].reshape(-1,2)
      # Nonzero fractional envelope, preserving exactly the same pixel centers
      # under both round and floor rasterizers; never turn a boundary sliver into background.
      points=(q//8).astype(float)+.1+.3*(q%8)/7
     else:points=c.reshape(-1,2).astype(float)+.25
     part=np.zeros_like(tile);cv2.fillPoly(part,[np.rint(points).astype(np.int32)],1)
     # Polygons are split into connected parts; no bridge is invented between disjoint fragments.
     target_parts.append(part);parent_ids.append(state['bands'][item['id']]['instances'][j]['id'])
     normalized=points/np.array([w,h],np.float64)
     lines.append('0 '+' '.join(f'{v:.8f}' for v in normalized.ravel()))
   raw_union=union[:,x1:x2];reconstructed=np.maximum.reduce(target_parts).astype(bool) if target_parts else np.zeros((h,w),bool)
   inter=int(np.count_nonzero(raw_union&reconstructed));total=int(np.count_nonzero(raw_union|reconstructed));coverage=inter/np.count_nonzero(raw_union) if raw_union.any() else 1.0
   if raw_union.any() and not target_parts:raise ValueError('Positive pixels cannot be exported as background: '+identifier)
   assert coverage>=.97,(identifier,coverage)
   assert cv2.imwrite(str(impath),image);label.write_text('\n'.join(lines)+('\n' if lines else ''),encoding='utf-8')
   np.savez_compressed(targetpath,masks=np.asarray(target_parts,np.uint8) if target_parts else np.empty((0,h,w),np.uint8),review_union=raw_union.astype(np.uint8))
   preview=image.copy();preview[reconstructed]=(0,230,100);cv2.imwrite(str(RUN/'crops/previews'/f'{identifier}.jpg'),cv2.addWeighted(image,.55,preview,.45,0))
   record={'id':identifier,'origin':'new_reviewed','frame_id':item['frame_id'],'camera':item['camera'],'band':item['band'],'tile':ti,'sources':item['sources'],'source':item['sources'][0],'source_sha256':item['source_sha256'],'source_group':item['frame_id'],'image':str(impath),'label':str(label),'target_masks':str(targetpath),'shape_hw':[h,w],'band_x1':x1,'band_x2':x2,'objects':len(lines),'background':not bool(raw_union.any()),'positive_pixels':int(raw_union.sum()),'image_sha256':sha(impath),'label_sha256':sha(label),'label_status':'user_reviewed_20261023','parent_instance_ids':parent_ids,'polygon_pixel_recall':coverage if raw_union.any() else 1.,'polygon_pixel_iou':inter/total if total else 1.,'degenerate_component_pixels':lost}
   records.append(record);conversion.append({k:record[k] for k in ['id','polygon_pixel_recall','polygon_pixel_iou','degenerate_component_pixels']})
   if record['background']:negative.append(record)
 assert sha(REVIEW/'annotations.json')==sha(RUN/'input/annotations.json'),'Labels changed during frozen export; review receipt needs update'
 write(RUN/'crops/manifest.json',records);write(RUN/'background_crops.json',negative);write(RUN/'mask_conversion_audit.json',conversion)
 write(RUN/'input/export_receipt.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'user_authorization':'All labels corrected/confirmed; crop every tile including backgrounds, combine with verified old dataset, train and evaluate; new split explicitly 70/30 by capture clusters with both cameras.','review_items_sha256':sha(RUN/'input/items.json'),'review_annotations_sha256':sha(RUN/'input/annotations.json'),'rasterization':'Editor-equivalent browser Canvas2D, round caps/joins, exact requested width, alpha >=128 to binary. Polygon components exported individually without invented bridges.','source_images_verified':len(source_hashes),'bands':len(payload['items']),'crops':len(records),'background_crops':len(negative),'minimum_polygon_pixel_recall':min(r['polygon_pixel_recall'] for r in records),'training_started':False})
 print(json.dumps({'crops':len(records),'backgrounds':len(negative),'min_polygon_recall':min(r['polygon_pixel_recall'] for r in records),'background_by_frame':{f:sum(r['background'] for r in records if r['frame_id']==f) for f in sorted({r['frame_id'] for r in records})}},indent=2),flush=True)
if __name__=='__main__':main()
