"""Hardware-off White replay using the campaign's frozen source and one segment model."""
import sys,os,argparse,json,hashlib,ast
from pathlib import Path
import cv2,numpy as np,decouple
ROOT=Path(os.environ.get('DISK_PROJECT_ROOT',str(Path.cwd()))).resolve();RUN=Path(os.environ.get('SEGMENT_SUPPLEMENT_ROOT',str(ROOT/'.work/segmentation_supplementary_20261023/training'))).resolve()
code=Path(__file__).resolve().parent;servicepath=code/'SERVICE' if (code/'SERVICE/src').exists() else ROOT/'SERVICE';sys.path[:0]=[str(servicepath),str(code),str(ROOT/'SERVICE/training')]
from smoke_local_white import normalize
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--model',required=True);a.add_argument('--tag',required=True);args=a.parse_args()
 protocol=json.loads((RUN/'protocol.json').read_text(encoding='utf-8'));before={p:sha(p) for p in protocol['protected_runtime_hashes']};assert before==protocol['protected_runtime_hashes']
 runtime=ROOT/'APP/DiskInspection/bin/BroSight_Disinspector/plugin';original=decouple.Config(decouple.RepositoryEnv(str(runtime/'config/config.env')))
 class Config:
  def __call__(self,key,*v,**kw):return str(Path(args.model).resolve()) if key=='DISK_SEGMENT_MODEL_PATH' else original(key,*v,**kw)
 decouple.config=Config();os.chdir(runtime)
 from src.service.check_disk_service_yolo import DiskCheckingService
 service=DiskCheckingService();seg=service.disk_segmentor_yolo;capture={'bands':[],'classes':[]};origseg=seg.segment_large_image_debug;origclass=service.point_classification_model.predict_batch;checked=set();debug=[]
 def segment(im,*v,**kw):
  value=origseg(im,*v,**kw);capture['bands'].append({'image':im.copy(),'mask':value[0].copy()})
  if capture['camera'] not in checked:
   normal=seg.segment_large_image(im)[0];ok=np.array_equal(normal,value[0]);assert ok,'production/debug segment mismatch';debug.append({'camera':capture['camera'],'mask_equal':ok});checked.add(capture['camera'])
  return value
 def classify(*v,**kw):
  result=origclass(*v,**kw);capture['classes'].append({'labels':[str(x) for x in result[0]],'confidences':[float(x) for x in result[1]]});return result
 seg.segment_large_image_debug=segment;service.point_classification_model.predict_batch=classify;rows=[]
 for cam in [1,2]:
  for p in sorted((ROOT/f'.work/NEW_IMAGES/ANH_18_10/white_cam{cam}').glob('*.bmp')):
   capture.update(camera=cam,bands=[],classes=[]);raw=cv2.imread(str(p));im=normalize(raw,cam,runtime.parent/'DiskInspection.exe.config');ok,jpg=cv2.imencode('.jpg',im);assert ok;im=cv2.imdecode(jpg,cv2.IMREAD_COLOR);r=service.check_disk_white(im.copy())
   geometry={k:getattr(r,k) for k in ['CropBox','UvBox1','UvBox2','PointGeometryStatus','InferredSlots']};counts=[len(ast.literal_eval(getattr(r,k))) if getattr(r,k) else None for k in ['Mid1','Mid2']]
   rows.append({'camera':cam,'source':str(p),'source_sha256':sha(p),'geometry':geometry,'classifier':capture['classes'],'counts':counts,'result':r.Result,'error_code':r.ErrorCode,'segment_bands':len(capture['bands']),'segment_pixels':[int(np.count_nonzero(b['mask'])) for b in capture['bands']]})
   print(args.tag,cam,p.name,counts,flush=True)
 after={p:sha(p) for p in before};assert after==before;assert checked=={1,2}
 target=RUN/'replay'/f'{args.tag}.json';target.parent.mkdir(exist_ok=True);target.write_text(json.dumps({'model':str(Path(args.model).resolve()),'frames':rows,'production_debug_parity':debug,'protected_runtime_unchanged':True,'hardware_calls':False},ensure_ascii=False,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)),encoding='utf-8')
if __name__=='__main__':main()
