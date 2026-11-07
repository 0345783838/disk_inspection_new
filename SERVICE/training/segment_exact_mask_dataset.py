"""Train/val formatter using reviewed raw multipart masks without polygon bridge pixels."""
import os,copy,math
from pathlib import Path
import cv2,numpy as np,torch
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.augment import Compose,LetterBox,RandomHSV,Format
from segment_supplementary_core import read_json

class ExactMaskFormat(Format):
 def __init__(self,*args,records=None,**kw):super().__init__(*args,**kw);self.records=records or {};self.cache={}
 def __call__(self,labels):
  key=str(Path(labels['im_file']).resolve());r=self.records.get(key);exact=None
  if r is not None:
   assert not self.mask_overlap
   path=r['target_masks']
   if path not in self.cache:
    with np.load(path) as f:self.cache[path]=f['masks'].astype(np.uint8)
   raw=self.cache[path];h,w=labels['img'].shape[:2];ratio=labels['ratio_pad'];gains,pad=ratio if isinstance(ratio[0],(tuple,list)) else (ratio,(0,0));h0,w0=labels['ori_shape'];rh,rw=round(h0*gains[0]),round(w0*gains[1]);left,top=map(int,pad)
   assert rh+top<=h and rw+left<=w,(path,(rh,rw),(h,w),pad)
   full=np.zeros((len(raw),h,w),np.uint8)
   for n,m in enumerate(raw):full[n,top:top+rh,left:left+rw]=cv2.resize(m,(rw,rh),interpolation=cv2.INTER_NEAREST)
   exact=np.asarray([cv2.resize(m,(w//self.mask_ratio,h//self.mask_ratio),interpolation=cv2.INTER_LINEAR) for m in full],np.uint8) if len(full) else np.empty((0,h//self.mask_ratio,w//self.mask_ratio),np.uint8)
  result=super().__call__(labels)
  if exact is not None:
   assert len(result['cls'])==len(exact),(key,len(result['cls']),len(exact))
   result['masks']=torch.from_numpy(exact)
  return result

class ExactYOLODataset(YOLODataset):
 def __init__(self,*args,**kw):
  run=Path(os.environ['SEGMENT_SUPPLEMENT_ROOT']);rows=read_json(run/'dataset/manifest.json');self.exact_records={str(Path(r['image']).resolve()):r for r in rows if r.get('target_masks')};super().__init__(*args,**kw)
 def build_transforms(self,hyp=None):
  # All geometric variation is precomputed/source-bound. Do not infer mask warps
  # from polygon proxy paths: image and exact masks share only load_image + LetterBox.
  assert hyp.mosaic==hyp.mixup==hyp.copy_paste==hyp.flipud==hyp.fliplr==hyp.degrees==hyp.translate==hyp.scale==hyp.shear==hyp.perspective==0
  transforms=[LetterBox(new_shape=(self.imgsz,self.imgsz),scaleup=False)]
  if self.augment:transforms.append(RandomHSV(hgain=hyp.hsv_h,sgain=hyp.hsv_s,vgain=hyp.hsv_v))
  transforms.append(ExactMaskFormat(bbox_format='xywh',normalize=True,return_mask=self.use_segments,return_keypoint=False,return_obb=False,batch_idx=True,mask_ratio=hyp.mask_ratio,mask_overlap=hyp.overlap_mask,bgr=hyp.bgr if self.augment else 0.,records=self.exact_records))
  return Compose(transforms)
