"""Mask union loss after native logit upsampling, calibrated by binary targets."""
import torch
import torch.nn.functional as F
import cv2,numpy as np
from pathlib import Path
from torchvision.ops import nms
from ultralytics.utils.tal import make_anchors
from segment_center_mask_dataset import CenterMaskFormat,CenterYOLODataset
from segment_mask_head_loss import MaskBackgroundLoss

def native_union(logits,support):
    # max(sigmoid) > .5 is exactly OR of the native >0 logit masks.
    return (logits.float().sigmoid()*support.float()).amax(0)

def calibrated_objective(pred,truth,roi):
    pred=pred.float().clamp(1e-6,1-1e-6);truth=truth.float()
    assert bool(((truth==0)|(truth==1)).all())
    # One spatial weight per pixel, shared between positive/negative BCE terms.
    # Do not rebalance the two terms: that shifts the optimal mask boundary.
    region=(roi|(truth>0)).float()
    # Explicit float32 probability BCE is mathematically equivalent here and
    # is safe inside CUDA autocast (F.binary_cross_entropy rejects AMP).
    pixel_bce=-(truth*pred.log()+(1-truth)*torch.log1p(-pred))
    bce=(pixel_bce*region).sum()/region.sum().clamp_min(1)
    dice=1-(2*(pred*truth).sum()+1)/(pred.sum()+truth.sum()+1)
    return bce+.1*dice

class NativeMaskFormat(CenterMaskFormat):
    def __call__(self,labels):
        r=self.records.get(str(Path(labels['im_file']).resolve()));reviewed=bool(r and r.get('origin')=='new_reviewed')
        h,w=labels['img'].shape[:2];full=np.zeros((h,w),np.float32)
        if reviewed:
            with np.load(r['target_masks']) as f:raw=f['review_union'].astype(np.float32)
            ratio=labels['ratio_pad'];gain,pad=ratio if isinstance(ratio[0],(tuple,list)) else (ratio,(0,0));h0,w0=labels['ori_shape'];rh,rw=round(h0*gain[0]),round(w0*gain[1]);left,top=map(int,pad)
            full[top:top+rh,left:left+rw]=cv2.resize(raw,(rw,rh),interpolation=cv2.INTER_NEAREST_EXACT)
        result=super().__call__(labels);result['dense_union_target']=torch.from_numpy(full);result['dense_reviewed']=reviewed;return result

class NativeYOLODataset(CenterYOLODataset):
    def build_transforms(self,hyp=None):
        result=super().build_transforms(hyp);result.transforms[-1].__class__=NativeMaskFormat;return result

class NativeUnionLoss:
    def __init__(self,model,weight=.5):
        self.previous=MaskBackgroundLoss(model,weight=.01);self.base=self.previous.base;self.weight=weight;self.last={}
    def __call__(self,preds,batch):
        loss,items=self.previous(preds,batch);feats,coeff,proto=preds if len(preds)==3 else preds[1];bs,_,mh,mw=proto.shape;flags=batch.get('dense_reviewed',())
        if not any(flags):return loss,items
        with torch.no_grad():
            packed=torch.cat([f.view(bs,self.base.no,-1) for f in feats],2);dist,scores=packed.split((self.base.reg_max*4,self.base.nc),1);anchors,stride=make_anchors(feats,self.base.stride,.5);boxes=self.base.bbox_decode(anchors,dist.permute(0,2,1).contiguous())*stride;scores=scores.sigmoid().amax(1)
        ih,iw=batch['img'].shape[-2:];yy=torch.arange(ih,device=proto.device).view(1,ih,1)+.5;xx=torch.arange(iw,device=proto.device).view(1,1,iw)+.5;terms=[];counts=[]
        for b,reviewed in enumerate(flags):
            if not reviewed:continue
            with torch.no_grad():
                ids=torch.where(scores[b]>=.5)[0]
                if not len(ids):continue
                ids=ids[scores[b,ids].argsort(descending=True)[:300]];ids=ids[nms(boxes[b,ids].float(),scores[b,ids].float(),.8)[:64]];bb=boxes[b,ids].float()
                support=(xx>=bb[:,0,None,None])&(xx<bb[:,2,None,None])&(yy>=bb[:,1,None,None])&(yy<bb[:,3,None,None])
            logits=torch.einsum('cn,chw->nhw',coeff[b,:,ids].float(),proto[b].float())
            logits=F.interpolate(logits[None],size=(ih,iw),mode='bilinear',align_corners=False)[0]
            probability=native_union(logits,support);truth=batch['dense_union_target'][b].to(proto.device,dtype=torch.float32);assert truth.shape==(ih,iw)
            terms.append(calibrated_objective(probability,truth,support.any(0)));counts.append(len(ids))
        if terms:
            extra=self.weight*self.base.hyp.box*torch.stack(terms).mean();loss=loss.clone();items=items.clone();loss[1]+=extra*bs;items[1]+=extra.detach();self.last={'reviewed_images':len(terms),'proposal_counts':counts,'dense_loss':float(extra.detach()),'target_grid':[ih,iw]}
        return loss,items
