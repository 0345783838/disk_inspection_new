"""Reviewed-union supervision, with the same proposal crop used at inference.

Only fully reviewed new masks receive this extra loss. Historical labels retain
their original instance loss. Detection scores/boxes and NMS are detached.
"""
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import nms
from ultralytics.utils.tal import make_anchors
from segment_mask_head_loss import MaskBackgroundLoss
from segment_center_mask_dataset import CenterMaskFormat, CenterYOLODataset


def union_from_logits(logits, support):
    """Differentiable OR of cropped proposal masks (float32 for AMP stability)."""
    if not len(logits):
        return logits.sum(0) * 0
    probs = logits.float().sigmoid() * support.float()
    return -torch.expm1(torch.log1p(-probs.clamp(max=1-1e-6)).sum(0))


def union_objective(pred, truth):
    """Balanced foreground/background BCE + soft Dice; handle empty truth."""
    pred = pred.float().clamp(1e-6, 1-1e-6)
    truth = truth.float().clamp(0, 1)
    positive = -(truth * pred.log()).sum() / truth.sum().clamp_min(1)
    negative = -((1-truth) * torch.log1p(-pred)).sum() / (1-truth).sum().clamp_min(1)
    bce = (positive + negative) / 2 if truth.sum() > 0 else negative
    dice = 1-(2*(pred*truth).sum()+1)/(pred.sum()+truth.sum()+1)
    return bce + dice


class DenseMaskFormat(CenterMaskFormat):
    def __call__(self, labels):
        record = self.records.get(str(Path(labels['im_file']).resolve()))
        reviewed = bool(record and record.get('origin') == 'new_reviewed')
        h, w = labels['img'].shape[:2]
        exact = np.zeros((h//self.mask_ratio,w//self.mask_ratio),np.float32)
        if reviewed:
            with np.load(record['target_masks']) as saved:
                raw = saved['review_union'].astype(np.float32)
            ratio = labels['ratio_pad']
            gains,pad = ratio if isinstance(ratio[0],(tuple,list)) else (ratio,(0,0))
            h0,w0 = labels['ori_shape']; rh,rw = round(h0*gains[0]),round(w0*gains[1])
            left,top = map(int,pad)
            full = np.zeros((h,w),np.float32)
            full[top:top+rh,left:left+rw] = cv2.resize(raw,(rw,rh),interpolation=cv2.INTER_NEAREST_EXACT)
            exact = cv2.resize(full,(w//self.mask_ratio,h//self.mask_ratio),interpolation=cv2.INTER_AREA)
        result = super().__call__(labels)
        result['dense_union_target'] = torch.from_numpy(exact)
        result['dense_reviewed'] = reviewed
        return result


class DenseYOLODataset(CenterYOLODataset):
    def build_transforms(self,hyp=None):
        result=super().build_transforms(hyp)
        formatter=result.transforms[-1]
        assert isinstance(formatter,CenterMaskFormat)
        # Identical settings/cache; only add separately resampled union truth.
        formatter.__class__=DenseMaskFormat
        return result


class DenseUnionLoss:
    def __init__(self, model, weight=.5):
        self.previous = MaskBackgroundLoss(model,weight=.01)
        self.base = self.previous.base
        self.weight = weight
        self.last = {}

    def __call__(self,preds,batch):
        loss,items=self.previous(preds,batch)
        feats,coeff,proto=preds if len(preds)==3 else preds[1]
        bs,_,mh,mw=proto.shape
        flags=batch.get('dense_reviewed',())
        if not any(flags):
            return loss,items
        with torch.no_grad():
            packed=torch.cat([x.view(bs,self.base.no,-1) for x in feats],2)
            distribution,scores=packed.split((self.base.reg_max*4,self.base.nc),1)
            anchors,stride=make_anchors(feats,self.base.stride,.5)
            boxes=self.base.bbox_decode(anchors,distribution.permute(0,2,1).contiguous())*stride
            scores=scores.sigmoid().amax(1)
        ih,iw=batch['img'].shape[-2:]
        yy=torch.arange(mh,device=proto.device).view(1,mh,1)+.5
        xx=torch.arange(mw,device=proto.device).view(1,1,mw)+.5
        terms=[];proposals=[]
        for b,reviewed in enumerate(flags):
            if not reviewed:
                continue
            with torch.no_grad():
                ids=torch.where(scores[b]>=.5)[0]
                if len(ids):
                    ids=ids[scores[b,ids].argsort(descending=True)[:300]]
                    keep=nms(boxes[b,ids].float(),scores[b,ids].float(),.8)[:64]
                    ids=ids[keep]
                    bb=boxes[b,ids].float()*boxes.new_tensor([mw/iw,mh/ih,mw/iw,mh/ih])
                    support=(xx>=bb[:,0,None,None])&(xx<bb[:,2,None,None])&(yy>=bb[:,1,None,None])&(yy<bb[:,3,None,None])
                else:
                    support=torch.empty((0,mh,mw),dtype=torch.bool,device=proto.device)
            if not len(ids):
                # No inference proposal: the unchanged assigned-instance loss
                # still learns positive targets. Union loss cannot invent boxes.
                continue
            logits=torch.einsum('cn,chw->nhw',coeff[b,:,ids].float(),proto[b].float())
            prediction=union_from_logits(logits,support)
            truth=batch['dense_union_target'][b].to(device=proto.device,dtype=torch.float32)
            assert truth.shape==(mh,mw)
            terms.append(union_objective(prediction,truth));proposals.append(len(ids))
        if terms:
            extra=self.weight*self.base.hyp.box*torch.stack(terms).mean()
            loss=loss.clone();items=items.clone();loss[1]+=extra*bs;items[1]+=extra.detach()
            self.last={'reviewed_images':len(terms),'proposal_counts':proposals,'dense_loss':float(extra.detach())}
        return loss,items
