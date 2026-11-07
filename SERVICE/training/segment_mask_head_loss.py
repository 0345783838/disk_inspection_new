"""Mask-only optimization with supervised empty-image mask loss."""
import torch
import torch.nn.functional as F
from ultralytics.utils.loss import v8SegmentationLoss

def freeze_except_masks(model):
 index=len(model.model)-1;prefixes=(f'model.{index}.proto.',f'model.{index}.cv4.');names=[]
 for name,p in model.named_parameters():
  p.requires_grad_(name.startswith(prefixes))
  if p.requires_grad:names.append(name)
 assert names and all(name.startswith(prefixes) for name in names)
 return names,prefixes

class MaskBackgroundLoss:
 def __init__(self,model,weight=.01):self.base=v8SegmentationLoss(model);self.weight=weight;self.background_images_seen=0
 def __call__(self,preds,batch):
  loss,items=self.base(preds,batch);feats,coeff,proto=preds if len(preds)==3 else preds[1];bs=proto.shape[0];positive=set(batch['batch_idx'].long().tolist());background=[i for i in range(bs) if i not in positive]
  if not background:return loss,items
  scores=torch.cat([x.view(bs,self.base.no,-1) for x in feats],2)[:,self.base.reg_max*4:,:].detach().sigmoid().amax(1);terms=[]
  for b in background:
   ids=scores[b].topk(min(8,scores.shape[1])).indices;selected=coeff[b].index_select(1,ids).transpose(0,1);logits=selected@proto[b].reshape(proto.shape[1],-1);terms.append(F.softplus(logits).mean())
  extra=self.weight*self.base.hyp.box*torch.stack(terms).mean();loss=loss.clone();items=items.clone();loss[1]+=extra*bs;items[1]+=extra.detach();self.background_images_seen+=len(background)
  return loss,items
