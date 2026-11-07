"""Frozen nano detail model plus a small target-aware raw-context correction."""
from copy import deepcopy
from pathlib import Path
import hashlib
import cv2
import numpy as np
import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.nn.tasks import ClassificationModel

LABELS=['ng','no_disk','ok']

def tensor(image):
    return np.transpose(cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),(224,224),interpolation=cv2.INTER_LINEAR).astype(np.float32)/255.,(2,0,1))

def make_contexts(image,rects,direction,span=5):
    """Clean context RGB plus queried physical slot column; no drawn overlays."""
    rects=np.asarray(rects,dtype=np.int32)
    if len(rects)<2:raise ValueError('Need ordered neighboring target rectangles')
    centers=(rects[:,0]+rects[:,2])/2.;pitch=float(np.median(np.diff(centers)))
    if pitch<=0:raise ValueError('Invalid slot order/pitch')
    results=[];descriptions=[]
    for rect,cx in zip(rects,centers):
        x1,y1,x2,y2=map(int,rect);left=int(np.floor(cx-span*pitch/2));width=max(1,int(round(span*pitch)));right=left+width
        height=y2-y1
        if height<=0:raise ValueError('Empty target rectangle')
        patch=np.zeros((height,width,3),np.uint8);mask=np.zeros((height,width),np.uint8)
        sx1,sx2=max(0,left),min(image.shape[1],right);sy1,sy2=max(0,y1),min(image.shape[0],y2)
        if sx2<=sx1 or sy2<=sy1:raise ValueError('Context outside frame')
        patch[sy1-y1:sy2-y1,sx1-left:sx2-left]=image[sy1:sy2,sx1:sx2]
        a=max(0,int(round(cx-pitch/2))-left);b=min(width,int(round(cx+pitch/2))-left)
        mask[sy1-y1:sy2-y1,a:b]=255
        if direction=='top':patch=cv2.rotate(patch,cv2.ROTATE_180);mask=cv2.rotate(mask,cv2.ROTATE_180)
        rgb=cv2.cvtColor(cv2.resize(patch,(224,224),interpolation=cv2.INTER_LINEAR),cv2.COLOR_BGR2RGB)
        marker=cv2.resize(mask,(224,224),interpolation=cv2.INTER_NEAREST)
        results.append(np.dstack([rgb,marker]));descriptions.append(dict(requested=[left,y1,right,y2],pitch=pitch,span=span,source_box=[sx1,sy1,sx2,sy2]))
    return results,descriptions

def augment(local,context,seed):
    rng=np.random.default_rng(seed);local=local.copy();context=context.copy()
    if rng.random()<.5:local=cv2.flip(local,1);context=cv2.flip(context,1)
    exposure=float(rng.uniform(.9,1.1));contrast=float(rng.uniform(.92,1.08));gamma=float(rng.uniform(.94,1.06))
    def photo(image):
        x=image.astype(np.float32);mean=x.mean((0,1),keepdims=True);x=np.clip(((x-mean)*contrast+mean)*exposure/255,0,1)
        return np.clip(np.power(x,gamma)*255,0,255).astype(np.uint8)
    local=photo(local);context[:,:,:3]=photo(context[:,:,:3]);return local,context

def block(cin,cout,stride=2):
    return nn.Sequential(nn.Conv2d(cin,cin,3,stride,1,groups=cin,bias=False),nn.Conv2d(cin,cout,1,bias=False),nn.GroupNorm(4,cout),nn.SiLU())

class ContextClassifier(nn.Module):
    def __init__(self,parent,control=False):
        super().__init__();self.control=bool(control);self.parent_yaml=deepcopy(parent.yaml)
        self.backbone=nn.Sequential(*list(parent.model.children())[:-1]);self.classify=list(parent.model.children())[-1]
        if self.classify.linear.in_features!=1280 or self.classify.linear.out_features!=3:raise ValueError('Wrong nano parent')
        for layer in [self.backbone,self.classify]:
            for value in layer.parameters():value.requires_grad_(False)
        self.context=nn.Sequential(nn.Conv2d(4,16,5,4,2,bias=False),nn.GroupNorm(4,16),nn.SiLU(),
            block(16,32),block(32,64),block(64,96),nn.AvgPool2d(4,3),nn.Flatten())
        self.context_head=nn.Linear(384,3)
        self.fusion=nn.Sequential(nn.LayerNorm(1664),nn.Linear(1664,64),nn.SiLU(),nn.Dropout(.05),nn.Linear(64,3))
        nn.init.zeros_(self.fusion[-1].weight);nn.init.zeros_(self.fusion[-1].bias)
        self.train(False)
    def train(self,mode=True):
        super().train(mode);self.backbone.eval();self.classify.eval();return self
    def local(self,images):
        features=self.classify.pool(self.classify.conv(self.backbone(images))).flatten(1)
        return features,self.classify.linear(features)
    def cached(self,features,logits,context):
        available=context[:,3:4].sum((2,3)).clamp(0,1)
        if self.control:context=torch.cat([torch.zeros_like(context[:,:3]),context[:,3:4]],1)
        context_features=self.context(context)
        delta=self.fusion(torch.cat([features,context_features],1))*available
        return logits+delta,self.context_head(context_features),available.flatten()
    def forward(self,images,context):
        features,logits=self.local(images);out,_,_=self.cached(features,logits,context);return out.softmax(1)
    def frozen_state(self):
        return {f'{name}.{k}':v.detach().cpu().clone() for name in ['backbone','classify'] for k,v in getattr(self,name).state_dict().items()}

def parent_model(path):return YOLO(str(path)).model.float().cpu().eval()
def save_model(path,model,**metadata):
    torch.save(dict(version=1,parent_yaml=model.parent_yaml,control=model.control,state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},metadata=metadata),path)
def load_model(path):
    value=torch.load(path,map_location='cpu',weights_only=False)
    parent=ClassificationModel(cfg=value['parent_yaml'],nc=3,verbose=False)
    model=ContextClassifier(parent,value['control']);model.load_state_dict(value['state_dict']);return model.eval()
def state_sha(state):
    result=hashlib.sha256()
    for key,value in sorted(state.items()):result.update(key.encode());result.update(value.contiguous().numpy().tobytes())
    return result.hexdigest()
