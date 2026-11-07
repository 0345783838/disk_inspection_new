"""A frozen nano classifier plus a zero-initialized spatial correction to NG only."""
from copy import deepcopy
from pathlib import Path
import hashlib
import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.nn.tasks import ClassificationModel

class SpatialNGClassifier(nn.Module):
    def __init__(self,parent):
        super().__init__()
        if any(getattr(layer,'f',-1)!=-1 for layer in parent.model):raise ValueError('Expected sequential classifier backbone')
        self.parent_yaml=deepcopy(parent.yaml)
        self.backbone=nn.Sequential(*list(parent.model.children())[:-1])
        self.classify=list(parent.model.children())[-1]
        self.feature_dim=self.classify.linear.in_features
        if self.feature_dim!=1280 or self.classify.linear.out_features!=3:raise ValueError('Expected three-class HA nano head')
        for module in (self.backbone,self.classify):
            for parameter in module.parameters():parameter.requires_grad_(False)
        # For fixed 224 input, the feature map is 7x7. This is exactly the
        # 2x2 adaptive-average bins (0:4,3:7), and exports with standard AvgPool.
        self.spatial_pool=nn.AvgPool2d(kernel_size=4,stride=3)
        self.ng_residual=nn.Linear(self.feature_dim*4,1,bias=False)
        nn.init.zeros_(self.ng_residual.weight)
        self.register_buffer('feature_mean',torch.zeros(self.feature_dim*4))
        self.register_buffer('feature_inv_std',torch.ones(self.feature_dim*4))
        self.register_buffer('ng_basis',torch.tensor([[1.,0.,0.]]))
        self.train(False)
    def train(self,mode=True):
        super().train(mode)
        self.backbone.eval();self.classify.eval()
        return self
    def extract(self,images):
        feature_map=self.classify.conv(self.backbone(images))
        if not torch.jit.is_tracing() and feature_map.shape[-2:]!=(7,7):raise ValueError('Spatial model requires 224x224 input')
        spatial=self.spatial_pool(feature_map).flatten(1)
        parent_logits=self.classify.linear(self.classify.drop(self.classify.pool(feature_map).flatten(1)))
        return spatial,parent_logits
    def logits_from_features(self,spatial,parent_logits):
        correction=self.ng_residual((spatial-self.feature_mean)*self.feature_inv_std)
        return parent_logits+correction*self.ng_basis,correction
    def forward(self,images):
        spatial,parent_logits=self.extract(images)
        logits,_=self.logits_from_features(spatial,parent_logits)
        return logits.softmax(-1)
    def freeze_state(self):
        return {k:v.detach().cpu().clone() for k,v in self.state_dict().items() if k.startswith(('backbone.','classify.'))}
    def assert_frozen(self,reference):
        actual=self.state_dict()
        changed=[k for k,v in reference.items() if not torch.equal(v,actual[k].detach().cpu())]
        if changed:raise ValueError('Frozen parent changed: '+str(changed))
        if any(p.requires_grad for m in (self.backbone,self.classify) for p in m.parameters()):raise ValueError('Parent requires gradients')

def from_parent(path):return SpatialNGClassifier(YOLO(str(path)).model.float().cpu().eval())
def state_digest(state):
    digest=hashlib.sha256()
    for k,v in sorted(state.items()):
        digest.update(k.encode());digest.update(str(v.dtype).encode());digest.update(str(tuple(v.shape)).encode());digest.update(v.contiguous().numpy().tobytes())
    return digest.hexdigest()
def save_model(model,path,**metadata):
    torch.save(dict(format='spatial_ng_v1',parent_yaml=model.parent_yaml,state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},labels=['ng','no_disk','ok'],**metadata),path)
def load_model(path,device='cpu'):
    value=torch.load(path,map_location='cpu',weights_only=False)
    if value.get('format')!='spatial_ng_v1' or value['labels']!=['ng','no_disk','ok']:raise ValueError('Wrong spatial artifact')
    parent=ClassificationModel(cfg=value['parent_yaml'],nc=3,verbose=False)
    model=SpatialNGClassifier(parent);model.load_state_dict(value['state_dict'],strict=True)
    return model.to(device).eval()
