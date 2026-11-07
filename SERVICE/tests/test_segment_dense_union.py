"""Loss must remove extra foreground and learn missing reviewed foreground."""
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from segment_dense_union import union_from_logits,union_objective

def test_union_loss_gradient_and_bbox_support():
    logits=torch.zeros((1,12,12),requires_grad=True)
    support=torch.ones_like(logits,dtype=torch.bool)
    support[:,0,:]=False
    target=torch.zeros((12,12));target[4:8,4:8]=1
    loss=union_objective(union_from_logits(logits,support),target)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad[0,5,5]<0  # gradient descent adds missing target pixels
    assert logits.grad[0,9,9]>0  # gradient descent removes false mask pixels
    assert logits.grad[0,0,:].abs().sum()==0  # no mask outside inference box

def test_empty_reviewed_crop_and_stable_overlapping_proposals():
    logits=torch.full((4,12,12),2.,requires_grad=True)
    prediction=union_from_logits(logits,torch.ones_like(logits,dtype=torch.bool))
    loss=union_objective(prediction,torch.zeros((12,12)))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert (logits.grad>0).all()
