import sys
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from segment_dense_native import native_union,calibrated_objective

def test_native_union_matches_binary_decoder_and_gradient():
    torch.manual_seed(4)
    logits=torch.randn(4,16,16,requires_grad=True)
    support=torch.rand(4,16,16)>.2
    p=native_union(logits,support)
    assert torch.equal(p>.5,((logits>0)&support).any(0))
    target=torch.zeros(16,16);target[5:9,5:9]=1
    loss=calibrated_objective(p,target,support.any(0));loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert (logits.grad[:,6,6]<=0).all()
    assert (logits.grad[:,12,12]>=0).all()
    assert (logits.grad[~support]==0).all()

def test_five_pixel_stripe_prefers_correct_boundary_over_eight_pixel_thickening():
    truth=torch.zeros(512,512);truth[:,253:258]=1
    occupancy=F.interpolate(truth[None,None],(128,128),mode='area')
    correct=F.interpolate(occupancy,(512,512),mode='bilinear',align_corners=False)[0,0]
    thick=F.interpolate((occupancy>0).float()*.99,(512,512),mode='bilinear',align_corners=False)[0,0]
    roi=torch.zeros_like(truth,dtype=torch.bool);roi[:,240:272]=True
    assert (correct[256]>.5).sum()==5 and (thick[256]>.5).sum()==8
    assert calibrated_objective(correct,truth,roi)<calibrated_objective(thick,truth,roi)

def test_empty_native_truth_reduces_false_foreground():
    logits=torch.ones(2,12,12,requires_grad=True);support=torch.ones_like(logits,dtype=torch.bool)
    loss=calibrated_objective(native_union(logits,support),torch.zeros(12,12),support.any(0));loss.backward()
    assert torch.isfinite(loss) and (logits.grad>0).all()

def test_calibrated_bce_is_equivalent_and_cuda_amp_safe():
    device='cuda' if torch.cuda.is_available() else 'cpu'
    logits=torch.linspace(-5,5,64,device=device).reshape(8,8).requires_grad_()
    truth=torch.zeros_like(logits);truth[:,3:5]=1;roi=torch.ones_like(truth,dtype=torch.bool)
    p=logits.sigmoid();dice=1-(2*(p*truth).sum()+1)/(p.sum()+truth.sum()+1)
    expected=F.binary_cross_entropy(p,truth)+.1*dice
    with torch.autocast(device_type=device,enabled=device=='cuda'):
        actual=calibrated_objective(p,truth,roi)
    assert torch.allclose(actual,expected,atol=1e-6,rtol=1e-6)
    actual.backward();assert torch.isfinite(logits.grad).all()
