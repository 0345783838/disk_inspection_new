from pathlib import Path
from copy import deepcopy
import sys,tempfile,unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from classifier_spatial_ng import from_parent,save_model,load_model,state_digest
PARENT=Path(__file__).resolve().parents[2]/'.work/classifier_finetune_ha_7030_20261022/artifacts/best.pt'
class SpatialNGTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(2);cls.model=from_parent(PARENT)
 def test_zero_residual_matches_parent_and_train_keeps_bn_eval(self):
  from ultralytics import YOLO
  torch.manual_seed(1);x=torch.rand(2,3,224,224);parent=YOLO(str(PARENT)).model.float().eval();model=deepcopy(self.model).train()
  with torch.no_grad():a=model(x);p=parent(x);p=p[0] if isinstance(p,tuple) else p
  self.assertTrue(torch.equal(a,p));self.assertFalse(model.classify.training)
  self.assertTrue(all(not m.training for m in model.backbone.modules() if isinstance(m,torch.nn.BatchNorm2d)))
 def test_backward_changes_only_new_branch_and_preserves_non_ng_logit_difference(self):
  model=deepcopy(self.model).train();reference=model.freeze_state();x=torch.rand(3,3,224,224)
  with torch.no_grad():features,old=model.extract(x)
  optimizer=torch.optim.AdamW(model.ng_residual.parameters(),lr=.001)
  logits,delta=model.logits_from_features(features,old);loss=torch.nn.functional.binary_cross_entropy_with_logits(delta[:,0],torch.tensor([1.,0.,1.]));loss.backward();optimizer.step()
  logits,delta=model.logits_from_features(features,old)
  self.assertTrue(torch.equal(logits[:,1:],old[:,1:]));self.assertGreater(float(model.ng_residual.weight.abs().sum()),0)
  model.assert_frozen(reference)
  self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),5120)
 def test_portable_roundtrip_keeps_predictions_and_hash(self):
  model=deepcopy(self.model).eval();x=torch.rand(2,3,224,224)
  with tempfile.TemporaryDirectory() as t:
   path=Path(t)/'model.pt';save_model(model,path);other=load_model(path)
   with torch.no_grad():self.assertTrue(torch.equal(model(x),other(x)))
   self.assertEqual(state_digest(model.freeze_state()),state_digest(other.freeze_state()))
 def test_pool_matches_adaptive_and_onnx_dynamic_batch(self):
  import onnxruntime as ort,numpy as np
  model=deepcopy(self.model).eval()
  with torch.no_grad():model.ng_residual.weight.normal_(0,.0002)
  x=torch.rand(2,3,224,224)
  f=torch.rand(2,1280,7,7);self.assertTrue(torch.allclose(model.spatial_pool(f),torch.nn.functional.adaptive_avg_pool2d(f,(2,2)),atol=1e-7))
  with tempfile.TemporaryDirectory() as t:
   path=Path(t)/'model.onnx'
   torch.onnx.export(model,x,path,input_names=['images'],output_names=['probabilities'],dynamic_axes={'images':{0:'batch'},'probabilities':{0:'batch'}},opset_version=13)
   session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
   for batch in (1,2,3):
    z=torch.rand(batch,3,224,224)
    with torch.no_grad():a=model(z).numpy()
    b=session.run(None,{'images':z.numpy()})[0];self.assertTrue(np.allclose(a,b,atol=1e-5));self.assertTrue(np.array_equal(a.argmax(1),b.argmax(1)))
if __name__=='__main__':unittest.main()
