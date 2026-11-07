from copy import deepcopy
from pathlib import Path
import sys,unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
import train_classifier_size_s as size
class SizeSafetyTests(unittest.TestCase):
 def metrics(self):
  return {d:dict(errors=4,ng_not_detected=0,false_ng=2,no_disk_as_ok=1,macro_f1=.9) for d in list(size.base.BALANCED_WEIGHTS)+['old_ha_Cam1','old_ha_Cam2','new_factory_Cam1','new_factory_Cam2']}
 def test_zero_ng_gate_is_stricter_than_baseline_retention(self):
  baseline=self.metrics();baseline['old_legacy_train']['ng_not_detected']=1
  candidate=deepcopy(baseline)
  for v in candidate.values():v['macro_f1']=.95
  self.assertIn('all/zero_ng_misses_required',size.acceptance(candidate,baseline,baseline,size.base.BALANCED_WEIGHTS,list(baseline)))
 def test_false_ng_regression_in_one_camera_cannot_hide_in_source_total(self):
  baseline=self.metrics();nsafety=deepcopy(baseline);nsafety['new_factory_Cam1']['false_ng']=1
  candidate=deepcopy(baseline)
  for v in candidate.values():v['macro_f1']=.95
  self.assertIn('new_factory_Cam1/false_ng_above_n_safety',size.acceptance(candidate,baseline,nsafety,size.base.BALANCED_WEIGHTS,list(baseline)))
 def test_safe_improvement_passes_without_changing_old_gates(self):
  old=self.metrics();new=deepcopy(old)
  for v in new.values():v.update(macro_f1=.95,errors=3,false_ng=1)
  self.assertEqual(size.acceptance(new,old,old,size.base.BALANCED_WEIGHTS,list(old)),[])
 def test_ng_miss_dominates_false_alarm_reduction(self):
  baseline=self.metrics();zero=deepcopy(baseline);miss=deepcopy(baseline)
  zero['new_factory']['false_ng']=100
  miss['new_factory'].update(ng_not_detected=1,false_ng=0)
  self.assertGreater(size.safety_fitness(zero,baseline),size.safety_fitness(miss,baseline))
 def test_loss_backprop_finite_for_three_class_batch(self):
  logits=torch.tensor([[0.,1.,2.],[2.,0.,1.],[1.,2.,0.]],requires_grad=True)
  loss,_=size.NGWeightedLoss()(logits,{'cls':torch.tensor([0,1,2])});loss.backward()
  self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(logits.grad).all());self.assertLess(logits.grad[0,0],0)
if __name__=='__main__':unittest.main()
