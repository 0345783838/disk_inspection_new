"""Safety priority, geometry-preserving AI augmentation and guarded KD."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
import numpy as np
import torch
from torch import nn
import train_classifier_safety_campaign as campaign


class SafetyCampaignTests(unittest.TestCase):
    def metrics(self):
        return {o:dict(errors=0,false_ng=0,ng_not_detected=0,no_disk_as_ok=0) for o in campaign.DOMAIN_BUDGET}

    def test_ng_misses_dominate_false_alarms(self):
        baseline=self.metrics();safe=self.metrics();safe['new_factory'].update(errors=100,false_ng=100)
        missed=self.metrics();missed['new_factory'].update(errors=1,ng_not_detected=1)
        self.assertGreater(campaign.safety_fitness(safe,baseline),campaign.safety_fitness(missed,baseline))

    def test_false_ng_breaks_equal_ng_priority(self):
        baseline=self.metrics();a=self.metrics();b=self.metrics();a['old_ha'].update(errors=2,false_ng=2);b['old_ha'].update(errors=3,false_ng=3)
        self.assertGreater(campaign.safety_fitness(a,baseline),campaign.safety_fitness(b,baseline))

    def test_ai_field_has_no_spatial_replacement(self):
        image=np.zeros((60,40,3),dtype=np.uint8);image[:,19:21]=100
        style=np.linspace(.8,1.2,64,dtype=np.float32).reshape(8,8)
        result=campaign.apply_illumination(image,style)
        self.assertTrue(np.array_equal(result==0,image==0))
        self.assertTrue(np.all(result[:,19:21]>=80));self.assertTrue(np.all(result[:,19:21]<=120))

    def test_generated_geometry_cannot_enter_style(self):
        source=np.full((80,60,3),100,dtype=np.uint8);generated=source.copy();generated[:,20:25]=0
        profile=campaign.illumination_profile(source,generated)
        self.assertEqual(profile.shape,(8,8));self.assertGreaterEqual(profile.min(),.8);self.assertLessEqual(profile.max(),1.2)
        output=campaign.apply_illumination(source,profile)
        self.assertTrue(np.all(output>0))

    def test_kd_never_teaches_wrong_new_labels(self):
        criterion=campaign.SafetyLoss(dict(kd=4.))
        logits=torch.tensor([[1.,2.,3.]],requires_grad=True)
        batch=dict(cls=torch.tensor([2]),population=torch.tensor([0]))
        plain=criterion(logits,batch)[0]
        batch['teacher_probabilities']=torch.tensor([[.999,.0005,.0005]])
        self.assertTrue(torch.equal(plain,criterion(logits,batch)[0]))

    def test_kd_keeps_teacher_detached_and_student_gradients(self):
        logits=torch.tensor([[.1,1.,.1]],requires_grad=True);teacher=torch.tensor([[.99,.005,.005]],requires_grad=True)
        loss,_=campaign.SafetyLoss(dict(kd=4.))(logits,dict(cls=torch.tensor([0]),population=torch.tensor([1]),teacher_probabilities=teacher))
        loss.backward();self.assertIsNotNone(logits.grad);self.assertIsNone(teacher.grad)

    def test_linear_scope_only_trains_final_linear(self):
        model=nn.Module();model.model=nn.Sequential(nn.Linear(2,2),nn.ModuleDict(dict(linear=nn.Linear(2,3))))
        campaign.set_trainable(model,'linear')
        self.assertFalse(model.model[0].weight.requires_grad);self.assertTrue(model.model[1]['linear'].weight.requires_grad)

    def test_sampler_preserves_camera_domain_and_class_budgets(self):
        rows=[]
        for o in campaign.DOMAIN_BUDGET:
            for camera in ([None] if o=='old_legacy_train' else [1,2]):
                for label in campaign.base.LABELS:
                    for i in range(2):rows.append(dict(id=f'{o}_{camera}_{label}_{i}',origin=o,camera=camera,label=label,split='train'))
        weights=campaign.train_weights(rows,{rows[0]['id']:4})
        self.assertAlmostEqual(sum(weights),1)
        for o,budget in campaign.DOMAIN_BUDGET.items():
            self.assertAlmostEqual(sum(w for r,w in zip(rows,weights) if r['origin']==o),budget)
            for i,label in enumerate(campaign.base.LABELS):
                self.assertAlmostEqual(sum(w for r,w in zip(rows,weights) if r['origin']==o and r['label']==label),budget*campaign.CLASS_BUDGET[o][i])
        rows[0]['split']='val'
        with self.assertRaisesRegex(ValueError,'Validation'):campaign.train_weights(rows,{})

    def test_status_failure_reaps_child_before_next_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);child=MagicMock();child.poll.return_value=None
            with patch.object(campaign.subprocess,'Popen',return_value=child),patch.object(campaign,'publish',side_effect=PermissionError('locked')):
                with self.assertRaises(PermissionError):campaign.checked(root,'fixture',['fake'],root/'log')
            child.terminate.assert_called_once();child.wait.assert_called_once_with(timeout=25)


if __name__=='__main__':unittest.main()
