"""Sampling, safety ranking, KD masking and immutable grouped splits."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

AVAILABLE = all(importlib.util.find_spec(x) is not None for x in ('torch','ultralytics'))
if AVAILABLE:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
    import torch
    from classifier_robust_core import RobustLoss, checkpoint_fitness, metrics_for, retention_failures, sampling_weights
    from train_classifier_robust import BUDGET, capture_folds, build_dataset, checked, reap_child, restore_stopper
    from train_classifier_supplementary import dump, read, digest


@unittest.skipUnless(AVAILABLE,'Optional training dependencies unavailable')
class RobustClassifierTests(unittest.TestCase):
    def test_status_write_failure_reaps_trainer_before_another_job_can_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); child=MagicMock(); child.poll.return_value=None
            with patch('train_classifier_robust.subprocess.Popen',return_value=child), patch(
                    'train_classifier_robust.publish',side_effect=PermissionError('status locked')):
                with self.assertRaises(PermissionError): checked(['fake-trainer'],root,'fixture',root/'job.log')
            child.terminate.assert_called_once()
            child.wait.assert_called_once_with(timeout=30)

    def test_reap_escalates_when_child_does_not_terminate(self):
        import subprocess
        child=MagicMock(); child.poll.return_value=None
        child.wait.side_effect=[subprocess.TimeoutExpired('fixture',30),0]
        reap_child(child)
        child.kill.assert_called_once()
        self.assertEqual(child.wait.call_count,2)

    def test_resume_preserves_existing_early_stopping_clock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            dump(root/'early_stopping_state.json',dict(epoch=140,best_epoch=90,best_fitness=.85))
            stopper=SimpleNamespace(best_epoch=0,best_fitness=0,possible_stop=False,patience=100)
            trainer=SimpleNamespace(save_dir=root,start_epoch=140,stopper=stopper,epochs=300)
            result=restore_stopper(trainer)
            self.assertEqual(result['best_epoch'],90)
            self.assertEqual(result['resume_start_epoch'],141)
            self.assertEqual(stopper.best_fitness,.85)
            self.assertEqual(result['patience'],100)

    def test_joint_budgets_survive_imbalance_corrections_and_uncertainty(self):
        records=[]
        for domain in BUDGET['domain_budget']:
            for label,n in [('ng',8),('no_disk',4),('ok',2)]:
                records += [dict(origin=domain,label=label,label_history=[{}] if i==0 else [],
                                 unresolved_similarity_candidate=i==1) for i in range(n)]
        weights,budget=sampling_weights(records,dict(BUDGET,robust=True))
        self.assertAlmostEqual(sum(weights),1)
        self.assertAlmostEqual(budget['old_legacy_train']['ok'],.14)
        self.assertAlmostEqual(budget['new_factory']['ng'],.105)
        self.assertEqual(records[0]['label'],'ng')

    def test_teacher_does_not_override_new_examples_or_disagreeing_old_labels(self):
        logits=torch.tensor([[1.,2.,3.],[1.,3.,2.]],requires_grad=True)
        target=torch.tensor([0,0])
        loss=RobustLoss(dict(distill=.5))
        plain=loss(logits,dict(cls=target))[0]
        masked=loss(logits,dict(cls=target,population=torch.tensor([0,1]),
                      teacher_probabilities=torch.tensor([[.9,.05,.05],[.05,.9,.05]])))[0]
        self.assertAlmostEqual(plain.item(),masked.item())
        masked.backward(); self.assertTrue(torch.isfinite(logits.grad).all())

    def test_robust_loss_is_finite_and_does_not_relabel_targets(self):
        logits=torch.tensor([[-100.,100.,0.],[50.,0.,-50.]],requires_grad=True)
        labels=torch.tensor([0,2]); before=labels.clone()
        loss,_=RobustLoss(dict(robust=True,distill=0))(logits,dict(cls=labels))
        loss.backward()
        self.assertTrue(torch.isfinite(loss)); self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.equal(labels,before))

    def test_false_ng_improvement_does_not_pass_ng_miss_gate(self):
        records=[dict(origin='new_factory',label=c) for c in ['ng','no_disk','ok']]
        baseline=metrics_for(records,[dict(label=c) for c in ['ng','no_disk','ng']])
        unsafe=metrics_for(records,[dict(label=c) for c in ['ok','no_disk','ok']])
        self.assertIn('new_factory/ng_not_detected',retention_failures(unsafe,baseline))
        self.assertGreater(checkpoint_fitness(baseline,baseline),checkpoint_fitness(unsafe,baseline))

    def test_group_split_uses_frozen_inputs_and_never_trains_legacy_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); frozen=root/'input_snapshot'; frozen.mkdir()
            records=[]
            for i,(origin,frame) in enumerate([('new_factory','Cam1_Image_20261018100000000'),
                         ('new_factory','Cam2_Image_20261018101000000'),
                         ('new_factory','Cam1_Image_20261018102000000'),
                         ('legacy_val',None),('old_ha',None),('old_legacy_train',None)]):
                file=f'{i}.png'; (frozen/file).write_bytes(bytes([i]))
                records.append(dict(id=str(i),origin=origin,frame=frame,image=file,label='ok',sha256=digest(frozen/file)))
            folds=capture_folds(records); dump(root/'frozen_records.json',records)
            baseline=dict(predictions=[dict(r,prediction=dict(label=r['label'])) for r in records])
            dump(root/'evaluations/baseline/result.json',baseline)
            recipe=root/'recipe'; recipe.mkdir()
            dump(recipe/'protocol.json',dict(campaign_root=str(root),frozen_pool=str(frozen),**folds))
            data=build_dataset(recipe,'group_cv_0',0); rows=read(data/'manifest.json')['records']
            self.assertTrue(all(r['split']=='val' for r in rows if r['origin']=='legacy_val'))
            self.assertTrue(all(r['split']=='train' for r in rows if r['origin'] in ('old_ha','old_legacy_train')))
            self.assertEqual(sum(r['origin']=='new_factory' and r['split']=='val' for r in rows),1)
            self.assertTrue(all((data/r['image']).read_bytes()==(frozen/f'{r["id"]}.png').read_bytes() for r in rows))
