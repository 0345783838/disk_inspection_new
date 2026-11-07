"""Canonical split preservation, balanced sampling, authorization and ranking."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training.classifier_dataset import StaleRevision,active_dataset,atomic_json,read
import test_classifier_canonical as canonical_tests
from training.train_classifier_canonical_holdout import validate_rows,sampling_weights,holdout_fitness,comparison_failures
import training.train_classifier_canonical_holdout as holdout


class HoldoutTests(unittest.TestCase):
    def test_frozen_batchnorm_keeps_statistics_and_affine_trainable(self):
        import torch
        model=torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.BatchNorm1d(4),torch.nn.ReLU(),torch.nn.Dropout(.03))
        model.train();buffers={k:v.clone() for k,v in model.state_dict().items() if k.endswith(('running_mean','running_var','num_batches_tracked'))}
        self.assertEqual(holdout.freeze_batchnorm_statistics(model),1)
        self.assertFalse(model[1].training);self.assertTrue(model[3].training)
        optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        optimizer.zero_grad();model(torch.randn(8,4)).sum().backward();optimizer.step()
        self.assertTrue(model[1].weight.requires_grad);self.assertIsNotNone(model[1].weight.grad)
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in buffers.items()))

    def test_full_balanced_evaluation_writes_reports_for_legacy_and_all_cameras(self):
        import test_classifier_rebalance as fixtures
        from training.rebalance_classifier_dataset import plan_split
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'input_snapshot';rows=fixtures.RebalanceTests().rows();policy=plan_split(rows,seeds=1)
            for r in rows:
                r['image']=f"{r['split']}/{r['label']}/{r['id']}.png"
                path=data/r['image'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(r['id'].encode())
                r['sha256']=holdout.digest(path)
            atomic_json(data/'manifest.json',dict(records=rows,split_policy=policy))
            (root/'artifacts').mkdir();(root/'reference').mkdir()
            for path in (root/'artifacts/best.pt',root/'artifacts/best.onnx',root/'reference/baseline.onnx',root/'reference/previous.onnx'):path.write_bytes(path.name.encode())
            domains=list(holdout.BALANCED_WEIGHTS)+[f'{o}_Cam{c}' for o in holdout.ORIGINS for c in (1,2)]
            protocol=dict(data=str(data),split_policy=policy,canonical_revision=10,pool_manifest_sha256='fixture',
                baseline_onnx=str(root/'reference/baseline.onnx'),fitness_weights=holdout.BALANCED_WEIGHTS,
                comparison_domains=domains,train_count=sum(r['split']=='train' for r in rows),val_count=sum(r['split']=='val' for r in rows),
                initialization='ha',previous_imagenet_onnx=str(root/'reference/previous.onnx'),previous_imagenet_sha256=holdout.digest(root/'reference/previous.onnx'))
            atomic_json(root/'protocol.json',protocol);atomic_json(root/'artifacts/completion.json',dict(epochs_completed=200))
            def infer(model,records):return None,[dict(label=r['label']) for r in records]
            with patch.object(holdout,'predict',side_effect=infer),patch.object(holdout,'torch_onnx_parity',return_value={'argmax_mismatches':0}):
                holdout.evaluate(root)
            release=read(root/'release_decision.json')
            self.assertEqual(release['candidate']['val']['old_legacy_train']['errors'],0)
            self.assertEqual(release['candidate']['val']['old_ha_Cam2']['errors'],0)
            self.assertIn('old_legacy_train',(root/'comparison.html').read_text(encoding='utf-8'))
            self.assertIn('Fine-tune HA / best',(root/'comparison.html').read_text(encoding='utf-8'))
            self.assertIn('ImageNet /',(root/'comparison.html').read_text(encoding='utf-8'))
            self.assertEqual(release['previous_imagenet']['val']['old_ha_Cam2']['errors'],0)
            self.assertTrue((root/'error_review.html').is_file())
            self.assertFalse(release['production_deployed'])

    def test_status_updates_only_matching_active_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            project=Path(tmp);(project/'.work').mkdir();root=project/'new_run';root.mkdir()
            pointer=project/'.work/ACTIVE_CLASSIFIER_TRAINING.json'
            atomic_json(pointer,dict(path=str(project/'other_run'),status='complete'))
            with patch.object(holdout,'PROJECT',project):
                holdout.publish(root,'running','training',child=123)
                self.assertEqual(read(pointer)['status'],'complete')
                atomic_json(pointer,dict(path=str(root),status='starting'))
                atomic_json(root/'release_decision.json',dict(recommended_for_review=False))
                holdout.publish(root,'complete','evaluated')
                self.assertEqual(read(pointer)['status'],'complete')
                self.assertIsNone(read(pointer)['campaign_pid'])
                self.assertFalse(read(pointer)['recommended_for_review'])

    def test_balanced_policy_keeps_cameras_and_legacy_in_training_and_fitness(self):
        import test_classifier_rebalance as fixtures
        from training.rebalance_classifier_dataset import plan_split
        rows=fixtures.RebalanceTests().rows();plan_split(rows,seeds=1)
        validate_rows(rows,holdout.BALANCED_POLICY)
        train=[r for r in rows if r['split']=='train'];weights=sampling_weights(train,holdout.BALANCED_POLICY)
        self.assertAlmostEqual(sum(weights),1)
        for origin,total in holdout.BALANCED_WEIGHTS.items():
            self.assertAlmostEqual(sum(w for r,w in zip(train,weights) if r['origin']==origin),total)
            if origin in holdout.ORIGINS:
                for camera in (1,2):
                    self.assertAlmostEqual(sum(w for r,w in zip(train,weights) if r['origin']==origin and r['camera']==camera),total/2)
        self.assertAlmostEqual(sum(w for r,w in zip(train,weights) if r['origin']=='old_legacy_train' and r['label']=='ng'),.14)
        metrics={o:dict(macro_f1=1) for o in holdout.BALANCED_WEIGHTS}
        metrics['old_legacy_train']['macro_f1']=0
        self.assertAlmostEqual(holdout_fitness(metrics,holdout.BALANCED_WEIGHTS),.8)
        broken=deepcopy(rows);broken[0]['target_group']='wrong'
        with self.assertRaisesRegex(ValueError,'provenance'):validate_rows(broken,holdout.BALANCED_POLICY)

    def test_scoped_evaluation_includes_each_camera_and_legacy(self):
        rows=[dict(origin=o,camera=c,label=label) for o,c in [('old_ha',1),('old_ha',2),('new_factory',1),('new_factory',2),('old_legacy_train',None)] for label in ('ng','no_disk','ok')]
        predictions=[dict(label=r['label']) for r in rows]
        # One HA Cam2 error must remain visible rather than cancel against Cam1.
        predictions[3]={'label':'ok'}
        metrics=holdout.scoped_metrics(rows,predictions)
        self.assertEqual(metrics['old_ha_Cam1']['errors'],0)
        self.assertEqual(metrics['old_ha_Cam2']['ng_not_detected'],1)
        self.assertEqual(metrics['old_legacy_train']['count'],3)

    def test_evaluation_creates_nested_output_and_reuses_hash_bound_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=root/'model.onnx';model.write_bytes(b'model')
            rows=self.rows();outputs=[dict(label=r['label']) for r in rows]
            output=root/'not_created/evaluations/baseline.json'
            with patch.object(holdout,'predict',return_value=(None,outputs)) as inference:
                result=holdout.evaluate_model(model,rows,output)
                self.assertEqual(result['metrics']['val']['all']['accuracy'],1)
                self.assertTrue(output.is_file())
                self.assertEqual(holdout.evaluate_model(model,rows,output),result)
                self.assertEqual(inference.call_count,1)
                model.write_bytes(b'changed model')
                with self.assertRaisesRegex(ValueError,'mismatch'):holdout.evaluate_model(model,rows,output)

    def rows(self):
        rows=[]
        for split in ('train','val'):
            for origin in ('old_ha','new_factory'):
                for label in ('ng','no_disk','ok'):
                    for index in range(1 if label=='ng' else 3):
                        i=f'{split}_{origin}_{label}_{index}'
                        rows.append(dict(id=i,split=split,origin=origin,label=label,sha256=i,decoded_sha256=i,
                                         frame=split+origin,capture_group=split+origin,split_group=split+origin))
        return rows

    def test_split_leaks_and_legacy_rejected(self):
        rows=self.rows();validate_rows(rows)
        for key in ('sha256','decoded_sha256','frame','capture_group','split_group'):
            wrong=deepcopy(rows);wrong[-1][key]=wrong[0][key]
            with self.assertRaisesRegex(ValueError,'Split leakage'):validate_rows(wrong)
        wrong=deepcopy(rows);wrong[0]['origin']='legacy_val'
        with self.assertRaises(ValueError):validate_rows(wrong)

    def test_sampler_uses_training_only_and_exact_domain_class_budgets(self):
        rows=[r for r in self.rows() if r['split']=='train'];weights=sampling_weights(rows)
        self.assertAlmostEqual(sum(weights),1)
        for origin in ('old_ha','new_factory'):
            for label,budget in [('ng',.3),('no_disk',.3),('ok',.4)]:
                self.assertAlmostEqual(sum(w for r,w in zip(rows,weights) if r['origin']==origin and r['label']==label),.5*budget)

    def test_checkpoint_fitness_balances_origins_and_gate_rejects_safety_regression(self):
        old={o:dict(macro_f1=.9,errors=4,false_ng=2,ng_not_detected=1,no_disk_as_ok=1) for o in ('old_ha','new_factory')}
        new=deepcopy(old);new['new_factory']['macro_f1']=1
        self.assertAlmostEqual(holdout_fitness(new),.95)
        self.assertEqual(comparison_failures(new,old),[])
        new['new_factory']['ng_not_detected']=2
        self.assertIn('new_factory/ng_not_detected',comparison_failures(new,old))

    def test_explicit_authorization_is_revision_bound_and_label_edit_revokes_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);store=canonical_tests.CanonicalDatasetTests().fixture(root)
            receipt=dict(canonical_revision=0,incomplete_review_acknowledged=True,user_statement='train bản này rồi đánh giá')
            with self.assertRaises(StaleRevision):store.authorize_training(receipt,1)
            with self.assertRaises(ValueError):store.authorize_training(dict(canonical_revision=0),0)
            store.authorize_training(receipt,0)
            registry=root/'registry.json';atomic_json(registry,dict(path=str(root)))
            with patch('training.classifier_dataset.REGISTRY',registry):
                self.assertEqual(active_dataset(),root.resolve())
                self.assertEqual(read(root/'manifest.json')['review_status'],'partial_review_training_authorized')
                store.save([dict(id='a',label='ng')],0)
                with self.assertRaises(ValueError):active_dataset()
