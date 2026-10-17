"""Whole-frame CV, honest full-fit overlap and hard-sampling contracts on fixtures."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'training'))
try:
    from train_classifier_fullha import fold_lookup, safety_rank, build_dataset, RECIPES, FullHATrainer
    from classifier_metrics import classification_metrics
except ImportError:
    fold_lookup = None


@unittest.skipIf(fold_lookup is None, 'Optional training dependencies unavailable')
class FullHAContractTests(unittest.TestCase):
    def test_three_camera_block_folds_cover_every_whole_frame(self):
        rows = [{'camera': c, 'frame': f'Cam{c}_{index:02}', 'slot': slot}
                for c in (1,2) for index in range(10) for slot in range(100)]
        lookup = fold_lookup(rows)
        self.assertEqual([sum(f == i for f in lookup.values()) for i in range(3)], [6,6,8])
        for fold in range(3):
            train = {r['frame'] for r in rows if lookup[r['frame']] != fold}
            val = {r['frame'] for r in rows if lookup[r['frame']] == fold}
            self.assertFalse(train & val)
        self.assertEqual(sum(sum(lookup[r['frame']] == f for r in rows) for f in range(3)), 2000)

    def test_ng_priority_and_reject_all_guard_survive_rounding(self):
        ha_gt = [0]*100+[1]*100+[2]*200
        old_gt = [0]*100+[1]*100+[2]*300
        old_bad = old_gt.copy(); old_bad[:5] = [2]*5
        old = classification_metrics(old_gt, old_bad)
        ha_bad = ha_gt.copy(); ha_bad[0] = 2
        ha_fp = ha_gt.copy(); ha_fp[-3:] = [0]*3
        missed, _ = safety_rank(classification_metrics(ha_gt, ha_bad), classification_metrics(old_gt, old_gt), 5)
        no_miss, eligible = safety_rank(classification_metrics(ha_gt, ha_fp), old, 5)
        self.assertTrue(eligible)
        self.assertGreater(round(no_miss, 5), round(missed, 5))
        reject_all, eligible = safety_rank(classification_metrics(ha_gt, [0]*400), old, 5)
        self.assertFalse(eligible)
        self.assertGreater(no_miss, reject_all)

    def protocol_fixture(self, root):
        records, sources = [], {}
        for camera in (1,2):
            for index in range(10):
                identifier = f'Cam{camera}_{index:02}'
                records.append({'id': identifier, 'frame': identifier, 'camera': camera,
                                'label': ['ng','no_disk','ok'][index%3], 'origin': 'ha', 'split': 'test'})
        for origin in ('legacy_train', 'legacy_val'):
            for label in ('ng','no_disk','ok'):
                records.append({'id': origin+'_'+label, 'label': label, 'origin': origin,
                                'split': 'train' if origin == 'legacy_train' else 'val'})
        folder = root/'sources'; folder.mkdir()
        for record in records:
            path = folder/(record['id']+'.png'); path.write_bytes(record['id'].encode())
            record['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            sources[record['id']] = str(path)
        lookup = fold_lookup([r for r in records if r['origin'] == 'ha'])
        (root/'protocol.json').write_text(json.dumps({'source_manifest': {'records': records},
            'source_paths': sources, 'fold_by_frame': lookup, 'hard_ids': [records[0]['id']],
            'cv_initialization': 'fixture original', 'final_protocol': 'all HA / in-sample',
            'cross_validation_limitation': 'same session'}))

    def test_final_includes_prior_eval_ha_but_not_legacy_val_in_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); self.protocol_fixture(root)
            cv = build_dataset(root, 'hard_focus', fold=0)
            m = json.loads((cv/'manifest.json').read_text())
            train = {r['sha256'] for r in m['records'] if r['split'] == 'train'}
            val = {r['sha256'] for r in m['records'] if r['split'] == 'val'}
            self.assertFalse(train & val)
            self.assertTrue(m['historical_test_is_now_training'])
            self.assertTrue(m['old_ha_test_protocol_abandoned'])
            final = build_dataset(root, 'hard_focus', full=True)
            m = json.loads((final/'manifest.json').read_text())
            self.assertEqual(sum(r['origin'] == 'ha' and r['split'] == 'train' for r in m['records']), 20)
            self.assertTrue(all(r['split'] == 'val' for r in m['records'] if r['origin'] == 'legacy_val'))
            self.assertEqual(sum(r['origin'] == 'ha_fit' for r in m['records']), 20)
            self.assertTrue(m['historical_test_is_now_training'])
            self.assertEqual(m['metric_population'], 'ha_fit_in_sample')
            self.assertEqual(m['sampling_recipe']['hard_boost'], 3)

    def test_hard_sampling_boost_respects_domain_budget(self):
        import torch
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            specs = [('ha', 'hard', 0), ('ha','normal',0), ('ha','empty',1), ('ha','ok',2),
                     ('legacy_train','old_ng',0), ('legacy_train','old_empty',1), ('legacy_train','old_ok',2)]
            records = [{'id': name, 'image': f'train/{label}/{name}.png', 'origin': group, 'split': 'train'}
                       for group,name,label in specs]
            (root/'manifest.json').write_text(json.dumps({'records': records,
                'sampling_recipe': RECIPES['hard_focus'], 'hard_ids': ['hard','heldout_not_in_training']}))
            class FixtureDataset:
                base = SimpleNamespace(classes=['ng','no_disk','ok'])
                samples = [[str(root/r['image']), spec[2]] for r,spec in zip(records,specs)]
                def __len__(self):
                    return len(self.samples)
                def __getitem__(self, index):
                    return {'img': torch.zeros((3,2,2)), 'cls': self.samples[index][1]}
            trainer = object.__new__(FullHATrainer)
            trainer.args = SimpleNamespace(data=str(root),seed=1005,workers=0)
            trainer.build_dataset = lambda *args: FixtureDataset()
            loader = trainer.get_dataloader(str(root/'train'), batch_size=4, rank=-1, mode='train')
            weights = loader.sampler.weights.numpy()
            self.assertAlmostEqual(weights[0]/weights[1], 3)
            self.assertAlmostEqual(sum(weights[:4]), .95)
            self.assertAlmostEqual(sum(weights[4:]), .05)
            self.assertAlmostEqual(weights[4], .05*.60)

    def test_single_network_interpolation_validates_topology_and_integer_buffers(self):
        import torch
        from blend_classifier_fullha import interpolate
        warm = {'weight': torch.tensor([0.,4.]), 'count': torch.tensor(2)}
        cold = {'weight': torch.tensor([4.,0.]), 'count': torch.tensor(9)}
        mixed = interpolate(warm, cold, .25)
        self.assertTrue(torch.equal(mixed['weight'], torch.tensor([1.,3.])))
        self.assertEqual(mixed['count'].item(), 2)
        self.assertTrue(torch.equal(warm['weight'], torch.tensor([0.,4.])))
        with self.assertRaises(ValueError):
            interpolate(warm, cold, 1)
        with self.assertRaises(ValueError):
            interpolate(warm, {'different': torch.tensor(1)}, .5)


if __name__ == '__main__':
    unittest.main()
