"""Fresh-pretrained validation must never use fitted crops for best selection."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

AVAILABLE = all(importlib.util.find_spec(n) is not None for n in ('torch', 'ultralytics'))
if AVAILABLE:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
    from train_classifier_yolo_pretrained import build_dataset, digest, validation_fitness


@unittest.skipUnless(AVAILABLE, 'Optional training dependencies unavailable')
class FreshPretrainedTests(unittest.TestCase):
    def fixture(self, root):
        pool = root / 'pool'; pool.mkdir()
        rows = []
        for index, (origin, frame) in enumerate([('old_ha', 'ha'), ('old_legacy_train', 'legacy_train'),
                                                ('legacy_val', 'legacy_val'), ('new_factory', 'a'), ('new_factory', 'b')]):
            image = f'{index}.png'; (pool / image).write_bytes(bytes([index]))
            rows.append(dict(id=str(index), image=image, sha256=digest(pool / image), origin=origin,
                             frame=frame, label='ok', human_reviewed=origin == 'new_factory'))
        (pool / 'manifest.json').write_text(json.dumps(dict(records=rows)), encoding='utf-8')
        run = root / 'run'; run.mkdir()
        (run / 'protocol.json').write_text(json.dumps(dict(pool=str(pool), frame_groups={'a': 0, 'b': 1},
                                                          group_folds={'0': 0, '1': 1}, gates={})), encoding='utf-8')
        return run

    def test_cv_separates_new_capture_and_keeps_old_ha_out_of_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.fixture(Path(temp))
            result = build_dataset(root, 'fold', 0)
            rows = json.loads((result / 'manifest.json').read_text())['records']
            self.assertEqual({r['id'] for r in rows if r['split'] == 'val'}, {'2', '3'})
            self.assertTrue(all(r['split'] == 'train' for r in rows if r['origin'] == 'old_ha'))

    def test_final_best_uses_legacy_validation_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.fixture(Path(temp))
            result = build_dataset(root, 'final', None)
            rows = json.loads((result / 'manifest.json').read_text())['records']
            self.assertEqual({r['origin'] for r in rows if r['split'] == 'val'}, {'legacy_val'})
            self.assertEqual(sum(r['split'] == 'train' for r in rows), 4)

    def test_validation_safety_regression_loses_eligibility(self):
        m = dict(legacy_val=dict(ng_not_detected=5, no_disk_as_ok=4, accuracy=.993,
                                 false_ng=10, errors=20, macro_f1=.99))
        gates = dict(legacy_ng_max=5, legacy_empty_max=4, legacy_accuracy_min=.987)
        good, eligible = validation_fitness(m, gates)
        self.assertTrue(eligible)
        m['legacy_val']['no_disk_as_ok'] = 5
        bad, eligible = validation_fitness(m, gates)
        self.assertFalse(eligible)
        self.assertGreater(good, bad)
