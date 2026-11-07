"""Supplementary selection must separate human truth from model proposals."""
import importlib.util
from pathlib import Path
import sys
import unittest

AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ('torch', 'ultralytics'))
if AVAILABLE:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
    from train_classifier_supplementary import population_metrics, score


@unittest.skipUnless(AVAILABLE, 'Optional training dependencies unavailable')
class SupplementarySelectionTests(unittest.TestCase):
    def metrics(self):
        records = []
        predictions = []
        for origin in ('new_factory', 'old_ha', 'legacy_val'):
            for label in ('ng', 'no_disk', 'ok'):
                records.append(dict(origin=origin, label=label, human_reviewed=origin == 'new_factory'))
                predictions.append(dict(label=label))
        records.append(dict(origin='new_factory', label='ok', human_reviewed=False))
        predictions.append(dict(label='ng'))
        return population_metrics(records, predictions)

    def test_proposal_errors_do_not_become_errors_on_human_labels(self):
        m = self.metrics()
        self.assertEqual(m['new_all']['false_ng'], 1)
        self.assertEqual(m['new_proposals']['false_ng'], 1)
        self.assertEqual(m['new_human']['false_ng'], 0)
        self.assertEqual(m['new_human']['count'], 3)

    def test_old_real_ng_miss_rejects_even_when_new_false_ng_improves(self):
        gates = dict(old_ha_ng_max=0, old_ha_empty_max=0, old_ha_errors_max=4,
                     legacy_ng_max=0, legacy_empty_max=0, legacy_accuracy_min=.98)
        m = self.metrics()
        good, failures = score(m, gates)
        self.assertFalse(failures)
        m['old_ha']['ng_not_detected'] = 1
        bad, failures = score(m, gates)
        self.assertIn('old_ha_ng_missed', failures)
        self.assertGreater(good, bad)


if __name__ == '__main__':
    unittest.main()
