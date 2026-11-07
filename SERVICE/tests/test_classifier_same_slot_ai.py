"""Synthetic exposure remains bounded without altering real source/camera/class budgets."""
import unittest
import tempfile,json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from collections import defaultdict
from training.train_classifier_same_slot_ai import supplementary_weights, report_metrics
from training.train_classifier_canonical_holdout import BALANCED_WEIGHTS, LABELS, sampling_weights, BALANCED_POLICY

class SupplementarySamplingTests(unittest.TestCase):
    def test_completed_phase_files_skip_training_and_publish_final_status(self):
        import training.train_classifier_same_slot_ai as module
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ['artifacts/completion.json','release_decision.json','white_replay.json','completion_verification.json','status.json']:
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(dict(passed=True,status='running')))
            with patch.object(module.subprocess,'run',return_value=SimpleNamespace(stdout='[]')),patch.object(module,'check',return_value={}),patch.object(module.base,'checked') as phases,patch.object(module.base,'publish') as publish:
                module.campaign(root,resume=True)
                phases.assert_not_called()
                publish.assert_called_once_with(root,'complete','AI supplementary training/evaluation/parity/White complete; no deployment')

    def test_report_total_handles_standard_all_without_derived_error_fields(self):
        metrics={origin:dict(errors=3,ng_not_detected=1,false_ng=2,no_disk_as_ok=0) for origin in BALANCED_WEIGHTS}
        metrics['all']=dict(count=1605,accuracy=.99)
        self.assertEqual(report_metrics(metrics,'all'),dict(errors=9,ng_not_detected=3,false_ng=6,no_disk_as_ok=0))
        self.assertIs(report_metrics(metrics,'old_ha'),metrics['old_ha'])

    def rows(self):
        rows=[]
        for origin in BALANCED_WEIGHTS:
            for camera in ((1,2) if origin!='old_legacy_train' else (None,)):
                for label in LABELS:
                    for _ in range(3):rows.append(dict(origin=origin,camera=camera,label=label,split='train'))
        for origin,camera in [('old_ha',2),('new_factory',1),('new_factory',2)]:
            for _ in range(2):rows.append(dict(origin=origin,camera=camera,label='ng',split='train',is_synthetic=True))
        return rows

    def test_preserves_strata_and_limits_ai_to_2point7_percent(self):
        rows=self.rows();before=sampling_weights(rows,BALANCED_POLICY);after=supplementary_weights(rows)
        groups=lambda values:{key:sum(w for r,w in zip(rows,values) if (r['origin'],r.get('camera'),r['label'])==key) for key in {(r['origin'],r.get('camera'),r['label']) for r in rows}}
        for key,mass in groups(before).items():self.assertAlmostEqual(mass,groups(after)[key],places=12)
        self.assertAlmostEqual(sum(after),1.,places=12)
        self.assertAlmostEqual(sum(w for r,w in zip(rows,after) if r.get('is_synthetic')),.027,places=12)
        self.assertTrue(all(w>0 for w in after))

    def test_rejects_validation_or_synthetic_non_ng(self):
        rows=self.rows();rows[0]['split']='val'
        with self.assertRaises(ValueError):supplementary_weights(rows)
        rows=self.rows();rows[-1]['label']='ok'
        with self.assertRaises(ValueError):supplementary_weights(rows)

if __name__=='__main__':unittest.main()
