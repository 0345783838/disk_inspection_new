"""Diagnostic predictions never become canonical labels or independent truth."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from training.classifier_dataset import CanonicalDataset, atomic_json, read
from training.classifier_diagnostic_review import live_diagnostics
from training.audit_classifier_dataset import audit


class DiagnosticReviewTests(unittest.TestCase):
    def fixture(self, root):
        model = root / 'model.onnx'; model.write_bytes(b'fixed model')
        rows = []
        for i, origin in enumerate(('old_ha', 'new_factory', 'legacy_val')):
            role = 'evaluation_only' if origin == 'legacy_val' else 'training_pool'
            base = 'evaluation/legacy_val' if role == 'evaluation_only' else 'train_pool'
            path = root / base / 'ok' / f'{i}.png'; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bytes([i]))
            rows.append(dict(id=str(i), origin=origin, role=role, label='ok', label_source='fixture',
                             image=path.relative_to(root).as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        m = dict(is_canonical=True, canonical_revision=0, review_status='in_progress',
                 training_authorized=False, records=rows, excluded=[], excluded_records=[])
        CanonicalDataset.refresh(m); atomic_json(root / 'manifest.json', m)
        store = CanonicalDataset(root); store.write_views(m)
        predictions = root / 'predictions.json'
        atomic_json(predictions, dict(origins=['old_ha', 'new_factory'], canonical_revision=0,
                    model_path=str(model), model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                    generated_at='fixture', predictions=[dict(id=r['id'], sha256=r['sha256'], label='ng',
                    confidence=.8, probabilities=dict(ng=.8, no_disk=.1, ok=.1)) for r in rows[:2]]))
        registry = root / 'evaluation.json'
        atomic_json(registry, dict(dataset_path=str(root), origins=['old_ha', 'new_factory'], predictions_path=str(predictions)))
        return store, registry

    def test_only_ha_and_new_are_scored_against_live_labels_after_edits_and_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); store, registry = self.fixture(root)
            frozen = (root / 'predictions.json').read_bytes()
            before = live_diagnostics(store.manifest(), root, registry)
            self.assertEqual(set(before['by_id']), {'0', '1'})
            self.assertEqual(before['counts_by_origin']['old_ha']['errors'], 1)
            store.save([dict(id='0', label='ng')], 0)
            after = live_diagnostics(store.manifest(), root, registry)
            self.assertEqual(after['canonical_revision'], 1)
            self.assertEqual(after['prediction_source_revision'], 0)
            self.assertEqual(after['counts_by_origin']['old_ha']['errors'], 0)
            store.save([dict(id='1', label='uncertain')], 1)
            after = live_diagnostics(store.manifest(), root, registry)
            self.assertEqual(after['count'], 1)
            self.assertNotIn('1', after['by_id'])
            self.assertEqual((root / 'predictions.json').read_bytes(), frozen)
            self.assertFalse(store.manifest()['training_authorized'])
            self.assertTrue(audit(root)['passed'])

    def test_changed_crop_hash_and_model_cannot_silently_reuse_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); store, registry = self.fixture(root)
            m = store.manifest(); m['records'][0]['sha256'] = 'different'
            result = live_diagnostics(m, root, registry)
            self.assertEqual(result['missing_prediction_ids'], ['0'])
            self.assertNotIn('0', result['by_id'])
            (root / 'model.onnx').write_bytes(b'replaced model')
            self.assertFalse(live_diagnostics(store.manifest(), root, registry)['available'])

    def test_audit_detects_actual_corruption_stale_csv_and_hash_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); store, _ = self.fixture(root)
            self.assertTrue(audit(root)['passed'])
            m = store.manifest()
            (root / m['records'][0]['image']).write_bytes(b'corrupt')
            (root / 'labels.csv').write_text('wrong', encoding='utf-8')
            m['records'][2]['sha256'] = m['records'][1]['sha256']
            atomic_json(root / 'manifest.json', m)
            result = audit(root)
            self.assertFalse(result['passed'])
            self.assertTrue(any('Hash mismatch' in e for e in result['errors']))
            self.assertIn('CSV differs from canonical manifest', result['errors'])
            self.assertIn('Exact train/validation hash overlap', result['errors'])

    def test_registry_for_other_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); store, registry = self.fixture(root)
            config = read(registry); config['dataset_path'] = str(root / 'other')
            atomic_json(registry, config)
            with self.assertRaises(ValueError):
                live_diagnostics(store.manifest(), root, registry)
