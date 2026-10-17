"""Versioning and split isolation on fixtures only."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'training'))
from apply_classifier_label_corrections import build
from classifier_metrics import classification_metrics, dual_safety_failures, dual_safety_fitness
from serve_classifier_result_review import review_payload
from make_classifier_sampling_variant import build as sampling_variant
from select_classifier_candidate import select


class ClassifierCorrectionTests(unittest.TestCase):
    def fixture(self, root):
        data, evaluation = root/'old_data', root/'old_evaluation'
        data.mkdir(); evaluation.mkdir()
        specs = [('a', 'train', 'ha', 'ok', 'ng'), ('b', 'test', 'ha', 'ng', 'ok'),
                 ('d', 'val', 'ha', 'ok', 'ok'), ('legacy_ng_c', 'val', 'legacy_val', 'ng', 'ok')]
        manifest, predictions = [], []
        for identifier, split, origin, label, prediction in specs:
            image = f'{split}/{label}/{identifier}.png'
            source = data/image
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(identifier.encode())
            record = {'id': identifier, 'split': split, 'origin': origin, 'label': label,
                      'image': image, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
            manifest.append(record)
            predicted = {'label': prediction, 'confidence': .9}
            evaluated = dict(record, path=str(source), predictions={'baseline': {'label': 'ng', 'confidence': .6},
                                                                    'candidate': predicted})
            predictions.append(evaluated)
            domain = 'legacy_val' if origin == 'legacy_val' else 'ha_'+split
            preview = evaluation/'errors'/domain/(identifier+'.png')
            preview.parent.mkdir(parents=True, exist_ok=True); preview.write_bytes(source.read_bytes())
        (data/'manifest.json').write_text(json.dumps({'records': manifest, 'domain': 'HA_dual_validation_fault_replay'}))
        (evaluation/'per_image.json').write_text(json.dumps(predictions))
        (evaluation/'report.json').write_text('{"domains":{}}')
        digest, _, _ = review_payload(evaluation)
        decisions = {'a': {'label': 'ng', 'original_label': 'ok'},
                     'b': {'label': 'ok', 'original_label': 'ng'},
                     'legacy_ng_c': {'label': 'ng', 'original_label': 'ng'}}
        folder = evaluation/'label_review'; folder.mkdir()
        (folder/'decisions.json').write_text(json.dumps({'evaluation_sha256': digest,
                                                       'decisions': decisions, 'revision': 4}))
        return data, evaluation

    def test_corrections_preserve_split_sources_and_reviewed_test_disclosure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); data, evaluation = self.fixture(root)
            originals = {p: p.read_bytes() for folder in (data, evaluation) for p in folder.rglob('*') if p.is_file()}
            output = root/'version2'/'dataset'
            build(data, evaluation, output)
            manifest = json.loads((output/'manifest.json').read_text())
            records = {r['id']: r for r in manifest['records']}
            self.assertEqual(records['a']['label'], 'ng')
            self.assertEqual(records['a']['split'], 'train')
            self.assertEqual(records['b']['label'], 'ok')
            self.assertEqual(records['b']['split'], 'test')
            self.assertFalse(any(p.name == 'b.png' for p in (output/'train').rglob('*')))
            protocol = manifest['label_correction_protocol']
            self.assertTrue(protocol['test_previously_viewed_and_labels_reviewed'])
            self.assertFalse(protocol['test_used_for_checkpoint_selection'])
            self.assertEqual(protocol['changed_labels'], 2)
            self.assertEqual(protocol['remaining_old_only_disagreements_not_reviewed'], 1)
            for path, content in originals.items():
                self.assertEqual(path.read_bytes(), content)
            rescored = json.loads((output.parent/'previous_models_corrected'/'report.json').read_text())
            self.assertEqual(rescored['domains']['ha_test']['candidate']['accuracy'], 1)
            with self.assertRaises(FileExistsError):
                build(data, evaluation, output)

    def test_incomplete_new_disagreements_rejected_before_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); data, evaluation = self.fixture(root)
            path = evaluation/'label_review'/'decisions.json'
            state = json.loads(path.read_text()); state['decisions'].pop('a'); path.write_text(json.dumps(state))
            output = root/'bad'/'dataset'
            with self.assertRaises(ValueError):
                build(data, evaluation, output)
            self.assertFalse(output.exists())

    def test_parameterized_retention_gates_keep_previous_defaults(self):
        ha_targets = [0]*10+[1]*10+[2]*20
        legacy_targets = [0]*100+[1]*100+[2]*300
        ha = classification_metrics(ha_targets, ha_targets)
        pred = legacy_targets.copy(); pred[:6] = [2]*6
        legacy = classification_metrics(legacy_targets, pred)
        self.assertEqual(dual_safety_failures(ha, legacy), [])
        gates = {'legacy_ng_misses_max': 5, 'legacy_empty_as_ok_max': 4}
        self.assertTrue(dual_safety_failures(ha, legacy, gates))
        self.assertGreater(dual_safety_fitness(ha, legacy), dual_safety_fitness(ha, legacy, gates))

    def test_sampling_variant_changes_only_probabilities(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); data, _ = self.fixture(root)
            original = json.loads((data/'manifest.json').read_text())
            output = root/'sampling'
            sampling_variant(data, output, .95)
            variant = json.loads((output/'manifest.json').read_text())
            self.assertEqual(variant['records'], original['records'])
            self.assertEqual(variant['ha_sampling_probability'], .95)
            self.assertAlmostEqual(variant['legacy_sampling_probability'], .05)
            for record in original['records']:
                self.assertEqual((output/record['image']).read_bytes(), (data/record['image']).read_bytes())

    def test_corrected_selection_discloses_previously_viewed_test_and_is_immutable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root/'trial'/'weights'/'best.onnx'
            model.parent.mkdir(parents=True); model.write_bytes(b'fixture onnx')
            model.with_suffix('.pt').write_bytes(b'fixture pt')
            (model.parent.parent/'args.yaml').write_text(json.dumps({'data': str(root/'training_variant')}))
            ha_targets = [0]*10+[1]*10+[2]*20
            old_targets = [0]*100+[1]*100+[2]*300
            report = {'dataset': str(root/'same_eval_data'),
                      'label_correction_protocol': {'test_previously_viewed_and_labels_reviewed': True},
                      'validation_safety_gates': {'legacy_ng_misses_max': 5, 'legacy_empty_as_ok_max': 4},
                      'models': {'candidate': {'path': str(model),
                                               'sha256': hashlib.sha256(model.read_bytes()).hexdigest()}},
                      'domains': {'ha_val': {'candidate': classification_metrics(ha_targets, ha_targets)},
                                  'legacy_val': {'candidate': classification_metrics(old_targets, old_targets)}}}
            evaluation = root/'evaluation.json'; evaluation.write_text(json.dumps(report))
            output = root/'selection'
            select([evaluation], output, 'fixture_reviewed')
            selection = json.loads((output/'selection.json').read_text())
            self.assertFalse(selection['selection_locked_before_ha_test'])
            self.assertTrue(selection['selection_locked_before_new_candidate_ha_test'])
            self.assertEqual(selection['dataset'], str(root/'same_eval_data'))
            self.assertEqual(selection['training_args']['data'], str(root/'training_variant'))
            self.assertFalse(selection['production_deployed'])
            before = (output/'selection.json').read_bytes()
            with self.assertRaises(FileExistsError):
                select([evaluation], output, 'fixture_reviewed')
            self.assertEqual((output/'selection.json').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
