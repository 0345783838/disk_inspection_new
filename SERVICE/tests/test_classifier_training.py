"""Training contract tests; optional torch dependencies, isolated fixtures only."""
import sys
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

TRAINING = Path(__file__).resolve().parents[1]/'training'
sys.path.insert(0, str(TRAINING))
from build_classifier_dataset import assign_splits
from classifier_metrics import classification_metrics, safety_fitness, dual_safety_fitness
try:
    from train_classifier_ha import runtime_tensor, safe_augment, RuntimeTrainer
    from src.inference.yolo_classify.yolo_classifier import YoloClassifier
except ImportError:
    runtime_tensor = None


class ClassifierMetricTests(unittest.TestCase):
    def test_false_ok_metrics(self):
        result = classification_metrics([0,0,1,1,2,2], [0,2,1,2,2,0])
        self.assertEqual(result['confusion_matrix_true_rows'], [[1,0,1],[0,1,1],[1,0,1]])
        self.assertEqual(result['unsafe_false_ok'], 2)
        self.assertEqual(result['unsafe_false_ok_rate'], .5)
        self.assertEqual(result['accuracy'], .5)
        self.assertAlmostEqual(result['macro_f1'], (.5+2/3+.4)/3)

    def test_split_whole_frames_and_chronological_test(self):
        items = [{'camera':c,'frame':f'Cam{c}_{f:02d}','slot':s}
                 for c in (1,2) for f in range(10) for s in range(1,101)]
        result = assign_splits(items)
        self.assertEqual(sum(s=='train' for s in result.values()), 12)
        self.assertEqual(sum(s=='val' for s in result.values()), 4)
        self.assertEqual(sum(s=='test' for s in result.values()), 4)
        for c in (1,2):
            self.assertEqual(result[f'Cam{c}_08'], 'test')
            self.assertEqual(result[f'Cam{c}_09'], 'test')
        with self.assertRaises(ValueError):
            assign_splits([{'camera':1,'frame':'one'}])

    def test_safety_ranking_prioritizes_defects_without_reject_all(self):
        targets = [0]*96+[1]*67+[2]*237
        all_correct = classification_metrics(targets, targets)
        one_false_ok = targets.copy();one_false_ok[0]=2
        other_mistakes = targets.copy();other_mistakes[:80]=[1]*80
        self.assertGreater(safety_fitness(classification_metrics(targets,other_mistakes)),
                           safety_fitness(classification_metrics(targets,one_false_ok)))
        self.assertGreater(safety_fitness(all_correct),safety_fitness(classification_metrics(targets,other_mistakes)))
        self.assertGreater(safety_fitness(classification_metrics(targets,one_false_ok)),
                           safety_fitness(classification_metrics(targets,[0]*400)))

    def test_dual_safety_rejects_legacy_empty_regression(self):
        ha_targets=[0]*96+[1]*67+[2]*237
        old_targets=[0]*821+[1]*512+[2]*1437
        ha=classification_metrics(ha_targets,ha_targets)
        old=classification_metrics(old_targets,old_targets)
        bad_predictions=old_targets.copy();bad_predictions[821:872]=[2]*51
        regressed=classification_metrics(old_targets,bad_predictions)
        ha_one=ha_targets.copy();ha_one[0]=2
        self.assertGreater(dual_safety_fitness(classification_metrics(ha_targets,ha_one),old),
                           dual_safety_fitness(ha,regressed))
        old_one=old_targets.copy();old_one[0]=2
        self.assertGreater(dual_safety_fitness(ha,old),dual_safety_fitness(ha,classification_metrics(old_targets,old_one)))
        warning_misses=old_targets.copy();warning_misses[:22]=[1]*22
        self.assertGreater(dual_safety_fitness(classification_metrics(ha_targets,ha_one),old),
                           dual_safety_fitness(ha,classification_metrics(old_targets,warning_misses)))


@unittest.skipIf(runtime_tensor is None, 'Optional torch/ultralytics training dependencies unavailable')
class ClassifierPreprocessTests(unittest.TestCase):
    def test_runtime_tensor_pixel_exact(self):
        generator = np.random.default_rng(1005)
        classifier = object.__new__(YoloClassifier)
        for shape in ((165,114,3),(140,95,3),(165,50,3),(100,100,3)):
            image = generator.integers(0,256,shape,dtype=np.uint8)
            expected = classifier._preprocess_batch([image], (224,224))[0]
            self.assertTrue(np.array_equal(runtime_tensor(image).numpy(), expected))

    def test_augmentation_preserves_native_dimensions(self):
        image = np.full((165,114,3), 85, np.uint8)
        original = image.copy()
        for seed in range(30):
            np.random.seed(seed)
            result = safe_augment(image)
            self.assertEqual(result.shape, image.shape)
            self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(np.array_equal(original, image))

    def test_replay_sampling_is_85_percent_ha_and_ng_priority(self):
        import torch
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            records, samples = [], []
            for group, counts in (('ha',[4,3,3]),('legacy_train',[8,2,1])):
                for label,count in enumerate(counts):
                    for index in range(count):
                        relative = f'train/{label}/{group}_{index}.png'
                        records.append({'image':relative,'origin':group,'split':'train'})
                        samples.append([str(root/relative),label])
            (root/'manifest.json').write_text(json.dumps({'domain':'HA_with_fault_priority_replay','records':records}))
            class FixtureDataset:
                base=SimpleNamespace(classes=['ng','no_disk','ok'])
                def __len__(self):
                    return len(self.samples)
                def __getitem__(self,index):
                    return {'img':torch.zeros((3,2,2)),'cls':self.samples[index][1]}
            dataset=FixtureDataset();dataset.samples=samples
            trainer=object.__new__(RuntimeTrainer)
            trainer.args=SimpleNamespace(data=str(root),seed=1005,workers=0)
            trainer.build_dataset=lambda *args:dataset
            loader=trainer.get_dataloader(str(root/'train'),batch_size=4,rank=-1,mode='train')
            weights=loader.sampler.weights.numpy()
            groups=[r['origin'] for r in records]
            self.assertAlmostEqual(sum(w for w,g in zip(weights,groups) if g=='ha'),.85)
            self.assertAlmostEqual(sum(w for w,g in zip(weights,groups) if g=='legacy_train'),.15)
            self.assertAlmostEqual(sum(w for w,g,s in zip(weights,groups,samples) if g=='legacy_train' and s[1]==0),.15*.75)


if __name__ == '__main__':
    unittest.main()
