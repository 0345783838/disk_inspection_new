"""Balanced replay and optional label-noise-resistant, teacher-preserving loss.

All labels remain canonical. This module never changes labels or live runtime.
"""
from collections import defaultdict
from copy import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from ultralytics import YOLO
from ultralytics.data.build import InfiniteDataLoader, seed_worker

from classifier_metrics import LABELS, classification_metrics
from train_classifier_ha import MacroF1Validator
from train_classifier_supplementary import SupplementDataset, SupplementValidator, SupplementTrainer, read, dump


def metrics_for(records, predictions):
    output = {}
    for name in ('new_factory', 'old_ha', 'old_legacy_train', 'legacy_val'):
        indices = [i for i, r in enumerate(records) if r['origin'] == name]
        if not indices:
            continue
        m = classification_metrics([LABELS.index(records[i]['label']) for i in indices],
                                   [LABELS.index(predictions[i]['label']) for i in indices])
        matrix = np.asarray(m['confusion_matrix_true_rows'])
        m.update(false_ng=int(matrix[1:, 0].sum()), errors=int(m['count'] - matrix.trace()),
                 ok_as_empty=int(matrix[2, 1]))
        output[name] = m
    return output


def retention_failures(metrics, baseline, include_new=True):
    failures = []
    for domain in ('old_ha', 'legacy_val') + (('new_factory',) if include_new else ()):
        if domain not in metrics:
            continue
        current, old = metrics[domain], baseline[domain]
        for key in ('ng_not_detected', 'no_disk_as_ok'):
            if current[key] > old[key]:
                failures.append(domain + '/' + key)
        tolerance = 4 if domain == 'old_ha' else (int(.005 * old['count']) if domain == 'legacy_val' else 0)
        if current['errors'] > old['errors'] + tolerance:
            failures.append(domain + '/errors')
        if domain in ('new_factory', 'legacy_val') and current['false_ng'] > old['false_ng']:
            failures.append(domain + '/false_ng')
    return failures


def checkpoint_fitness(metrics, baseline):
    """Rank held-out metrics; continuous penalties prevent unusable early winners."""
    domains = [d for d in ('new_factory', 'legacy_val') if d in metrics]
    value = 0.
    for domain in domains:
        m, old = metrics[domain], baseline[domain]
        supports = m['per_class']
        ng = max(1, supports['ng']['support'])
        empty = max(1, supports['no_disk']['support'])
        negatives = max(1, m['count'] - supports['ng']['support'])
        value += (m['macro_f1'] - 2 * m['ng_not_detected'] / ng
                  - m['no_disk_as_ok'] / empty - m['false_ng'] / negatives
                  - .5 * m['errors'] / m['count']) / len(domains)
        value -= (8 * max(0, m['ng_not_detected'] - old['ng_not_detected']) / ng
                  + 2 * max(0, m['no_disk_as_ok'] - old['no_disk_as_ok']) / empty
                  + 2 * max(0, m['errors'] - old['errors']) / m['count']) / len(domains)
    # Strict gates remain decisive once attainable; no fitted records enter here.
    return value + (3. if not retention_failures(metrics, baseline) else 0.)


def sampling_weights(records, recipe):
    """Explicit domain AND class budgets; correction boosts cannot steal budgets."""
    raw, totals = [], defaultdict(float)
    for r in records:
        domain, label = r['origin'], r['label']
        boost = 1.5 if (r.get('label_history') or
                       r.get('review_method') == 'explicit_saved_decision') else 1.
        if recipe.get('robust') and r.get('unresolved_similarity_candidate') and not r.get('label_history'):
            boost *= .6
        raw.append(boost); totals[(domain, label)] += boost
    weights = []
    for r, w in zip(records, raw):
        d, c = r['origin'], r['label']
        weights.append(w / totals[(d, c)] * recipe['domain_budget'][d] * recipe['class_budget'][c])
    expected = {(d, c) for d in recipe['domain_budget'] for c in LABELS}
    if set(totals) != expected:
        raise ValueError('Every training domain must contain all three classes')
    actual = defaultdict(float)
    for r, w in zip(records, weights): actual[(r['origin'], r['label'])] += w
    assert abs(sum(weights) - 1) < 1e-8
    for (d, c), w in actual.items():
        assert abs(w - recipe['domain_budget'][d] * recipe['class_budget'][c]) < 1e-8
    return weights, {d: {c: actual[(d, c)] for c in LABELS} for d in recipe['domain_budget']}


class RobustLoss:
    """Optional GCE mixture and KD only where an old-domain teacher agrees with GT."""
    def __init__(self, recipe):
        self.recipe = recipe

    def __call__(self, preds, batch):
        logits = preds[1] if isinstance(preds, (tuple, list)) else preds
        target = batch['cls'].long()
        ce = F.cross_entropy(logits, target, reduction='none')
        if self.recipe.get('robust'):
            probability = logits.softmax(-1).gather(1, target[:, None]).squeeze(1)
            gce = (1 - probability.clamp_min(1e-8).pow(.7)) / .7
            ce = .8 * ce + .2 * gce
        loss = ce.mean()
        if 'teacher_probabilities' in batch:
            teacher = batch['teacher_probabilities'].detach()
            population = batch['population'].to(target.device)
            mask = (population != 0) & (teacher.argmax(-1) == target) & (teacher.max(-1).values >= .6)
            if mask.any():
                temperature = 2.
                softened = teacher[mask].clamp_min(1e-8).pow(1 / temperature)
                softened = softened / softened.sum(-1, keepdim=True)
                kd = F.kl_div(F.log_softmax(logits[mask] / temperature, -1), softened,
                              reduction='batchmean') * temperature ** 2
                loss = loss + self.recipe['distill'] * kd
        return loss, loss.detach()


class RobustValidator(SupplementValidator):
    def get_stats(self):
        result = MacroF1Validator.get_stats(self)
        targets = torch.cat(self.targets).cpu().numpy()
        predictions = torch.cat(self.pred)[:, 0].cpu().numpy()
        records = [dict(origin=('new_factory', 'old_ha', 'legacy_val', 'old_legacy_train')[int(p)],
                        label=LABELS[int(t)]) for p, t in zip(torch.cat(self.populations).numpy(), targets)]
        metrics = metrics_for(records, [dict(label=LABELS[int(p)]) for p in predictions])
        for d, m in metrics.items():
            for k in ('accuracy', 'macro_f1', 'errors', 'ng_not_detected', 'no_disk_as_ok', 'false_ng'):
                result[f'metrics/{d}_{k}'] = m[k]
        result['fitness'] = checkpoint_fitness(metrics, self.manifest['baseline_validation'])
        result['metrics/eligible'] = float(not retention_failures(metrics, self.manifest['baseline_validation']))
        return result


class RobustTrainer(SupplementTrainer):
    def recipe(self):
        if not hasattr(self, '_recipe_cache'):
            self._recipe_cache = read(Path(self.args.data).parent.parent / 'protocol.json')['active_recipe']
        return self._recipe_cache

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        model.criterion = RobustLoss(self.recipe())
        return model

    def _model_train(self):
        super()._model_train()
        if self.recipe().get('freeze_bn'):
            for module in self.model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm): module.eval()

    def get_validator(self):
        self.loss_names = ['loss']
        return RobustValidator(self.test_loader, self.save_dir, args=copy(self.args), _callbacks=self.callbacks)

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        recipe = self.recipe()
        if recipe.get('distill'):
            if not hasattr(self, 'retention_teacher'):
                protocol = read(Path(self.args.data).parent.parent / 'protocol.json')
                self.retention_teacher = YOLO(protocol['baseline_pt']).model.float().to(self.device).eval()
                for p in self.retention_teacher.parameters(): p.requires_grad = False
            with torch.no_grad():
                output = self.retention_teacher(batch['img'])
                batch['teacher_probabilities'] = output[0] if isinstance(output, (tuple, list)) else output
        return batch

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        if mode != 'train':
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        if rank != -1: raise ValueError('Single GPU only')
        dataset = self.build_dataset(dataset_path, mode)
        if dataset.base.classes != LABELS: raise ValueError('Class order changed')
        records = [dataset.record_by_path[str(Path(s[0]).resolve())] for s in dataset.samples]
        weights, budget = sampling_weights(records, self.recipe())
        dump(self.save_dir / 'sampling.json', dict(joint_domain_class_budget=budget,
             legacy_ok_probability=budget['old_legacy_train']['ok'], replacement=True,
             training_records=len(records), target_labels_unchanged=True))
        generator = torch.Generator().manual_seed(self.args.seed)
        sampler = WeightedRandomSampler(weights, len(dataset), replacement=True, generator=generator)
        return InfiniteDataLoader(dataset, batch_size=min(batch_size, len(dataset)), sampler=sampler,
                                  num_workers=self.args.workers, pin_memory=True,
                                  worker_init_fn=seed_worker, generator=generator)
