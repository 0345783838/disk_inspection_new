"""Authorized supplementary learning: grouped new-frame CV, then all-data fitting.

Keeps historical models/datasets and installed runtime unchanged. Model-proposed
OK/no_disk labels remain identified separately from the user's 113 reviewed labels.
"""
import argparse
from collections import Counter
from copy import copy
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from ultralytics import YOLO
from ultralytics.data.build import InfiniteDataLoader, seed_worker

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classifier_metrics import LABELS, classification_metrics
from evaluate_classifier_ha import predict, torch_onnx_parity
from train_classifier_ha import RuntimeDataset, RuntimeTrainer, MacroF1Validator
from blend_classifier_fullha import interpolate
from classifier_dataset import active_dataset

cv2.setNumThreads(1)
FRACTIONS = (0., .25, .5, .75, 1.)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def population_metrics(records, predictions):
    groups = {
        'new_all': lambda r: r['origin'] == 'new_factory',
        'new_human': lambda r: r['origin'] == 'new_factory' and r.get('human_reviewed', False),
        'new_proposals': lambda r: r['origin'] == 'new_factory' and not r.get('human_reviewed', False),
        'old_ha': lambda r: r['origin'] == 'old_ha',
        'legacy_val': lambda r: r['origin'] == 'legacy_val',
    }
    metrics = {}
    for name, predicate in groups.items():
        indexes = [i for i, r in enumerate(records) if predicate(r)]
        if indexes:
            m = classification_metrics([LABELS.index(records[i]['label']) for i in indexes],
                                       [LABELS.index(predictions[i]['label']) for i in indexes])
            matrix = m['confusion_matrix_true_rows']
            m['false_ng'] = matrix[1][0] + matrix[2][0]
            m['errors'] = m['count'] - int(np.trace(matrix))
            metrics[name] = m
    return metrics


def score(metrics, gates):
    old, legacy, new = metrics['old_ha'], metrics['legacy_val'], metrics['new_human']
    failures = []
    for name, current, limit in (
        ('old_ha_ng_missed', old['ng_not_detected'], gates['old_ha_ng_max']),
        ('old_ha_empty_as_ok', old['no_disk_as_ok'], gates['old_ha_empty_max']),
        ('old_ha_errors', old['errors'], gates['old_ha_errors_max']),
        ('legacy_ng_missed', legacy['ng_not_detected'], gates['legacy_ng_max']),
        ('legacy_empty_as_ok', legacy['no_disk_as_ok'], gates['legacy_empty_max'])):
        if current > limit:
            failures.append(name)
    if legacy['accuracy'] < gates['legacy_accuracy_min']:
        failures.append('legacy_accuracy')
    value = ((2_000_000 if not failures else 0) + 1_000_000
             - 10_000 * new['ng_not_detected'] - 100 * new['false_ng']
             - metrics['new_all']['errors'] - old['errors'] * .01 - legacy['errors'] * .0001)
    return value, failures


class SupplementDataset(RuntimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        manifest = read(Path(self.root).parent / 'manifest.json')
        self.record_by_path = {str((Path(self.root).parent / r['image']).resolve()): r for r in manifest['records']}

    def __getitem__(self, index):
        result = super().__getitem__(index)
        record = self.record_by_path[str(Path(self.samples[index][0]).resolve())]
        result['population'] = {'new_factory': 0, 'old_ha': 1, 'legacy_val': 2, 'old_legacy_train': 3}[record['origin']]
        result['human_reviewed'] = int(record.get('human_reviewed', False))
        return result


class SupplementValidator(MacroF1Validator):
    def build_dataset(self, img_path):
        return SupplementDataset(root=img_path, args=self.args, augment=False, prefix=self.args.split)

    def init_metrics(self, model):
        super().init_metrics(model)
        self.populations, self.reviewed = [], []
        self.manifest = read(Path(self.args.data) / 'manifest.json')

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)
        self.populations.append(batch['population'].cpu())
        self.reviewed.append(batch['human_reviewed'].cpu())

    def get_stats(self):
        result = super().get_stats()
        targets = torch.cat(self.targets).cpu().numpy()
        predictions = torch.cat(self.pred)[:, 0].cpu().numpy()
        populations = torch.cat(self.populations).numpy()
        reviewed = torch.cat(self.reviewed).numpy()
        records = [{'origin': ('new_factory', 'old_ha', 'legacy_val', 'old_legacy_train')[int(p)],
                    'label': LABELS[int(t)], 'human_reviewed': bool(h)} for p, t, h in zip(populations, targets, reviewed)]
        metrics = population_metrics(records, [{'label': LABELS[int(p)]} for p in predictions])
        for name, m in metrics.items():
            for key in ('accuracy', 'errors', 'ng_not_detected', 'no_disk_as_ok', 'false_ng'):
                result[f'metrics/{name}_{key}'] = m[key]
        value, failures = score(metrics, self.manifest['gates'])
        result['fitness'] = value
        result['metrics/eligible'] = float(not failures)
        return result


class SupplementTrainer(RuntimeTrainer):
    def _model_train(self):
        super()._model_train()
        protocol = read(Path(self.args.data).parents[1] / 'protocol.json')
        if protocol.get('stability_recipe', False):
            # Ultralytics resets train() after epoch-start callbacks, so enforce
            # frozen running statistics here, including the classification head BN.
            for module in self.model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()

    def build_dataset(self, img_path, mode='train', batch=None):
        return SupplementDataset(root=img_path, args=self.args, augment=mode == 'train', prefix=mode)

    def get_validator(self):
        self.loss_names = ['loss']
        return SupplementValidator(self.test_loader, self.save_dir, args=copy(self.args), _callbacks=self.callbacks)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        if mode != 'train':
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        if rank != -1:
            raise ValueError('Single GPU training only')
        dataset = self.build_dataset(dataset_path, mode)
        if dataset.base.classes != LABELS:
            raise ValueError('Class order changed')
        groups = [dataset.record_by_path[str(Path(s[0]).resolve())]['origin'] for s in dataset.samples]
        counts = {g: np.bincount([s[1] for s, origin in zip(dataset.samples, groups) if origin == g], minlength=3)
                  for g in ('new_factory', 'old_ha', 'old_legacy_train')}
        if any((c == 0).any() for c in counts.values()):
            raise ValueError('A training domain is missing a class')
        raw = []
        for s, group in zip(dataset.samples, groups):
            r = dataset.record_by_path[str(Path(s[0]).resolve())]
            cls = s[1]
            if group == 'old_legacy_train':
                weight = [.6, .3, .1][cls] / counts[group][cls]
            else:
                weight = [1.2, 1.1, 1.0][cls] / np.sqrt(counts[group][cls])
                if group == 'new_factory' and r.get('human_reviewed'):
                    weight *= 4. if r['label'] != r['original_prediction'] else 2.
            raw.append(weight)
        totals = {g: sum(w for w, origin in zip(raw, groups) if origin == g) for g in counts}
        protocol = read(Path(self.args.data).parents[1] / 'protocol.json')
        probabilities = ({'new_factory': .25, 'old_ha': .50, 'old_legacy_train': .25}
                         if protocol.get('stability_recipe', False) else
                         {'new_factory': .45, 'old_ha': .45, 'old_legacy_train': .10})
        weights = [w / totals[g] * probabilities[g] for w, g in zip(raw, groups)]
        sampling = {g: sum(w for w, origin in zip(weights, groups) if origin == g) for g in counts}
        if not all(abs(sampling[g] - probabilities[g]) < 1e-8 for g in sampling):
            raise ValueError('Sampling domain budget changed')
        dump(self.save_dir / 'sampling.json', {'domain_probabilities': sampling, 'class_counts': {g: c.tolist() for g, c in counts.items()},
                                             'reviewed_correction_boost': 4, 'reviewed_other_boost': 2})
        generator = torch.Generator().manual_seed(self.args.seed)
        sampler = WeightedRandomSampler(weights, len(dataset), replacement=True, generator=generator)
        return InfiniteDataLoader(dataset, batch_size=min(batch_size, len(dataset)), sampler=sampler,
                                  num_workers=self.args.workers, pin_memory=True, worker_init_fn=seed_worker, generator=generator)


def make_dataset(root, name, fold, gates):
    protocol = read(root / 'protocol.json')
    pool = Path(protocol['pool'])
    source = read(pool / 'manifest.json')
    output = root / 'datasets' / name
    if output.exists():
        raise FileExistsError(output)
    records = []
    for original in source['records']:
        r = copy(original)
        source_image = pool / r['image']
        if digest(source_image) != r['sha256']:
            raise ValueError('Frozen source changed')
        if r['origin'] == 'legacy_val':
            splits = ['val']
        elif r['origin'] == 'new_factory':
            r['capture_group'] = protocol['frame_groups'][r['frame']]
            r['fold'] = protocol['group_folds'][str(r['capture_group'])]
            splits = ['train', 'val'] if fold is None else ['val' if r['fold'] == fold else 'train']
        elif r['origin'] == 'old_ha':
            splits = ['train', 'val']  # Explicit old-data retention, already fitted by the initialization model.
        else:
            splits = ['train']
        for split in splits:
            record = copy(r)
            record['split'] = split
            record['native_source'] = str(source_image)
            record['image'] = f'{split}/{record["label"]}/{record["id"]}{source_image.suffix.lower()}'
            record['seen_data_retention'] = split == 'val' and (r['origin'] == 'old_ha' or fold is None and r['origin'] == 'new_factory')
            records.append(record)
    train_hashes = {r['sha256'] for r in records if r['split'] == 'train'}
    for r in records:
        if r['split'] == 'val' and r['sha256'] in train_hashes and not r['seen_data_retention']:
            raise ValueError('Unintended train/validation hash overlap')
    if fold is not None:
        a = {r['capture_group'] for r in records if r['origin'] == 'new_factory' and r['split'] == 'train'}
        b = {r['capture_group'] for r in records if r['origin'] == 'new_factory' and r['split'] == 'val'}
        if a & b:
            raise ValueError('Capture group crossed a CV split')
    for split in ('train', 'val'):
        for label in LABELS:
            (output / split / label).mkdir(parents=True)
    for r in records:
        shutil.copy2(r['native_source'], output / r['image'])
    manifest = {'records': records, 'gates': gates, 'full_new_data_fitting': fold is None,
                'fold': fold, 'training_authorized_by_user': True, 'new_cv_hash_overlap': 0 if fold is not None else 'explicit fitting view',
                'old_ha_validation_is_retention_not_independent': True,
                'legacy_validation_never_in_new_training': True}
    dump(output / 'manifest.json', manifest)
    return output


def train(root, data, name, epochs, patience):
    protocol = read(root / 'protocol.json')
    if not protocol['training_authorized']:
        raise ValueError('Training is not authorized')
    if (root / 'training' / name).exists():
        raise FileExistsError(name)
    torch.set_num_threads(4)
    model = YOLO(protocol['baseline_pt'])
    stable = protocol.get('stability_recipe', False)
    learning_rate = .00001 if stable else .00003
    if [model.names[i] for i in range(3)] != LABELS:
        raise ValueError('Wrong checkpoint class order')
    model.train(trainer=SupplementTrainer, data=str(data), epochs=epochs, imgsz=224, batch=64, workers=2,
                device=0, seed=1020, deterministic=True, optimizer='AdamW', lr0=learning_rate, lrf=.1,
                cos_lr=True, weight_decay=.0005, warmup_epochs=3, warmup_bias_lr=learning_rate,
                patience=patience, dropout=0. if stable else .05,
                freeze=protocol['recipe'].get('freeze_first_modules',10) if stable else None,
                amp=False, auto_augment=None, erasing=0, scale=0,
                fliplr=.5, flipud=0, bgr=0, close_mosaic=0, project=str(root / 'training'),
                name=name, exist_ok=False, plots=True, verbose=False)
    best = Path(model.trainer.best)
    exported = YOLO(str(best)).export(format='onnx', dynamic=True, simplify=False, opset=13,
                                     imgsz=224, batch=1, device='cpu')
    print('TRAIN COMPLETE', name, exported, flush=True)


def blend(root, candidate_pt, fraction, name):
    protocol = read(root / 'protocol.json')
    folder = root / 'derived' / name
    folder.mkdir(parents=True)
    base, adapted = YOLO(protocol['baseline_pt']), YOLO(str(candidate_pt))
    base.model = base.model.cpu().float().eval()
    adapted.model = adapted.model.cpu().float().eval()
    base.model.load_state_dict(interpolate(base.model.state_dict(), adapted.model.state_dict(), fraction), strict=True)
    base.ckpt.update({'epoch': -1, 'optimizer': None, 'train_results': None, 'train_metrics': None,
                     'supplementary_blend': {'baseline': protocol['baseline_pt'], 'candidate': str(candidate_pt), 'candidate_fraction': fraction}})
    pt = folder / 'candidate.pt'
    base.save(str(pt))
    return Path(YOLO(str(pt)).export(format='onnx', dynamic=True, simplify=False, opset=13,
                                    imgsz=224, batch=1, device='cpu'))


def evaluate(model, records, folder, gates=None):
    if folder.exists():
        raise FileExistsError(folder)
    folder.mkdir(parents=True)
    _, predictions = predict(model, records)
    metrics = population_metrics(records, predictions)
    result = {'model': str(model), 'model_sha256': digest(model), 'metrics': metrics,
              'scope': 'Model proposals are not human ground truth; old HA is retention; full-new-data fit is not a test.'}
    if gates:
        value, failures = score(metrics, gates)
        result.update(score=value, retention_failures=failures, eligible=not failures)
    dump(folder / 'report.json', result)
    dump(folder / 'predictions.json', [dict(id=r['id'], original_id=r.get('original_id'), label=r['label'],
         origin=r['origin'], frame=r.get('frame'), human_reviewed=r.get('human_reviewed', False), image=r['path'],
         label_source=r.get('label_source'), prediction=p) for r, p in zip(records, predictions)])
    return result


def child_train(root, data, name, epochs, patience):
    log = root / 'logs' / f'{name}.log'
    log.parent.mkdir(exist_ok=True)
    print('START TRAIN', name, flush=True)
    command = [sys.executable, '-u', '-B', str(Path(__file__).resolve()), '--phase', 'train', '--root', str(root),
               '--data', str(data), '--name', name, '--epochs', str(epochs), '--patience', str(patience)]
    with log.open('w', encoding='utf-8') as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True,
                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    print('DONE TRAIN', name, flush=True)


def run(pool, root, baseline_pt, epochs=200, patience=100, fractions=FRACTIONS):
    pool = active_dataset(pool, required_split_policy='legacy_val_only')
    if epochs < 1 or patience < 1 or patience > epochs:
        raise ValueError('Require 1 <= patience <= epochs')
    if not fractions or tuple(sorted(set(fractions))) != tuple(fractions) or fractions[0] != 0 or fractions[-1] != 1:
        raise ValueError('Fractions must be unique, increasing, and include baseline 0 and full update 1')
    if root.exists():
        raise FileExistsError('Training output is immutable: ' + str(root))
    root.mkdir(parents=True)
    shutil.copy2(pool / 'manifest.json', root / 'source_manifest_at_training.json')
    source = read(pool / 'manifest.json')
    new = [r for r in source['records'] if r['origin'] == 'new_factory']
    frames = sorted({r['frame'] for r in new}, key=lambda f: f.split('Image_')[1])
    groups, previous, frame_groups = [], None, {}
    for frame in frames:
        stamp = datetime.strptime(frame.split('Image_')[1], '%Y%m%d%H%M%S%f')
        if previous is None or (stamp - previous).total_seconds() > 120:
            groups.append([])
        groups[-1].append(frame)
        frame_groups[frame] = len(groups) - 1
        previous = stamp
    loads = [0, 0, 0]
    group_folds = {}
    for i in sorted(range(len(groups)), key=lambda i: (-len(groups[i]), i)):
        fold = int(np.argmin(loads))
        group_folds[str(i)] = fold
        loads[fold] += len(groups[i]) * 100
    for group in source['exact_duplicate_groups']:
        if len({group_folds[str(frame_groups[f])] for f in group['frames']}) > 1:
            raise ValueError('Exact duplicate frames would cross CV folds')
    baseline_onnx = baseline_pt.with_suffix('.onnx')
    backup = root / 'baseline'
    backup.mkdir()
    for model in (baseline_pt, baseline_onnx):
        shutil.copy2(model, backup / model.name)
    expected_current = Path(source.get('new_review', pool.parent / 'review')) / 'classifier_before_review.onnx'
    if digest(baseline_onnx) != digest(expected_current):
        raise ValueError('Initialization model differs from deployed/reviewed baseline')
    protocol = {'created_at': datetime.now(timezone.utc).isoformat(), 'pool': str(pool),
                'pool_manifest_sha256': digest(pool / 'manifest.json'), 'training_authorized': True,
                'authorization': 'User authorized autonomous training and evaluation after preparation on 2026-10-20; no deployment requested.',
                'baseline_pt': str(backup / baseline_pt.name), 'baseline_onnx': str(backup / baseline_onnx.name),
                'baseline_pt_sha256': digest(baseline_pt), 'baseline_onnx_sha256': digest(baseline_onnx),
                'groups': groups, 'frame_groups': frame_groups, 'group_folds': group_folds, 'fold_new_crop_counts': loads,
                'grouping': 'Chronological frames from both cameras linked when consecutive captures are <=120 seconds apart; exact duplicate groups checked.',
                'recipe': {'lr0': .00003, 'new_probability': .45, 'old_ha_probability': .45, 'legacy_probability': .10,
                           'augmentation': 'Existing runtime RGB/stretch 224 plus mild illumination/noise/blur/flip; no crop/erasing/mixup.'},
                'candidate_fractions_predeclared': fractions,
                'training_schedule': {'cv_epochs': epochs, 'final_epochs': epochs,
                                      'patience': patience, 'monitor': 'validation safety fitness',
                                      'early_stopping': 'Strict fitness improvement; stop after patience consecutive epochs without improvement.',
                                      'authorization': 'User requested 200 epochs and patience 100 on 2026-10-21.'},
                'cv_scope': 'New collection grouped CV used for selection; labels reviewed with predictions visible; 2687 model proposals are not independently confirmed truth.',
                'final_scope': 'Final learns all 2800 new + old replay. Final new metrics are fitted-data metrics, not independent test accuracy.',
                'production_deployed': False}
    protocol['stability_recipe'] = os.environ.get('DISK_SUPPLEMENT_STABLE_HEAD') == '1'
    if protocol['stability_recipe']:
        protocol['recipe'].update(lr0=.00001, new_probability=.25, old_ha_probability=.50,
                                  legacy_probability=.25,
                                  freeze_first_modules=int(os.environ.get('DISK_SUPPLEMENT_FREEZE_MODULES','10')),
                                  frozen_batchnorm_running_statistics=True, dropout=0.)
    dump(root / 'protocol.json', protocol)
    shutil.copy2(pool / 'snapshots' / 'new_decisions.json', root / 'human_decisions_snapshot.json')
    all_records = [dict(r, path=str(pool / r['image'])) for r in source['records'] if r['origin'] != 'old_legacy_train']
    torch.set_num_threads(4)
    parity = torch_onnx_parity(Path(protocol['baseline_pt']), Path(protocol['baseline_onnx']), all_records[:32])
    dump(root / 'baseline' / 'export_parity.json', parity)
    baseline = evaluate(Path(protocol['baseline_onnx']), all_records, root / 'evaluations' / 'baseline')
    metrics = baseline['metrics']
    gates = {'old_ha_ng_max': metrics['old_ha']['ng_not_detected'],
             'old_ha_empty_max': metrics['old_ha']['no_disk_as_ok'], 'old_ha_errors_max': metrics['old_ha']['errors'] + 4,
             'legacy_ng_max': metrics['legacy_val']['ng_not_detected'],
             'legacy_empty_max': metrics['legacy_val']['no_disk_as_ok'],
             'legacy_accuracy_min': max(.987, metrics['legacy_val']['accuracy'] - .005)}
    protocol['gates'] = gates
    dump(root / 'protocol.json', protocol)
    cv_results = {str(f): [] for f in fractions}
    for fold in range(3):
        name = f'group_cv_{fold}'
        data = make_dataset(root, name, fold, gates)
        child_train(root, data, name, epochs, patience)
        pt = root / 'training' / name / 'weights' / 'best.pt'
        records = [dict(r, path=str(data / r['image'])) for r in read(data / 'manifest.json')['records'] if r['split'] == 'val']
        for fraction in fractions:
            model = Path(protocol['baseline_onnx']) if fraction == 0 else pt.with_suffix('.onnx') if fraction == 1 else blend(root, pt, fraction, f'cv{fold}_fraction{fraction}')
            folder = root / 'evaluations' / f'cv{fold}_fraction{fraction}'
            result = evaluate(model, records, folder, gates)
            result['predictions_file'] = str(folder / 'predictions.json')
            cv_results[str(fraction)].append(result)
        dump(root / 'cv_progress.json', cv_results)
    summaries = []
    for fraction in fractions:
        folds = cv_results[str(fraction)]
        rows = [r for f in folds for r in read(f['predictions_file']) if r['origin'] == 'new_factory']
        if len(rows) != 2800 or len({r['id'] for r in rows}) != 2800:
            raise ValueError('OOF coverage changed')
        combined = population_metrics(rows, [r['prediction'] for r in rows])
        m = combined['new_human']
        eligible = all(f['eligible'] for f in folds)
        key = (eligible, -m['ng_not_detected'], -m['false_ng'], -combined['new_all']['errors'],
               -max(f['metrics']['old_ha']['errors'] for f in folds), -fraction)
        summaries.append({'fraction': fraction, 'eligible_all_folds': eligible, 'metrics': combined,
                          'ranking': key, 'legacy_retention_per_fold': [f['metrics']['legacy_val'] for f in folds],
                          'old_ha_retention_per_fold': [f['metrics']['old_ha'] for f in folds]})
        dump(root / 'oof' / f'fraction{fraction}.json', rows)
    selected = max(summaries, key=lambda r: tuple(r['ranking']))
    decision = {'selection_locked_before_final_training': True, 'chosen_fraction': selected['fraction'],
                'summaries': summaries, 'recipe': protocol['recipe'], 'selected_at': datetime.now(timezone.utc).isoformat()}
    dump(root / 'cv_selection.json', decision)
    data = make_dataset(root, 'final_all_new', None, gates)
    child_train(root, data, 'final_all_new', epochs, patience)
    pt = root / 'training' / 'final_all_new' / 'weights' / 'best.pt'
    fraction = selected['fraction']
    final_model = Path(protocol['baseline_onnx']) if fraction == 0 else pt.with_suffix('.onnx') if fraction == 1 else blend(root, pt, fraction, f'final_fraction{fraction}')
    final = evaluate(final_model, all_records, root / 'evaluations' / 'final', gates)
    # Also report the unblended adaptation even if grouped CV selected a more conservative fraction.
    if final_model != pt.with_suffix('.onnx'):
        evaluate(pt.with_suffix('.onnx'), all_records, root / 'evaluations' / 'final_unblended', gates)
    parity = torch_onnx_parity(final_model.with_suffix('.pt'), final_model, all_records[:32])
    before, after = baseline['metrics']['new_human'], final['metrics']['new_human']
    recommend = final['eligible'] and after['ng_not_detected'] <= before['ng_not_detected'] and after['false_ng'] < before['false_ng']
    folder = root / 'selected'
    folder.mkdir()
    artifact = folder / 'c_p_supplementary_20261020.onnx'
    shutil.copy2(final_model, artifact)
    shutil.copy2(final_model.with_suffix('.pt'), artifact.with_suffix('.pt'))
    release = {'training_complete': True, 'recommended_for_review': recommend, 'production_deployed': False,
               'selected_onnx': str(artifact), 'selected_pt': str(artifact.with_suffix('.pt')),
               'onnx_sha256': digest(artifact), 'pt_sha256': digest(artifact.with_suffix('.pt')),
               'cv_fraction': fraction, 'cv_selection': str(root / 'cv_selection.json'),
               'baseline': baseline, 'candidate': final, 'onnx_pt_parity': parity,
               'new_human_labels': 113, 'new_model_proposals': 2687,
               'limitation': protocol['cv_scope'] + ' ' + protocol['final_scope']}
    dump(root / 'release_decision.json', release)
    render(root, release, selected)
    if digest(pool / 'manifest.json') != protocol['pool_manifest_sha256']:
        raise ValueError('Prepared pool metadata changed')
    print('COMPLETE', artifact, 'recommended', recommend, flush=True)


def render(root, release, selected):
    before, after = release['baseline']['metrics'], release['candidate']['metrics']
    parts = ['''<!doctype html><html lang="vi"><meta charset="utf-8"><title>Classifier bổ sung 20/10</title>
<style>body{font:16px system-ui;background:#17202a;color:#e4edf5;margin:26px;max-width:1400px}a{color:#8fcfff}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{border:1px solid #52667c;padding:10px;text-align:left}.note{background:#413723;padding:16px;line-height:1.5}img{max-width:180px;max-height:230px}figure{display:inline-block;vertical-align:top;width:230px;margin:12px;overflow-wrap:anywhere}small{color:#bac9da}</style>
<h1>Classifier train bổ sung — 20/10/2026</h1>''']
    parts.append(f'<p>Training và đánh giá hoàn tất. Đề xuất bản mới để kiểm tra: <b>{release["recommended_for_review"]}</b>. Model mới lưu riêng; bộ chạy hiện tại chưa thay.</p>')
    parts.append('<p class="note">113 crop mới có nhãn người rà; 2.687 crop OK/no_disk còn lại là nhãn model được yêu cầu đưa vào dataset. '
                 'CV giữ ngoài training theo nhóm lượt chụp mới, dùng để chọn mức cập nhật. Dữ liệu đã được xem nhãn/model; không phải kiểm chứng độc lập, mù. '
                 'Bản cuối học toàn bộ dữ liệu mới; số đo trên dữ liệu mới của bản cuối là kết quả fitting. HA cũ là retention, legacy validation là benchmark cũ.</p>')
    parts.append('<table><tr><th>Dữ liệu</th><th>Số crop</th><th>NG thật bỏ sót: cũ → mới</th><th>NG ảo: cũ → mới</th><th>no_disk → OK</th><th>Đúng nhãn</th></tr>')
    names = {'new_human': 'Mới: 113 crop đã sửa', 'new_all': 'Mới: toàn bộ, gồm nhãn model',
             'new_proposals': 'Mới: OK/no_disk chưa rà thủ công', 'old_ha': 'HA cũ — giữ kiến thức', 'legacy_val': 'Legacy validation'}
    for key in ('new_human', 'new_all', 'new_proposals', 'old_ha', 'legacy_val'):
        b, a = before[key], after[key]
        parts.append(f'<tr><td>{names[key]}</td><td>{a["count"]}</td><td>{b["ng_not_detected"]} → {a["ng_not_detected"]}</td><td>{b["false_ng"]} → {a["false_ng"]}</td><td>{b["no_disk_as_ok"]} → {a["no_disk_as_ok"]}</td><td>{b["accuracy"]:.2%} → {a["accuracy"]:.2%}</td></tr>')
    parts.append('</table><h2>Grouped CV trên lượt chụp mới</h2><pre>'+html.escape(json.dumps(selected['metrics'],ensure_ascii=False,indent=2))+'</pre>')
    parts.append('<p><a href="release_decision.json">Quyết định / hash / đầy đủ chỉ số</a> · <a href="protocol.json">Quy trình và cách chia nhóm</a> · <a href="cv_selection.json">So sánh CV</a> · <a href="selected/c_p_supplementary_20261020.onnx">Model ONNX</a> · <a href="selected/c_p_supplementary_20261020.pt">Checkpoint PT</a></p>')
    for label, path in [('Lỗi còn lại trên 113 crop đã sửa', root / 'evaluations/final/predictions.json')]:
        parts.append('<h2>'+label+'</h2>')
        for r in read(path):
            if r['origin']=='new_factory' and r.get('human_reviewed') and r['label']!=r['prediction']['label']:
                target = root / 'review_errors' / (r['id']+Path(r['image']).suffix)
                target.parent.mkdir(exist_ok=True)
                shutil.copy2(r['image'],target)
                parts.append(f'<figure><a href="review_errors/{target.name}"><img src="review_errors/{target.name}"></a><figcaption>{html.escape(r["original_id"])}<br>Nhãn {r["label"]}; model {r["prediction"]["label"]} ({r["prediction"]["confidence"]:.3f})</figcaption></figure>')
    parts.append('</html>')
    (root / 'index.html').write_text('\n'.join(parts),encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('run','train'), required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pool', type=Path)
    parser.add_argument('--pretrained', type=Path)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--name')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--fractions', default='0,.25,.5,.75,1', help='Predeclared CV update fractions')
    args = parser.parse_args()
    if args.phase == 'run':
        run(args.pool.resolve() if args.pool else None,args.root.resolve(),args.pretrained.resolve(),args.epochs,args.patience,
            tuple(float(f) for f in args.fractions.split(',')))
    else:
        train(args.root.resolve(),args.data.resolve(),args.name,args.epochs,args.patience)
