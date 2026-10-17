"""Grouped CV from pre-HA weights, then full-HA fitting with explicit seen-data metrics."""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from copy import copy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from ultralytics import YOLO
from ultralytics.data.build import InfiniteDataLoader, seed_worker

from classifier_metrics import LABELS, classification_metrics
from evaluate_classifier_ha import predict, torch_onnx_parity
from train_classifier_ha import RuntimeTrainer, MacroF1Validator

RECIPES = {
    'balanced': {'ha_probability': .90, 'ha_class_factors': [1.35, 1.15, 1.0], 'hard_boost': 1.0},
    'hard_focus': {'ha_probability': .95, 'ha_class_factors': [1.75, 1.15, 1.0], 'hard_boost': 3.0},
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def fold_lookup(records):
    """Three chronological camera-block folds: 6/6/8 whole frames, never random crops."""
    lookup = {}
    for camera in (1, 2):
        frames = sorted({r['frame'] for r in records if r['camera'] == camera})
        if len(frames) != 10:
            raise ValueError('Expected ten HA frames per camera')
        for index, frame in enumerate(frames):
            lookup[frame] = 0 if index < 3 else 1 if index < 6 else 2
    return lookup


def safety_rank(ha, legacy, legacy_ng_max=7, legacy_empty_max=4):
    """HA NG first, then legacy NG; guard against trivial reject-all and old-domain drift.

    Integer radix preserves priorities despite Ultralytics rounding fitness to five decimals.
    HA errors then legacy errors break equal NG counts. Return score and eligibility.
    """
    eligible = (ha['per_class']['ok']['recall'] >= .97 and legacy['per_class']['ok']['recall'] >= .95
                and legacy['accuracy'] >= .987 and legacy['ng_not_detected'] <= legacy_ng_max
                and legacy['no_disk_as_ok'] <= legacy_empty_max)
    ha_ng, old_ng = ha['per_class']['ng']['support'], legacy['per_class']['ng']['support']
    primary = ha['ng_not_detected']*(old_ng+1)+legacy['ng_not_detected']
    maximum_primary = ha_ng*(old_ng+1)+old_ng
    ha_errors = ha['count']-int(np.trace(ha['confusion_matrix_true_rows']))
    old_errors = legacy['count']-int(np.trace(legacy['confusion_matrix_true_rows']))
    rank = ((maximum_primary-primary)*(ha['count']+1)+(ha['count']-ha_errors))*(legacy['count']+1)
    rank += legacy['count']-old_errors + .1*ha['macro_f1']
    maximum = (maximum_primary+1)*(ha['count']+1)*(legacy['count']+1)+1
    return (maximum if eligible else 0)+rank, eligible


def prepare(source, previous_predictions, output):
    if output.exists():
        raise FileExistsError('Full-HA protocol is never overwritten')
    manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    ha = [r for r in manifest['records'] if r.get('origin') == 'ha']
    if len(ha) != 2000 or len({r['frame'] for r in ha}) != 20:
        raise ValueError('Expected all 2,000 HA crops / 20 frames')
    previous = {r['id']: r for r in json.loads(previous_predictions.read_text(encoding='utf-8'))}
    if not {r['id'] for r in ha}.issubset(previous):
        raise ValueError('Missing previous model predictions')
    hard_ids = []
    for r in ha:
        old = previous[r['id']]
        if old['label'] != r['label']:
            raise ValueError('Previous predictions use different corrected labels')
        if old['predictions']['candidate']['label'] != r['label'] or 'label_before_correction' in r:
            hard_ids.append(r['id'])
    folds = fold_lookup(ha)
    # Snapshot paths and checksums now; never use the changing review UI during training.
    sources = {}
    for r in manifest['records']:
        path = (source/r['image']).resolve()
        if digest(path) != r['sha256']:
            raise ValueError('Source image changed')
        sources[r['id']] = str(path)
    if len({r['id'] for r in manifest['records']}) != len(manifest['records']):
        raise ValueError('Duplicate record IDs')
    output.mkdir(parents=True)
    shutil.copy2(source/'post_evaluation_decisions_snapshot.json', output/'human_corrections_snapshot.json')
    protocol = {'created_at': datetime.now(timezone.utc).isoformat(),
                'source_dataset': str(source.resolve()), 'source_manifest_sha256': digest(source/'manifest.json'),
                'correction_sha256': digest(output/'human_corrections_snapshot.json'),
                'ha_records': ha, 'source_paths': sources, 'source_manifest': manifest,
                'fold_by_frame': folds, 'hard_ids': sorted(hard_ids), 'recipes': RECIPES,
                'cross_validation_limitation': 'same-session, already reviewed HA; checkpoint/recipe selection estimate, not independent blind test',
                'cv_initialization': 'pre-HA original checkpoint only; no previous HA weights in CV',
                'final_protocol': 'all 2,000 HA in training; full-HA metrics are in-sample fitting, NOT test accuracy',
                'old_holdout': 'all corrected 2,770 legacy val, never in new training; original pretraining had old train/val overlap',
                'legacy_training_sampling': '600 NG / 300 no_disk / 100 OK clean original TRAIN replay, unchanged',
                'historical_split_abandoned_by_user_request': True,
                'production_deployed': False}
    dump(output/'protocol.json', protocol)
    print('Prepared:', len(ha), 'HA crops;', len(hard_ids), 'hard/relabelled;', Counter(folds.values()), flush=True)


def build_dataset(root, recipe_name, fold=None, full=False):
    protocol = json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    recipe = RECIPES[recipe_name]
    output = root/'datasets'/(f'full_{recipe_name}' if full else f'{recipe_name}_fold{fold}')
    if output.exists():
        raise FileExistsError('Dataset variant never overwritten')
    manifest = protocol['source_manifest']
    records = []
    for original in manifest['records']:
        r = copy(original)
        origin = r.get('origin')
        source = Path(protocol['source_paths'][r['id']])
        if digest(source) != r['sha256']:
            raise ValueError('Frozen source crop changed')
        r['previous_split'] = r['split']
        r['native_source'] = str(source)
        if origin == 'ha':
            r['split'] = 'train' if full or protocol['fold_by_frame'][r['frame']] != fold else 'val'
        elif origin == 'legacy_train':
            r['split'] = 'train'
        elif origin == 'legacy_val':
            r['split'] = 'val'
        else:
            raise ValueError('Unknown origin')
        r['image'] = f'{r["split"]}/{r["label"]}/{r["id"]}{source.suffix.lower()}'
        records.append(r)
        if full and origin == 'ha':
            # Deliberate resubstitution view ONLY for final fitting checks, never called holdout/test.
            fit = copy(r)
            fit.update({'split': 'val', 'origin': 'ha_fit', 'id': 'fit__'+r['id'],
                        'source_ha_id': r['id'], 'image': f'val/{r["label"]}/fit__{r["id"]}{source.suffix.lower()}'})
            records.append(fit)
    train_hashes = {r['sha256'] for r in records if r['split'] == 'train'}
    overlap = [r for r in records if r['split'] == 'val' and r['sha256'] in train_hashes]
    if (not full and overlap) or (full and any(r['origin'] != 'ha_fit' for r in overlap)):
        raise ValueError('Unintended train/validation overlap')
    train_frames = {r['frame'] for r in records if r.get('origin') == 'ha' and r['split'] == 'train'}
    val_frames = {r['frame'] for r in records if r.get('origin') == 'ha' and r['split'] == 'val'}
    if train_frames & val_frames:
        raise ValueError('Whole frame crosses CV train/validation')
    output.mkdir(parents=True)
    for split in ('train', 'val'):
        for label in LABELS:
            (output/split/label).mkdir(parents=True)
    for r in records:
        target = output/r['image']
        shutil.copy2(r['native_source'], target)
        if digest(target) != r['sha256']:
            raise ValueError('Copy checksum mismatch')
    manifest = dict(manifest, domain='HA_fullfit_fault_replay' if full else 'HA_group_cv_fault_replay',
                    records=records, sampling_recipe=recipe, hard_ids=protocol['hard_ids'],
                    full_ha_fitting=full, fold=fold,
                    counts={s: dict(Counter(r['label'] for r in records if r['split'] == s)) for s in ('train','val')},
                    cv_initialization=protocol['cv_initialization'],
                    historical_test_is_now_training=any(r['origin']=='ha' and r['split']=='train' and
                                                       r['previous_split']=='test' for r in records),
                    old_ha_test_protocol_abandoned=True,
                    metric_population='ha_fit_in_sample' if full else 'ha_group_cv_validation',
                    expected_ha_val_count=2000 if full else 100*len(val_frames),
                    legacy_ng_max=5 if full else 7,
                    final_protocol=protocol['final_protocol'], cross_validation_limitation=protocol['cross_validation_limitation'])
    dump(output/'manifest.json', manifest)
    return output


class FullHAValidator(MacroF1Validator):
    def init_metrics(self, model):
        super().init_metrics(model)
        self.domains = []
        self.manifest = json.loads((Path(self.args.data)/'manifest.json').read_text(encoding='utf-8'))

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)
        self.domains.append(batch['domain'].cpu())

    def get_stats(self):
        result = super().get_stats()
        targets = torch.cat(self.targets).cpu().numpy()
        predictions = torch.cat(self.pred)[:,0].cpu().numpy()
        domains = torch.cat(self.domains).numpy()
        metrics = {name: classification_metrics(targets[domains == i], predictions[domains == i])
                   for i, name in enumerate(('ha', 'legacy'))}
        if metrics['ha']['count'] != self.manifest['expected_ha_val_count'] or metrics['legacy']['count'] != 2770:
            raise ValueError('Unexpected validation population')
        for name, m in metrics.items():
            for key in ('accuracy', 'macro_f1', 'ng_not_detected', 'ng_as_ok', 'ng_as_no_disk', 'no_disk_as_ok'):
                result[f'metrics/{name}_{key}'] = m[key]
        score, eligible = safety_rank(metrics['ha'], metrics['legacy'], self.manifest['legacy_ng_max'])
        result['fitness'], result['metrics/eligible'] = score, float(eligible)
        return result


class FullHATrainer(RuntimeTrainer):
    def get_validator(self):
        self.loss_names = ['loss']
        return FullHAValidator(self.test_loader, self.save_dir, args=copy(self.args), _callbacks=self.callbacks)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        if mode != 'train':
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        if rank != -1:
            raise ValueError('Single GPU only')
        dataset = self.build_dataset(dataset_path, mode)
        manifest = json.loads((Path(self.args.data)/'manifest.json').read_text(encoding='utf-8'))
        records = {str((Path(self.args.data)/r['image']).resolve()): r for r in manifest['records'] if r['split'] == 'train'}
        recipe = manifest['sampling_recipe']
        hard = set(manifest['hard_ids'])
        groups = [records[str(Path(s[0]).resolve())]['origin'] for s in dataset.samples]
        counts = {g: np.bincount([s[1] for s, group in zip(dataset.samples, groups) if group == g], minlength=3)
                  for g in ('ha', 'legacy_train')}
        if any((values == 0).any() for values in counts.values()) or dataset.base.classes != LABELS:
            raise ValueError('Missing class/domain')
        raw = []
        for sample, group in zip(dataset.samples, groups):
            record = records[str(Path(sample[0]).resolve())]
            label = sample[1]
            if group == 'ha':
                weight = recipe['ha_class_factors'][label]/np.sqrt(counts[group][label])
                if record['id'] in hard:
                    weight *= recipe['hard_boost']
            else:
                weight = [.60,.30,.10][label]/counts[group][label]
            raw.append(weight)
        totals = {g: sum(w for w, group in zip(raw, groups) if group == g) for g in counts}
        probability = recipe['ha_probability']
        weights = [w/totals[g]*(probability if g == 'ha' else 1-probability) for w,g in zip(raw,groups)]
        generator = torch.Generator().manual_seed(self.args.seed)
        sampler = WeightedRandomSampler(weights, len(dataset), replacement=True, generator=generator)
        return InfiniteDataLoader(dataset, batch_size=min(batch_size,len(dataset)), sampler=sampler,
                                  num_workers=self.args.workers, pin_memory=True,
                                  worker_init_fn=seed_worker, generator=generator)


def train(data, pretrained, project, name, lr0, epochs, patience):
    if (project/name).exists():
        raise FileExistsError('Training run never overwritten')
    torch.set_num_threads(4)
    model = YOLO(str(pretrained))
    if [model.names[i] for i in range(3)] != LABELS:
        raise ValueError('Class order changed')
    model.train(trainer=FullHATrainer, data=str(data), epochs=epochs, imgsz=224,
                batch=64, workers=2, device=0, seed=1005, deterministic=True,
                optimizer='AdamW', lr0=lr0, lrf=.1, cos_lr=True, weight_decay=.0005,
                warmup_epochs=5, warmup_bias_lr=lr0, patience=patience, dropout=.05,
                amp=False, auto_augment=None, erasing=0, scale=0, fliplr=.5, flipud=0,
                bgr=0, close_mosaic=0, project=str(project), name=name, exist_ok=False,
                plots=True, verbose=False)
    best = model.trainer.best
    onnx_path = YOLO(str(best)).export(format='onnx', dynamic=True, simplify=True, opset=13,
                                     imgsz=224, batch=1, device='cpu')
    print('EXPORT', best, onnx_path, flush=True)


def evaluate(data, model, output, full=False):
    if output.exists():
        raise FileExistsError('Evaluation never overwritten')
    manifest = json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    domains = {'ha_fit' if full else 'ha_cv': [dict(r, path=str(data/r['image'])) for r in manifest['records']
               if r['split'] == 'val' and r.get('origin') == ('ha_fit' if full else 'ha')],
               'legacy_val': [dict(r, path=str(data/r['image'])) for r in manifest['records']
                              if r['split'] == 'val' and r.get('origin') == 'legacy_val']}
    report = {'model': str(model), 'onnx_sha256': digest(model), 'dataset': str(data),
              'full_ha_training': full, 'ha_metrics_are_in_sample': full,
              'production_deployed': False, 'domains': {}}
    rows = []
    for name, records in domains.items():
        metrics, predictions = predict(model, records)
        report['domains'][name] = metrics
        rows.extend(dict(r, domain=name, prediction=p) for r,p in zip(records,predictions))
    score, eligible = safety_rank(report['domains']['ha_fit' if full else 'ha_cv'], report['domains']['legacy_val'],
                                  5 if full else 7)
    report['safety_rank'], report['eligible'] = score, eligible
    report['export_parity'] = torch_onnx_parity(model.with_suffix('.pt'), model, domains['ha_fit' if full else 'ha_cv'])
    output.mkdir(parents=True)
    dump(output/'report.json', report); dump(output/'per_image.json', rows)
    print('EVAL', json.dumps(report['domains']), flush=True)
    return report, rows


def run_child(root, data, pretrained, name, lr0, epochs, patience):
    log = root/'logs'/(name+'.log'); log.parent.mkdir(exist_ok=True)
    command = [sys.executable, '-u', '-B', str(Path(__file__).resolve()), '--phase', 'train',
               '--data', str(data), '--pretrained', str(pretrained), '--root', str(root), '--name', name,
               '--lr0', str(lr0), '--epochs', str(epochs), '--patience', str(patience)]
    print('START', name, 'source checkpoint', pretrained, flush=True)
    with log.open('w', encoding='utf-8') as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
    print('DONE TRAIN', name, flush=True)
    return root/'training'/name/'weights'/'best.onnx'


def run_cv(root, pretrained):
    protocol = json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    protocol['cv_pretrained_pt'] = str(pretrained)
    protocol['cv_pretrained_sha256'] = digest(pretrained)
    # Explicitly reject all previous HA checkpoints as CV initialization.
    if digest(pretrained) != '98fd8825e9c16f5e4ddd5ed96b6370ce09bbd182b0d43663ba819712dde91164':
        raise ValueError('CV must start from the verified pre-HA checkpoint')
    dump(root/'protocol.json', protocol)
    summaries = []
    for recipe in RECIPES:
        ha_rows, legacy_rows, fold_reports = [], [], []
        for fold in range(3):
            data = build_dataset(root, recipe, fold=fold)
            name = f'cv_{recipe}_fold{fold}'
            model = run_child(root, data, pretrained, name, .0001, 50, 10)
            report, rows = evaluate(data, model, root/'evaluations'/name)
            for row in rows:
                row['fold'] = fold
                (ha_rows if row['domain'] == 'ha_cv' else legacy_rows).append(row)
            fold_reports.append(report)
            print('FOLD SUMMARY', name, 'HA NG', report['domains']['ha_cv']['ng_not_detected'],
                  'legacy NG', report['domains']['legacy_val']['ng_not_detected'], flush=True)
        if len(ha_rows) != 2000 or len({r['id'] for r in ha_rows}) != 2000:
            raise ValueError('Every HA crop must be held out exactly once')
        def metrics(rows):
            return classification_metrics([LABELS.index(r['label']) for r in rows],
                                          [LABELS.index(r['prediction']['label']) for r in rows])
        ha_metrics, legacy_metrics = metrics(ha_rows), metrics(legacy_rows)
        # Legacy population is same 2,770 benchmark repeated across three models, NOT 8,310 independent images.
        score, aggregate_eligible = safety_rank(ha_metrics, legacy_metrics, 21, 12)
        eligible = all(r['eligible'] for r in fold_reports)
        summary = {'recipe': recipe, 'ha_group_cv_selection_estimate': ha_metrics,
                   'legacy_repeated_three_models': legacy_metrics, 'folds': fold_reports,
                   'all_folds_eligible': eligible, 'score': score, 'aggregate_eligible': aggregate_eligible,
                   'limitation': protocol['cross_validation_limitation']}
        summaries.append(summary)
        dump(root/(recipe+'_cv_summary.json'), summary)
        dump(root/(recipe+'_ha_oof_predictions.json'), ha_rows)
        print('RECIPE COMPLETE', recipe, json.dumps(ha_metrics), flush=True)
    eligible = [r for r in summaries if r['all_folds_eligible']]
    chosen = max(eligible or summaries, key=lambda r:r['score'])
    dump(root/'cv_selection.json', {'chosen_recipe': chosen['recipe'], 'summaries': summaries,
         'all_folds_retention_gate_satisfied': bool(eligible),
         'selection_is_cross_validation_estimate_not_independent_test': True,
         'initialization_pre_ha': True, 'production_deployed': False})
    print('CV CHOSEN', chosen['recipe'], 'all-fold retention', bool(eligible), flush=True)


def checked_cv_selection(root):
    """Recheck aggregate gates scale with three predictions per legacy image."""
    path = root/'cv_selection_checked.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    original = json.loads((root/'cv_selection.json').read_text(encoding='utf-8'))
    audits = []
    for folder in sorted((root/'datasets').iterdir()):
        manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        if manifest['full_ha_fitting']:
            continue
        records = manifest['records']
        train = [r for r in records if r['split']=='train']
        val = [r for r in records if r['split']=='val']
        if {r['sha256'] for r in train} & {r['sha256'] for r in val}:
            raise ValueError('CV crop overlap found during role audit')
        audits.append({'dataset': str(folder), 'cv_overlap': 0,
                       'former_ha_test_crops_now_in_this_cv_train': sum(r.get('origin')=='ha' and
                            r.get('previous_split')=='test' for r in train),
                       'old_ha_test_protocol_abandoned': True,
                       'note': 'actual roles below supersede early CV metadata boolean; new CV grouping is authoritative'})
    dump(root/'cv_role_audit.json', audits)
    summaries = original['summaries']
    for summary in summaries:
        score, eligible = safety_rank(summary['ha_group_cv_selection_estimate'],
                                     summary['legacy_repeated_three_models'], 21, 12)
        summary['score'], summary['aggregate_eligible'] = score, eligible
    eligible = [r for r in summaries if r['all_folds_eligible']]
    chosen = max(eligible or summaries, key=lambda r:r['score'])
    corrected = dict(original, chosen_recipe=chosen['recipe'], summaries=summaries,
                     aggregate_ng_gate=21, aggregate_empty_gate=12,
                     aggregate_population_is_three_repeated_benchmarks_not_independent_images=True)
    dump(path, corrected)
    return corrected


def run_final(root, original, previous):
    selection = checked_cv_selection(root)
    protocol = json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    if digest(original) != protocol['cv_pretrained_sha256']:
        raise ValueError('Original checkpoint changed')
    if digest(previous) != '210ca39fb60d7c53fbcb2f22ce8e785f8e531c1a95a6dbf781395899644d7002':
        raise ValueError('Preserved pre-retrain checkpoint changed')
    recipe = selection['chosen_recipe']
    data = build_dataset(root, recipe, full=True)
    candidates = []
    # Final full-HA models may warm-start previous weights; CV did not.
    for name, checkpoint, lr in [('full_from_original', original, .0001),
                                 ('full_from_backup', previous, .00003)]:
        model = run_child(root, data, checkpoint, name, lr, 70, 12)
        report, _ = evaluate(data, model, root/'evaluations'/name, full=True)
        candidates.append(report)
    # Compare both full-HA candidates with the preserved previous model on EXACTLY the same known data.
    previous_onnx = previous.with_suffix('.onnx')
    report, _ = evaluate(data, previous_onnx, root/'evaluations'/'retained_previous', full=True)
    candidates.append(report)
    eligible = [r for r in candidates if r['eligible']]
    if not eligible:
        dump(root/'release_decision.json', {'production_deployed': False, 'release_eligible': False,
                                          'reason': 'No model meets fault/retention gates', 'candidates': candidates})
        raise ValueError('No eligible candidate; keep previous backup unchanged')
    chosen = max(eligible, key=lambda r:r['safety_rank'])
    output = root/'selected'; output.mkdir()
    source = Path(chosen['model'])
    target = output/'c_p_ha_all_reviewed_20261005.onnx'
    shutil.copy2(source, target); shutil.copy2(source.with_suffix('.pt'), target.with_suffix('.pt'))
    if digest(target) != chosen['onnx_sha256']:
        raise ValueError('Selected model copy changed')
    dump(root/'release_decision.json', {'production_deployed': False, 'release_eligible': True,
        'chosen_source': str(source), 'selected_onnx': str(target), 'selected_pt': str(target.with_suffix('.pt')),
        'onnx_sha256': digest(target), 'pt_sha256': digest(target.with_suffix('.pt')),
        'all_ha_in_training': source != previous_onnx, 'all_ha_metrics_are_seen_data_not_test': True,
        'selection_scope': 'HA fitting error / legacy retention; independent future capture validation still required',
        'cv_recipe_selection': selection['chosen_recipe'], 'candidates': candidates,
        'previous_backup_unchanged': True})
    print('FINAL CHOSEN', source, json.dumps(chosen['domains']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['prepare','cv','final','train'], required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--previous-predictions', type=Path)
    parser.add_argument('--pretrained', type=Path)
    parser.add_argument('--previous-checkpoint', type=Path)
    parser.add_argument('--name'); parser.add_argument('--lr0', type=float, default=.0001)
    parser.add_argument('--epochs', type=int, default=50); parser.add_argument('--patience', type=int, default=10)
    args = parser.parse_args(); root = args.root.resolve()
    if args.phase == 'prepare':
        prepare(args.data.resolve(), args.previous_predictions.resolve(), root)
    elif args.phase == 'cv':
        run_cv(root, args.pretrained.resolve())
    elif args.phase == 'final':
        run_final(root, args.pretrained.resolve(), args.previous_checkpoint.resolve())
    else:
        train(args.data.resolve(), args.pretrained.resolve(), root/'training', args.name,
              args.lr0, args.epochs, args.patience)


if __name__ == '__main__':
    main()
