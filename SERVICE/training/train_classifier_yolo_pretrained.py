"""Train the reviewed factory pool from official ImageNet YOLO11n-cls weights.

Three grouped new-capture CV fits and one final fit; best checkpoints only.
The final checkpoint is selected on legacy validation, with new/HA fitting
metrics evaluated separately afterward. Installed runtime files are never edited.
"""
import argparse
from collections import Counter
from copy import copy
import csv
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_classifier_supplementary import (SupplementTrainer, SupplementValidator, population_metrics,
                                            read, dump, digest, evaluate)
from train_classifier_ha import MacroF1Validator
from classifier_metrics import LABELS
from evaluate_classifier_ha import torch_onnx_parity
from classifier_dataset import active_dataset

PROJECT = Path(__file__).resolve().parents[2]


def validation_fitness(metrics, limits):
    """Use held-out validation only; no fitted new/HA examples enter this score."""
    old = metrics['legacy_val']
    eligible = (old['ng_not_detected'] <= limits['legacy_ng_max'] and
                old['no_disk_as_ok'] <= limits['legacy_empty_max'] and
                old['accuracy'] >= limits['legacy_accuracy_min'])
    score = 1_000_000_000. + (10_000_000_000. if eligible else 0.)
    if 'new_human' in metrics:
        new = metrics['new_human']
        score -= (1_000_000 * new['ng_not_detected'] + 10_000 * old['ng_not_detected'] +
                  1_000 * new['no_disk_as_ok'] + 100 * old['no_disk_as_ok'] +
                  10 * new['false_ng'] + old['false_ng'])
        score += .1 * new['macro_f1']
    else:
        score -= (1_000_000 * old['ng_not_detected'] + 10_000 * old['no_disk_as_ok'] +
                  100 * old['false_ng'] + old['errors'])
        score += .1 * old['macro_f1']
    return score, eligible


class FreshValidator(SupplementValidator):
    def get_stats(self):
        result = MacroF1Validator.get_stats(self)
        targets = torch.cat(self.targets).cpu().numpy()
        predictions = torch.cat(self.pred)[:, 0].cpu().numpy()
        records = [{'origin': ('new_factory', 'old_ha', 'legacy_val', 'old_legacy_train')[int(p)],
                    'label': LABELS[int(t)], 'human_reviewed': bool(h)}
                   for p, t, h in zip(torch.cat(self.populations).numpy(), targets, torch.cat(self.reviewed).numpy())]
        metrics = population_metrics(records, [{'label': LABELS[int(p)]} for p in predictions])
        for name, m in metrics.items():
            for key in ('accuracy', 'errors', 'ng_not_detected', 'no_disk_as_ok', 'false_ng'):
                result[f'metrics/{name}_{key}'] = m[key]
        result['fitness'], eligible = validation_fitness(metrics, self.manifest['limits'])
        result['metrics/eligible'] = float(eligible)
        return result


class FreshTrainer(SupplementTrainer):
    def get_validator(self):
        self.loss_names = ['loss']
        return FreshValidator(self.test_loader, self.save_dir, args=copy(self.args), _callbacks=self.callbacks)


def build_dataset(root, name, fold):
    protocol = read(root / 'protocol.json')
    pool = Path(protocol['pool'])
    output = root / 'datasets' / name
    if output.exists():
        raise FileExistsError(output)
    records = []
    for original in read(pool / 'manifest.json')['records']:
        r = copy(original)
        source = pool / r['image']
        if digest(source) != r['sha256']:
            raise ValueError('Frozen crop changed: ' + r['id'])
        if r['origin'] == 'legacy_val':
            split = 'val'
        elif r['origin'] == 'new_factory':
            r['capture_group'] = protocol['frame_groups'][r['frame']]
            r['fold'] = protocol['group_folds'][str(r['capture_group'])]
            split = 'val' if fold is not None and r['fold'] == fold else 'train'
        else:
            split = 'train'
        r.update(split=split, native_source=str(source), seen_data_retention=False,
                 image=f'{split}/{r["label"]}/{r["id"]}{source.suffix.lower()}')
        records.append(r)
    train = {r['sha256'] for r in records if r['split'] == 'train'}
    if any(r['sha256'] in train for r in records if r['split'] == 'val'):
        raise ValueError('Train/validation exact duplicate')
    a = {r['capture_group'] for r in records if r['origin'] == 'new_factory' and r['split'] == 'train'}
    b = {r['capture_group'] for r in records if r['origin'] == 'new_factory' and r['split'] == 'val'}
    if a & b:
        raise ValueError('A capture group crosses train/validation')
    for split in ('train', 'val'):
        for label in LABELS:
            (output / split / label).mkdir(parents=True)
    for r in records:
        shutil.copy2(r['native_source'], output / r['image'])
    dump(output / 'manifest.json', {'records': records, 'limits': protocol['gates'], 'fold': fold,
          'final_checkpoint_validation': 'legacy_val only' if fold is None else 'held-out new capture groups plus legacy_val',
          'train_validation_hash_overlap': 0, 'fitted_examples_used_for_checkpoint_selection': False})
    return output


def train(root, data, name):
    protocol = read(root / 'protocol.json')
    if not protocol['training_authorized']:
        raise ValueError('Training not authorized')
    if (root / 'training' / name).exists():
        raise FileExistsError(name)
    torch.set_num_threads(4)
    model = YOLO(protocol['pretrained'])
    if len(model.names) != 1000 or digest(protocol['pretrained']) != protocol['pretrained_sha256']:
        raise ValueError('Expected frozen official 1000-class pretrained weights')

    def check_initialization(trainer):
        original = YOLO(protocol['pretrained']).model.float().cpu().state_dict()
        actual = trainer.model.state_dict()
        shared = [k for k in original if k in actual and original[k].shape == actual[k].shape]
        matching = [k for k in shared if torch.equal(original[k], actual[k].detach().float().cpu())]
        if shared != matching or len(shared) < 100:
            raise ValueError('Official pretrained transfer was not exact')
        if [trainer.model.names[i] for i in range(3)] != LABELS:
            raise ValueError('Dataset class order changed')
        dump(trainer.save_dir / 'initialization.json', {'source': protocol['pretrained'],
             'sha256': protocol['pretrained_sha256'], 'original_classes': 1000, 'target_classes': LABELS,
             'matching_transferred_tensors': len(matching), 'all_shared_tensors_match': True,
             'all_parameters_trainable': all(p.requires_grad for p in trainer.model.parameters())})

    def best_epoch(trainer):
        if trainer.fitness == trainer.best_fitness:
            dump(trainer.save_dir / 'best_checkpoint_epoch.json', {'epoch': trainer.epoch + 1, 'fitness': trainer.fitness})

    model.add_callback('on_pretrain_routine_end', check_initialization)
    model.add_callback('on_model_save', best_epoch)
    model.train(trainer=FreshTrainer, data=str(data), epochs=300, patience=100,
                imgsz=224, batch=64, workers=2, device=0, seed=1021, deterministic=True,
                pretrained=True, resume=False, freeze=None, optimizer='AdamW', lr0=.001, lrf=.01,
                cos_lr=True, weight_decay=.0005, warmup_epochs=3, warmup_bias_lr=.001,
                dropout=.05, amp=False, auto_augment=None, erasing=0, scale=0,
                fliplr=.5, flipud=0, bgr=0, close_mosaic=0, save_period=25,
                project=str(root / 'training'), name=name, exist_ok=False, plots=True, verbose=False)
    best = Path(model.trainer.best)
    export = YOLO(str(best)).export(format='onnx', dynamic=True, simplify=False, opset=13,
                                   imgsz=224, batch=1, device='cpu')
    dump(best.parent.parent / 'completion.json', {'epochs_completed': model.trainer.epoch + 1,
         'best': str(best), 'onnx': str(export), 'best_epoch': read(best.parent.parent / 'best_checkpoint_epoch.json'),
         'epochs_limit': 300, 'patience': 100, 'initialization': 'official YOLO11n-cls ImageNet pretrained'})


def publish(root, status, phase, active_pid=None, error=None):
    runs = []
    for name in ('group_cv_0', 'group_cv_1', 'group_cv_2', 'final_all_new'):
        p = root / 'training' / name
        f = p / 'results.csv'
        if f.exists():
            with f.open(encoding='utf-8') as handle:
                rows = [r for r in csv.DictReader(handle) if r.get('epoch', '').isdigit()]
            if rows:
                runs.append({'name': name, 'epoch': int(rows[-1]['epoch']), 'complete': (p / 'completion.json').exists()})
    dump(root / 'status.json', {'status': status, 'phase': phase, 'updated_at': datetime.now(timezone.utc).isoformat(),
         'pid': os.getpid(), 'active_pid': active_pid, 'epochs': 300, 'patience': 100, 'runs': runs, 'error': error})


def process(command, root, phase, log):
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    env = os.environ.copy()
    env.update(PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
    with log.open('w', encoding='utf-8') as handle:
        child = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=handle, stderr=subprocess.STDOUT, creationflags=flags)
        while True:
            code = child.poll()
            publish(root, 'running', phase, child.pid if code is None else None)
            if code is not None:
                if code:
                    raise RuntimeError(f'{phase} exited {code}: {log}')
                return
            time.sleep(15)


PAGE = '''<!doctype html><html lang="vi"><meta charset="utf-8"><title>YOLO pretrained — ANH_18_10 đã rà</title>
<style>body{font:16px system-ui;max-width:1400px;margin:30px;background:#142130;color:#e3edf6}a{color:#8ed1ff}td,th{border:1px solid #64768a;padding:12px}table{border-collapse:collapse}.note{padding:16px;background:#423a24;line-height:1.6}</style>
<h1>Train mới từ YOLO11n-cls pretrained gốc</h1>
<p>300 epoch / patience 100 · toàn bộ mạng được train · lấy best.pt · class ng / no_disk / ok</p>
<p class="note">Dataset sau rà: 5.799 crop train pool; 2.770 legacy validation tách riêng. ANH_18_10 có 2.799 crop được duyệt, 1 crop loại bỏ. Chạy 3 fold theo nhóm lần chụp, rồi train bản cuối. CV dùng ảnh mới giữ ngoài train; best của lượt cuối chọn trên legacy validation. Kết quả trên dữ liệu đã train được ghi riêng. Model đang dùng trên máy chưa bị thay.</p>
<p id="status">Đang tải…</p><div id="runs"></div><p id="error"></p>
<p><a href="http://127.0.0.1:8790/?origin=all_pool">Dataset sau rà</a> · <a href="protocol.json">Protocol / pretrained / checksum</a> · <a href="status.json">Trạng thái</a> · <a href="campaign.log">Log</a></p>
<script>async function refresh(){try{const s=await(await fetch('status.json?t='+Date.now(),{cache:'no-store'})).json();document.getElementById('status').textContent=s.status+' / '+s.phase+' — '+new Date(s.updated_at).toLocaleString('vi-VN');document.getElementById('runs').textContent=s.runs.map(r=>r.name+': '+r.epoch+'/300'+(r.complete?' (xong)':'')).join(' | ');document.getElementById('error').textContent=s.error||'';}catch(e){document.getElementById('error').textContent=e.message}}refresh();setInterval(refresh,10000);</script></html>'''


def report(root, release, oof, baseline_oof):
    page = PAGE.replace('</html>', '')
    page += '<h2>Kết quả best.pt</h2><p>Đủ điều kiện đề xuất rà tiếp: <b>' + str(release['recommended_for_review']) + '</b>. Chưa triển khai.</p>'
    page += '<table><tr><th>Tập đánh giá</th><th>Model</th><th>NG ảo</th><th>Bỏ sót NG</th><th>no_disk thành OK</th><th>Số crop</th></tr>'
    for name, before, after in [('New grouped OOF', baseline_oof['new_human'], oof['new_human'])] + [
            (n, release['baseline']['metrics'][n], release['candidate']['metrics'][n]) for n in ('new_human', 'old_ha', 'legacy_val')]:
        for title, m in [('HA cũ', before), ('YOLO pretrained → best', after)]:
            page += f'<tr><td>{name}</td><td>{title}</td><td>{m["false_ng"]}</td><td>{m["ng_not_detected"]}</td><td>{m["no_disk_as_ok"]}</td><td>{m["count"]}</td></tr>'
    page += '</table><p>New_human / old_ha của bản cuối là dữ liệu đã fit. Grouped OOF dùng checkpoint được chọn trên validation cùng collection đã rà với dự đoán hiển thị, không phải test mù độc lập. Legacy validation đã tham gia chọn checkpoint.</p>'
    page += '<p><a href="release_decision.json">Quyết định và metric đầy đủ</a> · <a href="selected/c_p_yolo_pretrained_reviewed_20261021.onnx">ONNX best</a> · <a href="selected/c_p_yolo_pretrained_reviewed_20261021.pt">PT best</a> · <a href="evaluations/final/predictions.json">Dự đoán từng crop</a></p></html>'
    (root / 'index.html').write_text(page, encoding='utf-8')


def run(pool, root, pretrained):
    pool = active_dataset(pool, required_split_policy='legacy_val_only')
    if root.exists():
        raise FileExistsError('Use a new output folder: ' + str(root))
    source = read(pool / 'manifest.json')
    if not source.get('new_review_completion', {}).get('review_complete'):
        raise ValueError('User-completed review receipt required')
    new = [r for r in source['records'] if r['origin'] == 'new_factory']
    if not all(r['human_reviewed'] for r in new):
        raise ValueError('New review is incomplete')
    root.mkdir(parents=True)
    shutil.copy2(pool / 'manifest.json', root / 'source_manifest_at_training.json')
    (root / 'index.html').write_text(PAGE, encoding='utf-8')
    publish(root, 'running', 'validating frozen inputs')
    try:
        if not all(digest(pool / r['image']) == r['sha256'] for r in source['records']):
            raise ValueError('Input checksum failed')
        protected = read(PROJECT / '.work/classifier_supplementary_20261020/protected_runtime_hashes.json')
        assert all(digest(p) == h for p, h in protected.items()), 'Installed runtime differs'
        dump(root / 'protected_runtime_hashes.json', protected)
        backup = root / 'initialization'; backup.mkdir()
        shutil.copy2(pretrained, backup / pretrained.name)
        if len(YOLO(str(backup / pretrained.name)).names) != 1000:
            raise ValueError('Expected official ImageNet weights, not old HA model')
        frames = sorted({r['frame'] for r in new}, key=lambda f: f.split('Image_')[1])
        groups, previous, frame_groups = [], None, {}
        for frame in frames:
            stamp = datetime.strptime(frame.split('Image_')[1], '%Y%m%d%H%M%S%f')
            if previous is None or (stamp - previous).total_seconds() > 120:
                groups.append([])
            groups[-1].append(frame); frame_groups[frame] = len(groups) - 1; previous = stamp
        counts = Counter(frame_groups[r['frame']] for r in new)
        loads, folds = [0, 0, 0], {}
        for group in sorted(counts, key=lambda g: (-counts[g], g)):
            fold = int(np.argmin(loads)); folds[str(group)] = fold; loads[fold] += counts[group]
        for d in source['exact_duplicate_groups']:
            f = [f for f in d['frames'] if f in frame_groups]
            if len({folds[str(frame_groups[x])] for x in f}) > 1:
                raise ValueError('Duplicate capture crosses folds')
        protocol = {'created_at': datetime.now(timezone.utc).isoformat(), 'pool': str(pool),
                    'pool_manifest_sha256': digest(pool / 'manifest.json'), 'training_authorized': True,
                    'authorization': 'User completed correction-only review and authorized 200-300 epoch training from YOLO pretrained on 2026-10-21.',
                    'pretrained': str(backup / pretrained.name), 'pretrained_sha256': digest(pretrained),
                    'initialization': 'Official Ultralytics YOLO11n-cls ImageNet 1000-class weights; full finetune, 3-class head',
                    'pretrained_url': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-cls.pt',
                    'epochs': 300, 'patience': 100, 'optimizer': 'AdamW', 'lr0': .001,
                    'stability_recipe': False, 'frame_groups': frame_groups, 'groups': groups,
                    'group_folds': folds, 'fold_new_crop_counts': loads, 'class_order': LABELS,
                    'final_checkpoint_selection': 'Legacy validation only, never fitted new/HA metrics',
                    'sampling_domain_probabilities': {'new_factory': .45, 'old_ha': .45, 'old_legacy_train': .10},
                    'production_deployed': False}
        dump(root / 'protocol.json', protocol)
        snapshot = root / 'code_snapshot'; snapshot.mkdir()
        protocol['source_hashes'] = {}
        for p in Path(__file__).resolve().parent.glob('*.py'):
            shutil.copy2(p, snapshot / p.name)
            protocol['source_hashes'][str(p)] = digest(p)
        dump(root / 'protocol.json', protocol)
        # Historical review receipts are provenance; the canonical manifest above is current truth.
        shutil.copy2(pool / 'snapshots/new_decisions.json', root / 'human_decisions_snapshot.json')
        shutil.copy2(pool / 'snapshots/new_review_completion.json', root / 'review_completion_snapshot.json')
        baseline = PROJECT / 'SERVICE/config/models/c_p.onnx'
        reference = root / 'reference'; reference.mkdir(); shutil.copy2(baseline, reference / 'c_p_ha_old.onnx')
        all_records = [dict(r, path=str(pool / r['image'])) for r in source['records'] if r['origin'] != 'old_legacy_train']
        torch.set_num_threads(4)
        before = evaluate(reference / 'c_p_ha_old.onnx', all_records, root / 'evaluations/baseline')
        m = before['metrics']
        gates = {'old_ha_ng_max': m['old_ha']['ng_not_detected'], 'old_ha_empty_max': m['old_ha']['no_disk_as_ok'],
                 'old_ha_errors_max': m['old_ha']['errors'] + 4, 'legacy_ng_max': m['legacy_val']['ng_not_detected'],
                 'legacy_empty_max': m['legacy_val']['no_disk_as_ok'], 'legacy_accuracy_min': max(.987, m['legacy_val']['accuracy'] - .005)}
        protocol['gates'] = gates; dump(root / 'protocol.json', protocol)
        results, oof_rows = [], []
        for fold in (0, 1, 2, None):
            assert all(digest(p) == h for p, h in protocol['source_hashes'].items()), 'Training source changed mid-run'
            name = 'final_all_new' if fold is None else f'group_cv_{fold}'
            publish(root, 'running', 'prepare ' + name)
            data = build_dataset(root, name, fold)
            logs = root / 'logs'; logs.mkdir(exist_ok=True)
            process([sys.executable, '-u', '-B', str(Path(__file__).resolve()), '--phase', 'train', '--root', str(root),
                     '--data', str(data), '--name', name], root, name, logs / (name + '.log'))
            pt = root / 'training' / name / 'weights/best.pt'
            publish(root, 'running', 'evaluate ' + name)
            records = all_records if fold is None else [r for r in all_records if r['origin'] != 'new_factory' or
                      folds[str(frame_groups[r['frame']])] == fold]
            result = evaluate(pt.with_suffix('.onnx'), records, root / 'evaluations' / name, gates)
            results.append(result); dump(root / 'cv_progress.json', results)
            if fold is not None:
                oof_rows.extend(r for r in read(root / 'evaluations' / name / 'predictions.json') if r['origin'] == 'new_factory')
        if len(oof_rows) != len(new) or len({r['id'] for r in oof_rows}) != len(new):
            raise ValueError('OOF coverage incomplete')
        dump(root / 'oof_predictions.json', oof_rows)
        oof = population_metrics(oof_rows, [r['prediction'] for r in oof_rows])
        dump(root / 'oof_metrics.json', oof)
        pt = root / 'training/final_all_new/weights/best.pt'
        after = results[-1]
        predictions = read(root / 'evaluations/final_all_new/predictions.json')
        def margin(r):
            probabilities = sorted(r['prediction']['probabilities'].values(), reverse=True)
            return probabilities[0] - probabilities[1]
        ambiguous = sorted([r for r in predictions if r['origin'] == 'new_factory'], key=margin)[:32]
        by_id = {r['id']: r for r in all_records}
        parity_records = [by_id[r['id']] for r in ambiguous]
        parity = torch_onnx_parity(pt, pt.with_suffix('.onnx'), parity_records)
        selected = root / 'selected'; selected.mkdir()
        artifact = selected / 'c_p_yolo_pretrained_reviewed_20261021.onnx'
        shutil.copy2(pt.with_suffix('.onnx'), artifact); shutil.copy2(pt, artifact.with_suffix('.pt'))
        oldnew, fitnew, cvnew = m['new_human'], after['metrics']['new_human'], oof['new_human']
        recommend = (after['eligible'] and all(x['eligible'] for x in results[:-1]) and
                     cvnew['ng_not_detected'] <= oldnew['ng_not_detected'] and cvnew['false_ng'] < oldnew['false_ng'] and
                     fitnew['ng_not_detected'] <= oldnew['ng_not_detected'] and fitnew['false_ng'] < oldnew['false_ng'])
        release = {'training_complete': True, 'recommended_for_review': recommend, 'production_deployed': False,
                   'selected_onnx': str(artifact), 'selected_pt': str(artifact.with_suffix('.pt')),
                   'onnx_sha256': digest(artifact), 'pt_sha256': digest(artifact.with_suffix('.pt')),
                   'baseline': before, 'candidate': after, 'new_grouped_oof_metrics': oof,
                   'onnx_pt_parity': parity, 'parity_review_sample_ids': [r['id'] for r in ambiguous],
                   'initialization': protocol['initialization'],
                   'scope': 'Final new/HA are fitted-data diagnostics. Grouped CV uses reviewed same-collection validation. Legacy validation selects best. No independent blind test or deployment.'}
        dump(root / 'release_decision.json', release)
        # Provide the final evaluation under a stable link for per-crop review.
        shutil.copytree(root / 'evaluations/final_all_new', root / 'evaluations/final')
        report(root, release, oof, before['metrics'])
        process([str(PROJECT / 'SERVICE/.venv/Scripts/python.exe'), '-u',
                 str(PROJECT / '.work/classifier_supplementary_20261020/verify_white_replay.py'), '--root', str(root)],
                root, 'White integration replay', root / 'white_replay.log')
        assert all(digest(p) == h for p, h in protected.items()), 'Installed runtime changed'
        assert digest(pool / 'manifest.json') == protocol['pool_manifest_sha256'], 'Frozen pool changed'
        publish(root, 'complete', 'Best checkpoint evaluated; runtime unchanged')
    except Exception as exc:
        publish(root, 'failed', 'Training/evaluation stopped', error=str(exc))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('run', 'train'), required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pool', type=Path)
    parser.add_argument('--pretrained', type=Path)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--name')
    args = parser.parse_args()
    if args.phase == 'run':
        run(args.pool.resolve() if args.pool else None, args.root.resolve(), args.pretrained.resolve())
    else:
        train(args.root.resolve(), args.data.resolve(), args.name)
