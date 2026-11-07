"""Autonomous canonical-dataset campaign, immutable run copies, no deployment."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import csv
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from ultralytics import YOLO
from classifier_dataset import active_dataset, atomic_json
from classifier_robust_core import RobustTrainer, checkpoint_fitness, metrics_for, retention_failures
from classifier_metrics import LABELS
from evaluate_classifier_ha import predict, torch_onnx_parity
from train_classifier_supplementary import read, dump, digest

PROJECT = Path(__file__).resolve().parents[2]
TRAINING = Path(__file__).resolve().parent
FLAGS = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
BUDGET = dict(domain_budget=dict(new_factory=.35, old_ha=.30, old_legacy_train=.35),
              class_budget=dict(ng=.30, no_disk=.30, ok=.40))
RECIPES = [dict(BUDGET, id='ha_retention', title='HA: balanced replay + frozen BN + distillation',
                initialization='ha', lr0=.00001, freeze_bn=True, freeze=None, distill=.5, robust=False),
           dict(BUDGET, id='imagenet_balanced', title='YOLO ImageNet: balanced domains and classes',
                initialization='imagenet', lr0=.0003, freeze_bn=False, freeze=None, distill=0., robust=False),
           dict(BUDGET, id='ha_head', title='HA: classification head with balanced replay',
                initialization='ha', lr0=.0001, freeze_bn=True, freeze=10, distill=0., robust=False),
           dict(BUDGET, id='imagenet_robust', title='YOLO ImageNet: robust loss + old-domain teacher',
                initialization='imagenet', lr0=.0001, freeze_bn=False, freeze=None, distill=.3, robust=True)]
PAGE = '''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Classifier — train và đánh giá tự động</title><style>body{font:16px system-ui;background:#14202e;color:#e6eef6;margin:28px;max-width:1550px}a{color:#9bd4ff}p{line-height:1.6}.note{background:#493d29;padding:16px}table{border-collapse:collapse;margin:20px 0;width:100%}td,th{border:1px solid #647991;padding:10px;text-align:left}pre{white-space:pre-wrap}.error{color:#ffb2aa}</style>
<h1>Classifier: so sánh 4 phương pháp và lấy best</h1><p>300 epoch tối đa / patience 100 mỗi lượt; 3 fold theo nhóm lần chụp rồi fit bản cuối. Dùng dataset chuẩn cùng các nhãn sửa mới nhất tại thời điểm bắt đầu.</p>
<p class="note">Người dùng cho phép train khi chưa rà xong toàn bộ nhãn tương tự. Kết quả là độ khớp nhãn hiện có; grouped CV cùng collection đã được dùng phát triển/chọn phương pháp, không phải kiểm tra mù độc lập. Legacy validation được dùng chọn checkpoint. Điểm trên dữ liệu đã train ghi riêng. Không triển khai tự động lên máy.</p>
<p id="status">Đang tải…</p><div id="progress"></div><p id="error" class="error"></p>
<p><a href="http://127.0.0.1:8791/">Dataset chuẩn / sửa nhãn</a> · <a href="protocol.json">Cấu hình và quyền train</a> · <a href="source_manifest_at_training.json">Nhãn chính xác dùng trong lần train này</a> · <a href="status.json">Trạng thái</a> · <a href="campaign.log">Log</a></p><div id="results"></div>
<script>const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));async function refresh(){try{const s=await(await fetch('status.json?t='+Date.now(),{cache:'no-store'})).json();document.getElementById('status').textContent=s.status+' / '+s.phase+' — '+new Date(s.updated_at).toLocaleString('vi-VN',{timeZone:'Asia/Ho_Chi_Minh'});document.getElementById('error').textContent=s.error||'';document.getElementById('progress').innerHTML='<table><tr><th>Phương pháp</th><th>Lượt / epoch</th><th>Kết quả</th></tr>'+s.recipes.map(r=>'<tr><td>'+esc(r.title)+'</td><td>'+r.runs.map(x=>esc(x.name)+': '+x.epoch+'/300'+(x.complete?' (xong)':'')).join('<br>')+'</td><td>'+esc(r.state)+'</td></tr>').join('')+'</table>';if(s.status==='complete')document.getElementById('results').innerHTML='<p><a href="comparison.html">MỞ ĐÁNH GIÁ CUỐI VÀ MODEL ĐƯỢC CHỌN</a></p>';}catch(e){document.getElementById('error').textContent=e.message}}refresh();setInterval(refresh,10000);</script></html>'''


def progress(root):
    rows = []
    for recipe in RECIPES:
        folder = root / recipe['id']; runs = []
        for name in ('group_cv_0', 'group_cv_1', 'group_cv_2', 'final_all_new'):
            path = folder / 'training' / name / 'results.csv'
            if path.exists():
                try:
                    with path.open(encoding='utf-8') as f: values = list(csv.DictReader(f))
                    valid = [v for v in values if v.get('epoch', '').strip().isdigit()]
                    if valid: runs.append(dict(name=name, epoch=int(valid[-1]['epoch']),
                                               complete=(folder / 'artifacts' / name / 'completion.json').exists()))
                except (OSError, ValueError, csv.Error): pass
        state = 'complete' if (folder / 'result.json').exists() else 'pending/running'
        if (folder / 'failure.json').exists(): state = 'failed; see failure.json'
        rows.append(dict(id=recipe['id'], title=recipe['title'], runs=runs, state=state))
    return rows


def publish(root, status, phase, error=None, child=None):
    now = datetime.now(timezone.utc).isoformat()
    atomic_json(root / 'status.json', dict(status=status, phase=phase, error=error,
                 pid=os.getpid(), active_pid=child, updated_at=now, recipes=progress(root)))


def log(root, text):
    with (root / 'campaign.log').open('a', encoding='utf-8') as f:
        f.write(datetime.now(timezone.utc).isoformat() + ' ' + text + '\n')
    print(text, flush=True)


class UnsafeChildState(RuntimeError):
    """Do not start another GPU job when a previous child cannot be reaped."""


def reap_child(child):
    if child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        child.kill()
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired as exc:
            raise UnsafeChildState('Cannot reap GPU child ' + str(child.pid)) from exc


def checked(command, root, phase, logfile):
    env = os.environ.copy(); env.update(PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open('a', encoding='utf-8') as handle:
        child = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=handle, stderr=subprocess.STDOUT, creationflags=FLAGS)
        try:
            while True:
                if (root / 'STOP_REQUESTED').exists():
                    raise InterruptedError('User stop marker exists')
                code = child.poll()
                publish(root, 'running', phase, child=child.pid if code is None else None)
                if code is not None:
                    if code: raise RuntimeError(f'{phase}: exit {code}; {logfile}')
                    return
                time.sleep(10)
        finally:
            # Includes status-write failures: never leave an untracked trainer
            # alive while the campaign advances to another recipe.
            reap_child(child)


def capture_folds(records):
    frames = sorted({r['frame'] for r in records if r['origin'] == 'new_factory'}, key=lambda f:f.split('Image_')[1])
    groups, previous, frame_groups = [], None, {}
    for frame in frames:
        stamp = datetime.strptime(frame.split('Image_')[1], '%Y%m%d%H%M%S%f')
        if previous is None or (stamp - previous).total_seconds() > 120: groups.append([])
        groups[-1].append(frame); frame_groups[frame] = len(groups)-1; previous = stamp
    if len(groups) < 3: raise ValueError('Need at least three independent capture groups')
    counts = Counter(frame_groups[r['frame']] for r in records if r['origin']=='new_factory')
    loads, folds = [0,0,0], {}
    for group in sorted(counts, key=lambda g:(-counts[g],g)):
        fold = min(range(3), key=lambda f:loads[f]); folds[str(group)] = fold; loads[fold] += counts[group]
    duplicate_folds = {}
    for r in records:
        if r['origin'] != 'new_factory': continue
        fold = folds[str(frame_groups[r['frame']])]
        if r['sha256'] in duplicate_folds and duplicate_folds[r['sha256']] != fold:
            raise ValueError('Duplicate new capture crosses folds')
        duplicate_folds[r['sha256']] = fold
    return dict(frame_groups=frame_groups, group_folds=folds, groups=groups, fold_counts=loads)


def prepare(root):
    if root.exists(): raise FileExistsError(root)
    pool = active_dataset(required_split_policy='legacy_val_only')
    root.mkdir(parents=True)
    shutil.copy2(pool / 'manifest.json', root / 'source_manifest_at_training.json')
    m = read(root / 'source_manifest_at_training.json')
    if not m.get('is_canonical'): raise ValueError('Canonical inputs required')
    snapshot = root / 'input_snapshot'
    for r in m['records']:
        source = pool / r['image']; target = snapshot / r['image']
        if digest(source) != r['sha256']: raise ValueError('Canonical crop checksum failed: '+r['id'])
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        if digest(target) != r['sha256']: raise ValueError('Frozen copy checksum failed')
    reference = root / 'reference'; reference.mkdir()
    inputs = {'baseline_onnx': PROJECT/'SERVICE/config/models/c_p.onnx',
              'baseline_pt': PROJECT/'.work/classifier_fullha_20261005/selected_v2/c_p_ha_all_reviewed_20261005.pt',
              'imagenet_pt': PROJECT/'.work/pretrained_ultralytics/yolo11n-cls.pt'}
    paths = {}
    for name, source in inputs.items():
        target = reference / (name + source.suffix); shutil.copy2(source, target); paths[name] = str(target)
    protected = read(PROJECT/'.work/classifier_supplementary_20261020/protected_runtime_hashes.json')
    assert all(digest(Path(p))==h for p,h in protected.items()), 'Installed runtime changed'
    dump(root / 'protected_runtime_hashes.json', protected)
    candidates = read(PROJECT/'.work/ACTIVE_CLASSIFIER_DATASET.json')['pair_candidates']
    pairs = read(Path(candidates))
    changed = {r['id'] for r in m['records'] if r.get('label_history')}
    unresolved = {p[k] for p in pairs for k in ('a','b') if p['a'] not in changed and p['b'] not in changed}
    records = [dict(r, unresolved_similarity_candidate=r['id'] in unresolved) for r in m['records']]
    dump(root / 'frozen_records.json', records)
    protocol = dict(schema_version=1, created_at=datetime.now(timezone.utc).isoformat(),
                    pool=str(pool), frozen_pool=str(snapshot), canonical_revision=m['canonical_revision'],
                    pool_manifest_sha256=digest(root/'source_manifest_at_training.json'),
                    training_authorized=True, authorization=m.get('training_authorization'),
                    incomplete_consistency_review=True, canonical_labels_changed_by_training=False,
                    epochs=300, patience=100, recipes=RECIPES, class_order=LABELS,
                    source_hashes={}, production_deployed=False, **paths, **capture_folds(records))
    code = root/'code_snapshot'; code.mkdir()
    for path in TRAINING.glob('*.py'):
        shutil.copy2(path, code/path.name); protocol['source_hashes'][str(path)] = digest(path)
    dump(root/'protocol.json', protocol)
    (root/'index.html').write_text(PAGE, encoding='utf-8')
    publish(root, 'running', 'inputs frozen; baseline evaluation')
    return protocol


def evaluation(model, records, folder):
    if (folder/'result.json').exists():
        result = read(folder/'result.json')
        assert result['model_sha256']==digest(model)
        return result
    _, outputs = predict(model, records)
    result = dict(model=str(model), model_sha256=digest(model), metrics=metrics_for(records, outputs),
                  predictions=[dict(id=r['id'], label=r['label'], origin=r['origin'], prediction=p)
                               for r,p in zip(records,outputs)])
    dump(folder/'result.json', result)
    return result


def subset_metrics(predictions, ids):
    selected = [r for r in predictions if r['id'] in ids]
    return metrics_for(selected, [r['prediction'] for r in selected])


def build_dataset(recipe_root, name, fold):
    p = read(recipe_root/'protocol.json'); root = Path(p['campaign_root'])
    output = recipe_root/'datasets'/name
    if (output/'manifest.json').exists(): return output
    records = []
    for original in read(root/'frozen_records.json'):
        r = deepcopy(original)
        if r['origin']=='legacy_val': split='val'
        elif r['origin']=='new_factory':
            r['capture_group'] = p['frame_groups'][r['frame']]
            r['fold'] = p['group_folds'][str(r['capture_group'])]
            split = 'val' if fold is not None and r['fold']==fold else 'train'
        else: split='train'
        source = Path(p['frozen_pool'])/r['image']
        if digest(source)!=r['sha256']: raise ValueError('Frozen image changed')
        r.update(split=split, native_source=str(source), image=f'{split}/{r["label"]}/{r["id"]}{source.suffix.lower()}')
        records.append(r)
    train_hashes = {r['sha256'] for r in records if r['split']=='train'}
    if any(r['sha256'] in train_hashes for r in records if r['split']=='val'):
        raise ValueError('Exact duplicate crosses train/validation')
    for split in ('train','val'):
        for label in LABELS: (output/split/label).mkdir(parents=True,exist_ok=True)
    for r in records:
        target = output/r['image']
        if not target.exists(): os.link(r['native_source'], target)
    validation_ids = {r['id'] for r in records if r['split']=='val'}
    baseline = read(root/'evaluations/baseline/result.json')
    dump(output/'manifest.json', dict(records=records, baseline_validation=subset_metrics(baseline['predictions'],validation_ids),
               fold=fold, train_validation_exact_hash_overlap=0, fitted_records_used_for_checkpoint_selection=False))
    return output


def restore_stopper(trainer):
    """Keep patience across a controlled resume (Ultralytics resets its stopper)."""
    folder=Path(trainer.save_dir)
    saved=folder/'early_stopping_state.json'
    state=read(saved) if saved.exists() else None
    if state and state['epoch']<=trainer.start_epoch:
        source='saved_stopper_state'
        best_epoch=state['best_epoch']; best_fitness=state['best_fitness']
    else:
        best_fitness=float(trainer.best_fitness)
        baseline=read(Path(trainer.args.data)/'manifest.json')['baseline_validation']
        matches=[]
        with (folder/'results.csv').open(encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                epoch=int(row['epoch'])
                if epoch>trainer.start_epoch: continue
                metrics=deepcopy(baseline)
                for domain,m in metrics.items():
                    for key in ('accuracy','macro_f1','errors','ng_not_detected','no_disk_as_ok','false_ng'):
                        m[key]=float(row[f'metrics/{domain}_{key}'])
                # The validator rounds returned fitness to five decimals.
                if round(checkpoint_fitness(metrics,baseline),5)==round(best_fitness,5):
                    matches.append(epoch)
        if matches:
            best_epoch=min(matches); source='earliest_matching_validation_fitness'
        else:
            best_epoch=min(read(folder/'best_checkpoint_epoch.json')['epoch'],trainer.start_epoch)
            source='best_checkpoint_epoch_fallback'
    trainer.stopper.best_epoch=best_epoch
    trainer.stopper.best_fitness=best_fitness
    trainer.stopper.possible_stop=(trainer.start_epoch-best_epoch)>=trainer.stopper.patience-1
    result=dict(resume_start_epoch=trainer.start_epoch+1,best_epoch=best_epoch,best_fitness=best_fitness,
                patience=trainer.stopper.patience,restoration_source=source,epochs_limit=trainer.epochs)
    dump(folder/'resume_stopper_verification.json',result)
    return result


def train(recipe_root, name, smoke=False):
    p = read(recipe_root/'protocol.json'); recipe=p['active_recipe']; data=recipe_root/'datasets'/name
    torch.set_num_threads(4)
    artifact = recipe_root/'artifacts'/name
    if (artifact/'completion.json').exists(): return
    last = recipe_root/'training'/name/'weights/last.pt'
    source = last if last.exists() else Path(p['baseline_pt'] if recipe['initialization']=='ha' else p['imagenet_pt'])
    model = YOLO(str(source))
    if not last.exists() and len(model.names)!=(3 if recipe['initialization']=='ha' else 1000):
        raise ValueError('Unexpected initialization class count')
    def verify_initialization(trainer):
        if last.exists():
            restore_stopper(trainer)
            return
        original=YOLO(str(source)).model.float().cpu().state_dict(); actual=trainer.model.state_dict()
        shared=[k for k in original if k in actual and original[k].shape==actual[k].shape]
        matching=[k for k in shared if torch.equal(original[k],actual[k].detach().float().cpu())]
        if shared!=matching or len(shared)<100: raise ValueError('Pretrained tensor transfer mismatch')
        dump(trainer.save_dir/'initialization.json',dict(source=str(source),sha256=digest(source),
             original_classes=len(YOLO(str(source)).names),matching_tensors=len(matching),target_classes=LABELS))
    def best_epoch(trainer):
        dump(trainer.save_dir/'early_stopping_state.json',dict(epoch=trainer.epoch+1,
             best_epoch=trainer.stopper.best_epoch,best_fitness=trainer.stopper.best_fitness))
        if trainer.fitness==trainer.best_fitness:
            dump(trainer.save_dir/'best_checkpoint_epoch.json',dict(epoch=trainer.epoch+1,fitness=trainer.fitness))
    model.add_callback('on_pretrain_routine_end',verify_initialization)
    model.add_callback('on_model_save',best_epoch)
    model.train(trainer=RobustTrainer,data=str(data),epochs=2 if smoke else 300,patience=100,
        imgsz=224,batch=64,workers=2,device=0,seed=1022,deterministic=True,
        pretrained=True,resume=bool(last.exists()),freeze=recipe['freeze'],optimizer='AdamW',
        lr0=recipe['lr0'],lrf=.05,cos_lr=True,weight_decay=.0005,warmup_epochs=3,
        warmup_bias_lr=recipe['lr0'],dropout=.03,amp=False,auto_augment=None,erasing=0,
        scale=0,fliplr=.5,flipud=0,bgr=0,close_mosaic=0,save_period=50,
        project=str(recipe_root/'training'),name=name,exist_ok=last.exists(),plots=True,verbose=False)
    raw_best=Path(model.trainer.best); clean=YOLO(str(raw_best))
    # Standard model class + no custom loss dependency in the portable PT artifact.
    clean.model.criterion=None
    artifact.mkdir(parents=True,exist_ok=True); pt=artifact/'best.pt'; clean.save(str(pt))
    onnx=Path(YOLO(str(pt)).export(format='onnx',dynamic=True,simplify=False,opset=13,imgsz=224,batch=1,device='cpu'))
    dump(artifact/'completion.json',dict(best_pt=str(pt),onnx=str(onnx),epochs_completed=model.trainer.epoch+1,
         best_epoch=read(raw_best.parent.parent/'best_checkpoint_epoch.json'),epochs_limit=2 if smoke else 300,patience=100))


def blended(recipe_root, name, fraction):
    raw=recipe_root/'artifacts'/name/'best.pt'
    if fraction==1: return raw
    from blend_classifier_fullha import interpolate
    out=recipe_root/'artifacts'/name/f'blend_{int(fraction*100):02}'/'best.pt'
    if out.with_suffix('.onnx').exists(): return out
    p=read(recipe_root/'protocol.json')
    warm=YOLO(p['baseline_pt']); cold=YOLO(str(raw))
    warm.model=warm.model.cpu().float().eval(); cold.model=cold.model.cpu().float().eval()
    if warm.names!=cold.names: raise ValueError('Blend class mismatch')
    warm.model.load_state_dict(interpolate(warm.model.state_dict(),cold.model.state_dict(),fraction),strict=True)
    warm.model.criterion=None; out.parent.mkdir(parents=True,exist_ok=True); warm.save(str(out))
    YOLO(str(out)).export(format='onnx',dynamic=True,simplify=False,opset=13,imgsz=224,batch=1,device='cpu')
    return out


def new_rank(metrics):
    m=metrics['new_factory']
    return (m['ng_not_detected'],m['false_ng'],m['no_disk_as_ok'],m['errors'])


def recipe_campaign(root, recipe, records, baseline):
    folder=root/recipe['id']; folder.mkdir(exist_ok=True)
    protocol=dict(read(root/'protocol.json'),active_recipe=recipe,campaign_root=str(root),stability_recipe=False)
    dump(folder/'protocol.json',protocol)
    fractions=(.25,.5,.75,1.) if recipe['initialization']=='ha' else (1.,)
    evaluations={fraction:[] for fraction in fractions}
    for fold in range(3):
        name=f'group_cv_{fold}'; data=build_dataset(folder,name,fold)
        checked([sys.executable,'-u','-B',str(Path(__file__).resolve()),'--phase','train','--root',str(folder),'--name',name],
                root,f'{recipe["id"]}/{name}',folder/'logs'/f'{name}.log')
        val_new_ids={r['id'] for r in read(data/'manifest.json')['records'] if r['origin']=='new_factory' and r['split']=='val'}
        selected_records=[r for r in records if r['origin']!='new_factory' or r['id'] in val_new_ids]
        for fraction in fractions:
            publish(root,'running',f'{recipe["id"]}/{name}: evaluate fraction {fraction}')
            pt=blended(folder,name,fraction)
            result=evaluation(pt.with_suffix('.onnx'),selected_records,folder/'evaluations'/name/f'f{fraction}')
            ref=subset_metrics(baseline['predictions'],{r['id'] for r in selected_records})
            evaluations[fraction].append(dict(metrics=result['metrics'],baseline=ref,
                  predictions=[r for r in result['predictions'] if r['origin']=='new_factory']))
    summaries=[]
    for fraction, fits in evaluations.items():
        oof=[r for fit in fits for r in fit['predictions']]
        if len({r['id'] for r in oof})!=len(oof): raise ValueError('OOF duplicate record')
        metrics=metrics_for(oof,[r['prediction'] for r in oof])
        reference=subset_metrics(baseline['predictions'],{r['id'] for r in oof})
        failures=retention_failures(metrics,reference)
        for i,fit in enumerate(fits):
            failures += [f'fold_{i}/'+x for x in retention_failures(fit['metrics'],fit['baseline'],include_new=False)]
        summaries.append(dict(fraction=fraction,metrics=metrics,failures=failures,eligible=not failures,
                              mean_fitness=sum(checkpoint_fitness(f['metrics'],f['baseline']) for f in fits)/3))
        dump(folder/'evaluations'/f'oof_f{fraction}.json',dict(metrics=metrics,predictions=oof))
    eligible=[s for s in summaries if s['eligible']]
    chosen=min(eligible,key=lambda s:new_rank(s['metrics'])) if eligible else max(summaries,key=lambda s:s['mean_fitness'])
    dump(folder/'cv_selection.json',dict(selection_locked_before_final_training=True,chosen_fraction=chosen['fraction'],summaries=summaries))
    name='final_all_new'; build_dataset(folder,name,None)
    checked([sys.executable,'-u','-B',str(Path(__file__).resolve()),'--phase','train','--root',str(folder),'--name',name],
            root,recipe['id']+'/'+name,folder/'logs'/f'{name}.log')
    pt=blended(folder,name,chosen['fraction'])
    final=evaluation(pt.with_suffix('.onnx'),records,folder/'evaluations'/'final')
    failures=chosen['failures']+retention_failures(final['metrics'],baseline['metrics'])
    improvement=new_rank(chosen['metrics'])<new_rank(baseline['metrics'])
    # A lower NG-miss count alone cannot justify increasing false NG (already gated).
    result=dict(recipe=recipe['id'],title=recipe['title'],fraction=chosen['fraction'],
                oof=chosen['metrics'],candidate=final['metrics'],failures=failures,
                recommended=not failures and improvement,pt=str(pt),onnx=str(pt.with_suffix('.onnx')),
                oof_file=str(folder/'evaluations'/f'oof_f{chosen["fraction"]}.json'),
                final_predictions=str(folder/'evaluations/final/result.json'))
    dump(folder/'result.json',result)
    return result


def report(root, release):
    rows=[]
    for result in release['comparison']:
        for cohort,m in [('New grouped CV (development)',result['oof']['new_factory']),
                         ('New fitted',result['candidate']['new_factory']),
                         ('HA fitted retention',result['candidate']['old_ha']),
                         ('Legacy train fitted retention',result['candidate']['old_legacy_train']),
                         ('Legacy checkpoint validation',result['candidate']['legacy_val'])]:
            rows.append(f'<tr><td>{html.escape(result["recipe"])}</td><td>{cohort}</td><td>{m["count"]}</td><td>{m["false_ng"]}</td><td>{m["ng_not_detected"]}</td><td>{m["no_disk_as_ok"]}</td><td>{m["errors"]}</td></tr>')
    for cohort,m in release['baseline'].items():
        rows.append(f'<tr><td>Baseline HA đang dùng</td><td>{cohort}</td><td>{m["count"]}</td><td>{m["false_ng"]}</td><td>{m["ng_not_detected"]}</td><td>{m["no_disk_as_ok"]}</td><td>{m["errors"]}</td></tr>')
    page=PAGE.split('<script>')[0]
    page+=f'<h2>Đã chọn: {html.escape(release["selected_recipe"])}</h2><p>Đủ điều kiện đề xuất bản mới: {release["recommended_for_review"]}. Chưa triển khai.</p>'
    page+='<table><tr><th>Model</th><th>Tập / vai trò</th><th>Crop</th><th>NG ảo</th><th>Bỏ sót NG</th><th>no_disk→OK</th><th>Tổng sai</th></tr>'+''.join(rows)+'</table>'
    page+='<p><a href="release_decision.json">Quyết định, lỗi giữ kiến thức và metric</a> · <a href="error_review.html">Ảnh sai / đã sửa được</a> · <a href="selected/best.pt">best.pt</a> · <a href="selected/best.onnx">best.onnx</a></p></html>'
    (root/'comparison.html').write_text(page,encoding='utf-8')


def error_review(root, selected, baseline, records):
    before={r['id']:r for r in baseline['predictions']}; by_id={r['id']:r for r in records}
    rows=[]
    for r in selected['predictions']:
        old=before[r['id']]; record=by_id[r['id']]
        if r['prediction']['label']==r['label'] and old['prediction']['label']==r['label']: continue
        source=Path(record['path']).relative_to(root).as_posix()
        tag='sai sau train' if r['prediction']['label']!=r['label'] else 'đã sửa được'
        rows.append(f'<article><b>{html.escape(r["id"])} · {tag}</b><p>Nhãn snapshot: {r["label"]}; cũ: {old["prediction"]["label"]}; mới: {r["prediction"]["label"]} ({r["prediction"]["confidence"]:.3f})</p><a href="{source}"><img loading="lazy" src="{source}"></a></article>')
    page='<!doctype html><meta charset="utf-8"><title>Ảnh sai / sửa được</title><style>body{font:16px system-ui;background:#14202e;color:white;margin:25px}article{display:inline-block;vertical-align:top;width:420px;padding:14px;border:1px solid #617991;margin:10px;overflow-wrap:anywhere}img{width:100%;height:220px;object-fit:contain}</style><h1>Đối chiếu model đã chọn với HA cũ</h1><p>Nhãn được đóng băng lúc bắt đầu train; chưa rà hết tính nhất quán. New / HA của bản cuối là dữ liệu đã train. Xem grouped CV trong báo cáo để đánh giá phát triển phương pháp.</p>'+''.join(rows)
    (root/'error_review.html').write_text(page,encoding='utf-8')


def campaign(root, resume=False):
    if (root/'STOP_REQUESTED').exists(): raise InterruptedError('Stop marker; do not restart')
    if resume:
        protocol=read(root/'protocol.json')
        if not protocol['training_authorized']: raise ValueError('Original run was not authorized')
    else: protocol=prepare(root)
    records=[dict(r,path=str(Path(protocol['frozen_pool'])/r['image'])) for r in read(root/'frozen_records.json')]
    baseline=evaluation(Path(protocol['baseline_onnx']),records,root/'evaluations/baseline')
    baseline_by_id={r['id']:r for r in baseline['predictions']}
    parity_records=sorted(records,key=lambda r:baseline_by_id[r['id']]['prediction']['label']==r['label'])[:32]
    dump(root/'reference/pt_onnx_parity.json',torch_onnx_parity(Path(protocol['baseline_pt']),Path(protocol['baseline_onnx']),parity_records))
    results=[]
    for recipe in RECIPES:
        folder=root/recipe['id']
        if (folder/'result.json').exists():
            results.append(read(folder/'result.json')); continue
        try:
            log(root,'START '+recipe['id'])
            result=recipe_campaign(root,recipe,records,baseline); results.append(result)
            if (folder/'failure.json').exists(): (folder/'failure.json').unlink()
            log(root,'DONE '+recipe['id']+' '+json.dumps(dict(recommended=result['recommended'],failures=result['failures'])))
        except (InterruptedError, UnsafeChildState): raise
        except Exception as e:
            dump(folder/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),time=datetime.now(timezone.utc).isoformat()))
            log(root,'FAILED '+recipe['id']+' '+repr(e))
    if len(results)!=len(RECIPES):
        raise RuntimeError('Some recipes failed; successful results retained. Repair and resume incomplete jobs.')
    eligible=[r for r in results if r['recommended']]
    chosen=min(eligible,key=lambda r:new_rank(r['oof'])) if eligible else None
    source_pt=Path(chosen['pt'] if chosen else protocol['baseline_pt'])
    source_onnx=Path(chosen['onnx'] if chosen else protocol['baseline_onnx'])
    selected=root/'selected'; selected.mkdir(exist_ok=True)
    shutil.copy2(source_pt,selected/'best.pt'); shutil.copy2(source_onnx,selected/'best.onnx')
    evaluated=evaluation(selected/'best.onnx',records,root/'evaluations/selected')
    hard=[by['id'] for by in evaluated['predictions'] if by['label']!=by['prediction']['label']]
    cases=sorted(records,key=lambda r:r['id'] not in hard)[:32]
    dump(selected/'export_parity.json',torch_onnx_parity(selected/'best.pt',selected/'best.onnx',cases))
    release=dict(training_complete=True,production_deployed=False,recommended_for_review=bool(chosen),
        selected_recipe=chosen['recipe'] if chosen else 'baseline_retained_no_candidate_passed',
        selected_onnx=str(selected/'best.onnx'),selected_pt=str(selected/'best.pt'),
        onnx_sha256=digest(selected/'best.onnx'),pt_sha256=digest(selected/'best.pt'),
        baseline=baseline['metrics'],candidate=evaluated['metrics'],comparison=results,
        canonical_revision=protocol['canonical_revision'],source_manifest_sha256=protocol['pool_manifest_sha256'],
        review_limitation='User explicitly deferred remaining consistency review; metrics use current labels, not independent blind GT.',
        selection='Grouped new-capture development CV and predeclared old/legacy/new safety-retention gates; no final fitted-error ranking.')
    dump(root/'release_decision.json',release); report(root,release); error_review(root,evaluated,baseline,records)
    # Hardware-off source integration replay; external script resolves the correct project root.
    wrapper="import sys,runpy;sys.path.append(sys.argv[1]);sys.argv=[sys.argv[2],'--root',sys.argv[3]];runpy.run_path(sys.argv[0],run_name='__main__')"
    checked([sys.executable,'-u','-B','-c',wrapper,str(PROJECT/'SERVICE/.venv/Lib/site-packages'),
             str(PROJECT/'.work/classifier_supplementary_20261020/verify_white_replay.py'),str(root)],
            root,'28 White images: source integration replay',root/'logs/white_replay.log')
    protected=read(root/'protected_runtime_hashes.json')
    assert all(digest(Path(p))==h for p,h in protected.items()), 'Installed runtime changed'
    m=read(Path(protocol['pool'])/'manifest.json')
    dump(root/'canonical_state_after_training.json',dict(start_revision=protocol['canonical_revision'],
        current_revision=m['canonical_revision'],canonical_changed_during_run=m['canonical_revision']!=protocol['canonical_revision'],
        training_used_frozen_inputs=True))
    publish(root,'complete','evaluation and White replay complete; no deployment')
    log(root,'CAMPAIGN COMPLETE '+release['selected_recipe'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['campaign','train'],required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--name'); parser.add_argument('--resume',action='store_true'); parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args(); root=args.root.resolve()
    if args.phase=='train': train(root,args.name,args.smoke)
    else:
        try: campaign(root,args.resume)
        except Exception as e:
            if root.exists():
                publish(root,'stopped' if isinstance(e,InterruptedError) else 'failed',repr(e),error=repr(e))
                log(root,'CAMPAIGN EXIT '+repr(e))
            raise


if __name__=='__main__': main()
