"""One fresh YOLO fit on the exact canonical HA/new train/val, then evaluation.

Run datasets are frozen evidence, not label sources. No deployment or label edits.
"""
import argparse
from collections import Counter, defaultdict
from copy import copy, deepcopy
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

PROJECT=next(p for p in Path(__file__).resolve().parents if (p/'SERVICE/training/classifier_dataset.py').is_file())
sys.path.insert(0,str(PROJECT/'SERVICE'))

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from ultralytics import YOLO
from ultralytics.data.build import InfiniteDataLoader, seed_worker

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classifier_dataset import active_dataset, atomic_json, atomic_text, read
from audit_classifier_dataset import audit
from adopt_classifier_dataset import decoded_fingerprint
from classifier_metrics import LABELS, classification_metrics
from classifier_robust_core import metrics_for
from train_classifier_ha import RuntimeTrainer, MacroF1Validator
from train_classifier_supplementary import SupplementDataset, SupplementValidator, digest
from evaluate_classifier_ha import predict, torch_onnx_parity

ORIGINS=('old_ha','new_factory')
POLICY='ha_new_capture_holdout_v1'
BALANCED_POLICY='camera_target_slot_legacy_ng_v1'
BALANCED_WEIGHTS={'old_ha':.4,'new_factory':.4,'old_legacy_train':.2}


def now(): return datetime.now(timezone.utc).isoformat()


def validate_rows(rows,policy_id=POLICY):
    if policy_id not in (POLICY,BALANCED_POLICY):raise ValueError('Unsupported canonical split policy')
    origins=ORIGINS+('old_legacy_train',) if policy_id==BALANCED_POLICY else ORIGINS
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate ID')
    if any(r['origin'] not in origins or r['split'] not in ('train','val') for r in rows):
        raise ValueError('Only canonical HA/new train/val allowed')
    for split in ('train','val'):
        for origin in origins:
            if {r['label'] for r in rows if r['split']==split and r['origin']==origin}!=set(LABELS):
                raise ValueError('Missing origin/class coverage')
    keys=('sha256','decoded_sha256','capture_group','split_group')+(('target_group',) if policy_id==BALANCED_POLICY else ('frame',))
    if policy_id==BALANCED_POLICY:
        for split in ('train','val'):
            for origin in ORIGINS:
                for camera in (1,2):
                    if {r['label'] for r in rows if r['split']==split and r['origin']==origin and r.get('camera')==camera}!=set(LABELS):
                        raise ValueError('Missing source/camera/class coverage')
        for r in rows:
            if r['origin'] in ORIGINS:
                expected=f"{r['origin']}|Cam{r['camera']}|{r['region']}|slot{int(r['slot']):02}"
                if r['target_group']!=expected or not r['frame'].startswith(f"Cam{r['camera']}_"):
                    raise ValueError('Incorrect target/camera provenance')
            elif r.get('camera') is not None:raise ValueError('Legacy camera must remain unknown')
    for key in keys:
        groups=defaultdict(set)
        for r in rows:
            if not r.get(key):raise ValueError('Missing grouping metadata: '+key)
            groups[r[key]].add(r['split'])
        if any(len(s)>1 for s in groups.values()):raise ValueError('Split leakage: '+key)


def sampling_weights(rows,policy_id=POLICY):
    if policy_id==BALANCED_POLICY:
        counts=Counter((r['origin'],r.get('camera'),r['label']) for r in rows)
        expected={(o,cam,c) for o in BALANCED_WEIGHTS for cam in ((1,2) if o in ORIGINS else (None,)) for c in LABELS}
        if set(counts)!=expected:raise ValueError('Missing camera/domain/class in training sampler')
        weights=[]
        for r in rows:
            o,cam,c=r['origin'],r.get('camera'),r['label']
            class_budget=dict(zip(LABELS,(.7,.15,.15) if o=='old_legacy_train' else (.3,.3,.4)))
            weights.append(BALANCED_WEIGHTS[o]*(.5 if o in ORIGINS else 1)*class_budget[c]/counts[o,cam,c])
        return weights
    counts=Counter((r['origin'],r['label']) for r in rows)
    if set(counts)!=set((o,c) for o in ORIGINS for c in LABELS):raise ValueError('Missing training domain/class')
    budget=dict(zip(LABELS,(.3,.3,.4)))
    return [.5*budget[r['label']]/counts[r['origin'],r['label']] for r in rows]


def holdout_fitness(metrics,weights=None):
    # Equal domain weight prevents the larger HA validation burst masking new data.
    # Macro-F1 balances NG precision (false NG) and recall (missed NG).
    weights=weights or dict.fromkeys(ORIGINS,.5)
    return sum(metrics[o]['macro_f1']*w for o,w in weights.items())


def scoped_metrics(rows,predictions):
    metrics=metrics_for(rows,predictions)
    for origin in ORIGINS:
        for camera in (1,2):
            indices=[i for i,r in enumerate(rows) if r['origin']==origin and r.get('camera')==camera]
            if indices:
                metrics[f'{origin}_Cam{camera}']=metrics_for([rows[i] for i in indices],[predictions[i] for i in indices])[origin]
    return metrics


class HoldoutDataset(SupplementDataset):
    def __getitem__(self,index):
        result=super().__getitem__(index)
        r=self.record_by_path[str(Path(self.samples[index][0]).resolve())]
        result['camera']=int(r.get('camera') or 0)
        return result


class HoldoutValidator(SupplementValidator):
    def build_dataset(self,img_path):
        return HoldoutDataset(root=img_path,args=self.args,augment=False,prefix=self.args.split)

    def init_metrics(self,model):
        super().init_metrics(model);self.cameras=[]

    def update_metrics(self,preds,batch):
        super().update_metrics(preds,batch);self.cameras.append(batch['camera'].cpu())

    def get_stats(self):
        result=MacroF1Validator.get_stats(self)
        targets=torch.cat(self.targets).cpu().numpy()
        predictions=torch.cat(self.pred)[:,0].cpu().numpy()
        origins=torch.cat(self.populations).cpu().numpy()
        cameras=torch.cat(self.cameras).cpu().numpy()
        rows=[dict(origin=('new_factory','old_ha','legacy_val','old_legacy_train')[int(o)],label=LABELS[int(t)],camera=int(c) or None)
              for o,t,c in zip(origins,targets,cameras)]
        metrics=scoped_metrics(rows,[dict(label=LABELS[int(p)]) for p in predictions])
        for origin,m in metrics.items():
            for key in ('macro_f1','accuracy','errors','ng_not_detected','false_ng','no_disk_as_ok'):
                result[f'metrics/{origin}_{key}']=m[key]
        result['fitness']=holdout_fitness(metrics,self.manifest.get('fitness_weights'))
        return result


class HoldoutTrainer(RuntimeTrainer):
    def _model_train(self):
        super()._model_train()
        if not hasattr(self,'_freeze_bn_cache'):
            self._freeze_bn_cache=read(Path(self.args.data).parent/'protocol.json').get('freeze_bn',False)
        if self._freeze_bn_cache:freeze_batchnorm_statistics(self.model)

    def build_dataset(self,img_path,mode='train',batch=None):
        return HoldoutDataset(root=img_path,args=self.args,augment=mode=='train',prefix=mode)

    def get_validator(self):
        self.loss_names=['loss']
        return HoldoutValidator(self.test_loader,self.save_dir,args=copy(self.args),_callbacks=self.callbacks)

    def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode='train'):
        if mode!='train':return super().get_dataloader(dataset_path,batch_size,rank,mode)
        if rank!=-1:raise ValueError('Single GPU only')
        dataset=self.build_dataset(dataset_path,mode)
        if dataset.base.classes!=LABELS:raise ValueError('Class order changed')
        rows=[dataset.record_by_path[str(Path(s[0]).resolve())] for s in dataset.samples]
        if any(r['split']!='train' for r in rows):raise ValueError('Validation entered sampler')
        policy=read(Path(self.args.data)/'manifest.json')['split_policy']['id']
        weights=sampling_weights(rows,policy)
        atomic_json(self.save_dir/'sampling.json',dict(training_count=len(rows),
            domain_probabilities={o:sum(w for r,w in zip(rows,weights) if r['origin']==o) for o in sorted({r['origin'] for r in rows})},
            camera_domain_probabilities={f'{o}/Cam{c}':sum(w for r,w in zip(rows,weights) if r['origin']==o and r.get('camera')==c)
                for o,c in sorted({(r['origin'],r.get('camera')) for r in rows})},
            class_probabilities={c:sum(w for r,w in zip(rows,weights) if r['label']==c) for c in LABELS},
            labels_changed=False))
        generator=torch.Generator().manual_seed(self.args.seed)
        sampler=WeightedRandomSampler(weights,len(dataset),replacement=True,generator=generator)
        return InfiniteDataLoader(dataset,batch_size=min(batch_size,len(dataset)),sampler=sampler,
            num_workers=self.args.workers,pin_memory=True,worker_init_fn=seed_worker,generator=generator)


def freeze_batchnorm_statistics(model):
    count=0
    for module in model.modules():
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval();count+=1
    return count


PAGE='''<!doctype html><html lang="vi"><meta charset="utf-8"><title>Train HA + ANH_18_10</title>
<style>body{font:16px system-ui;background:#14202e;color:#e6eef6;margin:28px;max-width:1400px}a{color:#9bd4ff}td,th{border:1px solid #63778d;padding:10px}table{border-collapse:collapse}p{line-height:1.5}.note{background:#294236;padding:16px}</style>
<h1>YOLO11n-cls pretrained — HA + ANH_18_10</h1>
<p>300 epoch tối đa · patience 100 · train 3.386 / val 1.397 · không legacy · giữ nguyên split canonical revision 9.</p>
<p class="note">Best chọn theo trung bình macro-F1 của HA và ANH_18_10 trên validation. Train/val tách theo nhóm chụp; dữ liệu từng dùng ở các lượt trước, không phải kiểm thử mù. ANH_18_10 val có 8 NG. Rà tính nhất quán còn một phần chưa xong. Không tự triển khai model.</p>
<p id="status">Đang tải…</p><p id="error"></p>
<p><a href="protocol.json">Cấu hình và nguồn pretrained</a> · <a href="status.json">Trạng thái</a> · <a href="logs/train.log">Log train</a> · <a href="comparison.html">Đánh giá best</a> · <a href="error_review.html">Ảnh dự đoán sai</a> · <a href="http://127.0.0.1:8791/?view=all&split=val">Dataset validation</a></p>
<script>async function refresh(){try{const s=await(await fetch('status.json?t='+Date.now(),{cache:'no-store'})).json();document.getElementById('status').textContent=s.status+' / '+s.phase+' · epoch '+s.epoch+'/300 · best '+s.best_epoch+' · '+new Date(s.updated_at).toLocaleString('vi-VN',{timeZone:'Asia/Ho_Chi_Minh'});document.getElementById('error').textContent=s.error||'';}catch(e){document.getElementById('error').textContent=e.message}}refresh();setInterval(refresh,10000);</script></html>'''


def publish(root,status,phase,child=None,error=None):
    progress={}
    state=root/'training/fit/early_stopping_state.json'
    if state.exists():progress=read(state)
    atomic_json(root/'status.json',dict(status=status,phase=phase,pid=os.getpid(),active_pid=child,
        epoch=progress.get('epoch',0),best_epoch=progress.get('best_epoch',0),epochs=300,patience=100,
        updated_at=now(),error=error))
    pointer=PROJECT/'.work/ACTIVE_CLASSIFIER_TRAINING.json'
    if pointer.exists():
        active=read(pointer)
        if Path(active['path']).resolve()==root.resolve():
            active.update(status=status,phase=phase,active_pid=child,error=error,
                          campaign_pid=None if status in ('complete','failed','stopped') else os.getpid())
            if status=='complete':
                release=read(root/'release_decision.json')
                active.update(completed_at=now(),recommended_for_review=release['recommended_for_review'],
                    installed_baseline_retained=True)
            atomic_json(pointer,active)


def run_page(protocol):
    if protocol['split_policy']['id']!=BALANCED_POLICY:return PAGE
    page=PAGE.replace('Train HA + ANH_18_10','Train 70/30 — HA + ảnh mới + legacy').replace(
        'YOLO11n-cls pretrained — HA + ANH_18_10','YOLO11n-cls pretrained — hai camera + legacy NG').replace(
        'train 3.386 / val 1.397 · không legacy · giữ nguyên split canonical revision 9.',
        f"train {protocol['train_count']} / val {protocol['val_count']} · legacy 600 (420 NG) · giữ nguyên split 70/30, revision {protocol['canonical_revision']}.").replace(
        'Best chọn theo trung bình macro-F1 của HA và ANH_18_10 trên validation. Train/val tách theo nhóm chụp; dữ liệu từng dùng ở các lượt trước, không phải kiểm thử mù. ANH_18_10 val có 8 NG.',
        'Best chọn theo macro-F1 validation: HA 40%, ANH_18_10 40%, legacy 20%. Cả train và val có đủ hai camera. Lịch sử cùng vị trí rãnh và ảnh trùng không vượt split; frame/phiên chụp có chung khung cảnh. Đây là validation theo vị trí cùng phiên chụp, không phải kiểm thử mù.').replace(
        '<a href="error_review.html">Ảnh dự đoán sai</a>', '<a href="error_review.html">Ảnh dự đoán sai</a> · <a href="white_replay.json">White replay</a>')
    if protocol.get('initialization')=='ha':
        page=page.replace('YOLO11n-cls pretrained — hai camera + legacy NG','Fine-tune HA — hai camera + legacy NG')
        page=page.replace('<p id="status">', '<p>Khởi tạo từ model HA đang cài · lr 1e-5 · toàn bộ weights trainable · giữ nguyên thống kê BatchNorm. HA validation đo giữ kiến thức cũ vì model gốc đã học HA.</p><p id="status">')
    return page


def prepare(root,policy_id=POLICY,initialization='imagenet',pretrained_override=None,protected_runtime=None):
    if initialization not in ('imagenet','ha','retained'):raise ValueError('Unsupported initialization')
    if initialization=='retained' and pretrained_override is None:raise ValueError('Retained initialization requires an explicit checkpoint')
    if initialization=='ha' and policy_id!=BALANCED_POLICY:raise ValueError('HA fine-tune requires current balanced split')
    pool=active_dataset(required_split_policy=policy_id)
    if root.exists():raise FileExistsError(root)
    root.mkdir(parents=True);(root/'logs').mkdir()
    raw=(pool/'manifest.json').read_bytes();(root/'source_manifest_at_training.json').write_bytes(raw)
    m=read(root/'source_manifest_at_training.json');rows=deepcopy(m['records']);validate_rows(rows,policy_id)
    integrity=audit(pool);atomic_json(root/'integrity_before.json',integrity)
    if not integrity['passed']:raise ValueError(integrity['errors'])
    data=root/'input_snapshot'
    for split in ('train','val'):
        for label in LABELS:(data/split/label).mkdir(parents=True)
    for r in rows:
        source=pool/r['image'];target=data/r['image']
        if digest(source)!=r['sha256']:raise ValueError('Canonical crop changed')
        if decoded_fingerprint(source)!=r['decoded_sha256']:raise ValueError('Decoded fingerprint changed')
        shutil.copy2(source,target)
        if digest(target)!=r['sha256']:raise ValueError('Snapshot checksum failed')
    if raw!=(pool/'manifest.json').read_bytes():raise ValueError('Canonical changed while freezing inputs')
    frozen=deepcopy(m);frozen['is_canonical']=False;frozen['immutable_run_snapshot']=True
    if policy_id==BALANCED_POLICY:frozen['fitness_weights']=BALANCED_WEIGHTS
    atomic_json(data/'manifest.json',frozen)
    reference=root/'reference';reference.mkdir()
    if initialization=='ha':
        pretrained=PROJECT/'.work/classifier_fullha_20261005/selected_v2/c_p_ha_all_reviewed_20261005.pt'
        if digest(pretrained)!='d8b82f801195cb9a3f24587f7cc5dac4e36200a1e6a336964bebf855a1835091':
            raise ValueError('HA parent checkpoint changed')
        provenance=dict(path=str(pretrained),sha256=digest(pretrained),classes=3,source='Installed HA parent; prior exposure to HA validation acknowledged')
        target_name='ha_parent.pt'
    else:
        pretrained=Path(pretrained_override) if pretrained_override is not None else PROJECT/'.work/pretrained_ultralytics/yolo11n-cls.pt'
        provenance=read(pretrained.with_suffix('.provenance.json'))
        if digest(pretrained)!=provenance['sha256']:raise ValueError('Official pretrained checksum changed')
        target_name='retained_parent.pt' if initialization=='retained' else 'imagenet.pt'
    shutil.copy2(pretrained,reference/target_name)
    shutil.copy2(PROJECT/'SERVICE/config/models/c_p.onnx',reference/'baseline.onnx')
    protected=protected_runtime if protected_runtime is not None else read(PROJECT/'.work/classifier_supplementary_20261020/protected_runtime_hashes.json')
    if not all(digest(p)==h for p,h in protected.items()):raise ValueError('Runtime hash differs before training')
    atomic_json(root/'protected_runtime_hashes.json',protected)
    protocol=dict(created_at=now(),pool=str(pool),canonical_revision=m['canonical_revision'],
        pool_manifest_sha256=digest(root/'source_manifest_at_training.json'),authorization=m['training_authorization'],
        training_authorized=True,review_incomplete=True,data=str(data),split_policy=m['split_policy'],
        epochs=300,patience=100,seed=1022,batch=64,lr0=.0003,
        pretrained=str(reference/target_name),pretrained_sha256=provenance['sha256'],pretrained_provenance=provenance,
        initialization=initialization,initial_classes=3 if initialization=='ha' else provenance.get('classes',1000),
        freeze_bn=initialization=='ha',dropout=.03,
        baseline_onnx=str(reference/'baseline.onnx'),baseline_sha256=digest(reference/'baseline.onnx'),
        checkpoint_selection='Mean per-origin validation macro-F1, HA 50% / ANH_18_10 50%; training examples never enter fitness.',
        sampling='HA 50% / ANH_18_10 50%; ng 30%, no_disk 30%, ok 40% within each domain.',
        recommendation_policy='Validation only: no increase in NG misses, no_disk-as-OK, false NG or errors in either origin; strict mean macro-F1 improvement. No deployment.',
        production_deployed=False)
    protocol.update(train_count=m['training_pool_size'],val_count=m['evaluation_size'])
    if policy_id==BALANCED_POLICY:
        protocol.update(fitness_weights=BALANCED_WEIGHTS,comparison_domains=list(BALANCED_WEIGHTS)+
            [f'{o}_Cam{c}' for o in ORIGINS for c in (1,2)],
            checkpoint_selection='Weighted validation macro-F1: HA 40%, new 40%, legacy 20%; training rows never enter fitness.',
            sampling='HA 40%, new 40%, legacy 20%; each HA/new camera 50/50 within source, classes 30/30/40; legacy classes NG 70%, no_disk 15%, OK 15%.',
            recommendation_policy='No increase in NG misses, no_disk-as-OK, false NG or errors in any source or HA/new camera; strict weighted macro-F1 improvement. No deployment.')
    if initialization=='ha':
        protocol.update(lr0=.00001,ha_validation_is_retention=True)
        parent_rows=[dict(r,path=str(data/r['image'])) for r in rows]
        atomic_json(root/'reference/parent_export_parity.json',torch_onnx_parity(reference/target_name,reference/'baseline.onnx',parent_rows))
        previous=PROJECT/'.work/classifier_camera_7030_legacy_20261022'
        previous_manifest=read(previous/'source_manifest_at_training.json')
        signature=lambda records:sorted((r['id'],r['label'],r['sha256'],r['split']) for r in records)
        if signature(rows)!=signature(previous_manifest['records']):raise ValueError('Previous ImageNet comparison has different inputs/splits')
        release=read(previous/'release_decision.json');model=Path(release['selected_onnx'])
        if digest(model)!=release['onnx_sha256']:raise ValueError('Previous ImageNet artifact changed')
        shutil.copy2(model,reference/'previous_imagenet.onnx')
        protocol.update(previous_imagenet_onnx=str(reference/'previous_imagenet.onnx'),
            previous_imagenet_sha256=release['onnx_sha256'],previous_imagenet_run=str(previous),
            comparison_inputs_and_splits_identical=True)
    code=root/'code_snapshot';code.mkdir()
    protocol['code_hashes']={}
    for path in Path(__file__).parent.glob('*.py'):
        shutil.copy2(path,code/path.name);protocol['code_hashes'][path.name]=digest(path)
    atomic_json(root/'protocol.json',protocol);atomic_text(root/'index.html',run_page(protocol))
    publish(root,'prepared','canonical inputs frozen')
    return protocol


def train(root,trainer_class=HoldoutTrainer):
    p=read(root/'protocol.json');data=Path(p['data'])
    if (root/'STOP_REQUESTED').exists():raise InterruptedError('STOP_REQUESTED exists')
    if not p['training_authorized']:raise ValueError('Training not authorized')
    validate_rows(read(data/'manifest.json')['records'],p['split_policy']['id'])
    for r in read(data/'manifest.json')['records']:
        if digest(data/r['image'])!=r['sha256']:raise ValueError('Frozen input changed')
    torch.set_num_threads(4)
    folder=root/'training/fit';last=folder/'weights/last.pt';resume=last.exists()
    source=last if resume else Path(p['pretrained'])
    model=YOLO(str(source))
    initial_classes=p.get('initial_classes',1000)
    if not resume and (len(model.names)!=initial_classes or digest(source)!=p['pretrained_sha256']):raise ValueError('Wrong pretrained')
    def initialize(trainer):
        if resume:
            s=read(folder/'early_stopping_state.json')
            if s['epoch']!=trainer.start_epoch:raise ValueError('Resume stopper/checkpoint epoch mismatch')
            trainer.stopper.best_epoch=s['best_epoch'];trainer.stopper.best_fitness=s['best_fitness']
            trainer.stopper.possible_stop=(trainer.start_epoch-s['best_epoch'])>=99
            atomic_json(folder/'resume_verification.json',s)
        else:
            original=YOLO(str(source)).model.float().cpu().state_dict();actual=trainer.model.state_dict()
            shared=[k for k in original if k in actual and original[k].shape==actual[k].shape]
            if len(shared)<100 or any(not torch.equal(original[k],actual[k].detach().float().cpu()) for k in shared):
                raise ValueError('Pretrained transfer mismatch')
            if [trainer.model.names[i] for i in range(3)]!=LABELS:raise ValueError('Wrong class order')
            atomic_json(folder/'initialization.json',dict(source=str(source),sha256=digest(source),
                original_classes=initial_classes,matching_tensors=len(shared),all_parameters_trainable=all(x.requires_grad for x in trainer.model.parameters()),
                classification_head_preserved=initial_classes==3,freeze_bn=p.get('freeze_bn',False)))
        if p.get('freeze_bn'):
            trainer._bn_reference={k:v.detach().cpu().clone() for k,v in trainer.model.state_dict().items()
                                   if k.endswith(('running_mean','running_var','num_batches_tracked'))}
    def save_state(trainer):
        if p.get('freeze_bn'):
            changed=[k for k,v in trainer._bn_reference.items() if not torch.equal(v,trainer.model.state_dict()[k].detach().cpu())]
            if changed:raise ValueError('Frozen BatchNorm buffers changed: '+str(changed))
            atomic_json(folder/'frozen_bn_verification.json',dict(epoch=trainer.epoch+1,
                buffers_checked=len(trainer._bn_reference),changed_buffers=changed,affine_parameters_trainable=True))
        atomic_json(folder/'early_stopping_state.json',dict(epoch=trainer.epoch+1,
            best_epoch=trainer.stopper.best_epoch,best_fitness=trainer.stopper.best_fitness))
        if trainer.fitness==trainer.best_fitness:
            atomic_json(folder/'best_checkpoint_epoch.json',dict(epoch=trainer.epoch+1,fitness=trainer.fitness))
    def stop_requested(trainer):
        if (root/'STOP_REQUESTED').exists():trainer.stop=True
    model.add_callback('on_pretrain_routine_end',initialize)
    model.add_callback('on_model_save',save_state)
    model.add_callback('on_train_epoch_end',stop_requested)
    model.train(trainer=trainer_class,data=str(data),epochs=300,patience=100,imgsz=224,batch=64,workers=2,
        device=0,seed=1022,deterministic=True,pretrained=True,resume=resume,freeze=None,optimizer='AdamW',
        lr0=p['lr0'],lrf=.05,cos_lr=True,weight_decay=.0005,warmup_epochs=3,warmup_bias_lr=p['lr0'],
        dropout=p.get('dropout',.03),amp=False,auto_augment=None,erasing=0,scale=0,fliplr=.5,flipud=0,bgr=0,close_mosaic=0,
        save_period=25,project=str(root/'training'),name='fit',exist_ok=resume,plots=True,verbose=False)
    if (root/'STOP_REQUESTED').exists():raise InterruptedError('User stopped training')
    artifact=root/'artifacts';artifact.mkdir(exist_ok=True)
    clean=YOLO(str(model.trainer.best));clean.model.criterion=None;pt=artifact/'best.pt';clean.save(str(pt))
    onnx=Path(YOLO(str(pt)).export(format='onnx',dynamic=True,simplify=False,opset=13,imgsz=224,batch=1,device='cpu'))
    atomic_json(artifact/'completion.json',dict(best_pt=str(pt),onnx=str(onnx),
        epochs_completed=model.trainer.epoch+1,best_checkpoint=read(folder/'best_checkpoint_epoch.json'),
        strict_improvement_best_epoch=read(folder/'early_stopping_state.json')['best_epoch'],epochs_limit=300,patience=100))


def evaluate_model(path,rows,output):
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        result=read(output)
        if result['model_sha256']!=digest(path):raise ValueError('Cached evaluation/model mismatch')
        return result
    _,predictions=predict(path,rows)
    metrics={}
    for split in ('train','val'):
        indices=[i for i,r in enumerate(rows) if r['split']==split]
        metrics[split]=scoped_metrics([rows[i] for i in indices],[predictions[i] for i in indices])
        metrics[split]['all']=classification_metrics([LABELS.index(rows[i]['label']) for i in indices],
            [LABELS.index(predictions[i]['label']) for i in indices])
    result=dict(model_path=str(path),model_sha256=digest(path),metrics=metrics,
        predictions=[dict(id=r['id'],truth=r['label'],split=r['split'],origin=r['origin'],prediction=p) for r,p in zip(rows,predictions)])
    atomic_json(output,result);return result


def comparison_failures(candidate,baseline,weights=None,domains=None):
    failures=[]
    for origin in (domains or ORIGINS):
        for key in ('ng_not_detected','no_disk_as_ok','false_ng','errors'):
            if candidate[origin][key]>baseline[origin][key]:failures.append(origin+'/'+key)
    if holdout_fitness(candidate,weights)<=holdout_fitness(baseline,weights):failures.append('mean_macro_f1_not_improved')
    return failures


def evaluate(root):
    p=read(root/'protocol.json');data=Path(p['data']);m=read(data/'manifest.json');validate_rows(m['records'],p['split_policy']['id'])
    rows=[dict(r,path=str(data/r['image'])) for r in m['records']]
    for r in rows:
        if digest(r['path'])!=r['sha256']:raise ValueError('Frozen evaluation image changed')
    candidate=root/'artifacts/best.onnx';pt=root/'artifacts/best.pt'
    baseline=evaluate_model(Path(p['baseline_onnx']),rows,root/'evaluations/baseline.json')
    after=evaluate_model(candidate,rows,root/'evaluations/candidate.json')
    previous=None
    if p.get('previous_imagenet_onnx'):
        if digest(p['previous_imagenet_onnx'])!=p['previous_imagenet_sha256']:raise ValueError('Previous model checksum changed')
        previous=evaluate_model(Path(p['previous_imagenet_onnx']),rows,root/'evaluations/previous_imagenet.json')
    val=[r for r,a in zip(rows,after['predictions']) if r['split']=='val' and r['label']!=a['prediction']['label']]
    cases=(val+[r for r in rows if r['split']=='val'])[:32]
    parity=torch_onnx_parity(pt,candidate,cases);atomic_json(root/'artifacts/export_parity.json',parity)
    failures=comparison_failures(after['metrics']['val'],baseline['metrics']['val'],p.get('fitness_weights'),p.get('comparison_domains'))
    release=dict(training_complete=True,production_deployed=False,recommended_for_review=not failures,
        selected_onnx=str(candidate),selected_pt=str(pt),onnx_sha256=digest(candidate),pt_sha256=digest(pt),
        recommendation_failures=failures,baseline=baseline['metrics'],candidate=after['metrics'],
        canonical_revision=p['canonical_revision'],source_manifest_sha256=p['pool_manifest_sha256'],
        checkpoint=read(root/'artifacts/completion.json'),limitations=p['split_policy']['limitations'],
        installed_baseline_retained=True,export_parity=parity,initialization=p.get('initialization','imagenet'))
    if previous:release['previous_imagenet']=previous['metrics']
    atomic_json(root/'release_decision.json',release)
    page=run_page(p).replace('</html>','')+'<h2>Đánh giá best checkpoint</h2><p>Đủ gate đề xuất rà tiếp: '+str(not failures)+'. Chưa triển khai.</p>'
    page+='<p>Gate chưa đạt: '+html.escape(', '.join(failures) or 'Không có')+'</p><table><tr><th>Tập</th><th>Nguồn</th><th>Model</th><th>Accuracy</th><th>Macro-F1</th><th>NG ảo</th><th>Bỏ sót NG</th><th>no_disk → OK</th><th>Tổng lỗi</th></tr>'
    for split in ('val','train'):
        for origin in p.get('comparison_domains',ORIGINS):
            models=[('HA đang dùng',baseline)]
            if previous:models.append(('ImageNet / lượt 70/30 trước',previous))
            models.append(('Fine-tune HA / best' if p.get('initialization')=='ha' else 'Pretrained mới / best',after))
            for name,result in models:
                q=result['metrics'][split][origin]
                page+=f'<tr><td>{split}</td><td>{origin}</td><td>{name}</td><td>{q["accuracy"]:.4%}</td><td>{q["macro_f1"]:.6f}</td><td>{q["false_ng"]}</td><td>{q["ng_not_detected"]}</td><td>{q["no_disk_as_ok"]}</td><td>{q["errors"]}</td></tr>'
    page+='</table><p>Train là dữ liệu đã học. Validation dùng chọn best; không phải test độc lập.</p></html>'
    atomic_text(root/'comparison.html',page)
    cards=[]
    for i,(r,b,a) in enumerate(zip(rows,baseline['predictions'],after['predictions'])):
        prev=previous['predictions'][i] if previous else None
        if r['label']==a['prediction']['label'] and r['label']==b['prediction']['label'] and (prev is None or prev['prediction']['label']==r['label']):continue
        rel='input_snapshot/'+r['image']
        text=f'{r["split"]} / {r["origin"]} / nhãn {r["label"]} / HA cũ {b["prediction"]["label"]}'
        if prev:text+=' / ImageNet '+prev['prediction']['label']
        text+=' / best '+a['prediction']['label']
        cards.append('<article><a href="'+html.escape(rel)+'"><img loading="lazy" src="'+html.escape(rel)+'"></a><p>'+html.escape(text)+'</p><small>'+html.escape(r['id'])+'</small></article>')
    atomic_text(root/'error_review.html','<!doctype html><meta charset="utf-8"><style>body{background:#14202e;color:white;font:15px system-ui}.grid{display:flex;flex-wrap:wrap;gap:14px}article{width:260px;padding:12px;background:#26394b}img{width:100%;height:190px;object-fit:contain}small{overflow-wrap:anywhere}</style><h1>Ảnh khác nhãn — nguồn truth cố định lúc train</h1><p>Train là fitted diagnostics; val dùng chọn checkpoint. Không tự sửa nhãn.</p><div class="grid">'+''.join(cards)+'</div>')


def checked(root,phase,command,logfile):
    env=os.environ.copy();env.update(PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1')
    with logfile.open('a',encoding='utf-8') as handle:
        child=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=handle,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        try:
            while True:
                if (root/'STOP_REQUESTED').exists():raise InterruptedError('User stop marker exists')
                code=child.poll();publish(root,'running',phase,child.pid if code is None else None)
                if code is not None:
                    if code:raise RuntimeError(f'{phase} exited {code}; see {logfile}')
                    return
                time.sleep(10)
        finally:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=20)


def campaign(root,resume=False,policy_id=POLICY,initialization='imagenet'):
    if (root/'STOP_REQUESTED').exists():raise InterruptedError('STOP_REQUESTED exists')
    if resume:
        if not (root/'protocol.json').exists():raise ValueError('No prepared protocol')
    else:prepare(root,policy_id,initialization)
    p=read(root/'protocol.json');code=root/'code_snapshot'
    if any(digest(code/name)!=sha for name,sha in p['code_hashes'].items()):raise ValueError('Frozen code changed')
    if not (root/'artifacts/completion.json').exists():
        checked(root,'training',[sys.executable,'-u','-B',str(code/Path(__file__).name),'--phase','train','--root',str(root)],root/'logs/train.log')
    checked(root,'evaluate best PT/ONNX',[sys.executable,'-u','-B',str(code/Path(__file__).name),'--phase','evaluate','--root',str(root)],root/'logs/evaluation.log')
    if not (root/'white_replay.json').exists():
        wrapper="import sys,runpy;sys.path.append(sys.argv[1]);sys.argv=[sys.argv[2],'--root',sys.argv[3]];runpy.run_path(sys.argv[0],run_name='__main__')"
        checked(root,'White replay 28 images; hardware off',[sys.executable,'-u','-B','-c',wrapper,
            str(PROJECT/'SERVICE/.venv/Lib/site-packages'),str(PROJECT/'.work/classifier_supplementary_20261020/verify_white_replay.py'),str(root)],root/'logs/white_replay.log')
    protected=read(root/'protected_runtime_hashes.json')
    if not all(digest(path)==sha for path,sha in protected.items()):raise ValueError('Runtime modified')
    current=read(Path(p['pool'])/'manifest.json')
    atomic_json(root/'canonical_state_after_training.json',dict(start_revision=p['canonical_revision'],
        current_revision=current['canonical_revision'],frozen_inputs_used=True,runtime_hashes_unchanged=True))
    # Replay appends its result section to index.html; retain that evidence.
    replay_page=(root/'index.html').read_text(encoding='utf-8')
    comparison=(root/'comparison.html').read_text(encoding='utf-8')
    marker='<h2>Chạy lại White trên 28 ảnh thực tế</h2>'
    if marker in replay_page:comparison=comparison.replace('</html>',replay_page[replay_page.index(marker):])
    atomic_text(root/'index.html',comparison)
    publish(root,'complete','best evaluated; White replay complete; no deployment')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('campaign','train','evaluate'),required=True)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--policy',choices=(POLICY,BALANCED_POLICY),default=POLICY)
    parser.add_argument('--initialization',choices=('imagenet','ha'),default='imagenet')
    args=parser.parse_args();root=args.root.resolve()
    try:
        if args.phase=='campaign':campaign(root,args.resume,args.policy,args.initialization)
        elif args.phase=='train':train(root)
        else:evaluate(root)
    except Exception as exc:
        if root.exists():publish(root,'stopped' if isinstance(exc,InterruptedError) else 'failed',args.phase,error=repr(exc))
        raise


if __name__=='__main__':main()
