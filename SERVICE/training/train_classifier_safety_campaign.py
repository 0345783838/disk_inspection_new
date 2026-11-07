"""Revision-bound NG-priority experiments, frozen inputs, no deployment."""
import argparse
from collections import Counter
from copy import copy, deepcopy
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

PROJECT = next(p for p in Path(__file__).resolve().parents if (p/'SERVICE/training/classifier_dataset.py').is_file())
sys.path.insert(0, str(PROJECT/'SERVICE'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from ultralytics import YOLO
from ultralytics.data.build import InfiniteDataLoader, seed_worker
import train_classifier_canonical_holdout as base
from classifier_dataset import CanonicalDataset, active_dataset, atomic_json, atomic_text, read
from train_classifier_ha import runtime_tensor, safe_augment
from train_classifier_supplementary import digest
from blend_classifier_fullha import interpolate

cv2.setNumThreads(1)
FLAGS = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
RECIPES = [
    dict(id='fc_anchor', scope='linear', lr0=3e-5, kd=4., ai=False),
    dict(id='late_anchor', scope='late', lr0=5e-6, kd=4., ai=False),
    dict(id='full_anchor', scope='full', lr0=3e-6, kd=4., ai=False),
    dict(id='full_anchor_ai', scope='full', lr0=3e-6, kd=4., ai=True),
]
DOMAIN_BUDGET = dict(old_ha=.4, new_factory=.45, old_legacy_train=.15)
CLASS_BUDGET = dict(old_ha=(.35,.25,.4), new_factory=(.4,.25,.35), old_legacy_train=(.7,.15,.15))
USER_STATEMENT = 'thôi được rồi giờ vận dụng hết khả năng của mày, làm thế nào tăng độ chính xác mô hình cho tao, ưu tiên không lọt lỗi và ít lỗi ảo nhất. Mày có thể tự tạo thêm data bằng cách dùng AI gen ảnh, tự augment, rồi train, tối ưu, mai tao kiểm tra kết quả, đừng làm tao thất vọng'


def totals(metrics):
    return {k:sum(metrics[o][k] for o in DOMAIN_BUDGET) for k in ('errors','false_ng','ng_not_detected','no_disk_as_ok')}


def safety_fitness(metrics, baseline):
    """Baseline NG regressions, total NG misses, then false NG dominate."""
    excess = sum(max(0,metrics[d]['ng_not_detected']-baseline[d]['ng_not_detected']) for d in baseline if d in metrics and d != 'all')
    t = totals(metrics)
    return -1e10*excess - 1e7*t['ng_not_detected'] - 1000*t['false_ng'] - t['no_disk_as_ok'] - .001*t['errors']


def illumination_profile(source, generated):
    """Only a smooth bounded intensity field is extracted; no generated edges."""
    a = cv2.resize(cv2.cvtColor(source,cv2.COLOR_BGR2GRAY),(8,8),interpolation=cv2.INTER_AREA).astype(np.float32)
    b = cv2.resize(cv2.cvtColor(generated,cv2.COLOR_BGR2GRAY),(8,8),interpolation=cv2.INTER_AREA).astype(np.float32)
    ratio = (b+32)/(a+32)
    ratio /= max(float(np.median(ratio)),1e-6)
    return np.clip(ratio,.8,1.2).astype(np.float32)


def apply_illumination(image, profile, strength=1.):
    # Per-pixel multiplication only: no warps, replacement pixels or inpainting.
    field = cv2.resize(np.asarray(profile,dtype=np.float32), (image.shape[1],image.shape[0]), interpolation=cv2.INTER_LINEAR)
    field = 1 + strength*(field-1)
    return np.clip(image.astype(np.float32)*field[:,:,None],0,255).astype(np.uint8)


def train_weights(rows, hard):
    if any(r['split']!='train' for r in rows):raise ValueError('Validation entered training sampler')
    counts = Counter((r['origin'],r.get('camera'),r['label']) for r in rows)
    raw = [float(hard.get(r['id'],1.)) for r in rows]
    sums = {}
    for r,w in zip(rows,raw):
        key=(r['origin'],r.get('camera'),r['label']);sums[key]=sums.get(key,0)+w
    weights=[]
    for r,w in zip(rows,raw):
        o,c,label=r['origin'],r.get('camera'),r['label'];key=(o,c,label)
        if key not in counts:raise ValueError('Missing training stratum')
        weights.append(w/sums[key]*DOMAIN_BUDGET[o]*(.5 if o!='old_legacy_train' else 1)*CLASS_BUDGET[o][base.LABELS.index(label)])
    return weights


class SafetyLoss:
    def __init__(self, recipe):self.recipe=recipe
    def __call__(self,preds,batch):
        logits=preds[1] if isinstance(preds,(tuple,list)) else preds
        target=batch['cls'].long()
        ce=F.cross_entropy(logits,target,weight=logits.new_tensor([2.,1.,1.]))
        loss=ce
        if 'teacher_probabilities' in batch:
            teacher=batch['teacher_probabilities'].detach();population=batch['population'].to(target.device)
            mask=((population!=0)|(target==0))&(teacher.argmax(-1)==target)&(teacher.max(-1).values>=.6)
            if mask.any():
                temperature=2.
                soft=teacher[mask].clamp_min(1e-8).pow(1/temperature);soft/=soft.sum(-1,keepdim=True)
                kd=F.kl_div(F.log_softmax(logits[mask]/temperature,-1),soft,reduction='none').sum(-1)*temperature**2
                strength=torch.where(population[mask]==0,1.,self.recipe['kd'])
                loss=loss+(strength*kd).mean()
        return loss,loss.detach()


class SafetyDataset(base.HoldoutDataset):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        p=read(Path(self.root).parent.parent/'protocol.json')
        self.recipe=p['recipe'];self.styles=p['illumination_profiles']
    def __getitem__(self,index):
        filename,label=self.samples[index][:2];image=cv2.imread(filename)
        if image is None:raise ValueError(filename)
        row=self.record_by_path[str(Path(filename).resolve())]
        if self.runtime_augment:
            image=safe_augment(image)
            if self.recipe['ai'] and np.random.random()<.35:
                image=apply_illumination(image,self.styles[np.random.randint(len(self.styles))],np.random.uniform(.3,1.))
        return dict(img=runtime_tensor(image),cls=label,
                    population={'new_factory':0,'old_ha':1,'legacy_val':2,'old_legacy_train':3}[row['origin']],
                    human_reviewed=int(bool(row.get('human_reviewed'))),camera=int(row.get('camera') or 0))


class SafetyValidator(base.HoldoutValidator):
    def build_dataset(self,img_path):return SafetyDataset(root=img_path,args=self.args,augment=False,prefix=self.args.split)
    def get_stats(self):
        result=super().get_stats()
        metrics={d:{k:result[f'metrics/{d}_{k}'] for k in ('errors','false_ng','ng_not_detected','no_disk_as_ok')}
                 for d in self.manifest['baseline_validation'] if d!='all'}
        result['fitness']=safety_fitness(metrics,self.manifest['baseline_validation'])
        return result


def set_trainable(model,scope):
    for name,p in model.named_parameters():
        p.requires_grad=(scope=='full' or (scope=='linear' and '.linear.' in name) or
                         (scope=='late' and (name.startswith('model.9.') or name.startswith('model.10.'))))
    if not any(p.requires_grad for p in model.parameters()):raise ValueError('No trainable parameters')


class SafetyTrainer(base.HoldoutTrainer):
    def recipe(self):
        if not hasattr(self,'_recipe'):self._recipe=read(Path(self.args.data).parent/'protocol.json')['recipe']
        return self._recipe
    def get_model(self,cfg=None,weights=None,verbose=True):
        model=super().get_model(cfg,weights,verbose);model.criterion=SafetyLoss(self.recipe());return model
    def _model_train(self):
        # Own protocol has freeze_bn=True, all running statistics remain fixed.
        super()._model_train()
    def build_dataset(self,img_path,mode='train',batch=None):
        return SafetyDataset(root=img_path,args=self.args,augment=mode=='train',prefix=mode)
    def get_validator(self):
        self.loss_names=['loss'];return SafetyValidator(self.test_loader,self.save_dir,args=copy(self.args),_callbacks=self.callbacks)
    def preprocess_batch(self,batch):
        batch=super().preprocess_batch(batch)
        if not hasattr(self,'_teacher'):
            p=read(Path(self.args.data).parent/'protocol.json')
            self._teacher=YOLO(p['parent_pt']).model.float().to(self.device).eval()
            for value in self._teacher.parameters():value.requires_grad=False
        with torch.no_grad():
            prediction=self._teacher(batch['img']);batch['teacher_probabilities']=prediction[0] if isinstance(prediction,(tuple,list)) else prediction
        return batch
    def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode='train'):
        if mode!='train':return super().get_dataloader(dataset_path,batch_size,rank,mode)
        if rank!=-1:raise ValueError('Single GPU only')
        dataset=self.build_dataset(dataset_path,mode);p=read(Path(self.args.data).parent/'protocol.json')
        rows=[dataset.record_by_path[str(Path(s[0]).resolve())] for s in dataset.samples]
        weights=train_weights(rows,p['hard_weights'])
        atomic_json(self.save_dir/'sampling.json',dict(count=len(rows),domain_budget=DOMAIN_BUDGET,
            class_budget=CLASS_BUDGET,total_probability=sum(weights),validation_in_sampler=False,
            hard_weight_ids=list(p['hard_weights']),class_probabilities={c:sum(w for r,w in zip(rows,weights) if r['label']==c) for c in base.LABELS}))
        generator=torch.Generator().manual_seed(self.args.seed)
        return InfiniteDataLoader(dataset,batch_size=min(batch_size,len(dataset)),
            sampler=WeightedRandomSampler(weights,len(dataset),replacement=True,generator=generator),
            num_workers=self.args.workers,pin_memory=True,worker_init_fn=seed_worker,generator=generator)


def check_code(root):
    if (root/'protocol.sha256').exists() and digest(root/'protocol.json')!=(root/'protocol.sha256').read_text().strip():raise ValueError('Frozen protocol checksum mismatch')
    p=read(root/'protocol.json')
    if any(digest(root/'code_snapshot'/name)!=sha for name,sha in p['code_hashes'].items()):raise ValueError('Frozen source checksum mismatch')


def prepare(root):
    if root.exists():raise FileExistsError(root)
    registry=read(PROJECT/'.work/ACTIVE_CLASSIFIER_DATASET.json');pool=Path(registry['path']);m=read(pool/'manifest.json')
    receipt=dict(authorized_at=base.now(),canonical_revision=m['canonical_revision'],incomplete_review_acknowledged=True,
        user_statement=USER_STATEMENT,scope='Autonomous NG-priority optimization, up to six fits max300/patience100, controlled on-the-fly augmentation and AI-derived smooth lighting only. Existing labels/splits unchanged. No deployment.')
    CanonicalDataset(pool).authorize_training(receipt,m['canonical_revision'])
    p=base.prepare(root,base.BALANCED_POLICY,'ha')
    records=read(root/'input_snapshot/manifest.json')['records'];ids={r['id']:r for r in records}
    previous=PROJECT/'.work/classifier_finetune_ha_7030_20261022'
    old=read(previous/'source_manifest_at_training.json')['records']
    signature=lambda rows:sorted((r['id'],r['label'],r['sha256'],r['split']) for r in rows)
    if signature(old)!=signature(records):raise ValueError('Previous fine-tune comparison inputs differ')
    reference=root/'reference';release=read(previous/'release_decision.json')
    shutil.copy2(release['selected_pt'],reference/'previous_finetune.pt')
    if digest(reference/'previous_finetune.pt')!=release['pt_sha256']:raise ValueError('Previous PT checksum differs')
    baseline=base.evaluate_model(Path(p['baseline_onnx']),[dict(r,path=str(root/'input_snapshot'/r['image'])) for r in records],root/'evaluations/baseline.json')
    prior=read(previous/'evaluations/candidate.json');prior_by_id={r['id']:r for r in prior['predictions']}
    old_by_id={r['id']:r for r in baseline['predictions']};hard={}
    for r in records:
        if r['split']!='train':continue
        predictions=[old_by_id[r['id']]['prediction'],prior_by_id[r['id']]['prediction']]
        weight=1.+2.*any(q['label']!=r['label'] for q in predictions)+float(min(q['probabilities'][r['label']] for q in predictions)<.6)
        if weight>1:hard[r['id']]=min(weight,4.)
    augmentation=root/'augmentation_sources';augmentation.mkdir()
    previews=PROJECT/'.work/classifier_synthetic_similar_20261022';sources=read(previews/'preview_provenance.json');styles=[];assets=[]
    for job in sources['jobs']:
        r=ids[job['source_id']]
        if r['split']!='train' or r['sha256']!=job['source_sha256']:raise ValueError('AI style source must be canonical train')
        original=root/'input_snapshot'/r['image'];generated=previews/job['generated_image']
        if digest(generated)!=job['generated_sha256']:raise ValueError('AI asset changed')
        target=augmentation/(job['id']+'.png');shutil.copy2(generated,target)
        styles.append(illumination_profile(cv2.imread(str(original)),cv2.imread(str(target))).tolist())
        assets.append(dict(id=job['id'],path=str(target),sha256=digest(target),source_id=r['id'],source_sha256=r['sha256'],
                           source_split='train',source_group=r['split_group'],raw_generated_pixels_used_for_training=False))
    shutil.copy2(previews/'generation_jobs.json',augmentation/'generation_jobs.json')
    p.update(campaign_type='ng_priority_20261022',recipes=RECIPES,repeat_top_k=2,repeat_seed=2027,
        epochs=300,patience=100,hard_weights=hard,baseline_validation=baseline['metrics']['val'],
        illumination_profiles=styles,augmentation_assets=assets,freeze_bn=True,checkpoint_selection=
        'Minimize NG regressions vs HA baseline, total NG misses, false NG, no_disk-as-OK, then total errors. Validation is development data.',
        fitness_formula='-1e10*NG_excess_across_sources_and_cameras -1e7*NG_misses -1000*false_NG -no_disk_as_OK -.001*errors',
        interpolation_fractions=[.25,.5,.75,1.],previous_finetune_fractions=[.1,.25,.5,.75,1.],
        synthetic_policy='Only bounded smooth 8x8 photometric fields from AI outputs, applied to original train pixels on the fly. No generated geometry or synthetic labels.',
        sampling=dict(domain_budget=DOMAIN_BUDGET,class_budget=CLASS_BUDGET),production_deployed=False,
        previous_finetune_pt=str(reference/'previous_finetune.pt'),previous_finetune_sha256=release['pt_sha256'])
    atomic_json(root/'protocol.json',p)
    atomic_text(root/'protocol.sha256',digest(root/'protocol.json'))
    atomic_text(root/'index.html',dashboard())
    atomic_json(root/'authorization_receipt.json',receipt)
    return p


def dashboard():
    return '''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NG-priority classifier campaign</title><style>body{font:16px system-ui;background:#14202e;color:#e6eef6;margin:28px;max-width:1300px}a{color:#9bd4ff}p{line-height:1.5}.note{background:#294236;padding:16px}td,th{border:1px solid #63778d;padding:10px}table{border-collapse:collapse}</style><h1>Tối ưu classifier — ưu tiên không lọt NG</h1><p>4 phương pháp và lặp seed cho 2 phương pháp tốt nhất; mỗi fit tối đa 300 epoch, patience 100. Giữ nguyên canonical 70/30, hai camera và legacy.</p><p class="note">Best ưu tiên NG bỏ sót, sau đó NG ảo. HA là retention; validation cùng phiên chụp, đã dùng phát triển nhiều lần, không phải test độc lập. Nhãn còn partial review. Ảnh AI chỉ cung cấp trường ánh sáng mượt để augment pixel gốc, không tự thêm nhãn ảnh AI.</p><p id="status">Đang tải…</p><p id="error"></p><p><a href="protocol.json">Cấu hình đã chốt</a> · <a href="candidates.json">Các ứng viên</a> · <a href="comparison.html">So sánh kết quả</a> · <a href="error_review.html">Ảnh sai của best</a> · <a href="RESULTS.md">Báo cáo</a> · <a href="http://127.0.0.1:8791/">Dataset chuẩn</a></p><script>async function refresh(){try{let s=await(await fetch('status.json?t='+Date.now(),{cache:'no-store'})).json();document.getElementById('status').textContent=s.status+' / '+s.phase+' · '+(s.job||'')+' · epoch '+s.epoch+' / 300 · best '+s.best_epoch;document.getElementById('error').textContent=s.error||''}catch(e){document.getElementById('error').textContent=e.message}}refresh();setInterval(refresh,10000)</script></html>'''


def publish(root,status,phase,job=None,child=None,error=None):
    progress={}
    if job and (root/'jobs'/job/'training/fit/early_stopping_state.json').exists():progress=read(root/'jobs'/job/'training/fit/early_stopping_state.json')
    state=dict(status=status,phase=phase,job=job,pid=os.getpid(),active_pid=child,epoch=progress.get('epoch',0),
               best_epoch=progress.get('best_epoch',0),epochs=300,patience=100,updated_at=base.now(),error=error)
    atomic_json(root/'status.json',state)
    pointer=PROJECT/'.work/ACTIVE_CLASSIFIER_TRAINING.json'
    if pointer.exists():
        current=read(pointer)
        if Path(current['path']).resolve()==root.resolve():
            current.update(status=status,phase=phase,job=job,active_pid=child,error=error,
                           campaign_pid=None if status in ('complete','failed','stopped') else os.getpid())
            if status=='complete':current['completed_at']=base.now()
            atomic_json(pointer,current)


def checked(root,phase,args,logfile,job=None):
    if (root/'STOP_REQUESTED').exists():raise InterruptedError('STOP_REQUESTED')
    env=os.environ.copy();env.update(PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1')
    logfile.parent.mkdir(parents=True,exist_ok=True)
    with logfile.open('a',encoding='utf-8') as handle:
        child=subprocess.Popen([sys.executable,'-u','-B']+list(map(str,args)),cwd=PROJECT,env=env,
                               stdout=handle,stderr=subprocess.STDOUT,creationflags=FLAGS)
        try:
            while True:
                if (root/'STOP_REQUESTED').exists():raise InterruptedError('STOP_REQUESTED')
                code=child.poll();publish(root,'running',phase,job,child.pid if code is None else None)
                if code is not None:
                    if code:raise RuntimeError(f'{phase} exited {code}; see {logfile}')
                    break
                time.sleep(10)
        finally:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=25)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=25)


def make_job(root,recipe,seed):
    name=recipe['id']+(f'_seed{seed}' if seed!=1022 else '')
    folder=root/'jobs'/name
    if (folder/'protocol.json').exists():return name
    p=read(root/'protocol.json');data=folder/'data';data.mkdir(parents=True)
    m=read(root/'input_snapshot/manifest.json');base.validate_rows(m['records'],base.BALANCED_POLICY)
    for r in m['records']:
        source=root/'input_snapshot'/r['image'];target=data/r['image'];target.parent.mkdir(parents=True,exist_ok=True)
        if digest(source)!=r['sha256']:raise ValueError('Frozen image changed')
        if not target.exists():os.link(source,target)
    m['baseline_validation']=p['baseline_validation'];atomic_json(data/'manifest.json',m)
    atomic_json(folder/'protocol.json',dict(recipe=recipe,seed=seed,freeze_bn=True,root=str(root),data=str(data),
        parent_pt=p['pretrained'],hard_weights=p['hard_weights'],illumination_profiles=p['illumination_profiles']))
    atomic_text(folder/'protocol.sha256',digest(folder/'protocol.json'))
    return name


def fit(root,name,smoke=False):
    check_code(root);folder=root/'jobs'/name;p=read(folder/'protocol.json');recipe=p['recipe']
    if digest(folder/'protocol.json')!=(folder/'protocol.sha256').read_text().strip():raise ValueError('Job protocol changed')
    output=folder/'training/fit';last=output/'weights/last.pt';resume=last.exists()
    source=last if resume else Path(p['parent_pt']);model=YOLO(str(source));torch.set_num_threads(4)
    def initialize(trainer):
        if resume:
            state=read(output/'early_stopping_state.json')
            if state['epoch']!=trainer.start_epoch:raise ValueError('Resume/checkpoint epoch mismatch')
            trainer.stopper.best_epoch=state['best_epoch'];trainer.stopper.best_fitness=state['best_fitness']
            trainer.stopper.possible_stop=(trainer.start_epoch-state['best_epoch'])>=99
        else:
            original=YOLO(p['parent_pt']).model.float().cpu().state_dict();actual=trainer.model.state_dict()
            if set(original)!=set(actual) or any(not torch.equal(v,actual[k].detach().float().cpu()) for k,v in original.items()):raise ValueError('HA initialization differs')
        set_trainable(trainer.model,recipe['scope'])
        trainer._frozen={k:v.detach().cpu().clone() for k,v in trainer.model.state_dict().items()
                         if k.endswith(('running_mean','running_var','num_batches_tracked')) or k in {n for n,w in trainer.model.named_parameters() if not w.requires_grad}}
        atomic_json(output/'initialization.json',dict(source=str(source),scope=recipe['scope'],
            trainable_parameters=sum(v.numel() for v in trainer.model.parameters() if v.requires_grad),
            frozen_parameters=sum(v.numel() for v in trainer.model.parameters() if not v.requires_grad),classes=base.LABELS,resume=resume))
    def save(trainer):
        changed=[k for k,v in trainer._frozen.items() if not torch.equal(v,trainer.model.state_dict()[k].detach().cpu())]
        if changed:raise ValueError('Frozen parameter or BN changed: '+str(changed))
        atomic_json(output/'frozen_verification.json',dict(epoch=trainer.epoch+1,buffers_and_parameters_checked=len(trainer._frozen),changed=[]))
        atomic_json(output/'early_stopping_state.json',dict(epoch=trainer.epoch+1,best_epoch=trainer.stopper.best_epoch,best_fitness=trainer.stopper.best_fitness))
        if trainer.fitness==trainer.best_fitness:
            atomic_json(output/'best_checkpoint_epoch.json',dict(epoch=trainer.epoch+1,fitness=trainer.fitness,metrics=trainer.metrics))
    def stop(trainer):
        if (root/'STOP_REQUESTED').exists():trainer.stop=True
    model.add_callback('on_pretrain_routine_end',initialize);model.add_callback('on_model_save',save);model.add_callback('on_train_epoch_end',stop)
    model.train(trainer=SafetyTrainer,data=p['data'],epochs=2 if smoke else 300,patience=100,imgsz=224,batch=64,workers=2,
        device=0,seed=p['seed'],deterministic=True,pretrained=True,resume=resume,freeze=None,
        optimizer='AdamW',lr0=recipe['lr0'],lrf=.05,cos_lr=True,weight_decay=.0005,warmup_epochs=3,
        warmup_bias_lr=recipe['lr0'],dropout=.03,amp=False,auto_augment=None,erasing=0,scale=0,
        fliplr=.5,flipud=0,bgr=0,close_mosaic=0,save_period=25,project=str(folder/'training'),name='fit',exist_ok=resume,plots=True,verbose=False)
    if (root/'STOP_REQUESTED').exists():raise InterruptedError('User stopped')
    artifacts=folder/'artifacts';artifacts.mkdir(exist_ok=True)
    clean=YOLO(str(model.trainer.best));clean.model.criterion=None;clean.save(str(artifacts/'best.pt'))
    atomic_json(artifacts/'completion.json',dict(epochs_completed=model.trainer.epoch+1,best_checkpoint=read(output/'best_checkpoint_epoch.json'),
        strict_improvement_best_epoch=read(output/'early_stopping_state.json')['best_epoch'],epochs_limit=300,patience=100,recipe=recipe,seed=p['seed']))


def all_rows(root):return [dict(r,path=str(root/'input_snapshot'/r['image'])) for r in read(root/'input_snapshot/manifest.json')['records']]


def candidates(root):
    p=read(root/'protocol.json');rows=all_rows(root);items=[]
    sources=[('previous_finetune',Path(p['previous_finetune_pt']),p['previous_finetune_fractions'])]
    sources += [(job.name,job/'artifacts/best.pt',p['interpolation_fractions']) for job in sorted((root/'jobs').iterdir()) if (job/'artifacts/completion.json').exists()]
    for name,source,fractions in sources:
        for alpha in fractions:
            if (root/'STOP_REQUESTED').exists():raise InterruptedError('STOP_REQUESTED')
            key=f'{name}_a{int(alpha*100):03d}';folder=root/'candidate_artifacts'/key;folder.mkdir(parents=True,exist_ok=True)
            pt=folder/'best.pt';onnx=folder/'best.onnx'
            publish(root,'running','export/evaluate '+key)
            if not pt.exists():
                warm=YOLO(p['pretrained']);warm.model=warm.model.float().cpu().eval();cold=YOLO(str(source));cold.model=cold.model.float().cpu().eval()
                if warm.names!=cold.names:raise ValueError('Blend class order differs')
                warm.model.load_state_dict(cold.model.state_dict() if alpha==1 else interpolate(warm.model.state_dict(),cold.model.state_dict(),alpha),strict=True)
                warm.model.criterion=None;warm.save(str(pt))
            if not onnx.exists():YOLO(str(pt)).export(format='onnx',dynamic=True,simplify=False,opset=13,imgsz=224,batch=1,device='cpu')
            result=base.evaluate_model(onnx,rows,root/'evaluations'/f'{key}.json')
            failures=base.comparison_failures(result['metrics']['val'],p['baseline_validation'],p['fitness_weights'],p['comparison_domains'])
            item=dict(id=key,source=name,alpha=alpha,pt=str(pt),onnx=str(onnx),metrics=result['metrics'],
                      evaluation=str(root/'evaluations'/f'{key}.json'),recommendation_failures=failures,eligible=not failures)
            items.append(item);atomic_json(root/'candidates.json',items)
    eligible=[c for c in items if c['eligible']]
    choose=lambda c:(totals(c['metrics']['val'])['ng_not_detected'],totals(c['metrics']['val'])['false_ng'],
                     totals(c['metrics']['val'])['no_disk_as_ok'],totals(c['metrics']['val'])['errors'],-base.holdout_fitness(c['metrics']['val'],p['fitness_weights']),c['id'])
    chosen=min(eligible,key=choose) if eligible else max(items,key=lambda c:safety_fitness(c['metrics']['val'],p['baseline_validation']))
    atomic_json(root/'selection.json',dict(selected=chosen,eligible_count=len(eligible),candidate_count=len(items),validation_is_development=True,production_deployed=False))
    artifact=root/'artifacts';artifact.mkdir(exist_ok=True)
    shutil.copy2(chosen['pt'],artifact/'best.pt');shutil.copy2(chosen['onnx'],artifact/'best.onnx')
    atomic_json(artifact/'completion.json',dict(selection_id=chosen['id'],source=chosen['source'],alpha=chosen['alpha'],
        epochs_limit=300,patience=100,checkpoint_selection=p['checkpoint_selection']))
    base.evaluate(root)
    release=read(root/'release_decision.json');release.update(campaign_selection=read(root/'selection.json'),objective='NG misses first, false NG second')
    atomic_json(root/'release_decision.json',release)
    table='<h2>Tất cả ứng viên đã chốt trước trong protocol</h2><table><tr><th>Ứng viên</th><th>Lọt NG</th><th>NG ảo</th><th>no_disk → OK</th><th>Tổng lỗi</th><th>Qua gate</th></tr>'
    for c in sorted(items,key=choose):
        t=totals(c['metrics']['val']);table+=f'<tr><td>{c["id"]}</td><td>{t["ng_not_detected"]}</td><td>{t["false_ng"]}</td><td>{t["no_disk_as_ok"]}</td><td>{t["errors"]}</td><td>{c["eligible"]}</td></tr>'
    page=(root/'comparison.html').read_text(encoding='utf-8');atomic_text(root/'comparison.html',page.replace('</html>',table+'</table></html>'))
    atomic_json(root/'candidate_evaluation_complete.json',dict(selected=chosen['id'],completed_at=base.now()))


def finalize(root):
    p=read(root/'protocol.json');release=read(root/'release_decision.json');rows=[r for r in all_rows(root) if r['split']=='val']
    pt=root/'artifacts/best.pt';onnx=root/'artifacts/best.onnx';model=YOLO(str(pt)).model.float().cpu().eval()
    from evaluate_classifier_ha import YoloClassifier
    classifier=YoloClassifier(str(onnx),base.LABELS,batch_size=32);maximum=0.;wrong=[];torch.set_num_threads(4)
    with torch.inference_mode():
        for start in range(0,len(rows),32):
            part=rows[start:start+32];batch=classifier._preprocess_batch([cv2.imread(r['path']) for r in part],(224,224))
            result=model(torch.from_numpy(batch));result=(result[0] if isinstance(result,(tuple,list)) else result).numpy()
            exported=classifier.ort_sess.run(None,{classifier.ort_sess.get_inputs()[0].name:batch})[0]
            maximum=max(maximum,float(np.max(np.abs(result-exported))))
            wrong.extend(r['id'] for r,a,b in zip(part,result.argmax(1),exported.argmax(1)) if a!=b)
    parity=dict(count=len(rows),argmax_mismatches=len(wrong),max_probability_error=maximum,passed=not wrong and maximum<1e-3)
    atomic_json(root/'artifacts/full_validation_export_parity.json',parity)
    if not parity['passed']:raise ValueError('Full validation PT/ONNX parity failed')
    protected=read(root/'protected_runtime_hashes.json')
    if not all(digest(path)==sha for path,sha in protected.items()):raise ValueError('Runtime files changed')
    check_code(root)
    for asset in p['augmentation_assets']:
        if digest(asset['path'])!=asset['sha256']:raise ValueError('AI augmentation asset changed')
    frozen=read(root/'input_snapshot/manifest.json');base.validate_rows(frozen['records'],base.BALANCED_POLICY)
    for r in frozen['records']:
        if digest(root/'input_snapshot'/r['image'])!=r['sha256']:raise ValueError('Frozen input changed')
    if digest(root/'source_manifest_at_training.json')!=p['pool_manifest_sha256']:raise ValueError('Frozen manifest changed')
    white=read(root/'white_replay.json')
    if len(white['records'])!=28 or white['hardware_calls'] or not white['all_geometry_and_counts_unchanged'] or not white['both_camera_debug_production_parity']:raise ValueError('White replay failed')
    current=read(Path(p['pool'])/'manifest.json')
    verification=dict(verified_at=base.now(),passed=True,canonical_revision_at_start=p['canonical_revision'],current_canonical_revision=current['canonical_revision'],
        frozen_inputs_used=True,frozen_hashes_verified=True,runtime_hashes_unchanged=True,white_count=28,hardware_calls=False,
        recommended_for_review=release['recommended_for_review'],production_deployed=False,full_validation_export_parity=parity)
    atomic_json(root/'completion_verification.json',verification)
    lines=['# Tối ưu classifier ưu tiên NG','',f"Ứng viên: {read(root/'selection.json')['selected']['id']}",
           f"Qua gate đề xuất rà tiếp: {release['recommended_for_review']}. Không triển khai; giữ HA đang cài.",'',
           '| Nguồn | HA gốc: lỗi / NG lọt / NG ảo | Ứng viên: lỗi / NG lọt / NG ảo |','| --- | ---: | ---: |']
    for domain in p['comparison_domains']:
        a,b=release['baseline']['val'][domain],release['candidate']['val'][domain]
        lines.append(f"| {domain} | {a['errors']} / {a['ng_not_detected']} / {a['false_ng']} | {b['errors']} / {b['ng_not_detected']} / {b['false_ng']} |")
    lines+=['','Gate chưa đạt: '+(', '.join(release['recommendation_failures']) or 'Không có'),'',
        'Ảnh AI chỉ cung cấp trường ánh sáng 8×8 bị chặn 0,8–1,2; pixel hình học gốc giữ nguyên vị trí. Nhãn/split không đổi.',
        'HA là retention do parent đã học HA. Validation đã dùng phát triển và chọn best, có chung frame context; không phải test độc lập.',
        'Nhãn còn partial review. Không tuyên bố không lọt NG trên sản xuất từ các chỉ số này.',
        'PT/ONNX khớp trên toàn validation; White replay đủ 28 ảnh với hardware tắt; runtime hashes giữ nguyên.',
        '', '- So sánh: http://127.0.0.1:8797/comparison.html','- Ảnh lỗi: http://127.0.0.1:8797/error_review.html']
    atomic_text(root/'RESULTS.md','\n'.join(lines)+'\n')


def campaign(root,resume=False):
    if (root/'STOP_REQUESTED').exists():raise InterruptedError('STOP_REQUESTED exists')
    # Verify there is no other Python trainer before spawning a new trainer tree.
    command="Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    inventory=subprocess.run(['powershell.exe','-NoProfile','-Command',command],capture_output=True,text=True,creationflags=FLAGS,check=True).stdout.strip()
    processes=json.loads(inventory) if inventory else [];processes=processes if isinstance(processes,list) else [processes]
    active=[x for x in processes if x['ProcessId']!=os.getpid() and 'train_classifier_' in (x.get('CommandLine') or '') and '--phase' in x['CommandLine']]
    if active:raise RuntimeError('Another classifier parent/child is alive: '+str(active))
    p=read(root/'protocol.json') if resume else prepare(root)
    if (root/'status.json').exists() and read(root/'status.json')['status']=='complete':raise ValueError('Completed campaign must not resume')
    check_code(root);script=root/'code_snapshot'/Path(__file__).name
    names=[]
    for recipe in p['recipes']:
        name=make_job(root,recipe,1022);names.append(name)
        if not (root/'jobs'/name/'artifacts/completion.json').exists():checked(root,'training', [script,'--phase','fit','--root',root,'--job',name],root/'logs'/f'{name}.log',name)
    ranked=sorted(names,key=lambda name:read(root/'jobs'/name/'artifacts/completion.json')['best_checkpoint']['fitness'],reverse=True)
    repeated=ranked[:p['repeat_top_k']]
    atomic_json(root/'seed_selection.json',dict(selected_for_repeat=repeated,policy='top two by predeclared NG-priority validation fitness',seed=p['repeat_seed']))
    for original in repeated:
        recipe=read(root/'jobs'/original/'protocol.json')['recipe'];name=make_job(root,recipe,p['repeat_seed'])
        if not (root/'jobs'/name/'artifacts/completion.json').exists():checked(root,'training repeat seed',[script,'--phase','fit','--root',root,'--job',name],root/'logs'/f'{name}.log',name)
    if not (root/'candidate_evaluation_complete.json').exists():checked(root,'compare fixed weight interpolations',[script,'--phase','candidates','--root',root],root/'logs/evaluation.log')
    if not (root/'white_replay.json').exists():
        wrapper="import sys,runpy;sys.path.append(sys.argv[1]);sys.argv=[sys.argv[2],'--root',sys.argv[3]];runpy.run_path(sys.argv[0],run_name='__main__')"
        checked(root,'White replay 28 images; hardware off',['-c',wrapper,str(PROJECT/'SERVICE/.venv/Lib/site-packages'),str(PROJECT/'.work/classifier_supplementary_20261020/verify_white_replay.py'),root],root/'logs/white_replay.log')
    checked(root,'final full parity and hash verification',[script,'--phase','finalize','--root',root],root/'logs/finalize.log')
    publish(root,'complete','six fits and evaluation complete; no deployment')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--phase',choices=('campaign','fit','candidates','finalize'),required=True)
    parser.add_argument('--job');parser.add_argument('--resume',action='store_true');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();root=args.root.resolve()
    try:
        if args.phase=='campaign':campaign(root,args.resume)
        elif args.phase=='fit':fit(root,args.job,args.smoke)
        elif args.phase=='candidates':candidates(root)
        else:finalize(root)
    except Exception as error:
        if root.exists():
            try:publish(root,'stopped' if isinstance(error,InterruptedError) else 'failed',args.phase,args.job,error=str(error))
            except Exception:pass
        traceback.print_exc();raise
