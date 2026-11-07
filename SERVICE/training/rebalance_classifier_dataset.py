"""Preserve canonical labels, add a small curated NG-heavy legacy subset, resplit.

Target-slot histories and decoded/flip/180 aliases stay together. Frames can be
shared as context: this is same-session spatial validation, not tray holdout.
"""
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import itertools
import json
from pathlib import Path
import shutil

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

try:
    from .classifier_dataset import CanonicalDataset,LABELS,read,atomic_json
    from .adopt_classifier_dataset import source_inventory,decoded_fingerprint,digest
except ImportError:
    from classifier_dataset import CanonicalDataset,LABELS,read,atomic_json
    from adopt_classifier_dataset import source_inventory,decoded_fingerprint,digest

POLICY='camera_target_slot_legacy_ng_v1'
LEGACY_QUOTAS={'ng':420,'no_disk':90,'ok':90}


def camera_of(r):
    if r['origin'] in ('old_legacy_train','legacy_val'):return None
    camera=r.get('camera')
    if camera is None:camera=int(r['frame'].split('_')[0].removeprefix('Cam'))
    if int(camera) not in (1,2):raise ValueError('Unknown camera: '+r['id'])
    return int(camera)


def select_legacy(source_rows,current_rows,excluded_ids=(),quotas=None):
    quotas=quotas or LEGACY_QUOTAS
    labels_by_hash=defaultdict(set)
    for r in current_rows+source_rows:labels_by_hash[r['decoded_sha256']].add(r['label'])
    current_ids={r['id'] for r in current_rows}|set(excluded_ids)
    selected=[];used=set();skipped=[]
    for label,wanted in quotas.items():
        candidates=sorted((r for r in source_rows if r['label']==label),
            key=lambda r:hashlib.sha256(('legacy_1022|'+r['id']).encode()).hexdigest())
        chosen=[]
        for r in candidates:
            sha=r['decoded_sha256']
            if r['id'] in current_ids:continue
            if len(labels_by_hash[sha])>1:
                skipped.append(dict(id=r['id'],reason='conflicting_decoded_alias_labels'));continue
            if sha in used:continue
            chosen.append(deepcopy(r));used.add(sha)
            if len(chosen)==wanted:break
        if len(chosen)!=wanted:raise ValueError(f'Insufficient unique legacy {label}: {len(chosen)}/{wanted}')
        selected.extend(chosen)
    return selected,skipped


def assign_groups(rows):
    for r in rows:
        r['camera']=camera_of(r)
        if r['camera'] is None:
            target='legacy|'+r['decoded_sha256']
        else:
            if r.get('region') is None or r.get('slot') is None:raise ValueError('Missing target-slot provenance')
            target=f"{r['origin']}|Cam{r['camera']}|{r['region']}|slot{int(r['slot']):02}"
        r['target_group']=target
    parent={r['target_group']:r['target_group'] for r in rows}
    def find(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        return a
    by_hash={}
    for r in rows:
        key=r['decoded_sha256'];g=r['target_group']
        if key in by_hash:
            a,b=find(g),find(by_hash[key]);parent[max(a,b)]=min(a,b)
        else:by_hash[key]=g
    for r in rows:
        r.setdefault('collection_burst',r.get('capture_group'))
        r['capture_group']=r['target_group']
        r['split_group']=find(r['target_group'])


def stratum(r):return f"{r['origin']}|Cam{r['camera'] if r['camera'] is not None else 'unknown'}|{r['label']}"


def plan_split(rows,seeds=16):
    assign_groups(rows)
    labels=np.array([stratum(r) for r in rows]);groups=np.array([r['split_group'] for r in rows])
    totals=Counter(labels);best=None
    for seed in range(1022,1022+seeds):
        splitter=StratifiedGroupKFold(n_splits=10,shuffle=True,random_state=seed)
        folds=[val for _,val in splitter.split(np.zeros(len(rows)),labels,groups)]
        for fold in itertools.combinations(range(10),3):
            val_indices=np.concatenate([folds[i] for i in fold])
            counts=Counter(labels[val_indices])
            if any(counts[k]==0 or counts[k]>=v for k,v in totals.items()):continue
            fractions=np.array([counts[k]/v for k,v in sorted(totals.items())])
            score=float(np.mean((fractions-.3)**2)+abs(len(val_indices)/len(rows)-.3)**2)
            key=(score,seed,fold)
            if best is None or key<best[0]:best=(key,set(val_indices.tolist()))
    if best is None:raise ValueError('No balanced split with every source/camera/class on both sides')
    for i,r in enumerate(rows):
        r['split']='val' if i in best[1] else 'train'
        r['role']='evaluation_only' if r['split']=='val' else 'training_pool'
        r['canonical_original_role']=r['role']
    for key in ('sha256','decoded_sha256','target_group','split_group'):
        sides=defaultdict(set)
        for r in rows:sides[r[key]].add(r['split'])
        if any(len(v)>1 for v in sides.values()):raise ValueError('Split leakage: '+key)
    by_stratum={k:{s:sum(r['split']==s and stratum(r)==k for r in rows) for s in ('train','val')} for k in sorted(totals)}
    return dict(id=POLICY,authoritative=True,target_fraction=.3,actual_fraction=len(best[1])/len(rows),
        grouping_unit='origin_camera_target_slot',allow_frame_overlap=True,
        frame_overlap_reason='Same-session spatial validation: different target slots of a frame may be on opposite sides; all repeated views of each target slot stay together.',
        validation_origins=sorted({r['origin'] for r in rows}),seed=best[0][1],fold=best[0][2],selection_score=best[0][0],
        candidate_seed_count=seeds,selection_inputs='Labels, camera, target IDs and decoded image hashes only; no model predictions or metrics.',
        strata_counts=by_stratum,legacy_quotas=LEGACY_QUOTAS,
        limitations=['Frames and capture sessions are shared as context; this is not independent full-frame/tray/temporal validation.',
            'Physical target-slot histories across all frames in each source/camera stay together, including all their changing states.',
            'Decoded identical images, including flips and 180-degree rotations, never cross splits.',
            'Legacy camera/frame provenance is unavailable and is recorded as unknown, never assigned a fake camera.',
            'Previously viewed development data with partial label-consistency review; old run metrics retain their original inputs.'])


def prepare(current,source,staging,report_dir):
    current,source,staging,report_dir=[Path(p).resolve() for p in (current,source,staging,report_dir)]
    if staging.exists():raise FileExistsError(staging)
    before=(current/'manifest.json').read_bytes();m=read(current/'manifest.json')
    rows=deepcopy(m['records'])
    if any(r['origin'] not in ('old_ha','new_factory') for r in rows):raise ValueError('Expected HA/new input canonical dataset')
    for r in rows:
        r['_copy_source']=str(current/r['image'])
        if digest(r['_copy_source'])!=r['sha256']:raise ValueError('Canonical image changed')
        if decoded_fingerprint(r['_copy_source'])!=r['decoded_sha256']:raise ValueError('Decoded fingerprint changed')
    incoming,inventory=source_inventory(source)
    candidates=[r for r in incoming if r['origin']=='old_legacy_train']
    for r in candidates:r['decoded_sha256']=decoded_fingerprint(source/r['adopted_source_image'])
    excluded_ids={r['id'] for r in m.get('excluded',[])+m.get('excluded_records',[])}
    legacy,skipped=select_legacy(candidates,rows,excluded_ids)
    for r in legacy:
        r['_copy_source']=str(source/r['adopted_source_image'])
        r['legacy_import_source']=str(source)
        if r['label']!=r['source_manifest_label']:
            r['label_source']='user_selected_folder_label'
    rows.extend(legacy);policy=plan_split(rows)
    staging.mkdir(parents=True);report_dir.mkdir(parents=True,exist_ok=True)
    for split in ('train','val'):
        for label in LABELS:(staging/split/label).mkdir(parents=True)
    for r in rows:
        origin=Path(r.pop('_copy_source'));r['image']=f"{r['split']}/{r['label']}/{origin.name}"
        shutil.copy2(origin,staging/r['image'])
        if digest(staging/r['image'])!=r['sha256']:raise ValueError('Copy checksum failed')
        for key in ('context','full_context'):
            relative=r.get(key)
            if not relative:continue
            old=current/relative
            if not old.is_file():r[key]=None;continue
            dest=staging/relative
            if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old,dest)
    for r in deepcopy(m.get('excluded_records',[])):
        dest=staging/r['image'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(current/r['image'],dest)
    if before!=(current/'manifest.json').read_bytes():raise ValueError('Canonical labels changed while preparing')
    _,after=source_inventory(source)
    if inventory!=after:raise ValueError('Selected legacy source changed while preparing')
    new=deepcopy(m);new.update(records=rows,canonical_revision=m['canonical_revision']+1,
        updated_at=datetime.now(timezone.utc).isoformat(),review_status='in_progress',training_authorized=False,
        training_started=False,status='camera_balanced_ng_heavy_legacy_prepared',split_policy=policy,
        role_directories={'training_pool':'train','evaluation_only':'val'},authoritative_splits=True)
    new.pop('training_authorization',None)
    new['last_repartition']=dict(source_revision=m['canonical_revision'],previous_manifest_sha256=digest(current/'manifest.json'),
        user_instruction='Thêm một ít legacy ưu tiên NG vào train/val và chia đều hai camera.',
        existing_canonical_labels_preserved=True,legacy_source=inventory,legacy_ids=[r['id'] for r in legacy],
        skipped_ambiguous_legacy=skipped,training_started=False)
    CanonicalDataset.refresh(new);atomic_json(staging/'manifest.json',new);CanonicalDataset(staging).write_views(new)
    report=dict(canonical_revision=new['canonical_revision'],current_manifest_sha256=digest(current/'manifest.json'),
        total=len(rows),legacy_added=len(legacy),legacy_class_counts=dict(Counter(r['label'] for r in legacy)),
        existing_labels_preserved=len(m['records']),policy=policy,
        counts={split:{domain:dict(Counter(r['label'] for r in rows if r['split']==split and
            (r['origin']+'|Cam'+str(r['camera']))==domain)) for domain in sorted({r['origin']+'|Cam'+str(r['camera']) for r in rows})}
            for split in ('train','val')},
        train=new['training_pool_size'],val=new['evaluation_size'],training_started=False)
    atomic_json(report_dir/'plan.json',report);print(json.dumps(report,ensure_ascii=True));return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('current','source','staging','report-dir'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();prepare(a.current,a.source,a.staging,a.report_dir)
