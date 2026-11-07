"""Adopt only a user-selected ImageFolder and hold out complete HA/new bursts.

Actual source class folders override stale source-manifest labels. Missing files
are reported, never filled from other datasets. No training or deployment.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import shutil

import cv2

try:
    from .classifier_dataset import CanonicalDataset, LABELS, atomic_json, atomic_text, read
except ImportError:
    from classifier_dataset import CanonicalDataset, LABELS, atomic_json, atomic_text, read


POLICY = 'ha_new_capture_holdout_v1'
ELIGIBLE = ('old_ha', 'new_factory')


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_inventory(root):
    root = Path(root).resolve()
    manifest = read(root / 'manifest.json')
    metadata = {r['id']: r for r in manifest['records']}
    if len(metadata) != len(manifest['records']): raise ValueError('Duplicate source manifest IDs')
    found, overrides = {}, []
    for split in ('train', 'val'):
        for label in LABELS:
            for path in sorted((root / split / label).rglob('*')):
                if not path.is_file() or path.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'):
                    continue
                identifier = path.stem
                if identifier not in metadata: raise ValueError('Unregistered source crop: ' + identifier)
                if identifier in found: raise ValueError('Source crop appears twice: ' + identifier)
                r = deepcopy(metadata[identifier])
                if digest(path) != r['sha256']: raise ValueError('Source pixel file changed: ' + identifier)
                if r['label'] != label or r['split'] != split:
                    overrides.append(dict(id=identifier, manifest_label=r['label'], folder_label=label,
                                          manifest_split=r['split'], folder_split=split))
                r.update(source_manifest_label=r['label'], label=label, adopted_source_image=path.relative_to(root).as_posix(),
                         adopted_source_split=split, adopted_source_label=label)
                found[identifier] = r
    missing = [dict(id=i, origin=r['origin'], image=r['image'], label=r['label'])
               for i, r in metadata.items() if i not in found]
    inventory = [dict(id=i, image=r['adopted_source_image'], sha256=r['sha256'], label=r['label'])
                 for i, r in sorted(found.items())]
    fingerprint = hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()
    return list(found.values()), dict(source_path=str(root), source_manifest_sha256=digest(root/'manifest.json'),
        inventory_sha256=fingerprint, present=len(found), declared=len(metadata),
        missing_from_selected_source=missing, actual_folder_overrides=overrides,
        counts_by_original_split={s:dict(Counter(r['label'] for r in found.values() if r['adopted_source_split']==s))
                                  for s in ('train','val')})


def decoded_fingerprint(path):
    image = cv2.imread(str(path))
    if image is None: raise ValueError('Unreadable source image: ' + str(path))
    return min(hashlib.sha256(str(x.shape).encode()+x.tobytes()).hexdigest()
               for x in (image, cv2.flip(image, 1), cv2.flip(image, 0), cv2.flip(image, -1)))


def capture_groups(rows):
    groups = {}
    for origin in ELIGIBLE:
        frames = sorted({r['frame'] for r in rows if r['origin']==origin}, key=lambda f:f.split('Image_')[1])
        previous = None; index = -1
        for frame in frames:
            stamp = datetime.strptime(frame.split('Image_')[1], '%Y%m%d%H%M%S%f')
            if previous is None or (stamp-previous).total_seconds()>120: index += 1
            groups[(origin, frame)] = f'{origin}_burst_{index:02}'
            previous = stamp
    for r in rows:
        r['capture_group'] = groups[(r['origin'], r['frame'])] if r['origin'] in ELIGIBLE else 'legacy_train_only'


def choose_split(rows, target_fraction=.2):
    """Exhaustive deterministic grouped choice; no inference/model scores used."""
    capture_groups(rows)
    parent = {r['capture_group']:r['capture_group'] for r in rows}
    def find(g):
        while parent[g]!=g:
            parent[g]=parent[parent[g]];g=parent[g]
        return g
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[max(a,b)]=min(a,b)
    aliases = defaultdict(list)
    for r in rows: aliases[r.get('decoded_sha256', r['sha256'])].append(r)
    for copies in aliases.values():
        for r in copies[1:]: union(copies[0]['capture_group'], r['capture_group'])
    for r in rows:r['split_group']=find(r['capture_group'])
    groups = defaultdict(list)
    for r in rows: groups[r['split_group']].append(r)
    eligible = sorted(g for g,items in groups.items() if all(r['origin'] in ELIGIBLE for r in items))
    if len(eligible)>20: raise ValueError('Too many groups for the audited exhaustive split planner')
    counts = {o:Counter(r['label'] for r in rows if r['origin']==o) for o in ELIGIBLE}
    totals = {o:sum(c.values()) for o,c in counts.items()}
    candidates = []
    for mask in itertools.product((False,True),repeat=len(eligible)):
        selected = {g for g,keep in zip(eligible,mask) if keep}
        val = [r for g in selected for r in groups[g]]
        by_origin = {o:Counter(r['label'] for r in val if r['origin']==o) for o in ELIGIBLE}
        valid = True
        for origin in ELIGIBLE:
            c = by_origin[origin]
            for label,minimum in [('ng',8),('no_disk',40),('ok',100)]:
                needed = min(minimum,max(1,counts[origin][label]//5))
                if c[label]<needed or counts[origin][label]-c[label]<min(20,max(1,counts[origin][label]//4)):
                    valid=False
        if not valid: continue
        size_error = abs(len(val)/len(rows)-target_fraction)
        # Prefer coverage of each source's classes, subject to whole-capture
        # constraints. HA has only two bursts, so its 20% target is unattainable.
        source_error = sum(abs(sum(by_origin[o].values())/totals[o]-target_fraction) for o in ELIGIBLE)/2
        class_error = sum(abs(by_origin[o][c]/counts[o][c]-target_fraction)
                          for o in ELIGIBLE for c in LABELS)/6
        score = size_error + source_error*.5 + class_error*.5
        candidates.append((score,tuple(sorted(selected)),by_origin,len(val)))
    if not candidates: raise ValueError('No grouped HA/new holdout has the required class coverage')
    score,selected,val_counts,val_size = min(candidates,key=lambda x:(x[0],x[1]))
    chosen = set(selected)
    for r in rows:
        r['split'] = 'val' if r['split_group'] in chosen else 'train'
        r['role'] = 'evaluation_only' if r['split']=='val' else 'training_pool'
    if any(r['origin'] not in ELIGIBLE and r['split']=='val' for r in rows): raise AssertionError('Legacy entered validation')
    for key in ('sha256','decoded_sha256','frame','capture_group','split_group'):
        sides=defaultdict(set)
        for r in rows:
            if r.get(key) is not None: sides[r[key]].add(r['split'])
        if any(len(s)>1 for s in sides.values()): raise ValueError('Train/validation overlap: '+key)
    return dict(id=POLICY,authoritative=True,target_fraction=target_fraction,actual_fraction=val_size/len(rows),
        validation_origins=list(ELIGIBLE),legacy_policy='All legacy crops are omitted from the active dataset at user request.',
        selected_validation_groups=list(selected),validation_origin_class_counts={o:dict(v) for o,v in val_counts.items()},
        training_origin_counts=dict(Counter(r['origin'] for r in rows if r['split']=='train')),
        validation_origin_counts=dict(Counter(r['origin'] for r in rows if r['split']=='val')),
        feasible_group_assignments=len(candidates),selection_score=score,
        grouping='Within each origin, consecutive frames <=120 seconds form a burst; exact decoded flip/180 aliases connect bursts. Entire groups stay on one side.',
        limitations=['HA has only two capture bursts. Holding out a whole burst includes an entire HA camera burst.',
                    'These images have been used in previous training/development. Old models do not yield an independent test on this newly assigned validation set.'])


def prepare(source, current, staging, audit_dir):
    source,current,staging,audit_dir=map(lambda p:Path(p).resolve(),(source,current,staging,audit_dir))
    if staging.exists():raise FileExistsError(staging)
    old=read(current/'manifest.json')
    rows,inventory=source_inventory(source)
    original_inventory=deepcopy(inventory)
    removed_legacy=[r for r in rows if r['origin'] in ('old_legacy_train','legacy_val')]
    if any(r['origin'] not in (*ELIGIBLE,'old_legacy_train','legacy_val') for r in rows):
        raise ValueError('Unexpected source origin')
    rows=[r for r in rows if r['origin'] in ELIGIBLE]
    for r in rows:r['decoded_sha256']=decoded_fingerprint(source/r['adopted_source_image'])
    policy=choose_split(rows)
    now=datetime.now(timezone.utc).isoformat()
    old_by_id={r['id']:r for r in old['records']}
    resets=[dict(id=r['id'],previous_canonical_label=old_by_id[r['id']]['label'],selected_source_label=r['label'])
            for r in rows if r['id'] in old_by_id and old_by_id[r['id']]['label']!=r['label']]
    staging.mkdir(parents=True)
    for split in ('train','val'):
        for label in LABELS:(staging/split/label).mkdir(parents=True)
    for r in rows:
        source_image=source/r['adopted_source_image']
        r['image']=f'{r["split"]}/{r["label"]}/{source_image.name}'
        shutil.copy2(source_image,staging/r['image'])
        if digest(staging/r['image'])!=r['sha256']:raise ValueError('Staging copy checksum failed')
        r.pop('native_source',None);r.pop('fold',None);r.pop('seen_data_retention',None)
        r['canonical_original_role']=r['role']
        r['adopted_at']=now
        if r['label']!=r['source_manifest_label']:
            r['label_source']='user_selected_folder_label'
            r['review']=dict(label=r['label'],source='user_selected_curated_source_folder',updated_at=now)
        # Context is auxiliary display data, not another crop/label source. Only
        # keep contexts already referenced by the selected manifest for this ID.
        for key in ('context','full_context'):
            relative=r.get(key)
            if not relative:continue
            origin=current/relative
            if not origin.is_file():r[key]=None;continue
            if r['id'] not in old_by_id or old_by_id[r['id']]['sha256']!=r['sha256']:
                r[key]=None;continue
            target=staging/relative
            if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(origin,target)
    # Re-scan the chosen folder to reject concurrent manual file/class changes.
    _,final_inventory=source_inventory(source)
    if final_inventory!=original_inventory:raise ValueError('Selected source changed during preparation')
    missing=[dict(id=r['id'],label='exclude',reason='removed_by_user_as_unsuitable',source_image=r['image'])
             for r in inventory['missing_from_selected_source']]
    m=dict(schema_version=2,is_canonical=True,canonical_revision=old['canonical_revision']+1,
        created_at=now,updated_at=now,status='adopted_user_selected_source_holdout_prepared',
        review_status='in_progress',training_authorized=False,training_started=False,
        role_directories=dict(training_pool='train',evaluation_only='val'),authoritative_splits=True,
        split_policy=policy,records=rows,excluded_records=[],excluded=deepcopy(old.get('excluded',[]))+missing,
        adopted_source=inventory,adoption=dict(user_requested_source=str(source),adopted_at=now,
          previous_canonical_revision=old['canonical_revision'],previous_canonical_manifest_sha256=digest(current/'manifest.json'),
          previous_labels_replaced_by_selected_source=resets,other_training_crop_sources_merged=False,
          source_curation_user_confirmed=True,
          user_confirmation='The user confirmed the missing crops were intentionally deleted as unsuitable and class labels were edited in this selected folder.',
          auxiliary_context_policy='Existing referenced same-ID/same-hash context only; not training examples.',
          legacy_removed_at_user_request=True,
          legacy_removed_counts=dict(Counter(r['origin'] for r in removed_legacy)),
          old_validation_now_training=False),
        canonical_authority='Actual selected source class folders at adoption, then this manifest and train/val directories. No historical label overlay.')
    CanonicalDataset.refresh(m);atomic_json(staging/'manifest.json',m)
    CanonicalDataset(staging).write_views(m)
    report=dict(source=inventory,split_policy=policy,current_source_label_differences=resets,
                total=len(rows),train=m['training_pool_size'],val=m['evaluation_size'],
                counts_by_role_origin=m['counts_by_role_origin'],training_class_counts=m['training_pool_counts'],
                validation_class_counts=m['validation_counts'],canonical_revision=m['canonical_revision'],
                images_imported_from_other_datasets=0,training_started=False,
                legacy_removed_counts=dict(Counter(r['origin'] for r in removed_legacy)),
                staging=str(staging),current=str(current),current_manifest_sha256=digest(current/'manifest.json'))
    audit_dir.mkdir(parents=True,exist_ok=True);atomic_json(audit_dir/'adoption_plan.json',report)
    atomic_json(audit_dir/'selected_source_inventory.json',inventory)
    print(json.dumps(report,ensure_ascii=True))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--current',type=Path,required=True)
    parser.add_argument('--staging',type=Path,required=True)
    parser.add_argument('--audit-dir',type=Path,required=True)
    args=parser.parse_args();prepare(args.source,args.current,args.staging,args.audit_dir)
