"""Small legacy TRAIN replay, prioritizing NG hard negatives; no val/test mining."""
import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2

from evaluate_classifier_ha import legacy_records, predict


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose(records, predictions, counts):
    selected = []
    for label, count in counts.items():
        candidates = [(r,p) for r,p in zip(records,predictions) if r['label']==label]
        if label != 'ok':
            # True NG/no_disk predicted OK comes first; then other mistakes; then low certainty.
            candidates.sort(key=lambda pair:(0 if pair[1]['label']=='ok' else
                                            1 if pair[1]['label']!=label else 2,
                                            -pair[1]['confidence'] if pair[1]['label']!=label else pair[1]['confidence'],
                                            pair[0]['id']))
        else:
            candidates.sort(key=lambda pair:(pair[1]['label']==label,pair[1]['confidence'],pair[0]['id']))
        hard_count = count//2
        hard, remaining = candidates[:hard_count], candidates[hard_count:]
        random.Random(1005).shuffle(remaining)
        picked = hard+remaining[:count-hard_count]
        if len(picked)!=count:
            raise ValueError(f'Insufficient clean legacy training examples: {label}')
        for record, prediction in picked:
            selected.append(dict(record,mining_prediction=prediction))
    return selected


def build(data, legacy, mining_model, output, counts=None, ha_probability=.85, dual_validation=False):
    if output.exists():
        raise FileExistsError('Replay dataset is never overwritten')
    manifest = json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    blocked_hashes = {r['sha256'] for r in manifest['records'] if r['split']!='train'}
    blocked_hashes.update(digest(Path(r['path'])) for r in legacy_records(legacy/'val'))
    pool, skipped, seen = [], [], set()
    for record in legacy_records(legacy/'train'):
        path = Path(record['path'])
        image = cv2.imread(str(path))
        if image is None:
            skipped.append({'path':str(path),'reason':'unreadable'})
            continue
        fingerprint = digest(path)
        if fingerprint in blocked_hashes or fingerprint in seen:
            skipped.append({'path':str(path),'reason':'duplicate or validation/test overlap'})
            continue
        seen.add(fingerprint)
        pool.append(dict(record,sha256=fingerprint))
    # Mine only old TRAIN labels. Original legacy val and HA val/test remain untouched.
    _, predictions = predict(mining_model, pool)
    counts = counts or {'ng':400,'no_disk':100,'ok':50}
    selected = choose(pool,predictions,counts)
    shutil.copytree(data,output)
    records = manifest['records']
    for r in records:
        r['origin']='ha'
    for r in selected:
        source = Path(r['path'])
        relative = f'train/{r["label"]}/{r["id"]}{source.suffix.lower()}'
        target = output/relative
        shutil.copy2(source,target)
        assert digest(target)==r['sha256']
        records.append({'id':r['id'],'label':r['label'],'split':'train','origin':'legacy_train',
                        'image':relative,'source_crop':str(source),'sha256':r['sha256'],
                        'mining_prediction':r['mining_prediction']})
    if dual_validation:
        for r in legacy_records(legacy/'val'):
            source=Path(r['path'])
            relative=f'val/{r["label"]}/{r["id"]}{source.suffix.lower()}'
            shutil.copy2(source,output/relative)
            records.append({'id':r['id'],'label':r['label'],'split':'val','origin':'legacy_val',
                            'image':relative,'source_crop':str(source),'sha256':digest(source)})
    manifest.update({'domain':'HA_dual_validation_fault_replay' if dual_validation else 'HA_with_fault_priority_replay',
                     'legacy_replay_counts':counts,
                     'legacy_sampling_probability':1-ha_probability,'ha_sampling_probability':ha_probability,
                     'legacy_class_sampling':[.60,.30,.10] if dual_validation else [.75,.15,.10],
                     'legacy_validation_in_selection':dual_validation,
                     'legacy_train_pool_counts':dict(Counter(r['label'] for r in pool)),
                     'legacy_replay_mining_model':str(mining_model.resolve()),
                     'legacy_replay_mining_model_sha256':digest(mining_model),
                     'skipped_legacy':skipped,'old_demo_training_images':sum(counts.values()),
                     'selection_policy':('weighted NG->non-NG risk (OK OR no_disk): HA 75%, legacy 25%; legacy NG misses <=21 and empty->OK <=2; '
                                         'HA OK recall >=90%, legacy OK recall >=95%, legacy accuracy >=97%; '
                                         'then other NG misses, empty misses, macro-F1') if dual_validation else
                                        'minimize HA val NG->OK, NG->empty, empty->OK, macro-F1; OK recall >=0.90',
                     'records':records,
                     'counts':{s:dict(Counter(r['label'] for r in records if r['split']==s)) for s in ('train','val','test')}})
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ('domain','counts','legacy_replay_counts','legacy_train_pool_counts',
                                           'ha_sampling_probability','legacy_sampling_probability')},indent=2))
    print('Skipped legacy:',len(skipped),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--legacy',type=Path,required=True)
    parser.add_argument('--mining-model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--counts',type=int,nargs=3,default=[400,100,50],metavar=('NG','NO_DISK','OK'))
    parser.add_argument('--ha-probability',type=float,default=.85)
    parser.add_argument('--dual-validation',action='store_true')
    args=parser.parse_args()
    assert 0<args.ha_probability<1 and min(args.counts)>0
    build(args.data.resolve(),args.legacy.resolve(),args.mining_model.resolve(),args.output.resolve(),
          dict(zip(['ng','no_disk','ok'],args.counts)),args.ha_probability,args.dual_validation)
