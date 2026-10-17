"""Freeze human-confirmed HA labels; split whole frames, never individual crops."""
import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

LABELS = ['ng', 'no_disk', 'ok']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assign_splits(items):
    result = {}
    for camera in (1, 2):
        frames = sorted({i['frame'] for i in items if i['camera'] == camera})
        if len(frames) != 10:
            raise ValueError('This frozen HA protocol requires ten frames per camera')
        for index, frame in enumerate(frames):
            result[frame] = 'test' if index >= 8 else ('val' if index in (3, 6) else 'train')
    return result


def build(review, output):
    if output.exists():
        raise FileExistsError('Existing training dataset is never overwritten')
    payload_file, decisions_file = review/'items.json', review/'decisions.json'
    payload = json.loads(payload_file.read_text(encoding='utf-8'))
    frozen_decisions = decisions_file.read_bytes()
    decisions = json.loads(frozen_decisions)
    if decisions['items_sha256'] != digest(payload_file):
        raise ValueError('Decisions and proposal dataset do not match')
    items = payload['items']
    if set(decisions['decisions']) != {i['id'] for i in items}:
        raise ValueError('Unknown or unreviewed IDs remain; never train on implicit proposals')
    if any(d['label'] not in LABELS for d in decisions['decisions'].values()):
        raise ValueError('Uncertain/excluded labels need a separate audited dataset protocol')
    split_lookup = assign_splits(items)
    records, changed, hashes = [], 0, {}
    # Validate before creating the output tree; detect exact-byte duplicates across frames/splits.
    for item in items:
        label = decisions['decisions'][item['id']]['label']
        source = review/item['image']
        actual_hash = digest(source)
        if actual_hash != item['sha256']:
            raise ValueError(f'Crop changed: {item["id"]}')
        split = split_lookup[item['frame']]
        if actual_hash in hashes and hashes[actual_hash] != split:
            raise ValueError('Exact crop duplicate crosses dataset splits')
        hashes[actual_hash] = split
        changed += int(label != item['proposed_label'])
        records.append({'id':item['id'], 'camera':item['camera'], 'frame':item['frame'],
                        'region':item['region'], 'slot':item['slot'], 'label':label, 'split':split,
                        'image':f'{split}/{label}/{item["id"]}.png', 'source_crop':str(source.resolve()),
                        'sha256':actual_hash, 'original_prediction':item['proposed_label'],
                        'original_confidence':item['confidence'], 'review':decisions['decisions'][item['id']]})
    if decisions_file.read_bytes() != frozen_decisions:
        raise ValueError('Human review changed during preparation; freeze again')
    output.mkdir(parents=True)
    for split in ('train','val','test'):
        for label in LABELS:
            (output/split/label).mkdir(parents=True)
    for record in records:
        shutil.copy2(record['source_crop'], output/record['image'])
        assert digest(output/record['image']) == record['sha256']
    shutil.copy2(payload_file, output/'review_items_snapshot.json')
    (output/'review_decisions_snapshot.json').write_bytes(frozen_decisions)
    counts = {s:dict(Counter(r['label'] for r in records if r['split']==s)) for s in ('train','val','test')}
    report = {'schema_version':1, 'created_at':datetime.now(timezone.utc).isoformat(),
              'training_authorized_by_user':True, 'domain':'human_reviewed_HA_only',
              'review_items_sha256':digest(payload_file),
              'review_decisions_sha256':hashlib.sha256(frozen_decisions).hexdigest(),
              'class_order':LABELS, 'crop_contract':payload['crop_contract'], 'old_model':payload['model'],
              'counts':counts, 'changed_labels':changed, 'split_by_frame':split_lookup,
              'split_protocol':'per camera, sorted capture time: train indices 0,1,2,4,5,7; val 3,6; test 8,9',
              'test_limitation':'chronological held-out frames within same HA capture sessions, not independent factory test',
              'cross_split_exact_duplicates':0, 'old_demo_training_images':0, 'records':records}
    (output/'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','old_model')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.review.resolve(), args.output.resolve())
