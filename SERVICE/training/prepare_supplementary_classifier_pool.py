"""Freeze a reviewable classifier pool without training or modifying old data.

Includes all new crops at the user's request. Human-reviewed decisions override
proposals; unreviewed OK/no_disk proposals retain explicit model provenance.
The historical ha_fit duplicate view is excluded; legacy validation stays separate.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

try:
    from .classifier_dataset import REGISTRY
except ImportError:
    from classifier_dataset import REGISTRY


LABELS = ('ng', 'no_disk', 'ok')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def build(old_dataset, new_review, output, page_template, review_completion=None):
    if REGISTRY.exists():
        raise ValueError('A canonical classifier dataset is active. Update DATASET_CLASSIFIER directly; do not create another prepared pool.')
    if output.exists():
        raise FileExistsError('Use a new folder; prepared pools and reviews are never overwritten')
    old_manifest = old_dataset / 'manifest.json'
    items_file, decisions_file = new_review / 'items.json', new_review / 'decisions.json'
    frozen = {str(p): p.read_bytes() for p in (old_manifest, items_file, decisions_file)}
    old = json.loads(frozen[str(old_manifest)])
    new = json.loads(frozen[str(items_file)])
    decisions = json.loads(frozen[str(decisions_file)])
    completion = None
    if review_completion:
        completion = json.loads(Path(review_completion).read_text(encoding='utf-8'))
        if (not completion.get('review_complete') or
                completion.get('items_sha256') != digest(items_file) or
                completion.get('decisions_sha256') != digest(decisions_file)):
            raise ValueError('Review completion does not match the frozen review inputs')
    if decisions['items_sha256'] != digest(items_file):
        raise ValueError('Human decisions belong to a different set of crop proposals')
    item_by_id = {item['id']: item for item in new['items']}
    if len(item_by_id) != len(new['items']) or not set(decisions['decisions']) <= set(item_by_id):
        raise ValueError('Duplicate or unknown crop IDs')
    ng_ids = {i['id'] for i in new['items'] if i['proposed_label'] == 'ng'}
    if not ng_ids <= set(decisions['decisions']):
        raise ValueError('Some original NG proposals remain unreviewed')
    historical_review = Path(next(r['source_crop'] for r in old['records'] if r.get('origin') == 'ha')).parents[2]
    old_review_items = json.loads((historical_review / 'items.json').read_text(encoding='utf-8'))
    old_contexts = {i['id']: i for i in old_review_items['items']}
    output.mkdir(parents=True)
    snapshots = output / 'snapshots'
    snapshots.mkdir()
    if completion:
        shutil.copy2(review_completion, snapshots / 'new_review_completion.json')
    for source, name in ((old_manifest, 'old_dataset_manifest.json'), (items_file, 'new_items.json'),
                         (decisions_file, 'new_decisions.json')):
        (snapshots / name).write_bytes(frozen[str(source)])
    records, excluded, source_hashes = [], [], {}
    copied_contexts = {}

    def copy_checked(source, relative, expected=None):
        source = Path(source)
        actual = digest(source)
        if expected and actual != expected:
            raise ValueError('Source crop hash changed: ' + str(source))
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if digest(target) != actual:
                raise ValueError('Conflicting destination: ' + str(target))
        else:
            shutil.copy2(source, target)
        if digest(target) != actual:
            raise OSError('Copy checksum mismatch: ' + str(target))
        source_hashes[str(source)] = actual
        return actual

    def copy_context(context_root, item, origin):
        result = {}
        for key in ('context', 'full_context'):
            if not item or not item.get(key):
                continue
            source = context_root / item[key]
            if not source.is_file():
                raise FileNotFoundError(source)
            if str(source) not in copied_contexts:
                relative = 'context/' + origin + '__' + source.name
                copy_checked(source, relative)
                copied_contexts[str(source)] = relative
            result[key] = copied_contexts[str(source)]
        return result

    for r in old['records']:
        if r.get('origin') == 'ha_fit':
            continue
        origin = {'ha': 'old_ha', 'legacy_train': 'old_legacy_train', 'legacy_val': 'legacy_val'}[r['origin']]
        pool = 'evaluation/legacy_val' if origin == 'legacy_val' else 'train_pool'
        filename = origin + '__' + r['id'] + Path(r['image']).suffix.lower()
        relative = f'{pool}/{r["label"]}/{filename}'
        sha = copy_checked(old_dataset / r['image'], relative, r['sha256'])
        record = {'id': origin + '__' + r['id'], 'original_id': r['id'], 'label': r['label'],
                  'origin': origin, 'role': 'evaluation_only' if origin == 'legacy_val' else 'training_pool',
                  'image': relative, 'sha256': sha, 'source_dataset': str(old_dataset),
                  'source_image': str(old_dataset / r['image']), 'original_source_crop': r.get('source_crop'),
                  'historical_split': r.get('previous_split', r.get('split')),
                  'camera': r.get('camera'), 'frame': r.get('frame'), 'region': r.get('region'), 'slot': r.get('slot'),
                  'label_source': 'historical_human_review' if origin == 'old_ha' else 'historical_dataset_label',
                  'previous_review': r.get('review'), 'correction_review': r.get('post_evaluation_review')}
        if origin == 'old_ha':
            record.update(copy_context(historical_review, old_contexts.get(r['id']), origin))
        records.append(record)
    for item in new['items']:
        decision = decisions['decisions'].get(item['id'])
        label = decision['label'] if decision else item['proposed_label']
        if label not in LABELS:
            excluded.append({'id': item['id'], 'label': label, 'reason': 'uncertain_or_excluded_by_human'})
            continue
        if decision is None and item['proposed_label'] not in ('ok', 'no_disk'):
            raise ValueError('Unreviewed NG cannot enter the pool')
        relative = f'train_pool/{label}/new_factory__{item["id"]}.png'
        sha = copy_checked(new_review / item['image'], relative, item['sha256'])
        record = {'id': 'new_factory__' + item['id'], 'original_id': item['id'], 'label': label,
                  'origin': 'new_factory', 'role': 'training_pool', 'image': relative, 'sha256': sha,
                  'source_dataset': str(new_review), 'source_image': str(new_review / item['image']),
                  'raw_frame': item.get('source'), 'camera': item['camera'], 'frame': item['frame'],
                  'region': item['region'], 'slot': item['slot'],
                  'label_source': ('new_human_review' if decision else
                                   'new_human_reviewed_model_default' if completion else 'model_proposal_user_requested_inclusion'),
                  'human_reviewed': bool(decision) or bool(completion), 'review': decision,
                  'review_method': ('explicit_saved_decision' if decision else
                                    'user_declared_complete_correction_only_review' if completion else 'unreviewed_model_proposal'),
                  'original_prediction': item['proposed_label'], 'original_confidence': item['confidence'],
                  'original_probabilities': item['probabilities'],
                  'capture_group': None, 'split': 'unassigned_pending_dataset_approval'}
        record.update(copy_context(new_review, item, 'new_factory'))
        records.append(record)
    source_hashes.update({p: hashlib.sha256(raw).hexdigest() for p, raw in frozen.items()})
    for path, sha in source_hashes.items():
        if digest(path) != sha:
            raise ValueError('A source file changed during preparation: ' + path)
    hashes = defaultdict(list)
    for r in records:
        hashes[r['sha256']].append(r)
    conflicts = [[r['id'] for r in group] for group in hashes.values() if len({r['label'] for r in group}) > 1]
    cross_role = [[r['id'] for r in group] for group in hashes.values() if len({r['role'] for r in group}) > 1]
    if conflicts or cross_role:
        raise ValueError(f'Conflicting labels or train/evaluation exact duplicates: {conflicts[:3]}, {cross_role[:3]}')
    duplicate_groups = []
    for sha, group in hashes.items():
        if len(group) > 1:
            duplicate_groups.append({'sha256': sha, 'ids': [r['id'] for r in group],
                                     'frames': sorted({r['frame'] for r in group if r.get('frame')})})
            for r in group:
                r['exact_duplicate_group'] = sha
    counts = {origin: dict(Counter(r['label'] for r in records if r['origin'] == origin))
              for origin in ('old_ha', 'old_legacy_train', 'new_factory', 'legacy_val')}
    manifest = {'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
                'status': ('prepared_review_complete_training_authorized' if completion and completion.get('training_authorized')
                           else 'prepared_waiting_for_user_dataset_verification_and_training_permission'),
                'training_authorized': bool(completion and completion.get('training_authorized')), 'training_started': False,
                'new_review_completion': completion,
                'preparation_authorization': 'User requested all new OK/no_disk positions in addition to reviewed NG proposals.',
                'old_dataset': str(old_dataset), 'new_review': str(new_review),
                'reviewed_new_original_classes': dict(Counter(item_by_id[k]['proposed_label'] for k in decisions['decisions'])),
                'new_human_labels': dict(Counter(d['label'] for d in decisions['decisions'].values())),
                'new_label_provenance_counts': dict(Counter(r['label_source'] for r in records if r['origin'] == 'new_factory')),
                'counts_by_origin': counts,
                'training_pool_counts': dict(Counter(r['label'] for r in records if r['role'] == 'training_pool')),
                'training_pool_size': sum(r['role'] == 'training_pool' for r in records),
                'evaluation_size': sum(r['role'] == 'evaluation_only' for r in records),
                'historical_ha_fit_duplicate_view_copied': False,
                'source_files_and_manifests_unchanged': True,
                'source_hashes': source_hashes, 'exact_duplicate_groups_within_role': sum(len(g) > 1 for g in hashes.values()),
                'exact_duplicate_groups': duplicate_groups,
                'conflicting_label_hashes': 0, 'train_evaluation_exact_duplicate_groups': 0,
                'new_split_policy': 'Not assigned yet. Group real capture cycles and near-duplicates before training; never split individual crops.',
                'evaluation_note': 'Legacy retention benchmark is historical, not an independent test; former HA splits were all learned by the current full-HA model.',
                'excluded': excluded, 'records': records}
    dump(output / 'manifest.json', manifest)
    dump(output / 'catalog.json', {'counts': counts, 'records': records,
                                 'training_pool_size': manifest['training_pool_size'], 'evaluation_size': manifest['evaluation_size']})
    with (output / 'labels.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        fields = ('id', 'origin', 'role', 'label', 'label_source', 'frame', 'region', 'slot', 'image', 'source_image', 'sha256')
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: r.get(key) for key in fields} for r in records)
    shutil.copy2(page_template, output / 'index.html')
    print(json.dumps({k: manifest[k] for k in ('status', 'counts_by_origin', 'new_label_provenance_counts',
          'training_pool_size', 'evaluation_size', 'exact_duplicate_groups_within_role')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-dataset', type=Path, required=True)
    parser.add_argument('--new-review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--review-completion', type=Path, help='User completion receipt bound to exact items and decisions hashes')
    args = parser.parse_args()
    build(args.old_dataset.resolve(), args.new_review.resolve(), args.output.resolve(),
          Path(__file__).with_name('classifier_pool_catalog.html'), args.review_completion)
