"""Read-only integrity audit of the registered canonical classifier dataset."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
from datetime import datetime, timezone

try:
    from .classifier_dataset import CanonicalDataset, REGISTRY, atomic_json, read
except ImportError:
    from classifier_dataset import CanonicalDataset, REGISTRY, atomic_json, read


def audit(root):
    root = Path(root)
    raw = (root / 'manifest.json').read_bytes()
    m = json.loads(raw)
    rows = m['records'] + m.get('excluded_records', [])
    errors, assets, hashes, frames = [], set(), defaultdict(list), defaultdict(set)
    if len({r['id'] for r in rows}) != len(rows):
        errors.append('Duplicate stable IDs')
    for r in rows:
        path = root / r['image']
        if not path.resolve().is_relative_to(root.resolve()):
            errors.append('Image escapes dataset: ' + r['id'])
            continue
        assets.add(r['image'])
        if not path.is_file():
            errors.append('Missing crop: ' + r['id'])
        elif hashlib.sha256(path.read_bytes()).hexdigest() != r['sha256']:
            errors.append('Hash mismatch: ' + r['id'])
        if path.parent.name != r['label']:
            errors.append('Class folder mismatch: ' + r['id'])
        role = r['role']
        expected = {'training_pool': 'train_pool/', 'evaluation_only': 'evaluation/',
                    'reference_only': 'reference/', 'excluded': 'excluded/'}
        expected.update({role: folder.rstrip('/') + '/' for role, folder in m.get('role_directories', {}).items()})
        if role not in expected or not r['image'].startswith(expected[role]):
            errors.append('Role folder mismatch: ' + r['id'])
        hashes[r['sha256']].append(r)
        if r.get('frame'):
            frames[r['frame']].add(role)
        for key in ('context', 'full_context'):
            if r.get(key):
                assets.add(r[key])
                asset = (root / r[key]).resolve()
                if not asset.is_relative_to(root.resolve()) or not asset.is_file():
                    errors.append('Missing/unsafe context: ' + r['id'] + ':' + key)
    group_overlaps = {}
    if m.get('authoritative_splits'):
        for r in rows:
            wanted = {'training_pool':'train','evaluation_only':'val','excluded':'excluded'}.get(r['role'])
            if r.get('split') != wanted:
                errors.append('Authoritative split disagrees with role: ' + r['id'])
            if r['role']=='evaluation_only' and r['origin'] not in m['split_policy']['validation_origins']:
                errors.append('Disallowed validation origin: ' + r['id'])
        for key in ('decoded_sha256','capture_group','split_group','target_group'):
            sides=defaultdict(set)
            for r in rows:
                if r.get(key) and r['role'] in ('training_pool','evaluation_only'):
                    sides[r[key]].add(r['role'])
            group_overlaps[key]=[k for k,v in sides.items() if len(v)>1]
            if group_overlaps[key]:errors.append('Train/validation group overlap: '+key)
    overlaps, conflicts = [], []
    for sha, copies in hashes.items():
        roles = {r['role'] for r in copies}
        if {'training_pool', 'evaluation_only'} <= roles:
            overlaps.append([r['id'] for r in copies])
        if len({r['label'] for r in copies}) > 1:
            conflicts.append([r['id'] for r in copies])
    if overlaps:
        errors.append('Exact train/validation hash overlap')
    if conflicts:
        errors.append('Conflicting identical-image labels')
    frame_overlap = [frame for frame, roles in frames.items()
                     if {'training_pool', 'evaluation_only'} <= roles]
    policy=m.get('split_policy',{})
    frame_overlap_allowed=(policy.get('id')=='camera_target_slot_legacy_ng_v1'
                           and policy.get('grouping_unit')=='origin_camera_target_slot'
                           and policy.get('allow_frame_overlap') is True)
    if frame_overlap_allowed:
        active=[r for r in rows if r['role'] in ('training_pool','evaluation_only')]
        if any(not r.get('target_group') for r in active):errors.append('Missing target-slot group')
        for r in active:
            if r['origin'] in ('old_ha','new_factory'):
                expected_camera=int(r['frame'].split('_')[0].removeprefix('Cam'))
                expected_target=f"{r['origin']}|Cam{expected_camera}|{r['region']}|slot{int(r['slot']):02}"
                if r.get('camera')!=expected_camera or r.get('target_group')!=expected_target:
                    errors.append('Incorrect target/camera provenance: '+r['id'])
        for origin in ('old_ha','new_factory'):
            for camera in (1,2):
                for split in ('train','val'):
                    labels={r['label'] for r in active if r['origin']==origin and r.get('camera')==camera and r['split']==split}
                    if labels!={'ng','no_disk','ok'}:errors.append(f'Missing source/camera/class coverage: {origin}/{camera}/{split}')
    if frame_overlap and not frame_overlap_allowed:
        errors.append('Source frame crosses train/validation')
    catalog = read(root / 'catalog.json')
    if catalog.get('canonical_revision') != m['canonical_revision'] or catalog['records'] != m['records']:
        errors.append('Catalog differs from canonical manifest')
    csv_rows = list(csv.DictReader(io.StringIO((root / 'labels.csv').read_text(encoding='utf-8-sig'))))
    if len(csv_rows) != len(m['records']) or any(
            any(c.get(k, '') != ('' if r.get(k) is None else str(r.get(k, ''))) for k in c)
            for c, r in zip(csv_rows, m['records'])):
        errors.append('CSV differs from canonical manifest')
    expected_manifest = json.loads(raw)
    CanonicalDataset.refresh(expected_manifest)
    for key in ('counts_by_origin', 'training_pool_counts', 'training_pool_size', 'evaluation_size',
                'new_label_provenance_counts', 'canonical_excluded_count'):
        if m.get(key) != expected_manifest[key]:
            errors.append('Stale manifest count: ' + key)
    listed = {r['image'] for r in rows}
    unlisted = []
    folders = set(('train_pool', 'evaluation', 'reference', 'excluded')) | set(m.get('role_directories', {}).values())
    for folder in folders:
        for path in (root / folder).rglob('*'):
            if path.is_file() and path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp'):
                if path.relative_to(root).as_posix() not in listed:
                    unlisted.append(path.relative_to(root).as_posix())
    if unlisted:
        errors.append('Unlisted class-folder images')
    if raw != (root / 'manifest.json').read_bytes():
        errors.append('Dataset changed during audit; retry')
    return dict(checked_at=datetime.now(timezone.utc).isoformat(), dataset_path=str(root.resolve()),
                canonical_revision=m['canonical_revision'], manifest_sha256=hashlib.sha256(raw).hexdigest(),
                passed=not errors, errors=errors, records_checked=len(rows), unique_assets_checked=len(assets),
                counts_by_origin=m['counts_by_origin'], roles=dict(Counter(r['role'] for r in rows)),
                role_class_counts={role: dict(Counter(r['label'] for r in rows if r['role'] == role))
                                   for role in sorted({r['role'] for r in rows})},
                validation_origins=dict(Counter(r['origin'] for r in rows if r['role'] == 'evaluation_only')),
                exact_duplicate_groups=sum(len(v) > 1 for v in hashes.values()),
                train_validation_hash_overlap=overlaps, conflicting_hash_labels=conflicts,
                train_validation_frame_overlap=frame_overlap, unlisted_images=unlisted,
                shared_frame_context_allowed=frame_overlap_allowed,
                shared_frame_context_count=len(frame_overlap),
                train_validation_group_overlap=group_overlaps,
                historical_exclusions_without_canonical_crop=m.get('excluded', []),
                review_status=m['review_status'], human_consistency_review_complete=False,
                limitation='Integrity checks verify files and metadata, not visual label correctness.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(read(REGISTRY)['path'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=True, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
