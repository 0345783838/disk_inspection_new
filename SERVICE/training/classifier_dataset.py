"""Single writable classifier dataset: labels, files and views commit together."""
from collections import Counter, defaultdict
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
import uuid

LABELS = ('ng', 'no_disk', 'ok')
PROJECT = Path(__file__).resolve().parents[2]
REGISTRY = PROJECT / '.work/ACTIVE_CLASSIFIER_DATASET.json'


class StaleRevision(ValueError):
    pass


class DatasetLock:
    """Reentrant process/thread lock for concurrent canonical editor windows."""
    def __init__(self, root):
        self.path = Path(root) / '.canonical-write.lock'
        self.thread = threading.RLock()
        self.depth = 0
        self.stream = None

    def __enter__(self):
        self.thread.acquire()
        try:
            if self.depth == 0:
                self.stream = self.path.open('a+b')
                if self.stream.seek(0, 2) == 0:
                    self.stream.write(b'0'); self.stream.flush()
                deadline = time.monotonic() + 30
                while True:
                    try:
                        self.stream.seek(0)
                        if os.name == 'nt':
                            import msvcrt
                            msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
                        else:
                            import fcntl
                            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError('Dataset is busy; try saving again')
                        time.sleep(.05)
            self.depth += 1
            return self
        except Exception:
            if self.depth == 0 and self.stream:
                self.stream.close(); self.stream = None
            self.thread.release()
            raise

    def __exit__(self, *exc):
        self.depth -= 1
        try:
            if self.depth == 0:
                self.stream.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.stream, fcntl.LOCK_UN)
                self.stream.close(); self.stream = None
        finally:
            self.thread.release()


_dataset_locks = {}
_dataset_locks_guard = threading.Lock()


def dataset_lock(root):
    with _dataset_locks_guard:
        return _dataset_locks.setdefault(str(root), DatasetLock(root))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_text(path, text):
    path = Path(path)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.canonical-tmp')
    try:
        temp.write_text(text, encoding='utf-8')
        for attempt in range(9):
            try:
                os.replace(temp, path)
                break
            except PermissionError as exc:
                # Windows HTTP/file readers can briefly hold a handle without
                # FILE_SHARE_DELETE. Retry sharing violations, not other errors.
                if getattr(exc, 'winerror', None) not in (5, 32, 33) or attempt == 8:
                    raise
                time.sleep(min(.02 * 2 ** attempt, .25))
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass


def atomic_json(path, value, *, compact=False):
    options = {'separators': (',', ':')} if compact else {'indent': 2}
    atomic_text(path, json.dumps(value, ensure_ascii=False, **options))


def active_dataset(requested=None, required_split_policy=None):
    """Prevent old prepared pools or an in-progress review from training silently."""
    if not REGISTRY.exists():
        if requested is None:
            raise ValueError('No canonical classifier dataset registered')
        return Path(requested).resolve()
    registry = read(REGISTRY)
    path = Path(registry['path']).resolve()
    if requested is not None and Path(requested).resolve() != path:
        raise ValueError('Use the canonical classifier dataset: ' + str(path))
    manifest = read(path / 'manifest.json')
    actual_policy = manifest.get('split_policy', {}).get('id', 'legacy_val_only')
    if required_split_policy is not None and actual_policy != required_split_policy:
        raise ValueError('Trainer split policy is incompatible with the canonical dataset: ' + actual_policy
                         + '. Use the canonical train/val assignments; do not reconstruct legacy validation.')
    partial = (manifest.get('review_status') == 'partial_review_training_authorized'
               and manifest.get('training_authorization', {}).get('incomplete_review_acknowledged') is True
               and manifest.get('training_authorization', {}).get('canonical_revision') == manifest.get('canonical_revision'))
    if (manifest.get('review_status') != 'complete' and not partial) or not manifest.get('training_authorized'):
        raise ValueError('Canonical dataset is being reviewed; finish labels and authorize training first')
    return path


class CanonicalDataset:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest_path = self.root / 'manifest.json'
        self.journal = self.root / '.canonical-transaction.json'
        self.lock = dataset_lock(self.root)
        with self.lock:
            self.recover()
        if not read(self.manifest_path).get('is_canonical'):
            raise ValueError('Dataset is not registered as canonical')

    def path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Path outside canonical dataset')
        return path

    @staticmethod
    def refresh(manifest):
        rows = manifest['records']
        origins = list(dict.fromkeys(['old_ha', 'old_legacy_train', 'new_factory', 'legacy_val'] + [r['origin'] for r in rows]))
        manifest['counts_by_origin'] = {origin: dict(Counter(r['label'] for r in rows if r['origin'] == origin))
                                      for origin in origins}
        manifest['training_pool_counts'] = dict(Counter(r['label'] for r in rows if r['role'] == 'training_pool'))
        manifest['training_pool_size'] = sum(r['role'] == 'training_pool' for r in rows)
        manifest['evaluation_size'] = sum(r['role'] == 'evaluation_only' for r in rows)
        manifest['validation_counts'] = dict(Counter(r['label'] for r in rows if r['role'] == 'evaluation_only'))
        manifest['counts_by_role_origin'] = {role: dict(Counter(r['origin'] for r in rows if r['role'] == role))
                                            for role in ('training_pool', 'evaluation_only')}
        manifest['new_label_provenance_counts'] = dict(Counter(r['label_source'] for r in rows if r['origin'] == 'new_factory'))
        manifest['canonical_excluded_count'] = len(manifest.get('excluded_records', [])) + len(manifest.get('excluded', []))

    def write_views(self, manifest):
        catalog = {k: manifest[k] for k in ('counts_by_origin', 'training_pool_size', 'evaluation_size')}
        catalog.update(counts=manifest['counts_by_origin'], records=manifest['records'],
                       canonical_revision=manifest['canonical_revision'])
        atomic_json(self.root / 'catalog.json', catalog, compact=True)
        fields = ('id', 'origin', 'role', 'split', 'camera', 'label', 'label_source', 'frame', 'region', 'slot',
                  'target_group', 'split_group', 'image', 'sha256')
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r.get(k) for k in fields} for r in manifest['records'])
        atomic_text(self.root / 'labels.csv', '\ufeff' + stream.getvalue())

    def recover(self):
        if not self.journal.exists():
            return
        transaction = read(self.journal)
        current = read(self.manifest_path)
        if current.get('canonical_revision', 0) != transaction['target_revision']:
            for addition in transaction.get('copies', []):
                if addition.get('temporary'):
                    self.path(addition['temporary']).unlink(missing_ok=True)
                target = self.path(addition['after'])
                if target.exists():
                    if hashlib.sha256(target.read_bytes()).hexdigest() != addition['sha256']:
                        raise RuntimeError('Cannot remove changed uncommitted addition: ' + addition['after'])
                    target.unlink()
            for move in reversed(transaction['moves']):
                before, after = self.path(move['before']), self.path(move['after'])
                if after.exists() and not before.exists():
                    before.parent.mkdir(parents=True, exist_ok=True)
                    after.rename(before)
                elif not before.exists():
                    raise RuntimeError('Cannot recover dataset transaction: ' + move['before'])
            before_manifest = transaction['before_manifest']
            if current != before_manifest:
                atomic_json(self.manifest_path, before_manifest, compact=True)
            current = before_manifest
        self.write_views(current)
        self.journal.unlink()

    def manifest(self):
        with self.lock:
            return read(self.manifest_path)

    def authorize_training(self, receipt, expected_revision):
        """Record explicit user authorization without claiming labels fully reviewed."""
        with self.lock:
            self.recover()
            m = read(self.manifest_path)
            if m['canonical_revision'] != expected_revision:
                raise StaleRevision('Dataset changed before training authorization')
            if (receipt.get('canonical_revision') != expected_revision
                    or receipt.get('incomplete_review_acknowledged') is not True
                    or not str(receipt.get('user_statement', '')).strip()):
                raise ValueError('Explicit revision-bound authorization receipt required')
            m.update(review_status='partial_review_training_authorized', training_authorized=True,
                     training_started=False, training_authorization=deepcopy(receipt),
                     updated_at=datetime.now(timezone.utc).isoformat())
            atomic_json(self.manifest_path, m)
            return m

    @staticmethod
    def public_record(r):
        keys = ('id', 'original_id', 'label', 'origin', 'role', 'image', 'context', 'full_context',
                'label_source', 'frame', 'region', 'slot', 'sha256', 'canonical_updated_at', 'split', 'capture_group', 'camera', 'target_group',
                'human_reviewed', 'review', 'review_import')
        return {k: r.get(k) for k in keys}

    def state(self):
        with self.lock:
            m = read(self.manifest_path)
            return {'revision': m['canonical_revision'], 'dataset_path': str(self.root),
                    'review_status': m['review_status'], 'training_authorized': m['training_authorized'],
                    'records': [self.public_record(r) for r in m['records'] + m.get('excluded_records', [])],
                    'counts': m['counts_by_origin'], 'train_count': m['training_pool_size'],
                    'split_policy': m.get('split_policy'),
                    'counts_by_role_origin': m.get('counts_by_role_origin', {}),
                    'validation_count': m['evaluation_size'], 'excluded_count': m.get('canonical_excluded_count', 0)}

    def save(self, changes, expected_revision, additions=None, repartition=None, real_intake=None):
        """Commit labels and additions in the single canonical transaction.

        Additions are {source: external image path, record: canonical metadata}.
        Synthetic additions inherit train parents. Confirmed real crops stay
        reference_only with no train/val split until an authorized repartition.
        A revision-bound real_intake can assign new crops and promote existing
        references in one transaction without moving any existing train/val row.
        The journal rolls back copied files as well as label moves on failure.
        """
        with self.lock:
            self.recover()
            before_manifest = self.manifest_path.read_text(encoding='utf-8')
            m = json.loads(before_manifest)
            if expected_revision != m['canonical_revision']:
                raise StaleRevision('Dataset changed in another tab; reload and choose the label again')
            additions = [] if additions is None else additions
            if real_intake is not None and (changes or repartition is not None or not isinstance(real_intake, dict)):
                raise ValueError('Real intake cannot include label edits or a full repartition')
            if (not isinstance(changes, list) or not isinstance(additions, list)
                    or (real_intake is None and repartition is None and not 1 <= len(changes) + len(additions) <= 1000)):
                raise ValueError('Expected 1..1000 label changes or additions')
            if repartition is not None and (changes or additions or not isinstance(repartition, dict)):
                raise ValueError('Repartition must be an exclusive full-dataset transaction')
            records = m['records'] + m.get('excluded_records', [])
            by_id = {r['id']: r for r in records}
            intake_receipt = None
            if real_intake is not None:
                intake_receipt = real_intake['receipt']
                old_hash = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()
                policy = real_intake['policy']
                if (intake_receipt.get('canonical_revision') != expected_revision
                        or intake_receipt.get('manifest_sha256') != old_hash
                        or not intake_receipt.get('user_statement', '').strip()
                        or policy.get('id') != m.get('split_policy', {}).get('id')):
                    raise ValueError('Real intake requires the exact current manifest, acceptance and policy')
            aliases = defaultdict(list)
            for r in records:
                aliases[r['sha256']].append(r['id'])
            copies, new_records = [], []
            verified_evidence = {}
            known_ids = set(by_id)
            known_hashes = set(aliases)
            known_decoded = {r.get('decoded_sha256') for r in records}
            new_labels_by_hash = {}
            for addition in additions:
                r = deepcopy(addition['record'])
                identifier = r.get('id', '')
                if not re.fullmatch(r'[A-Za-z0-9_.-]{1,220}', identifier) or identifier in known_ids:
                    raise ValueError('Invalid or existing addition ID')
                real_review = r.get('review_import', {}).get('kind') == 'real_classifier_review'
                parent = by_id.get(r.get('synthetic', {}).get('parent_id'))
                if real_review:
                    split = r.get('split')
                    role = {'train': 'training_pool', 'val': 'evaluation_only'}.get(split)
                    assigned = intake_receipt is not None and role is not None
                    allowed_provenance = ((r.get('label_source') == 'user_batch_reviewed_model_default'
                                           and r.get('label') in ('ng', 'no_disk'))
                                          or (r.get('label_source') == 'user_assumed_model_ok'
                                              and r.get('label') == 'ok'))
                    valid_assignment = (assigned and r.get('label') in LABELS and allowed_provenance
                                        and r.get('human_reviewed') is False and r.get('role') == role
                                        and r.get('canonical_original_role') == role)
                    valid_reference = (not assigned and r.get('human_reviewed') is True
                                       and r.get('canonical_original_role') == 'reference_only'
                                       and r.get('role') == ('reference_only' if r['label'] in LABELS else 'excluded')
                                       and split == (None if r['label'] in LABELS else 'excluded'))
                    if (r.get('is_synthetic') is not False or not (valid_assignment or valid_reference)
                            or r.get('label') not in (*LABELS, 'uncertain', 'exclude')
                            or r.get('camera') not in (1, 2) or not r.get('target_group')
                            or not r.get('capture_group') or not r.get('split_group')):
                        raise ValueError('Invalid reference review or authorized real intake row')
                    provenance = r['review_import']
                    for key, hash_key in (('raw_frame', 'raw_frame_sha256'), ('proposal_manifest', 'proposal_manifest_sha256')):
                        evidence = Path(provenance.get(key, '')).resolve()
                        if evidence not in verified_evidence and evidence.is_file():
                            verified_evidence[evidence] = hashlib.sha256(evidence.read_bytes()).hexdigest()
                        if verified_evidence.get(evidence) != provenance.get(hash_key) or not evidence.is_file():
                            raise ValueError('Real review provenance missing or changed: ' + key)
                    m.setdefault('role_directories', {'training_pool': 'train_pool', 'evaluation_only': 'evaluation/legacy_val'}).setdefault('reference_only', 'reference')
                elif (intake_receipt is not None or not r.get('is_synthetic') or parent is None or parent.get('split') != 'train'
                        or parent.get('role') != 'training_pool' or r.get('split') != 'train'
                        or r.get('role') != 'training_pool' or r.get('label') not in LABELS
                        or r.get('human_reviewed') is not True):
                    raise ValueError('Addition must be a reviewed synthetic descendant of an existing train row')
                for key in (() if real_review else ('origin', 'camera', 'frame', 'region', 'slot', 'target_group', 'capture_group', 'split_group')):
                    if r.get(key) != parent.get(key):
                        raise ValueError('Addition changes parent target/split provenance: ' + key)
                source = Path(addition['source']).resolve()
                if source.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.bmp'):
                    raise ValueError('Unsupported addition image format')
                if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != r.get('sha256'):
                    raise ValueError('Addition image missing or changed')
                if (not r.get('decoded_sha256') or (not real_review and (r['sha256'] in known_hashes
                        or r['decoded_sha256'] in known_decoded))):
                    raise ValueError('Duplicate addition or missing decoded fingerprint')
                if real_review and any(by_id[i]['label'] != r['label'] for i in aliases[r['sha256']]):
                    raise ValueError('Identical canonical crop has a different label; review its existing record first')
                if real_review and new_labels_by_hash.get(r['sha256'], r['label']) != r['label']:
                    raise ValueError('Identical new crops cannot receive opposing labels')
                base = 'excluded' if r['label'] not in LABELS else m.get('role_directories', {}).get(r['role'], 'train_pool')
                relative = f"{base}/{r['label']}/{identifier}{source.suffix.lower()}"
                if r.get('image') != relative or self.path(relative).exists():
                    raise ValueError('Addition destination invalid or already exists')
                if source.is_relative_to(self.root):
                    raise ValueError('Addition source must be external to the active dataset')
                copies.append(dict(source=str(source), after=relative, sha256=r['sha256'],
                                   temporary=relative + '.' + uuid.uuid4().hex + '.canonical-tmp'))
                new_records.append(r)
                new_labels_by_hash[r['sha256']] = r['label']
                known_ids.add(identifier); known_hashes.add(r['sha256']); known_decoded.add(r['decoded_sha256'])
            planned = {}
            for change in changes:
                if not isinstance(change, dict) or change.get('id') not in by_id:
                    raise ValueError('Unknown crop ID')
                row = by_id[change['id']]
                for field in ('sha256', 'split'):
                    expected = 'expected_' + field
                    if expected in change and row.get(field) != change[expected]:
                        raise ValueError('Crop no longer matches the requested image or split: ' + change['id'])
                label = change.get('label')
                if label not in (*LABELS, 'exclude', 'uncertain'):
                    raise ValueError('Invalid label')
                action = change.get('review_action', 'confirm')
                targets = [change['id']] if action == 'unreview' else aliases[by_id[change['id']]['sha256']]
                if action == 'unreview' and any(by_id[i]['label'] != label for i in aliases[by_id[change['id']]['sha256']] if i != change['id']):
                    raise ValueError('Unreview cannot change the label of an identical canonical image')
                for identifier in targets:
                    if identifier in planned and planned[identifier]['label'] != label:
                        raise ValueError('Identical image copies cannot receive opposing labels')
                    if action not in ('confirm', 'unreview'):
                        raise ValueError('Invalid review action')
                    if action == 'unreview' and by_id[identifier].get('review_import', {}).get('kind') != 'real_classifier_review':
                        raise ValueError('Unreview is only supported for new real review crops')
                    planned[identifier] = {'label': label, 'notes': str(change.get('notes', ''))[:2000], 'review_action': action}
            moves, updated = [], []
            now = datetime.now(timezone.utc).isoformat()
            if real_intake is not None:
                for identifier, assignment in real_intake['promotions'].items():
                    r = by_id[identifier]
                    split, group = assignment['split'], assignment['split_group']
                    if (r['role'] != 'reference_only' or not r.get('review_import') or r['label'] not in LABELS
                            or r.get('human_reviewed') is not True or split not in ('train', 'val') or not group):
                        raise ValueError('Only explicitly reviewed real references may be promoted')
                    role = 'training_pool' if split == 'train' else 'evaluation_only'
                    before = self.path(r['image'])
                    relative = f"{m['role_directories'][role]}/{r['label']}/{before.name}"
                    if (self.path(relative).exists() or not before.is_file()
                            or hashlib.sha256(before.read_bytes()).hexdigest() != r['sha256']):
                        raise ValueError('Reference crop missing, changed, or destination exists: ' + identifier)
                    moves.append({'before': r['image'], 'after': relative})
                    r.update(split=split, role=role, canonical_original_role=role, image=relative, split_group=group,
                             canonical_updated_at=now)
                    r['review_import']['split_state'] = 'authorized_real_intake'
                    updated.append(r)
                active = [r for r in m['records'] + new_records if r.get('split') in ('train', 'val')]
                for key in ('sha256', 'decoded_sha256', 'target_group', 'capture_group', 'split_group'):
                    sides = defaultdict(set)
                    for r in active:
                        if r.get(key): sides[r[key]].add(r['split'])
                    if any(len(splits) > 1 for splits in sides.values()):
                        raise ValueError('Real intake leaks groups across train and val: ' + key)
                m['split_policy'] = deepcopy(policy)
                m['last_real_intake'] = {'receipt': deepcopy(intake_receipt), 'promoted': len(updated),
                                         'added': len(new_records), 'evidence': real_intake['evidence']}
            if repartition is not None:
                assignment = repartition.get('assignments', {})
                if set(assignment) != {r['id'] for r in m['records']} or any(v not in ('train','val') for v in assignment.values()):
                    raise ValueError('Repartition requires exactly all current active IDs')
                policy = repartition.get('policy', {})
                if not policy.get('id') or policy['id'] != m.get('split_policy', {}).get('id'):
                    raise ValueError('Repartition must preserve the target grouping policy')
                for key in ('sha256','decoded_sha256','target_group','capture_group','split_group'):
                    sides = defaultdict(set)
                    for r in m['records']:
                        if r.get(key): sides[r[key]].add(assignment[r['id']])
                    if any(len(v)>1 for v in sides.values()):
                        raise ValueError('Repartition splits an image/target family: ' + key)
                for r in m['records']:
                    if r.get('is_synthetic'):
                        parent_id = r.get('synthetic', {}).get('parent_id')
                        if assignment[r['id']] != 'train' or assignment.get(parent_id) != 'train':
                            raise ValueError('Synthetic images and original train parents must stay train')
                    split = assignment[r['id']]
                    role = 'training_pool' if split == 'train' else 'evaluation_only'
                    base = m.get('role_directories', {})[role]
                    relative = f"{base}/{r['label']}/{Path(r['image']).name}"
                    before = self.path(r['image']); after = self.path(relative)
                    if not before.is_file() or hashlib.sha256(before.read_bytes()).hexdigest() != r['sha256']:
                        raise ValueError('Repartition image missing or changed: ' + r['id'])
                    if before != after:
                        if after.exists(): raise ValueError('Repartition destination exists: ' + relative)
                        moves.append(dict(before=r['image'],after=relative))
                        r.setdefault('split_history', []).append(dict(before=r.get('split'),after=split,
                            updated_at=now,source='explicit_user_repartition',evidence=repartition.get('evidence')))
                        updated.append(r)
                    r.update(split=split,role=role,canonical_original_role=role,image=relative)
                m['split_policy'] = deepcopy(policy)
                m['last_repartition'] = dict(source_revision=expected_revision,
                    previous_manifest_sha256=hashlib.sha256(self.manifest_path.read_bytes()).hexdigest(),
                    user_instruction=repartition.get('user_statement'),evidence=repartition.get('evidence'),
                    all_labels_hashes_ids_preserved=True,training_started=False)
                m.pop('training_authorization',None)
            for identifier, change in planned.items():
                r = by_id[identifier]
                label = change['label']
                before = self.path(r['image'])
                if not before.is_file() or hashlib.sha256(before.read_bytes()).hexdigest() != r['sha256']:
                    raise ValueError('Image missing or changed: ' + identifier)
                role = r.get('canonical_original_role', r['role'])
                directories = m.get('role_directories', {'training_pool': 'train_pool',
                                                        'evaluation_only': 'evaluation/legacy_val',
                                                        'reference_only': 'reference'})
                base = directories[role] if label in LABELS else 'excluded'
                relative = f'{base}/{label}/{before.name}'
                after = self.path(relative)
                if before != after:
                    if after.exists():
                        raise ValueError('Destination already exists: ' + relative)
                    moves.append({'before': r['image'], 'after': relative})
                r.setdefault('canonical_original_role', role)
                r.setdefault('canonical_original_label', r['label'])
                r.setdefault('label_history', []).append({'from': r['label'], 'to': label, 'updated_at': now,
                                                         'notes': change['notes'], 'source': 'canonical_dataset_editor'})
                r.update(label=label, role=role if label in LABELS else 'excluded', image=relative,
                         human_reviewed=True, label_source='human_canonical_review', canonical_updated_at=now)
                if m.get('authoritative_splits'):
                    r['split'] = {'training_pool': 'train', 'evaluation_only': 'val'}.get(role) if label in LABELS else 'excluded'
                r['review'] = {'label': label, 'notes': change['notes'], 'updated_at': now,
                               'source': 'canonical_dataset_editor'}
                if change['review_action'] == 'unreview':
                    r.update(human_reviewed=False, label_source='model_proposal_pending_review')
                    r['review']['source'] = 'canonical_review_undo'
                updated.append(r)
            for r in new_records:
                r['canonical_updated_at'] = now
            updated.extend(new_records)
            m['records'] = [r for r in records if r['role'] != 'excluded'] + new_records
            m['excluded_records'] = [r for r in records if r['role'] == 'excluded']
            m.update(canonical_revision=expected_revision + 1, updated_at=now,
                     review_status='in_progress', training_authorized=False, training_started=False)
            m.pop('training_authorization', None)
            self.refresh(m)
            # Reuse the existing JSON for the before-image and keep the journal
            # readable by other running canonical editors during recovery.
            journal = json.dumps({'target_revision': m['canonical_revision'], 'moves': moves, 'copies': copies},
                                 ensure_ascii=False, separators=(',', ':'))
            atomic_text(self.journal, journal[:-1] + ',"before_manifest":' + before_manifest + '}')
            try:
                for addition in copies:
                    target = self.path(addition['after'])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = self.path(addition['temporary'])
                    with Path(addition['source']).open('rb') as source, temporary.open('xb') as dest:
                        shutil.copyfileobj(source, dest)
                    if hashlib.sha256(temporary.read_bytes()).hexdigest() != addition['sha256']:
                        raise ValueError('Addition copy changed: ' + addition['after'])
                    # Atomic exclusive creation: no partial target, no overwrite.
                    os.link(temporary, target)
                    temporary.unlink()
                for move in moves:
                    source, target = self.path(move['before']), self.path(move['after'])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source.rename(target)
                self.write_views(m)
                # The manifest revision is the commit marker; recovery rolls back until this is replaced.
                atomic_json(self.manifest_path, m, compact=True)
                self.journal.unlink()
            except Exception:
                self.recover()
                raise
            return {'revision': m['canonical_revision'], 'updated': [self.public_record(r) for r in updated],
                    'counts': m['counts_by_origin'], 'train_count': m['training_pool_size'],
                    'validation_count': m['evaluation_size'], 'excluded_count': m['canonical_excluded_count'],
                    'saved': len(updated)}
