"""Canonical edits change actual labels and class folders, with crash recovery."""
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from functools import partial
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from training.classifier_dataset import CanonicalDataset, StaleRevision, active_dataset, atomic_json, atomic_text, read
from training.serve_classifier_dataset import DatasetHandler


class CanonicalDatasetTests(unittest.TestCase):
    def split_fixture(self,root):
        store=self.fixture(root);m=store.manifest()
        m.update(authoritative_splits=True,role_directories={'training_pool':'train_pool','evaluation_only':'evaluation/legacy_val'},split_policy={'id':'fixture_groups'})
        for r in m['records']:r['split']='train' if r['role']=='training_pool' else 'val'
        atomic_json(store.manifest_path,m);store.write_views(m)
        return store

    def test_repartition_moves_alias_families_and_preserves_labels_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);store=self.split_fixture(root);before=store.manifest()
            proposal=dict(assignments=dict(a='val',b='val',c='train'),policy={'id':'fixture_groups'})
            store.save([],0,repartition=proposal);after=store.manifest()
            self.assertEqual([(r['id'],r['label'],r['sha256']) for r in before['records']],[(r['id'],r['label'],r['sha256']) for r in after['records']])
            self.assertTrue((root/'evaluation/legacy_val/ok/a.png').exists())
            self.assertTrue((root/'train_pool/no_disk/c.png').exists())
            self.assertEqual(read(root/'catalog.json')['records'],after['records'])
            self.assertEqual(after['canonical_revision'],1)
            self.assertFalse(after['training_authorized'])

    def test_repartition_rejects_incomplete_or_split_aliases_and_recovers_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);store=self.split_fixture(root);original=store.manifest()
            for assignments in [dict(a='val'),dict(a='val',b='train',c='train')]:
                with self.assertRaises(ValueError):store.save([],0,repartition=dict(assignments=assignments,policy={'id':'fixture_groups'}))
            actual=store.write_views;calls=[]
            def fail_once(m):
                calls.append(1)
                if len(calls)==1:raise OSError('transaction failed')
                actual(m)
            with patch.object(store,'write_views',side_effect=fail_once):
                with self.assertRaises(OSError):store.save([],0,repartition=dict(assignments=dict(a='val',b='val',c='train'),policy={'id':'fixture_groups'}))
            self.assertEqual(store.manifest(),original)
            self.assertTrue((root/'train_pool/ok/a.png').exists())

    def synthetic_addition(self, root, store):
        import copy
        source = root.parent / (root.name + '_synthetic.png')
        source.write_bytes(b'new synthetic content')
        self.addCleanup(lambda: source.unlink(missing_ok=True))
        r = copy.deepcopy(store.manifest()['records'][0])
        r.update(id='synthetic_a', image='train_pool/ng/synthetic_a.png', label='ng',
                 split='train', is_synthetic=True, human_reviewed=True, label_source='user_accepted_ai',
                 sha256=hashlib.sha256(source.read_bytes()).hexdigest(), decoded_sha256='synthetic-decoded',
                 synthetic={'parent_id':'a'})
        m = store.manifest(); m['records'][0]['split'] = 'train'
        atomic_json(store.manifest_path, m); store.write_views(m)
        return dict(source=str(source), record=r)

    def test_synthetic_addition_commits_views_and_revokes_authorization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); store=self.fixture(root); addition=self.synthetic_addition(root,store)
            store.authorize_training(dict(canonical_revision=0,incomplete_review_acknowledged=True,user_statement='train'),0)
            result=store.save([],0,additions=[addition])
            self.assertEqual((result['revision'],result['train_count'],result['validation_count']),(1,3,1))
            self.assertFalse(store.manifest()['training_authorized'])
            self.assertTrue((root/'train_pool/ng/synthetic_a.png').exists())
            self.assertEqual(read(root/'catalog.json')['records'],store.manifest()['records'])
            self.assertIn('synthetic_a',(root/'labels.csv').read_text(encoding='utf-8'))
            self.assertTrue(Path(addition['source']).exists())

    def test_synthetic_addition_failure_and_crash_restore_original_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); store=self.fixture(root); addition=self.synthetic_addition(root,store)
            original=store.write_views; calls=[]
            def fail_once(m):
                calls.append(1)
                if len(calls)==1:raise OSError('disk error')
                original(m)
            with patch.object(store,'write_views',side_effect=fail_once):
                with self.assertRaises(OSError):store.save([],0,additions=[addition])
            self.assertEqual(store.state()['revision'],0)
            self.assertFalse((root/'train_pool/ng/synthetic_a.png').exists())
            self.assertFalse(store.journal.exists())
            # Simulate a process dying during the temporary copy, before target installation.
            temporary='train_pool/ng/crash.canonical-tmp'
            (root/temporary).write_bytes(b'partial')
            atomic_json(store.journal,dict(before_manifest=store.manifest(),target_revision=1,moves=[],
                copies=[dict(after=addition['record']['image'],temporary=temporary,sha256=addition['record']['sha256'])]))
            recovered=CanonicalDataset(root)
            self.assertFalse((root/temporary).exists())
            self.assertEqual(recovered.state()['revision'],0)

    def test_additions_reject_stale_revision_val_parent_duplicates_and_changed_groups(self):
        from copy import deepcopy
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); store=self.fixture(root); addition=self.synthetic_addition(root,store)
            for mutate in [lambda r:r['synthetic'].update(parent_id='c'),
                           lambda r:r.update(split='val'),lambda r:r.update(camera=2),
                           lambda r:r.update(id='a'),lambda r:r.update(id='../outside')]:
                bad=deepcopy(addition); mutate(bad['record'])
                with self.assertRaises(ValueError):store.save([],0,additions=[bad])
            with self.assertRaises(StaleRevision):store.save([],9,additions=[addition])
            with self.assertRaises(ValueError):store.save([],0,additions=[addition,addition])
            self.assertEqual(store.state()['revision'],0)

    def test_atomic_write_retries_windows_reader_lock_and_cleans_temporary_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); target=root/'status.json'; target.write_text('old')
            import os
            actual_replace=os.replace; calls=[]
            def transient_lock(source,destination):
                calls.append(True)
                if len(calls)==1:
                    error=PermissionError('Windows reader lock'); error.winerror=32
                    raise error
                actual_replace(source,destination)
            with patch('training.classifier_dataset.os.replace',side_effect=transient_lock), patch('training.classifier_dataset.time.sleep'):
                atomic_text(target,'new')
            self.assertEqual(target.read_text(),'new')
            self.assertEqual(len(calls),2)
            self.assertEqual(list(root.glob('*.canonical-tmp')),[])

    def fixture(self, root):
        rows = []
        for identifier, role, label, data in [('a', 'training_pool', 'ok', b'alias'),
                                               ('b', 'training_pool', 'ok', b'alias'),
                                               ('c', 'evaluation_only', 'no_disk', b'other')]:
            base = 'train_pool' if role == 'training_pool' else 'evaluation/legacy_val'
            path = root / base / label / (identifier + '.png')
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
            rows.append(dict(id=identifier, original_id=identifier, image=path.relative_to(root).as_posix(),
                             sha256=hashlib.sha256(data).hexdigest(), origin='new_factory' if role == 'training_pool' else 'legacy_val',
                             label=label, role=role, label_source='fixture'))
        m = dict(is_canonical=True, canonical_revision=0, review_status='in_progress', training_authorized=False,
                 records=rows, excluded=[], excluded_records=[])
        CanonicalDataset.refresh(m); atomic_json(root / 'manifest.json', m)
        store = CanonicalDataset(root); store.write_views(m)
        return store

    def test_direct_edit_moves_images_and_syncs_aliases_manifest_catalog_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); store = self.fixture(root)
            result = store.save([dict(id='a', label='ng')], 0)
            self.assertEqual(result['saved'], 2)
            self.assertEqual(read(root / 'manifest.json')['canonical_revision'], 1)
            self.assertEqual(read(root / 'catalog.json')['canonical_revision'], 1)
            for identifier in ('a', 'b'):
                self.assertTrue((root / f'train_pool/ng/{identifier}.png').is_file())
                self.assertFalse((root / f'train_pool/ok/{identifier}.png').exists())
            self.assertIn('train_pool/ng/a.png', (root / 'labels.csv').read_text(encoding='utf-8'))
            self.assertFalse((root / 'decisions.json').exists())
            self.assertEqual(CanonicalDataset(root).state()['train_count'], 2)

    def test_stale_tab_and_conflicting_alias_labels_do_not_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); store = self.fixture(root)
            before = (root / 'manifest.json').read_bytes()
            with self.assertRaises(StaleRevision): store.save([dict(id='a', label='ng')], 5)
            with self.assertRaises(ValueError): store.save([dict(id='a', label='ng'), dict(id='b', label='ok')], 0)
            self.assertEqual((root / 'manifest.json').read_bytes(), before)

    def test_exclude_and_restore_preserve_original_train_validation_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); store = self.fixture(root)
            store.save([dict(id='c', label='uncertain')], 0)
            self.assertEqual(store.state()['validation_count'], 0)
            self.assertTrue((root / 'excluded/uncertain/c.png').is_file())
            store.save([dict(id='c', label='ok')], 1)
            self.assertEqual(store.state()['validation_count'], 1)
            self.assertTrue((root / 'evaluation/legacy_val/ok/c.png').is_file())

    def test_failed_commit_rolls_back_file_moves_and_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); store = self.fixture(root)
            original = store.write_views; calls = []
            def fail_once(m):
                calls.append(True)
                if len(calls) == 1: raise OSError('fixture disk error')
                original(m)
            with patch.object(store, 'write_views', side_effect=fail_once):
                with self.assertRaises(OSError): store.save([dict(id='a', label='ng')], 0)
            self.assertEqual(store.state()['revision'], 0)
            self.assertTrue((root / 'train_pool/ok/a.png').exists())
            self.assertTrue((root / 'train_pool/ok/b.png').exists())
            self.assertFalse(store.journal.exists())

    def test_restart_recovers_interrupted_uncommitted_move(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); store = self.fixture(root); before = read(root / 'manifest.json')
            moved = root / 'train_pool/ng/a.png'; moved.parent.mkdir(parents=True)
            (root / 'train_pool/ok/a.png').rename(moved)
            atomic_json(store.journal, dict(before_manifest=before, target_revision=1,
                         moves=[dict(before='train_pool/ok/a.png', after='train_pool/ng/a.png')]))
            recovered = CanonicalDataset(root)
            self.assertEqual(recovered.state()['revision'], 0)
            self.assertTrue((root / 'train_pool/ok/a.png').exists())
            self.assertFalse(moved.exists())

    def test_training_uses_only_registered_dataset_after_review_authorization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fixture(root)
            registry = root / 'registry.json'; atomic_json(registry, dict(path=str(root)))
            with patch('training.classifier_dataset.REGISTRY', registry):
                with self.assertRaises(ValueError): active_dataset(root)
                m = read(root / 'manifest.json')
                m.update(review_status='complete', training_authorized=True)
                atomic_json(root / 'manifest.json', m)
                self.assertEqual(active_dataset(), root.resolve())
                with self.assertRaises(ValueError): active_dataset(root / 'old_dataset')

    def test_incomplete_review_requires_explicit_revision_bound_authorization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.fixture(root)
            registry=root/'registry.json'; atomic_json(registry,dict(path=str(root)))
            m=read(root/'manifest.json'); m.update(review_status='partial_review_training_authorized',training_authorized=True)
            atomic_json(root/'manifest.json',m)
            with patch('training.classifier_dataset.REGISTRY',registry):
                with self.assertRaises(ValueError): active_dataset()
                m['training_authorization']=dict(incomplete_review_acknowledged=True,canonical_revision=0)
                atomic_json(root/'manifest.json',m)
                self.assertEqual(active_dataset(),root.resolve())
                CanonicalDataset(root).save([dict(id='a',label='ng')],0)
                with self.assertRaises(ValueError): active_dataset()

    def test_http_edit_writes_dataset_and_rejects_stale_tabs_and_other_origins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); store = self.fixture(root)
            class Handler(DatasetHandler):
                def log_message(self, *args): pass
            Handler.store = store
            server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Handler, directory=str(root)))
            thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
            base = 'http://127.0.0.1:' + str(server.server_port)
            def post(revision, origin=base):
                data = json.dumps(dict(revision=revision, changes=[dict(id='a', label='ng')])).encode()
                return urlopen(Request(base + '/api/labels', data=data,
                                      headers={'Content-Type': 'application/json', 'Origin': origin}))
            try:
                with post(0) as response:
                    self.assertEqual(json.load(response)['saved'], 2)
                with urlopen(base + '/api/state') as response:
                    self.assertEqual(json.load(response)['revision'], 1)
                with self.assertRaises(HTTPError) as stale: post(0)
                self.assertEqual(stale.exception.code, 409)
                with self.assertRaises(HTTPError) as origin: post(1, 'http://other.invalid')
                self.assertEqual(origin.exception.code, 403)
                self.assertEqual(store.state()['revision'], 1)
                self.assertTrue((root / 'train_pool/ng/a.png').exists())
            finally:
                server.shutdown(); server.server_close(); thread.join()
