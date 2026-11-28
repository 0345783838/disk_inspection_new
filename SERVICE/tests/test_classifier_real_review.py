"""New-crop review saves, undo, conflicts and recovery on isolated datasets only."""
import csv
import json
import sys
import tempfile
import unittest
import subprocess
from pathlib import Path
from unittest.mock import patch

from SERVICE.tests import test_classifier_canonical as canonical_fixtures
from training.classifier_dataset import CanonicalDataset, StaleRevision, atomic_json, read
from training.audit_classifier_dataset import audit

RUN=Path(__file__).resolve().parents[2]/'.work/classifier_factory_20261118'
sys.path.insert(0,str(RUN))
from review_store import ReviewStore, canonical_id


class RealReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.canonical=canonical_fixtures.CanonicalDatasetTests().fixture(self.root)
        m=self.canonical.manifest()
        m.update(authoritative_splits=True,role_directories={'training_pool':'train_pool','evaluation_only':'evaluation/legacy_val'},
                 split_policy={'id':'fixture_groups','validation_origins':['legacy_val']})
        for r in m['records']:r['split']='train' if r['role']=='training_pool' else 'val'
        atomic_json(self.canonical.manifest_path,m);self.canonical.write_views(m)
        self.review=ReviewStore(RUN,self.root)
        self.item=next(iter(self.review.items.values()))
        self.original=read(self.root/'manifest.json')

    def save(self,label,revision=0,**extra):
        return self.review.save({'revision':revision,'changes':[dict(id=self.item['id'],label=label,**extra)]})

    def test_confirm_persists_reference_only_all_views_and_preserves_old_rows(self):
        result=self.save('ok')
        m=read(self.root/'manifest.json');r=m['records'][-1]
        self.assertEqual(m['records'][:-1],self.original['records'])
        self.assertEqual((r['role'],r['split'],r['is_synthetic']),('reference_only',None,False))
        self.assertTrue(r['human_reviewed']);self.assertFalse(m['training_authorized'])
        self.assertEqual((m['training_pool_size'],m['evaluation_size']),(2,1))
        self.assertTrue((self.root/r['image']).is_file())
        self.assertEqual(read(self.root/'catalog.json')['records'],m['records'])
        with (self.root/'labels.csv').open(encoding='utf-8-sig',newline='') as stream:
            self.assertEqual(len(list(csv.DictReader(stream))),4)
        self.assertEqual(result['reviewed'],1)
        report=audit(self.root);self.assertTrue(report['passed'],report['errors'])

    def test_changed_label_undo_reload_and_uncertain_restore(self):
        first=self.save('ok')
        undone=self.review.save({'revision':1,'changes':first['undo_changes']})
        self.assertEqual(undone['reviewed'],0)
        self.assertFalse(ReviewStore(RUN,self.root).state()['records'][self.item['id']]['human_reviewed'])
        self.save('ng',2)
        self.save('uncertain',3)
        self.assertTrue((self.root/f"excluded/uncertain/{canonical_id(self.item['id'])}.png").is_file())
        self.save('no_disk',4)
        self.assertEqual(self.review.state()['records'][self.item['id']]['label'],'no_disk')
        report=audit(self.root);self.assertTrue(report['passed'],report['errors'])

    def test_first_exclusion_starts_in_excluded_records(self):
        self.save('exclude')
        manifest = self.canonical.manifest()
        self.assertEqual(manifest['records'], self.original['records'])
        record, = manifest['excluded_records']
        self.assertEqual((record['role'], record['split'], record['label']),
                         ('excluded', 'excluded', 'exclude'))
        self.assertEqual(read(self.root/'catalog.json')['records'], manifest['records'])
        self.assertTrue((self.root/record['image']).is_file())
        self.assertEqual(self.review.state()['reviewed'], 1)

    def test_stale_tab_and_failed_write_leave_truth_unchanged(self):
        self.save('ok')
        before=(self.root/'manifest.json').read_bytes()
        with self.assertRaises(StaleRevision):self.save('ng',0)
        self.assertEqual((self.root/'manifest.json').read_bytes(),before)
        original=self.review.store.write_views;calls=[]
        def fail_once(m):
            calls.append(True)
            if len(calls)==1:raise OSError('simulated disk failure')
            return original(m)
        with patch.object(self.review.store,'write_views',side_effect=fail_once):
            with self.assertRaises(OSError):self.save('ng',1)
        self.assertEqual((self.root/'manifest.json').read_bytes(),before)
        self.assertFalse(self.review.store.journal.exists())

    def test_addition_failure_rolls_back_copies_and_existing_rows(self):
        original=self.review.store.write_views;calls=[]
        def fail_once(m):
            calls.append(True)
            if len(calls)==1:raise OSError('simulated copy transaction failure')
            return original(m)
        with patch.object(self.review.store,'write_views',side_effect=fail_once):
            with self.assertRaises(OSError):self.save('ok')
        self.assertEqual(self.review.store.manifest(),self.original)
        self.assertFalse(list((self.root/'reference').rglob('*.png')))

    def test_rejects_changed_provenance_and_duplicate_request_ids(self):
        addition=self.review.addition(self.item,'ok')
        addition['record']['review_import']['raw_frame_sha256']='0'*64
        with self.assertRaises(ValueError):self.canonical.save([],0,additions=[addition])
        with self.assertRaises(ValueError):self.review.save({'revision':0,'changes':[{'id':self.item['id'],'label':'ok'}]*2})
        self.assertEqual(self.review.state()['reviewed'],0)

    def test_two_processes_cannot_commit_the_same_revision(self):
        code="""import sys
from training.classifier_dataset import CanonicalDataset,StaleRevision
s=CanonicalDataset(sys.argv[1])
try:
 s.save([{'id':'a','label':sys.argv[2]}],0)
 print('saved')
except StaleRevision:print('stale')
"""
        processes=[subprocess.Popen([sys.executable,'-B','-c',code,str(self.root),label],
                    stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for label in ('ng','no_disk')]
        results=[p.communicate(timeout=20) for p in processes]
        self.assertEqual(sorted(out.strip() for out,err in results),['saved','stale'],results)
        self.assertTrue(all(p.returncode==0 for p in processes),results)
        self.assertEqual(self.canonical.manifest()['canonical_revision'],1)

if __name__=='__main__':unittest.main()
