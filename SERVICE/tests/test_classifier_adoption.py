"""Curated source adoption and grouped holdout invariants, using temp fixtures."""
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training.adopt_classifier_dataset import choose_split, source_inventory
from training.audit_classifier_dataset import audit
from training.classifier_dataset import CanonicalDataset, active_dataset, atomic_json, read
import test_classifier_canonical as canonical_tests


class AdoptionTests(unittest.TestCase):
    def test_actual_folder_label_wins_and_missing_is_not_recovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'train/ng').mkdir(parents=True)
            (root/'train/ng/present.png').write_bytes(b'crop')
            rows=[dict(id=i,origin='old_ha',label='ok',split='train',image=f'train/ok/{i}.png',
                       sha256=hashlib.sha256(b'crop').hexdigest()) for i in ('present','removed')]
            atomic_json(root/'manifest.json',dict(records=rows))
            actual,report=source_inventory(root)
            self.assertEqual([(r['id'],r['label']) for r in actual],[('present','ng')])
            self.assertEqual(report['missing_from_selected_source'][0]['id'],'removed')
            self.assertEqual(report['actual_folder_overrides'][0]['folder_label'],'ng')

    def test_whole_bursts_and_decoded_aliases_never_cross_split(self):
        rows=[]
        for origin in ('old_ha','new_factory'):
            for burst in range(4):
                for index in range(30):
                    identifier=f'{origin}_{burst}_{index}'
                    rows.append(dict(id=identifier,origin=origin,label=('ng','no_disk','ok')[index%3],
                        sha256=identifier,decoded_sha256=identifier,frame=f'Cam1_Image_20261018{10+burst:02}0000000'))
        # Same decoded image joins capture groups, including across origins.
        rows[0]['decoded_sha256']=rows[120]['decoded_sha256']='same_pixels'
        a,b=deepcopy(rows),deepcopy(rows)
        policy=choose_split(a);self.assertEqual(policy,choose_split(b))
        for key in ('frame','capture_group','split_group','decoded_sha256'):
            groups={}
            for r in a:groups.setdefault(r[key],set()).add(r['split'])
            self.assertTrue(all(len(s)==1 for s in groups.values()),key)
        for origin in ('old_ha','new_factory'):
            self.assertEqual({r['label'] for r in a if r['origin']==origin and r['split']=='val'}, {'ng','no_disk','ok'})

    def test_new_layout_edits_exclude_restore_keep_split_and_old_trainer_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); store=canonical_tests.CanonicalDatasetTests().fixture(root)
            m=store.manifest()
            m.update(role_directories={'training_pool':'train','evaluation_only':'val'},authoritative_splits=True,
                     split_policy={'id':'ha_new_capture_holdout_v1','validation_origins':['old_ha','new_factory']})
            for r in m['records']:
                r['origin']='new_factory';r['split']='train' if r['role']=='training_pool' else 'val'
                new=f"{r['split']}/{r['label']}/{r['id']}.png"
                (root/new).parent.mkdir(parents=True,exist_ok=True);(root/r['image']).rename(root/new);r['image']=new
            CanonicalDataset.refresh(m);atomic_json(root/'manifest.json',m);store.write_views(m)
            self.assertTrue(audit(root)['passed'])
            store.save([dict(id='c',label='exclude')],0)
            self.assertEqual(store.manifest()['excluded_records'][0]['split'],'excluded')
            store.save([dict(id='c',label='ng')],1)
            self.assertTrue((root/'val/ng/c.png').is_file());self.assertTrue(audit(root)['passed'])
            m=store.manifest();m.update(review_status='complete',training_authorized=True)
            atomic_json(root/'manifest.json',m)
            registry=root/'registry.json';atomic_json(registry,dict(path=str(root)))
            with patch('training.classifier_dataset.REGISTRY',registry):
                self.assertEqual(active_dataset(),root.resolve())
                with self.assertRaisesRegex(ValueError,'incompatible'):
                    active_dataset(required_split_policy='legacy_val_only')
            m['records'][-1]['origin']='legacy_val';CanonicalDataset.refresh(m)
            atomic_json(root/'manifest.json',m);store.write_views(m)
            self.assertTrue(any('Disallowed validation origin' in e for e in audit(root)['errors']))
