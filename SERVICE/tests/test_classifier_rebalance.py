"""Camera coverage, repeated-slot grouping and NG-heavy legacy import."""
from collections import Counter,defaultdict
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest

from training.rebalance_classifier_dataset import plan_split,select_legacy
from training.classifier_dataset import CanonicalDataset,atomic_json
from training.audit_classifier_dataset import audit


class RebalanceTests(unittest.TestCase):
    def rows(self):
        rows=[]
        for origin in ('old_ha','new_factory'):
            for camera in (1,2):
                for slot in range(1,21):
                    for frame in range(3):
                        i=f'{origin}_{camera}_{slot}_{frame}'
                        rows.append(dict(id=i,origin=origin,camera=camera,region='row1_bottom',slot=slot,
                            frame=f'Cam{camera}_Image_{origin}_{frame}',label=('ng','no_disk','ok')[(slot+frame)%3],
                            sha256=i,decoded_sha256=i,label_source='fixture',capture_group='old_burst'))
        for label in ('ng','no_disk','ok'):
            for i in range(20):
                identifier=f'legacy_{label}_{i}'
                rows.append(dict(id=identifier,origin='old_legacy_train',label=label,frame=None,
                    sha256=identifier,decoded_sha256=identifier,label_source='fixture'))
        return rows

    def test_every_camera_source_class_present_and_all_target_histories_together(self):
        a=self.rows();b=deepcopy(a)
        # Force a decoded alias to join two different target groups.
        a[0]['decoded_sha256']=a[3]['decoded_sha256']='alias';b=deepcopy(a)
        policy=plan_split(a,seeds=2);self.assertEqual(policy,plan_split(b,seeds=2))
        for counts in policy['strata_counts'].values():
            self.assertGreater(counts['train'],0);self.assertGreater(counts['val'],0)
        for key in ('target_group','split_group','decoded_sha256'):
            by=defaultdict(set)
            for r in a:by[r[key]].add(r['split'])
            self.assertTrue(all(len(v)==1 for v in by.values()))
        self.assertTrue(policy['allow_frame_overlap'])
        self.assertTrue(all(r['camera'] is None for r in a if r['origin']=='old_legacy_train'))

    def test_legacy_sampling_quotas_do_not_restore_exclusions_or_ambiguous_aliases(self):
        rows=[r for r in self.rows() if r['origin']=='old_legacy_train']
        rows[0]['decoded_sha256']=rows[20]['decoded_sha256']='conflict'
        selected,skipped=select_legacy(rows,[],excluded_ids=[rows[1]['id']],quotas={'ng':4,'no_disk':2,'ok':2})
        self.assertEqual(Counter(r['label'] for r in selected),{'ng':4,'no_disk':2,'ok':2})
        self.assertNotIn(rows[1]['id'],{r['id'] for r in selected})
        self.assertNotIn('conflict',{r['decoded_sha256'] for r in selected})
        self.assertEqual(selected,select_legacy(rows,[],[rows[1]['id']],{'ng':4,'no_disk':2,'ok':2})[0])

    def test_audit_requires_target_provenance_for_shared_frame_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=self.rows();policy=plan_split(rows,seeds=1)
            for r in rows:
                data=r['id'].encode();r['sha256']=hashlib.sha256(data).hexdigest()
                r['image']=f"{r['split']}/{r['label']}/{r['id']}.png"
                p=root/r['image'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
            m=dict(is_canonical=True,canonical_revision=10,records=rows,excluded=[],excluded_records=[],
                review_status='in_progress',authoritative_splits=True,split_policy=policy,
                role_directories={'training_pool':'train','evaluation_only':'val'})
            def save():
                CanonicalDataset.refresh(m);atomic_json(root/'manifest.json',m);CanonicalDataset(root).write_views(m)
            save();result=audit(root)
            self.assertTrue(result['passed'],result['errors']);self.assertGreater(result['shared_frame_context_count'],0)
            rows[0]['target_group']='wrong';save()
            self.assertTrue(any('Incorrect target/camera' in e for e in audit(root)['errors']))
