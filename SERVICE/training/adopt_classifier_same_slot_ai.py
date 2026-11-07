"""Adopt the ten user-accepted ImageGen candidates through CanonicalDataset.save."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classifier_dataset import CanonicalDataset, REGISTRY, atomic_json, read
from adopt_classifier_dataset import decoded_fingerprint
from audit_classifier_dataset import audit

PROJECT = Path(__file__).resolve().parents[2]
PILOT = PROJECT / '.work/classifier_ai_same_slot_pilot_20261029'
EVIDENCE = PROJECT / '.work/classifier_ai_same_slot_adoption_20261029'
USER_STATEMENT = 'ok cho vào train bổ sung đi'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    registry = read(REGISTRY)
    pool = Path(registry['path'])
    store = CanonicalDataset(pool)
    before = store.manifest()
    preview = read(PILOT / 'candidate_provenance.json')
    assert before['canonical_revision'] == preview['canonical_revision'] == 10
    assert digest(pool/'manifest.json') == preview['canonical_manifest_sha256']
    assert len(preview['records']) == 10
    if EVIDENCE.exists():
        raise FileExistsError('Adoption evidence exists; inspect before any retry')
    EVIDENCE.mkdir(parents=True)
    for name in ['manifest.json','catalog.json','labels.csv']:
        shutil.copy2(pool/name,EVIDENCE/('before_'+name))
    atomic_json(EVIDENCE/'registry_before.json',registry)
    integrity = audit(pool)
    atomic_json(EVIDENCE/'integrity_before.json',integrity)
    assert integrity['passed'], integrity['errors']
    sources = {r['id']:r for r in before['records']}
    stamp = datetime.now(timezone.utc).isoformat()
    ids = {r['id']:'ai_same_slot_20261029__'+r['id'].lower() for r in preview['records']}
    additions = []
    for generated in preview['records']:
        parent = sources[generated['source_id']]
        source = PILOT/generated['generated_image']
        assert parent['split']=='train' and parent['label']=='ng'
        assert digest(pool/parent['image'])==generated['source_sha256']==parent['sha256']
        assert digest(source)==generated['generated_sha256']
        record = {key:deepcopy(parent[key]) for key in
                  ['origin','camera','frame','region','slot','target_group','capture_group','split_group','collection_burst']
                  if key in parent}
        identifier = ids[generated['id']]
        record.update(id=identifier,original_id=generated['id'],label='ng',role='training_pool',
            split='train',canonical_original_role='training_pool',canonical_original_label='ng',
            image=f'train/ng/{identifier}.png',sha256=digest(source),decoded_sha256=decoded_fingerprint(source),
            label_source='human_accepted_ai_generation',human_reviewed=True,is_synthetic=True,
            context=parent['context'],full_context=parent.get('full_context'),
            context_is_source_reference=True,frame_semantics='original_train_reference_not_synthetic_capture',
            source_dataset=str(PILOT),source_image=str(source),created_at=stamp,
            review=dict(label='ng',source='explicit_user_acceptance',updated_at=stamp,
                        notes=USER_STATEMENT,scope='All ten generated candidates in gallery8801 accepted for supplementary NG training.'),
            synthetic=dict(method='built_in_ImageGen_reference_edit',parent_id=parent['id'],
                parent_sha256=parent['sha256'],source_canonical_revision=10,parent_split='train',
                source_split_group=parent['split_group'],source_candidate_id=generated['id'],
                generated_parent_id=ids.get(generated.get('generated_parent_id')),
                generation_provenance=str(PILOT/'candidate_provenance.json'),
                original_generated_path=generated['original_generated_path'],prompt=generated['prompt'],
                focus=generated['focus'],generated_size=generated['generated_size'],
                root_reference_train_only=True,user_acceptance=USER_STATEMENT,accepted_at=stamp))
        additions.append(dict(source=str(source),record=record))
    atomic_json(EVIDENCE/'adoption_plan.json',dict(user_statement=USER_STATEMENT,accepted_at=stamp,
        previous_revision=10,new_revision=11,validation_unchanged=True,partial_review_remains=True,
        additions=additions))
    result = store.save([],expected_revision=10,additions=additions)
    after = store.manifest()
    updated = {r['id']:r for r in after['records']}
    assert all(updated[r['id']]==r for r in before['records'])
    assert after.get('excluded_records')==before.get('excluded_records')
    assert [r for r in after['records'] if r['split']=='val']==[r for r in before['records'] if r['split']=='val']
    integrity = audit(pool)
    atomic_json(EVIDENCE/'integrity_after.json',integrity)
    assert integrity['passed'], integrity['errors']
    assert result['train_count']==3788 and result['validation_count']==1605
    registry.update(current_canonical_revision=11,updated_at=stamp,
                    last_synthetic_adoption=str(EVIDENCE/'adoption_plan.json'))
    atomic_json(REGISTRY,registry)
    receipt=dict(completed_at=stamp,canonical_revision=11,added_count=10,train_count=3788,val_count=1605,
        all_previous_5383_records_identical=True,all_validation_records_identical=True,
        synthetic_train_only=True,parent_groups_preserved=True,integrity_passed=True,
        canonical_manifest_sha256=digest(pool/'manifest.json'),review_status=after['review_status'],
        training_authorization_revoked_pending_new_receipt=not after['training_authorized'],
        user_statement=USER_STATEMENT,production_deployed=False)
    atomic_json(EVIDENCE/'adoption_verification.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False))

if __name__=='__main__':
    main()
