"""Commit the tested coverage proposal through the sole canonical transaction API."""
from pathlib import Path
import json,shutil,sys
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parent))
from classifier_dataset import REGISTRY,CanonicalDataset,read,atomic_json
from audit_classifier_dataset import audit
from train_classifier_supplementary import digest
from train_classifier_canonical_holdout import validate_rows,BALANCED_POLICY

PROJECT=Path(__file__).resolve().parents[2]

def main():
    evidence=PROJECT/'.work/classifier_coverage_split_20261030';proposal=read(evidence/'proposal.json')
    registry=read(REGISTRY);pool=Path(registry['path']);store=CanonicalDataset(pool);before=store.manifest()
    assert before['canonical_revision']==proposal['expected_revision']==11
    assert digest(pool/'manifest.json')==proposal['source_manifest_sha256']==digest(evidence/'manifest_before.json')
    for name in ['catalog.json','labels.csv']:shutil.copy2(pool/name,evidence/('before_'+name))
    atomic_json(evidence/'registry_before.json',registry)
    check=audit(pool);atomic_json(evidence/'integrity_before.json',check);assert check['passed'],check['errors']
    result=store.save([],11,repartition=proposal);after=store.manifest();new={r['id']:r for r in after['records']}
    assert len(new)==len(before['records'])==5393
    # Only split metadata, class-folder path and split history may differ.
    allowed={'split','role','canonical_original_role','image','split_history'}
    for old in before['records']:
        r=new[old['id']]
        assert {k:v for k,v in r.items() if k not in allowed}=={k:v for k,v in old.items() if k not in allowed},old['id']
    assert before.get('excluded_records')==after.get('excluded_records') and before.get('excluded')==after.get('excluded')
    validate_rows(after['records'],BALANCED_POLICY)
    check=audit(pool);atomic_json(evidence/'integrity_after.json',check);assert check['passed'],check['errors']
    registry.update(current_canonical_revision=12,updated_at=datetime.now(timezone.utc).isoformat(),current_repartition_report=str(evidence/'summary.json'))
    atomic_json(REGISTRY,registry)
    receipt=dict(canonical_revision=12,total=5393,train=result['train_count'],val=result['validation_count'],
        moved=result['saved'],all_ids_labels_hashes_provenance_preserved=True,exclusions_preserved=True,
        no_hash_or_group_overlap=True,synthetic_train_only=True,manifest_sha256=digest(pool/'manifest.json'),
        review_status=after['review_status'],training_authorized=after['training_authorized'],split_summary=read(evidence/'summary.json'))
    atomic_json(evidence/'adoption_verification.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='split_summary'}))

if __name__=='__main__':main()
