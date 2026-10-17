"""Lock a candidate using validation ONLY before opening the HA test split."""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

from classifier_metrics import dual_safety_fitness, dual_safety_failures


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(evaluations, output, artifact_name='c_p_ha_fault_priority_20261005'):
    if output.exists():
        raise FileExistsError('Selection is locked; never reselect after viewing test results')
    candidates=[]
    dataset=None
    gates=None
    correction_protocol=None
    for path in evaluations:
        report=json.loads(path.read_text(encoding='utf-8'))
        if 'ha_test' in report['domains']:
            raise ValueError('Test was already opened for this candidate')
        candidate_dataset=Path(report['dataset']).resolve()
        candidate_gates=report.get('validation_safety_gates')
        if dataset is not None and (dataset!=candidate_dataset or gates!=candidate_gates):
            raise ValueError('Candidates were not evaluated on the same dataset/gates')
        dataset,gates=candidate_dataset,candidate_gates
        correction_protocol=report.get('label_correction_protocol')
        ha,legacy=report['domains']['ha_val']['candidate'],report['domains']['legacy_val']['candidate']
        for metrics in (ha,legacy):
            metrics.setdefault('ng_as_no_disk',metrics['confusion_matrix_true_rows'][0][1])
        score=dual_safety_fitness(ha,legacy,gates)
        model=Path(report['models']['candidate']['path'])
        if digest(model)!=report['models']['candidate']['sha256']:
            raise ValueError('Candidate changed after validation')
        failures=dual_safety_failures(ha,legacy,gates)
        candidates.append({'name':path.parent.name,'onnx':str(model),'sha256':digest(model),
                           'evaluation':str(path.resolve()),'evaluation_sha256':digest(path),
                           'ha_validation':ha,'legacy_validation':legacy,'score':score,
                           'eligible':not failures,'rejected_reasons':failures})
    eligible=[c for c in candidates if c['eligible']]
    if not eligible:
        raise ValueError('No candidate meets safety/retention gates; no model released')
    chosen=max(eligible,key=lambda c:c['score'])
    source=Path(chosen['onnx'])
    pt=source.with_suffix('.pt')
    assert pt.exists()
    args=yaml.safe_load((source.parent.parent/'args.yaml').read_text(encoding='utf-8'))
    output.mkdir(parents=True)
    if not artifact_name or Path(artifact_name).name!=artifact_name or artifact_name in {'.','..'}:
        raise ValueError('Artifact name must be a simple filename stem')
    destination=output/(artifact_name+'.onnx')
    destination_pt=destination.with_suffix('.pt')
    shutil.copy2(source,destination)
    shutil.copy2(pt,destination_pt)
    assert digest(destination)==chosen['sha256']
    payload={'created_at':datetime.now(timezone.utc).isoformat(), 'selection_locked_before_ha_test':True,
             'policy':'NG->non-NG risk (OK OR no_disk) weighted HA 75%, legacy 25%; legacy NG misses <=21 and empty->OK <=2; '
                      'HA OK recall >=90%, legacy OK recall >=95%, legacy accuracy >=97%; then direct NG->OK and macro-F1',
             'reason':'APP Warning sends OK to PLC; no_disk predictions on true NG are unsafe too',
             'chosen':chosen['name'],'chosen_source':chosen['onnx'],'onnx':str(destination.resolve()),
             'pt':str(destination_pt.resolve()),'onnx_sha256':digest(destination),'pt_sha256':digest(destination_pt),
             'dataset':str(dataset),'training_args':args,'candidates':candidates,'production_deployed':False}
    if correction_protocol:
        payload['label_correction_protocol']=correction_protocol
        payload['validation_safety_gates']=gates
        payload['selection_locked_before_new_candidate_ha_test']=True
        payload['selection_locked_before_ha_test']=False
        payload['policy']='Corrected validation only; HA 75% / legacy 25% NG->non-NG risk; manifest retention gates'
    (output/'selection.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in payload.items() if k not in ('training_args','candidates')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluations',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--artifact-name',default='c_p_ha_fault_priority_20261005')
    args=parser.parse_args()
    select(args.evaluations,args.output.resolve(),args.artifact_name)
