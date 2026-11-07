"""Plan a 70/30 group split using labels/regions and fixed ImageNet features only."""
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
from pathlib import Path
import hashlib,json,sys
import cv2
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import lil_matrix

sys.path.insert(0,str(Path(__file__).resolve().parent))
from classifier_dataset import REGISTRY,read,atomic_json
from train_classifier_supplementary import digest
from train_classifier_ha import runtime_tensor

PROJECT=Path(__file__).resolve().parents[2]
SEED=1030

def cell(r):
    return '|'.join(map(str,(r['origin'],r.get('camera') or 'unknown',r.get('region') or 'unknown',r['label'])))

def extract(pool,rows,out):
    from ultralytics import YOLO
    checkpoint=PROJECT/'.work/pretrained_ultralytics/yolo11n-cls.pt'
    provenance=read(checkpoint.with_suffix('.provenance.json'))
    assert digest(checkpoint)==provenance['sha256']
    torch.set_num_threads(4);cv2.setNumThreads(1)
    model=YOLO(str(checkpoint)).model.float().cuda().eval()
    captured=[]
    hook=model.model[-1].pool.register_forward_hook(lambda module,args,value:captured.append(value.detach().flatten(1).cpu().numpy()))
    features=[];photos=[]
    with torch.inference_mode():
        for start in range(0,len(rows),64):
            part=rows[start:start+64];images=[cv2.imread(str(pool/r['image'])) for r in part]
            assert all(im is not None for im in images)
            batch=torch.stack([runtime_tensor(im) for im in images]).cuda();captured.clear();model(batch)
            assert len(captured)==1;features.append(captured[0])
            for im in images:
                gray=cv2.cvtColor(cv2.resize(im,(64,64)),cv2.COLOR_BGR2GRAY).astype(np.float32)/255
                photos.append([gray.mean(),gray.std(),np.quantile(gray,.1),np.quantile(gray,.9),float((gray>.9).mean()),float((gray<.15).mean())])
            if start%640==0:print('FEATURES',min(start+64,len(rows)),'/',len(rows),flush=True)
    hook.remove();features=np.concatenate(features);features/=np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-8)
    photos=np.asarray(photos,dtype=np.float32)
    np.savez_compressed(out/'fixed_features.npz',ids=np.array([r['id'] for r in rows]),features=features,photos=photos)
    atomic_json(out/'feature_provenance.json',dict(checkpoint=str(checkpoint),sha256=provenance['sha256'],classes=1000,
        fitting=False,preprocessing='Runtime RGB/stretch224',scope='Frozen official ImageNet features for split design only; no inspection model predictions or historic error lists.',real_rows=len(rows)))
    return features,photos

def visual_cells(rows,features,photos):
    reduced=PCA(n_components=min(32,len(rows)-1,features.shape[1]),random_state=SEED).fit_transform(features)
    # Scale each block by its RMS, avoiding tiny PCA dimensions being amplified.
    reduced/=max(float(np.sqrt(np.mean(reduced**2))),1e-6)
    photos=(photos-photos.mean(0))/np.maximum(photos.std(0),.05)
    embedded=np.concatenate([reduced,.35*photos],axis=1)
    by=defaultdict(list)
    for i,r in enumerate(rows):by[cell(r)].append(i)
    clusters={};description={}
    for key,indices in sorted(by.items()):
        groups={rows[i]['split_group'] for i in indices}
        k=min(4,max(1,round(len(indices)/30)),max(1,len(groups)//3))
        labels=KMeans(n_clusters=k,n_init=10,random_state=SEED).fit_predict(embedded[indices]) if k>1 else np.zeros(len(indices),dtype=int)
        # Small clusters cannot independently cover train/val. Merge by feature
        # distance until every remaining visual cluster spans >=3 target groups.
        while len(set(labels))>1:
            counts={label:len({rows[i]['split_group'] for i,l in zip(indices,labels) if l==label}) for label in set(labels)}
            small=[label for label,n in counts.items() if n<3]
            if not small:break
            source=min(small,key=lambda l:(counts[l],l))
            centers={label:embedded[np.array(indices)[labels==label]].mean(0) for label in set(labels)}
            target=min((l for l in centers if l!=source),key=lambda l:np.linalg.norm(centers[l]-centers[source]))
            labels[labels==source]=target
        mapping={v:i for i,v in enumerate(sorted(set(labels)))}
        for i,label in zip(indices,labels):clusters[rows[i]['id']]=key+'|visual'+str(mapping[label])
        description[key]=dict(real_count=len(indices),independent_groups=len(groups),visual_clusters=len(mapping))
    return clusters,description

def subsets(rows,clusters):
    groups=defaultdict(list)
    for i,r in enumerate(rows):
        groups['region/'+cell(r)].append(i)
        groups['visual/'+clusters[r['id']]].append(i)
        groups['source_camera_class/'+'|'.join(map(str,(r['origin'],r.get('camera') or 'unknown',r['label'])))].append(i)
    groups['all']=list(range(len(rows)))
    return groups

def solve_split(rows,clusters,forced_train,time_limit=90):
    groups=sorted({r['split_group'] for r in rows});gi={g:i for i,g in enumerate(groups)}
    subsets_by_name=subsets(rows,clusters);names=sorted(subsets_by_name);G=len(groups);S=len(names)
    count=np.zeros((S,G),dtype=np.float64)
    for j,key in enumerate(names):
        for i in subsets_by_name[key]:count[j,gi[rows[i]['split_group']]]+=1
    totals=count.sum(1);target=.3*totals
    # Binary group membership plus absolute row-count deviations per stratum.
    A=lil_matrix((3*S,G+S),dtype=np.float64);lower=[];upper=[]
    for j,key in enumerate(names):
        A[3*j,:G]=count[j];A[3*j,G+j]=-1;lower.append(-np.inf);upper.append(target[j])
        A[3*j+1,:G]=-count[j];A[3*j+1,G+j]=-1;lower.append(-np.inf);upper.append(-target[j])
        A[3*j+2,:G]=count[j]
        if key=='all':lo,hi=.29*totals[j],.31*totals[j]
        elif len(np.flatnonzero(count[j]))>=2:lo,hi=1,totals[j]-1
        else:lo,hi=0,totals[j]
        lower.append(lo);upper.append(hi)
    objective=np.zeros(G+S)
    for j,key in enumerate(names):
        importance=8 if key=='all' else 3 if key.startswith('source_camera_class/') else 1
        objective[G+j]=importance/totals[j]
    for g,index in gi.items():objective[index]=int(hashlib.sha256((str(SEED)+g).encode()).hexdigest()[:8],16)/2**32*1e-8
    lb=np.zeros(G+S);ub=np.concatenate([np.ones(G),np.full(S,np.inf)])
    for g in forced_train:ub[gi[g]]=0
    result=milp(objective,integrality=np.r_[np.ones(G),np.zeros(S)],bounds=Bounds(lb,ub),
        constraints=LinearConstraint(A.tocsc(),np.asarray(lower),np.asarray(upper)),
        options=dict(time_limit=time_limit,mip_rel_gap=.01))
    if result.x is None:raise ValueError('No feasible group/region/visual coverage split: '+result.message)
    assignments={g:'val' if result.x[index]>.5 else 'train' for g,index in gi.items()}
    for j,key in enumerate(names):
        actual=sum(count[j,index] for g,index in gi.items() if assignments[g]=='val')
        assert lower[3*j+2]-1e-6<=actual<=upper[3*j+2]+1e-6,key
    return assignments,dict(status=int(result.status),message=result.message,objective=float(result.fun),mip_gap=float(result.mip_gap),time_limit_seconds=time_limit,used_predictions=False)

def coverage(rows,clusters,assignment):
    result={}
    for key,indices in sorted(subsets(rows,clusters).items()):
        train=sum(assignment[rows[i]['split_group']]=='train' for i in indices);val=len(indices)-train
        result[key]=dict(train=train,val=val,total=len(indices),val_fraction=val/len(indices),independent_groups=len({rows[i]['split_group'] for i in indices}),both=bool(train and val))
    return result

def main(out):
    pool=Path(read(REGISTRY)['path']);manifest=read(pool/'manifest.json')
    assert manifest['canonical_revision']==11
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);raw=(pool/'manifest.json').read_bytes();(out/'manifest_before.json').write_bytes(raw)
    rows=[deepcopy(r) for r in manifest['records'] if not r.get('is_synthetic')]
    atomic_json(out/'design_before_features.json',dict(source_revision=11,seed=SEED,target_val_fraction=.3,
        grouping='Preserve existing origin/camera/target-slot, decoded/flip/180 aliases and synthetic parent families.',
        strata='source x camera x region x label plus fixed ImageNet/photometric visual clusters',
        synthetic_policy='All ten AI and their real root target groups forced train; no AI in validation.',
        objective='Minimize normalized 70/30 deviation per cell/visual cluster/source-camera-class; every multi-group cell/cluster must appear in both splits.',
        selection_inputs='Current canonical labels, metadata and frozen official ImageNet features only. No inspection predictions, old missed-NG IDs or evaluation metrics.'))
    features,photos=extract(pool,rows,out);clusters,description=visual_cells(rows,features,photos)
    forced={r['split_group'] for r in manifest['records'] if r.get('is_synthetic')}
    old={r['split_group']:r['split'] for r in manifest['records']}
    proposed,solver=solve_split(rows,clusters,forced)
    before=coverage(rows,clusters,old);after=coverage(rows,clusters,proposed)
    assignments={r['id']:proposed[r['split_group']] for r in manifest['records']}
    assert all(assignments[r['id']]=='train' for r in manifest['records'] if r.get('is_synthetic'))
    assert raw==(pool/'manifest.json').read_bytes()
    policy=deepcopy(manifest['split_policy']);policy.update(partition_strategy='region_visual_group_milp_v2',
        target_fraction=.3,actual_fraction=sum(v=='val' for v in assignments.values())/len(assignments),
        seed=SEED,selection_inputs='Canonical labels/source/camera/region/target groups + frozen official ImageNet embedding/photometric clusters only; no inspection predictions or error IDs.',
        selection_method=solver,synthetic_train_only=True,visual_feature_provenance=read(out/'feature_provenance.json'),
        coverage_report=str(out/'coverage_after.json'))
    policy.pop('fold',None);policy.pop('selection_score',None);policy.pop('candidate_seed_count',None);policy.pop('strata_counts',None)
    proposal=dict(expected_revision=11,source_manifest_sha256=hashlib.sha256(raw).hexdigest(),assignments=assignments,
        policy=policy,user_statement='không ổn lắm nhỉ, thử phân bổ lại tập dữ liệu train/val rồi train lại từ đầu được không, làm sao để đảm bảo phân phối của tập train và val bao phủ nhau, model học được hết',
        evidence=str(out),labels_hashes_ids_unchanged=True)
    atomic_json(out/'proposal.json',proposal);atomic_json(out/'coverage_before.json',before);atomic_json(out/'coverage_after.json',after)
    atomic_json(out/'visual_clusters.json',dict(labels_by_id=clusters,cells=description))
    summary=dict(total=len(assignments),real_count=len(rows),train=sum(v=='train' for v in assignments.values()),val=sum(v=='val' for v in assignments.values()),
        moved=sum(assignments[r['id']]!=r['split'] for r in manifest['records']),forced_train_target_groups=sorted(forced),solver=solver,
        region_cells=sum(k.startswith('region/') for k in after),region_missing_before=[k for k,v in before.items() if k.startswith('region/') and not v['both']],
        region_missing_after=[k for k,v in after.items() if k.startswith('region/') and not v['both']],
        visual_clusters=sum(k.startswith('visual/') for k in after),visual_missing_before=sum(k.startswith('visual/') and not v['both'] for k,v in before.items()),
        visual_missing_after=sum(k.startswith('visual/') and not v['both'] for k,v in after.items()))
    atomic_json(out/'summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();main(args.output.resolve())
