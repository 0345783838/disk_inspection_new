"""Reconstruct clean contextual views, with canonical labels as sole authority."""
from pathlib import Path
from collections import defaultdict,Counter
import itertools,json,hashlib
import cv2,numpy as np
from classifier_dataset import read,atomic_json
from classifier_context import make_contexts

def find_project(path):
    return next(p for p in Path(path).resolve().parents if (p/'SERVICE/training/classifier_dataset.py').is_file())

PROJECT=find_project(__file__)

def raw_frames():
    ha=read(PROJECT/'.work/HA_classify_wide_20261001/manifest.json')
    new=read(PROJECT/'.work/classifier_factory_20261020/review/items.json')
    metadata={};geometries={}
    for frame in ha['frames']:
        identifier=f"Cam{frame['camera']}_{Path(frame['source']).stem}"
        metadata[identifier]=dict(kind='ha',source=frame['source'],normalization=frame['normalization'],
            width=frame['rectified_shape'][1],height=frame['rectified_shape'][0],matrix=frame['warp_matrix'])
        for item in frame['crops']:geometries[f'{identifier}__{item["region"]}__slot{item["index"]:02}']=dict(box=item['effective_box'],rotated_180=item['rotated_180'])
    quad={r['frame']:r['crop_quad'] for r in new['geometry_audit']}
    for item in new['items']:
        metadata.setdefault(item['frame'],dict(kind='new',source=item['source'],camera=item['camera'],quad=quad[item['frame']]))
        geometries[item['id']]=item['geometry']
    return metadata,geometries

def rectify(meta):
    image=cv2.imread(meta['source']);assert image is not None,meta['source']
    if meta['kind']=='ha':
        norm=meta['normalization'];codes={'Clockwise90':cv2.ROTATE_90_CLOCKWISE,'CounterClockwise90':cv2.ROTATE_90_COUNTERCLOCKWISE,'Rotate180':cv2.ROTATE_180}
        if norm['rotation']!='None':image=cv2.rotate(image,codes[norm['rotation']])
        if norm.get('roi'):
            x,y,w,h=norm['roi'];image=image[y:y+h,x:x+w]
        return cv2.warpPerspective(image,np.array(meta['matrix']),(meta['width'],meta['height']))
    from training.audit_new_factory_images import load
    report=read(PROJECT/'.work/factory_audit_20261020_run/report.json')
    image=load(Path(meta['source']),meta['camera'],Path(report['app_config']))
    q=np.asarray(meta['quad'],np.float32);tl,tr,br,bl=q
    w=int(round(max(np.linalg.norm(tr-tl),np.linalg.norm(br-bl))));h=int(round(max(np.linalg.norm(bl-tl),np.linalg.norm(br-tr))))
    matrix=cv2.getPerspectiveTransform(q,np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]]))
    return cv2.warpPerspective(image,matrix,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)

def build(root):
    m=read(root/'source_manifest_at_training.json');pool=root/'input_snapshot';metadata,geometries=raw_frames()
    groups=defaultdict(list)
    for r in m['records']:
        if r.get('frame') and not r.get('is_synthetic'):groups[r['frame']].append(r)
    records=[];raw_provenance=[];matched=0
    (root/'context').mkdir(exist_ok=True)
    for frame,items in sorted(groups.items()):
        meta=metadata[frame];image=rectify(meta);raw_provenance.append(dict(frame=frame,metadata=meta,source_sha256=hashlib.sha256(Path(meta['source']).read_bytes()).hexdigest()))
        # Include deleted/unselected geometry only for physical neighbor positions,
        # never their historical labels or their image membership.
        for region in ['row1_bottom','row2_top','row2_bottom','row3_top']:
            all_ids=[f'{frame}__{region}__slot{s:02}' for s in range(1,26)]
            boxes=[geometries[i]['box'] for i in all_ids];rotated=geometries[all_ids[0]]['rotated_180']
            contexts,desc=make_contexts(image,boxes,'top' if rotated else 'bottom')
            for r in [v for v in items if v['region']==region]:
                index=int(r['slot'])-1;box=boxes[index];x1,y1,x2,y2=box;local=image[y1:y2,x1:x2]
                if rotated:local=cv2.rotate(local,cv2.ROTATE_180)
                actual=cv2.imread(str(pool/r['image']))
                if not np.array_equal(local,actual):raise ValueError('Canonical crop reconstruction mismatch: '+r['id'])
                matched+=1;path=root/'context'/f"{r['id']}.png";rgba=contexts[index]
                # Four-channel PNG stores RGB+target mask without drawing on RGB.
                encoded=cv2.cvtColor(rgba,cv2.COLOR_RGBA2BGRA)
                assert cv2.imwrite(str(path),encoded)
                decoded=cv2.imread(str(path),cv2.IMREAD_UNCHANGED);decoded=cv2.cvtColor(decoded,cv2.COLOR_BGRA2RGBA)
                assert np.array_equal(decoded,rgba)
                records.append(dict(id=r['id'],context=path.relative_to(root).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),available=True,geometry=desc[index],frame=frame))
        print('CONTEXT',frame,'matched',len(items),flush=True)
    by_id={r['id']:r for r in records}
    for r in m['records']:
        if r['id'] not in by_id:records.append(dict(id=r['id'],context=None,available=False,reason='AI geometry has no matching raw context' if r.get('is_synthetic') else 'Legacy raw/geometry unavailable'))
    assert matched==4783 and len(records)==5393
    atomic_json(root/'context_manifest.json',dict(records=records,raw_sources=raw_provenance,all_available_locals_pixel_exact=True,available=matched,unavailable=610))
    return records

def holdout(rows):
    """Quarantine same-burst canonical-train rows; never move val into train."""
    real=[r for r in rows if r.get('frame') and not r.get('is_synthetic')]
    ha=sorted({r['collection_burst'] for r in real if r['origin']=='old_ha'})
    # Fixed seed, selected before evaluating any model predictions.
    rng=np.random.default_rng(1031);held_ha=ha[int(rng.integers(len(ha)))]
    new_groups=sorted({r['collection_burst'] for r in real if r['origin']=='new_factory'});best=None
    new=[r for r in real if r['origin']=='new_factory'];totals=Counter((r['camera'],r['label']) for r in new if r['split']=='val')
    for count in [2,3]:
        for chosen in itertools.combinations(new_groups,count):
            val=[r for r in new if r['split']=='val' and r['collection_burst'] in chosen]
            train=[r for r in new if r['split']=='train' and r['collection_burst'] not in chosen]
            vc=Counter((r['camera'],r['label']) for r in val);tc=Counter((r['camera'],r['label']) for r in train)
            if any(vc[k]==0 or tc[k]==0 for k in totals):continue
            score=sum((vc[k]/n-.3)**2 for k,n in totals.items())+((len(val)/sum(totals.values()))-.3)**2
            key=(score,chosen)
            if best is None or key<best[0]:best=(key,chosen)
    if best is None:raise ValueError('No capture holdout with new camera/class coverage')
    held={held_ha,*best[1]};fit=[r['id'] for r in real if r['split']=='train' and r['collection_burst'] not in held]
    primary=[r['id'] for r in real if r['split']=='val' and r['collection_burst'] in held]
    quarantine=[r['id'] for r in real if r['split']=='train' and r['collection_burst'] in held]
    fit_frames={r['frame'] for r in rows if r['id'] in set(fit)};val_frames={r['frame'] for r in rows if r['id'] in set(primary)}
    assert not fit_frames&val_frames
    return dict(seed=1031,held_bursts=sorted(held),fit_ids=fit,primary_validation_ids=primary,quarantined_train_ids=quarantine,
        fit_count=len(fit),primary_count=len(primary),quarantined_count=len(quarantine),fit_frames=sorted(fit_frames),primary_frames=sorted(val_frames),
        canonical_assignments_unchanged=True,scope='Capture-disjoint context adaptation only. Parent has historical exposure; not a blind independent test. Current val outside held captures is secondary development only.',
        selection_inputs='Current labels, origin/camera and capture metadata only; no model predictions/error IDs.')
