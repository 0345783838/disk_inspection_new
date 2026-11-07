"""Two bounded context-adaptation fits, immutable canonical inputs, no deployment."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import html,json,os,shutil,subprocess,sys,time,traceback
PROJECT=next(p for p in Path(__file__).resolve().parents if (p/'SERVICE/training/classifier_dataset.py').is_file())
sys.path.insert(0,str(PROJECT/'SERVICE'));sys.path.insert(0,str(Path(__file__).resolve().parent));sys.path.append(str(PROJECT/'SERVICE/.venv/Lib/site-packages'))
import cv2,numpy as np,torch
import torch.nn.functional as F
from classifier_dataset import CanonicalDataset,active_dataset,read,atomic_json,atomic_text
from audit_classifier_dataset import audit
from adopt_classifier_dataset import decoded_fingerprint
from train_classifier_supplementary import digest
from classifier_metrics import classification_metrics
import train_classifier_canonical_holdout as base
from classifier_context import ContextClassifier,parent_model,load_model,save_model,state_sha,tensor,augment,make_contexts
from build_classifier_context import build,holdout

FLAGS=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
USER='vậy thử đi bro, trông cậy vào mày, tao ngu bất lực rồi'
NAMES=['control','context']

def metrics(rows,pred):
    result=base.scoped_metrics(rows,pred)
    all_m=classification_metrics([base.LABELS.index(r['label']) for r in rows],[base.LABELS.index(r['label']) for r in pred]);cm=all_m['confusion_matrix_true_rows']
    all_m.update(errors=sum(r['label']!=p['label'] for r,p in zip(rows,pred)),false_ng=cm[1][0]+cm[2][0],ok_as_empty=cm[2][1]);result['all']=all_m
    return result

def score(current,anchor):
    keys=[k for k in anchor if k!='all' and anchor[k]['count']>0]
    excess=sum(max(0,current[k]['ng_not_detected']-anchor[k]['ng_not_detected']) for k in keys)
    m=current['all'];return -1e9*excess-1e6*m['ng_not_detected']-1000*m['false_ng']-10*m['no_disk_as_ok']-.001*m['errors']

def publish(root,status,phase,child=None,error=None,**extra):
    state=dict(status=status,phase=phase,pid=os.getpid(),active_pid=child,updated_at=base.now(),error=error,**extra)
    atomic_json(root/'status.json',state)
    pointer=PROJECT/'.work/ACTIVE_CLASSIFIER_TRAINING.json'
    if pointer.exists():
        item=read(pointer)
        if Path(item['path']).resolve()==root.resolve():
            item.update(status=status,phase=phase,active_pid=child,campaign_pid=None if status in ['complete','failed','stopped'] else item.get('campaign_pid') or os.getpid(),error=error)
            if status=='complete':item.update(completed_at=base.now(),recommended_for_review=read(root/'release_decision.json')['recommended_for_review'])
            atomic_json(pointer,item)

def check(root):
    if (root/'STOP_REQUESTED').exists():raise InterruptedError('STOP_REQUESTED')
    p=read(root/'protocol.json')
    if digest(root/'protocol.json')!=(root/'protocol.sha256').read_text().strip():raise ValueError('Protocol changed')
    for name,expected in p['code_hashes'].items():
        if digest(root/'code_snapshot'/name)!=expected:raise ValueError('Frozen code changed: '+name)
    if digest(root/'source_manifest_at_training.json')!=p['manifest_sha256']:raise ValueError('Frozen canonical manifest changed')
    for path,expected in read(root/'protected_runtime.json').items():
        if digest(path)!=expected:raise ValueError('Protected runtime/model changed: '+path)
    return p

def page(p):
    return '''<!doctype html><html lang="vi"><meta charset="utf-8"><title>Context + khe mục tiêu</title><style>body{font:16px system-ui;background:#14202e;color:#e6eef6;margin:25px;max-width:1500px}a{color:#9bd4ff}p{line-height:1.6}table{border-collapse:collapse}td,th{border:1px solid #65798e;padding:9px}img{max-width:100%;height:220px;object-fit:contain}.grid{display:flex;flex-wrap:wrap;gap:15px}article{width:290px;padding:15px;background:#24364a;overflow-wrap:anywhere}</style><h1>Thử nghiệm crop chi tiết + context5khe + mask khe mục tiêu</h1><p>Giữ backbone/head best111 đóng băng; thêm contextCNN và đầu hiệu chỉnh nhỏ. Hai fit tối đa: control chỉ mask/vị trí, context dùng cả ảnh xung quanh. Cùng dữ liệu/seed/loss; epoch0 là parent nguyên trạng. 300epoch/patience100 mỗi fit. Không thay SERVICE đang test.</p><p>Ảnh context dựng từ raw, crop gốc phải khớp pixel. Giữ nguyên canonical revision12 và các split. Train cùng phiên với primary-val bị cách ly khỏi phần học context. Primary-val chỉ dùng các canonical-val trong phiên giữ riêng. Parent đã từng học một phần dữ liệu nên đây là đánh giá thích nghi, không phải test mù độc lập. Val còn lại là development phụ. Legacy/AI thiếu context dùng nguyên kết quả parent.</p><p id="status">Đang tải…</p><p><a href="protocol.json">Protocol</a> · <a href="eligibility.json">Phân tách phiên chụp</a> · <a href="input_preview.html">Ảnh đầu vào</a> · <a href="comparison.html">So sánh</a> · <a href="RESULTS.md">Kết quả</a> · <a href="logs/control.log">Log control</a> · <a href="logs/context.log">Log context</a></p><script>async function r(){try{let s=await(await fetch('status.json?t='+Date.now(),{cache:'no-store'})).json();document.getElementById('status').textContent=s.status+' / '+s.phase+' · epoch '+(s.epoch??'')+' · best '+(s.best_epoch??'')+' · '+(s.error??'')}catch(e){}}r();setInterval(r,10000)</script></html>'''

def prepare(root):
    if root.exists():raise FileExistsError(root)
    registry=read(PROJECT/'.work/ACTIVE_CLASSIFIER_DATASET.json');pool=Path(registry['path']);before=read(pool/'manifest.json');assert before['canonical_revision']==12
    receipt=dict(authorized_at=base.now(),canonical_revision=12,incomplete_review_acknowledged=True,user_statement=USER,
        scope='Bounded context pilot: raw5-slot context+explicit target mask with frozen best111 local nano, matched mask-only control; max2fits300/patience100. No canonical label/split edits, same-capture train rows quarantined from context primary-val, parent prior exposure disclosed. Current SERVICE/retained models untouched; export parity and actual White latency<=2x required.')
    CanonicalDataset(pool).authorize_training(receipt,12);pool=active_dataset(required_split_policy=base.BALANCED_POLICY)
    root.mkdir(parents=True);(root/'logs').mkdir();(root/'reference').mkdir();(root/'input_snapshot').mkdir()
    raw=(pool/'manifest.json').read_bytes();(root/'source_manifest_at_training.json').write_bytes(raw);m=read(root/'source_manifest_at_training.json')
    assert [(r['id'],r['label'],r['sha256'],r['split']) for r in m['records']]==[(r['id'],r['label'],r['sha256'],r['split']) for r in before['records']]
    integrity=audit(pool);atomic_json(root/'integrity_before.json',integrity);assert integrity['passed'],integrity['errors']
    for r in m['records']:
        source=pool/r['image'];assert digest(source)==r['sha256'] and decoded_fingerprint(source)==r['decoded_sha256']
        dest=root/'input_snapshot'/r['image'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    assert raw==(pool/'manifest.json').read_bytes()
    frozen=dict(m,is_canonical=False,immutable_run_snapshot=True);atomic_json(root/'input_snapshot/manifest.json',frozen)
    parent=PROJECT/'SERVICE/config/models/c_p_finetune_ha_r10_best111_20261029.pt';assert digest(parent)=='51e002c68b4c64b64fd74d8f0987c6df987b4fc3eec7a779bb6464cc516be252'
    for source,name in [(parent,'parent.pt'),(parent.with_suffix('.onnx'),'parent.onnx'),(PROJECT/'SERVICE/config/models/c_p.onnx','service.onnx')]:shutil.copy2(source,root/'reference'/name)
    protected={str(f.resolve()):digest(f) for f in (PROJECT/'SERVICE/src').rglob('*.py')}
    for basepath in [PROJECT/'SERVICE/config',PROJECT/'APP/DiskInspection/bin/BroSight_Disinspector/plugin/config']:
        for f in basepath.rglob('*'):
            if f.is_file() and f.suffix.lower() in ['.env','.onnx','.pt','.json']:protected[str(f.resolve())]=digest(f)
    app_config=PROJECT/'APP/DiskInspection/bin/BroSight_Disinspector/DiskInspection.exe.config';protected[str(app_config)]=digest(app_config)
    atomic_json(root/'protected_runtime.json',protected)
    p=dict(created_at=base.now(),pool=str(pool),canonical_revision=12,manifest_sha256=digest(root/'source_manifest_at_training.json'),
        authorization=receipt,review_incomplete=True,epochs=300,patience=100,seed=1031,train_views=3,lr=.001,batch=64,
        fits=NAMES,parent=str(root/'reference/parent.pt'),parent_sha256=digest(parent),local_frozen=True,
        architecture='Frozen YOLO11n-cls local parent1280features + small4-channel context encoder + 3-logit residual; control zeros context RGB but keeps target mask.',
        selection='Primary capture-disjoint canonical-val safety fitness; epoch0 parent eligible. No thresholds or interpolation.',
        loss='NG CE2/1/1 +0.5aux context CE +0.5 teacher KL on correct confident parent only.',
        full_fits_max=2,context_span=5,unavailable_policy='Legacy and AI have zero context/mask and exactly unchanged parent predictions.',
        prior_parent_exposure=True,canonical_splits_unchanged=True,production_deployed=False)
    code=root/'code_snapshot';code.mkdir();p['code_hashes']={}
    for source in Path(__file__).parent.glob('*.py'):shutil.copy2(source,code/source.name);p['code_hashes'][source.name]=digest(source)
    atomic_json(root/'protocol.json',p);atomic_text(root/'protocol.sha256',digest(root/'protocol.json'));atomic_text(root/'index.html',page(p));atomic_json(root/'authorization_receipt.json',receipt)
    atomic_json(root/'previous_active_training.json',read(PROJECT/'.work/ACTIVE_CLASSIFIER_TRAINING.json'))
    atomic_json(PROJECT/'.work/ACTIVE_CLASSIFIER_TRAINING.json',dict(path=str(root),entrypoint='SERVICE/training/train_classifier_context.py',report_url='http://127.0.0.1:8806/',canonical_revision=12,status='prepared',phase='context inputs frozen',campaign_pid=os.getpid(),active_pid=None,authorization=receipt,production_deployed=False))
    publish(root,'prepared','frozen canonical inputs, building clean context')
    return p

def cache(root):
    p=check(root);m=read(root/'source_manifest_at_training.json');records=m['records'];elig=holdout(records);atomic_json(root/'eligibility.json',elig)
    if not (root/'context_manifest.json').exists():build(root)
    context={r['id']:r for r in read(root/'context_manifest.json')['records']};fitset=set(elig['fit_ids']);fit_indices=[i for i,r in enumerate(records) if r['id'] in fitset]
    entries=[(i,0) for i in range(len(records))]+[(i,v) for i in fit_indices for v in range(1,p['train_views'])]
    folder=root/'features';folder.mkdir(exist_ok=True);total=len(entries)
    features=np.lib.format.open_memmap(folder/'local.npy',mode='w+',dtype=np.float32,shape=(total,1280));logits=np.lib.format.open_memmap(folder/'logits.npy',mode='w+',dtype=np.float32,shape=(total,3));contexts=np.lib.format.open_memmap(folder/'context.npy',mode='w+',dtype=np.uint8,shape=(total,4,224,224))
    targets=np.array([base.LABELS.index(records[i]['label']) for i,v in entries],np.int64);np.save(folder/'targets.npy',targets)
    indexmap=defaultdict(list)
    for j,(i,v) in enumerate(entries):indexmap[records[i]['id']].append(j)
    atomic_json(folder/'positions.json',dict(by_id=dict(indexmap),records=[i for i,v in entries]))
    torch.set_num_threads(4);cv2.setNumThreads(1);model=ContextClassifier(parent_model(p['parent'])).cuda().eval()
    with torch.inference_mode():
        for start in range(0,total,64):
            local_batch=[];context_batch=[]
            for i,view in entries[start:start+64]:
                r=records[i];im=cv2.imread(str(root/'input_snapshot'/r['image']));c=context[r['id']]
                ctx=cv2.cvtColor(cv2.imread(str(root/c['context']),cv2.IMREAD_UNCHANGED),cv2.COLOR_BGRA2RGBA) if c['available'] else np.zeros((224,224,4),np.uint8)
                if view:im,ctx=augment(im,ctx,1031+i*31+view)
                local_batch.append(tensor(im));context_batch.append(np.transpose(ctx,(2,0,1)))
            f,l=model.local(torch.from_numpy(np.array(local_batch)).cuda());features[start:start+len(local_batch)]=f.cpu().numpy();logits[start:start+len(local_batch)]=l.cpu().numpy();contexts[start:start+len(local_batch)]=np.array(context_batch)
            if start%640==0:print('CACHE',min(start+64,total),'/',total,flush=True)
    features.flush();logits.flush();contexts.flush()
    atomic_json(folder/'complete.json',dict(entries=total,originals=len(records),fit_images=len(fit_indices),train_views=p['train_views'],parent_frozen_sha256=state_sha(model.frozen_state()),files={name:digest(folder/name) for name in ['local.npy','logits.npy','context.npy','targets.npy','positions.json']}))
    preview=[];(root/'preview').mkdir(exist_ok=True)
    for r in [r for r in records if r['label']=='ng' and context[r['id']]['available']][:12]:
        bgra=cv2.imread(str(root/context[r['id']]['context']),cv2.IMREAD_UNCHANGED)
        rgb_path=f"preview/{r['id']}_rgb.png";mask_path=f"preview/{r['id']}_mask.png"
        assert cv2.imwrite(str(root/rgb_path),bgra[:,:,:3]) and cv2.imwrite(str(root/mask_path),bgra[:,:,3])
        preview.append(f'<article><p>{html.escape(r["id"])}</p><p>Crop chi tiết</p><img src="input_snapshot/{r["image"]}"><p>Context raw 5khe</p><img src="{rgb_path}"><p>Mask khe mục tiêu</p><img src="{mask_path}"></article>')
    atomic_text(root/'input_preview.html',page(p).split('<h1>')[0]+'<h1>Crop chi tiết / context raw / mask mục tiêu</h1><p>Ảnh context và mask được tách riêng khi hiển thị; file bốn kênh dùng train giữ nguyên.</p><div class="grid">'+''.join(preview)+'</div>')

def arrays(root):
    folder=root/'features'
    return [np.load(folder/name,mmap_mode='r') for name in ['local.npy','logits.npy','context.npy','targets.npy']]

def predict_cached(model,arr,positions):
    local,logits,ctx,targets=arr;out=[]
    with torch.inference_mode():
        for start in range(0,len(positions),128):
            ids=np.asarray(positions[start:start+128]);a=torch.from_numpy(np.asarray(local[ids])).cuda();b=torch.from_numpy(np.asarray(logits[ids])).cuda();c=torch.from_numpy(np.asarray(ctx[ids])).cuda().float()/255
            probs=model.cached(a,b,c)[0].softmax(1).cpu().numpy()
            for values in probs:
                index=int(values.argmax());out.append(dict(label=base.LABELS[index],confidence=float(values[index]),probabilities={k:float(v) for k,v in zip(base.LABELS,values)}))
    return out

def fit(root,name):
    p=check(root);records=read(root/'source_manifest_at_training.json')['records'];by_id={r['id']:i for i,r in enumerate(records)};elig=read(root/'eligibility.json');arr=arrays(root);positions=read(root/'features/positions.json')['by_id']
    primary=[by_id[i] for i in elig['primary_validation_ids']];primary_rows=[records[i] for i in primary]
    fitrows=[records[by_id[i]] for i in elig['fit_ids']];counts=Counter((r['origin'],r['camera'],r['label']) for r in fitrows)
    cameras={o:{r['camera'] for r in fitrows if r['origin']==o} for o in ['old_ha','new_factory']}
    weights=np.array([.5/len(cameras[r['origin']])*dict(zip(base.LABELS,[.3,.3,.4]))[r['label']]/counts[r['origin'],r['camera'],r['label']] for r in fitrows]);weights/=weights.sum()
    folder=root/'jobs'/name;folder.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.manual_seed(p['seed']);np.random.seed(p['seed']);rng=np.random.default_rng(p['seed'])
    model=ContextClassifier(parent_model(p['parent']),control=name=='control').cuda();frozen=model.frozen_state();optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=p['lr'],weight_decay=1e-4)
    parent_pred=[dict(label=base.LABELS[int(np.argmax(arr[1][i]))]) for i in primary];anchor=metrics(primary_rows,parent_pred)
    best_score=score(anchor,anchor);best_epoch=0;start_epoch=1
    if not (folder/'last.pt').exists():save_model(folder/'best.pt',model,epoch=0,fitness=best_score)
    else:
        resume=torch.load(folder/'last.pt',map_location='cpu',weights_only=False);model.load_state_dict(resume['model']);optimizer.load_state_dict(resume['optimizer']);best_score=resume['best_score'];best_epoch=resume['best_epoch'];start_epoch=resume['epoch']+1;rng.bit_generator.state=resume['rng'];torch.set_rng_state(resume['torch_rng']);torch.cuda.set_rng_state(resume['cuda_rng'])
    atomic_json(folder/'sampling.json',dict(eligible_train=len(fitrows),epoch_draws=2*len(fitrows),every_fit_image_once=True,quarantined_train=elig['quarantined_count'],primary_validation=len(primary),validation_or_quarantine_in_sampler=False,context_rgb=name=='context',parent_weights_trainable=False))
    for epoch in range(start_epoch,p['epochs']+1):
        if (root/'STOP_REQUESTED').exists():raise InterruptedError('STOP_REQUESTED')
        model.train();indices=np.r_[rng.permutation(len(fitrows)),rng.choice(len(fitrows),len(fitrows),p=weights)];rng.shuffle(indices);losses=[]
        for start in range(0,len(indices),p['batch']):
            ids=[positions[fitrows[int(i)]['id']][int(rng.integers(p['train_views']))] for i in indices[start:start+p['batch']]];ids=np.array(ids)
            local=torch.from_numpy(np.asarray(arr[0][ids])).cuda();parent=torch.from_numpy(np.asarray(arr[1][ids])).cuda();context=torch.from_numpy(np.asarray(arr[2][ids])).cuda().float()/255;target=torch.from_numpy(np.asarray(arr[3][ids])).cuda()
            prediction,aux,available=model.cached(local,parent,context);loss=F.cross_entropy(prediction,target,weight=prediction.new_tensor([2.,1.,1.]))+.5*F.cross_entropy(aux,target,weight=prediction.new_tensor([2.,1.,1.]))
            teacher=parent.softmax(1);mask=(teacher.argmax(1)==target)&(teacher.max(1).values>=.8)
            if mask.any():loss+=.5*F.kl_div(F.log_softmax(prediction[mask]/2,1),F.softmax(parent[mask]/2,1),reduction='batchmean')*4
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],5.);optimizer.step();losses.append(float(loss.detach()))
        model.eval();pred=predict_cached(model,arr,primary);current=metrics(primary_rows,pred);fitness=score(current,anchor)
        if fitness>best_score:
            best_score=fitness;best_epoch=epoch;save_model(folder/'best.pt',model,epoch=epoch,fitness=fitness);atomic_json(folder/'best_metrics.json',current)
        current_frozen=model.frozen_state();assert all(torch.equal(value,current_frozen[key]) for key,value in frozen.items())
        summary=dict(epoch=epoch,best_epoch=best_epoch,fitness=fitness,best_fitness=best_score,loss=float(np.mean(losses)),**{k:current['all'][k] for k in ['errors','ng_not_detected','false_ng','no_disk_as_ok']})
        atomic_json(folder/'progress.json',summary)
        with (folder/'history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(summary)+'\n')
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,best_score=best_score,best_epoch=best_epoch,rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),folder/'last.pt')
        publish(root,'running',name,child=os.getpid(),epoch=epoch,best_epoch=best_epoch,metrics=summary);print(name,json.dumps(summary),flush=True)
        if epoch-best_epoch>=p['patience']:break
    assert state_sha(model.frozen_state())==read(root/'features/complete.json')['parent_frozen_sha256']
    atomic_json(folder/'completion.json',dict(epochs_completed=epoch,best_epoch=best_epoch,best_fitness=best_score,parent_all_weights_bn_unchanged=True))

def evaluate(root):
    p=check(root);records=read(root/'source_manifest_at_training.json')['records'];arr=arrays(root);elig=read(root/'eligibility.json');primary=set(elig['primary_validation_ids']);val=[i for i,r in enumerate(records) if r['split']=='val'];primary_indices=[i for i,r in enumerate(records) if r['id'] in primary]
    (root/'evaluations').mkdir(exist_ok=True);results={}
    # Re-evaluate frozen parent and frozen current SERVICE snapshot on identical rows.
    data_rows=[dict(r,path=str(root/'input_snapshot'/r['image'])) for r in records]
    for name,path in [('parent',root/'reference/parent.onnx'),('service',root/'reference/service.onnx')]:
        output=base.evaluate_model(path,data_rows,root/'evaluations'/f'{name}.json');pred=[x['prediction'] for x in output['predictions']]
        results[name]=dict(primary=metrics([records[i] for i in primary_indices],[pred[i] for i in primary_indices]),val=metrics([records[i] for i in val],[pred[i] for i in val]))
    for name in NAMES:
        model=load_model(root/'jobs'/name/'best.pt').cuda().eval();pred=predict_cached(model,arr,list(range(len(records))))
        results[name]=dict(primary=metrics([records[i] for i in primary_indices],[pred[i] for i in primary_indices]),val=metrics([records[i] for i in val],[pred[i] for i in val]))
        atomic_json(root/'evaluations'/f'{name}.json',dict(metrics=results[name],predictions=[dict(id=r['id'],truth=r['label'],split=r['split'],prediction=a) for r,a in zip(records,pred)]))
        model=model.cpu();export=root/'jobs'/name/'best.onnx'
        torch.onnx.export(model,(torch.zeros(2,3,224,224),torch.zeros(2,4,224,224)),str(export),input_names=['local','context'],output_names=['probabilities'],dynamic_axes={'local':{0:'batch'},'context':{0:'batch'},'probabilities':{0:'batch'}},opset_version=17)
        import onnxruntime as ort
        options=ort.SessionOptions();options.intra_op_num_threads=4;session=ort.InferenceSession(str(export),sess_options=options,providers=['CPUExecutionProvider']);maximum=0.;wrong=0
        with torch.inference_mode():
            for start in range(0,len(val),32):
                ids=val[start:start+32];local=np.array([tensor(cv2.imread(data_rows[i]['path'])) for i in ids]);context=np.asarray(arr[2][ids]).astype(np.float32)/255
                expected=model(torch.from_numpy(local),torch.from_numpy(context)).numpy();actual=session.run(None,dict(local=local,context=context))[0]
                maximum=max(maximum,float(np.max(np.abs(actual-expected))));wrong+=int((actual.argmax(1)!=expected.argmax(1)).sum())
        parity=dict(count=len(val),argmax_mismatches=wrong,max_probability_error=maximum,passed=wrong==0 and maximum<1e-3);atomic_json(root/'jobs'/name/'parity.json',parity);assert parity['passed']
    failures=[];context=results['context'];parent=results['parent'];service=results['service']
    if score(context['primary'],parent['primary'])<=score(results['control']['primary'],parent['primary']):failures.append('context_not_better_than_mask_only_control')
    for scope in ['primary','val']:
        if context[scope]['all']['ng_not_detected']!=0:failures.append(scope+'/zero_ng_required')
        for ref in ['parent','service']:
            for domain,m in context[scope].items():
                if m['count'] and m['ng_not_detected']>results[ref][scope][domain]['ng_not_detected']:failures.append(scope+'/'+domain+'/ng_above_'+ref)
                if m['count'] and m['false_ng']>results[ref][scope][domain]['false_ng']:failures.append(scope+'/'+domain+'/false_ng_above_'+ref)
    atomic_json(root/'release_decision.json',dict(recommended_for_review=not failures,recommendation_failures=failures,metrics=results,selected_onnx=str(root/'jobs/context/best.onnx'),selected_pt=str(root/'jobs/context/best.pt'),production_deployed=False,prior_parent_exposure=True))
    page_html=page(p).replace('</html>','')+'<h2>Kết quả</h2><table><tr><th>Scope</th><th>Model</th><th>N</th><th>NG lọt</th><th>NG ảo</th><th>Trống→OK</th><th>Lỗi</th></tr>'
    for scope in ['primary','val']:
        for name,m in results.items():
            q=m[scope]['all'];page_html+=f'<tr><td>{scope}</td><td>{name}</td><td>{q["count"]}</td><td>{q["ng_not_detected"]}</td><td>{q["false_ng"]}</td><td>{q["no_disk_as_ok"]}</td><td>{q["errors"]}</td></tr>'
    atomic_text(root/'comparison.html',page_html+'</table><p>'+html.escape(', '.join(failures) or 'Đạt gate rà tiếp')+'</p></html>')

class RuntimeContext:
    def __init__(self,path):
        import onnxruntime as ort
        options=ort.SessionOptions();options.intra_op_num_threads=4;self.session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider']);self.pending=None
    def capture(self,image,rects,direction):self.pending=make_contexts(image,rects,direction)[0]
    def predict_batch(self,crops):
        if self.pending is None or len(crops)!=len(self.pending):raise ValueError('Context not paired with current target row')
        labels=[];confidence=[]
        for start in range(0,len(crops),2):
            local=np.array([tensor(im) for im in crops[start:start+2]]);context=np.transpose(np.array(self.pending[start:start+2]),(0,3,1,2)).astype(np.float32)/255
            out=self.session.run(None,dict(local=local,context=context))[0]
            labels.extend(base.LABELS[int(x.argmax())] for x in out);confidence.extend(float(x.max()) for x in out)
        self.pending=None;return labels,confidence

def replay(root):
    p=check(root);import decouple
    from unittest.mock import patch
    decouple.config=decouple.Config(decouple.RepositoryEnv(str(PROJECT/'SERVICE/config/config.env')));os.chdir(PROJECT/'SERVICE')
    from src.communication.modbus_client import MbClient
    def forbid(*a,**k):raise RuntimeError('Hardware forbidden')
    with patch.object(MbClient,'connect',forbid),patch.object(MbClient,'_MbClient__write_bit',forbid):
        from src.service.check_disk_service_yolo import DiskCheckingService
        from training.audit_new_factory_images import load
        service=DiskCheckingService();baseline=service.point_classification_model;candidate=RuntimeContext(root/'jobs/context/best.onnx');original=service.get_line_boxes_ratio_shift
        def capture(image,boxes,direction,*a,**k):
            rects=original(image,boxes,direction,*a,**k)
            if service.point_classification_model is candidate:candidate.capture(image,rects,direction)
            return rects
        service.get_line_boxes_ratio_shift=capture;data=read(PROJECT/'.work/factory_audit_20261020_run/report.json');rows=[];debug=set()
        for item in data['white']:
            image=load(Path(item['source']),item['camera'],Path(data['app_config']));times=[];outputs=[]
            for clf in [baseline,candidate]:
                service.point_classification_model=clf;start=time.perf_counter();outputs.append(service.check_disk_white(image.copy()));times.append((time.perf_counter()-start)*1000)
            b,a=outputs;fields=['CropBox','UvBox1','UvBox2','Mid1','Mid2','MinDiskDistance','MaxDiskDistance','InferredSlots','PointGeometryStatus'];assert all(b.model_dump()[k]==a.model_dump()[k] for k in fields)
            if item['camera'] not in debug:
                result=service.check_disk_debug(image.copy(),service._default_white_params());assert result.FinalImg==a.ResImg and result.Result==a.Result;debug.add(item['camera'])
            rows.append(dict(camera=item['camera'],source=item['source'],baseline_ms=times[0],candidate_ms=times[1],ratio=times[1]/times[0],geometry_counts_unchanged=True));atomic_json(root/'white_progress.json',rows);print('WHITE',len(rows),rows[-1]['ratio'],flush=True)
    ratios=np.array([r['ratio'] for r in rows]);assert len(rows)==28 and debug=={1,2}
    latency=dict(frames=28,hardware_calls=False,both_camera_debug_parity=True,geometry_counts_unchanged=True,median_paired_ratio=float(np.median(ratios)),max_paired_ratio=float(ratios.max()),within_2x=bool((ratios<=2).all()),records=rows)
    atomic_json(root/'white_replay.json',latency);release=read(root/'release_decision.json')
    if not latency['within_2x']:release['recommendation_failures'].append('white_latency_over_2x')
    release['recommended_for_review']=not release['recommendation_failures'];atomic_json(root/'release_decision.json',release)

def finalize(root):
    p=check(root);m=read(root/'source_manifest_at_training.json');assert all(digest(root/'input_snapshot'/r['image'])==r['sha256'] for r in m['records'])
    contexts=read(root/'context_manifest.json');assert all(digest(root/r['context'])==r['sha256'] for r in contexts['records'] if r['available'])
    assert all(digest(s['metadata']['source'])==s['source_sha256'] for s in contexts['raw_sources'])
    cache_info=read(root/'features/complete.json');assert all(digest(root/'features'/name)==value for name,value in cache_info['files'].items())
    for name in NAMES:
        assert read(root/'jobs'/name/'parity.json')['passed'];model=load_model(root/'jobs'/name/'best.pt');assert state_sha(model.frozen_state())==cache_info['parent_frozen_sha256']
    white=read(root/'white_replay.json');assert white['frames']==28
    current=audit(Path(p['pool']));atomic_json(root/'integrity_after.json',current);assert current['passed'];release=read(root/'release_decision.json')
    atomic_json(root/'completion_verification.json',dict(passed=True,verified_at=base.now(),canonical_revision_at_start=12,current_revision=current['canonical_revision'],input_code_protocol_context_and_raw_hashes_verified=True,runtime_unchanged=True,parent_all_weights_bn_unchanged=True,capture_disjoint_adaptation=True,parent_historical_exposure=True,parity={name:read(root/'jobs'/name/'parity.json') for name in NAMES},white_replay=white,recommended_for_review=release['recommended_for_review'],production_deployed=False))
    lines=['# Context experiment','',f"Recommended for further review: {release['recommended_for_review']}. No deployment.",'','| Scope | Model | N | Missed NG | False NG | Empty as OK | Errors |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for scope in ['primary','val']:
        for name,value in release['metrics'].items():
            q=value[scope]['all'];lines.append(f"| {scope} | {name} | {q['count']} | {q['ng_not_detected']} | {q['false_ng']} | {q['no_disk_as_ok']} | {q['errors']} |")
    lines+=['','Failures: '+(', '.join(release['recommendation_failures']) or 'None'),'',f"Whole White median ratio{white['median_paired_ratio']:.3f}x/max{white['max_paired_ratio']:.3f}x.",'','Parent local model is frozen best111; contextless legacy/AI remain parent predictions. Context input raw crops reconstructed pixel-exact. Current canonical labels/splits unchanged.','Primary adaptation fitting excludes all canonical-train rows from held capture bursts. Parent already saw historical data; this is not an independent blind test. Full canonical val is secondary spatial development. No zero-miss production guarantee. SERVICE field-test model/retained HA unchanged.']
    atomic_text(root/'RESULTS.md','\n'.join(lines)+'\n');atomic_text(root/'index.html',(root/'comparison.html').read_text(encoding='utf-8'))

def campaign(root,resume=False):
    cmd="Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    raw=subprocess.run(['powershell.exe','-NoProfile','-Command',cmd],capture_output=True,text=True,creationflags=FLAGS,check=True).stdout.strip();processes=json.loads(raw) if raw else [];processes=processes if isinstance(processes,list) else [processes]
    if any(x['ProcessId']!=os.getpid() and 'train_classifier_' in (x.get('CommandLine') or '') and '--phase' in x['CommandLine'] for x in processes):raise RuntimeError('Another trainer alive')
    if resume and read(root/'status.json')['status']=='complete':raise ValueError('Do not resume completed trial')
    p=check(root) if resume else prepare(root);script=root/'code_snapshot'/Path(__file__).name
    phases=[('cache','features/complete.json')]+[(name,f'jobs/{name}/completion.json') for name in NAMES]+[('evaluate','release_decision.json'),('replay','white_replay.json'),('finalize','completion_verification.json')]
    for phase,done in phases:
        if (root/done).exists():continue
        if (root/'STOP_REQUESTED').exists():raise InterruptedError('STOP_REQUESTED')
        with (root/'logs'/f'{phase}.log').open('a',encoding='utf-8') as handle:
            child=subprocess.Popen([sys.executable,'-X','utf8','-u','-B',str(script),'--phase',phase,'--root',str(root)],cwd=PROJECT,stdout=handle,stderr=subprocess.STDOUT,creationflags=FLAGS)
            while child.poll() is None:
                if (root/'STOP_REQUESTED').exists():child.terminate();child.wait();raise InterruptedError('STOP_REQUESTED')
                if not (root/'status.json').exists() or read(root/'status.json').get('phase')!=phase:publish(root,'running',phase,child.pid)
                time.sleep(5)
            if child.returncode:raise RuntimeError(phase+' failed; inspect '+str(root/'logs'/f'{phase}.log'))
    assert read(root/'completion_verification.json')['passed'];publish(root,'complete','context/control evaluated, parity and White verified; no deployment')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--phase',choices=['campaign','cache','control','context','evaluate','replay','finalize'],required=True);parser.add_argument('--resume',action='store_true');args=parser.parse_args();root=args.root.resolve()
    try:
        if args.phase=='campaign':campaign(root,args.resume)
        elif args.phase in NAMES:fit(root,args.phase)
        else:globals()[args.phase](root)
    except Exception as exc:
        if root.exists():publish(root,'stopped' if isinstance(exc,InterruptedError) else 'failed',args.phase,error=repr(exc))
        traceback.print_exc();raise

if __name__=='__main__':main()
