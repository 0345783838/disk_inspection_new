"""One-to-one mask precision/recall and continuity at fixed confidence values."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from evaluate_edge_continuity import read_instances


def iter_predictions(records, model_path, imgsz, min_conf=.1):
    if model_path.suffix.lower()=='.onnx':
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
        from inference.yolo_segment.onnx_segmentor import OnnxSegmentor
        model=OnnxSegmentor(str(model_path.resolve()),['disk'],preprocess_mode='letterbox',
                            rgb_input=True,mask_decode='native')
        if model.input_width!=imgsz or model.input_height!=imgsz:
            raise ValueError('ONNX shape does not match requested input size')
        for record in records:
            image=cv2.imread(record['image'])
            _,scores,_,masks=model.segment_objects_debug(image,min_conf,.8)
            yield record,[mask>0 for mask in masks],np.asarray(scores)
    else:
        model=YOLO(str(model_path.resolve()))
        predictions=model.predict([r['image'] for r in records],imgsz=imgsz,rect=False,batch=4,
                conf=min_conf,iou=.8,retina_masks=True,device=0,verbose=False,stream=True)
        for record,prediction in zip(records,predictions):
            masks=[] if prediction.masks is None else list(prediction.masks.data.cpu().numpy()>0)
            yield record,masks,prediction.boxes.conf.cpu().numpy()


def compare(targets, predicted, scores, conf, threshold):
    predictions=[mask for mask,score in zip(predicted,scores) if score>=conf]
    ious=np.zeros((len(targets),len(predictions)),np.float32)
    for i,target in enumerate(targets):
        for j,prediction in enumerate(predictions):
            ious[i,j]=np.count_nonzero(target&prediction)/max(1,np.count_nonzero(target|prediction))
    pairs=[]
    if ious.size:
        rows,columns=linear_sum_assignment(-(ious+100*(ious>=threshold)))
        pairs=[(i,j) for i,j in zip(rows,columns) if ious[i,j]>=threshold]
    full=0
    for i,j in pairs:
        target,prediction=targets[i],predictions[j]
        rows=np.flatnonzero(target.any(axis=1))
        hits=(target&prediction).any(axis=1)[rows]
        if hits.mean()>=.9 and hits[:12].any() and hits[-12:].any():
            full+=1
    return {'tp':len(pairs),'fp':len(predictions)-len(pairs),'fn':len(targets)-len(pairs),
            'gt':len(targets),'predictions':len(predictions),'full_length':full}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--imgsz',type=int,default=320)
    p.add_argument('--split',choices=['train','val'],default='val')
    p.add_argument('--conf-values',type=float,nargs='+',default=[.1,.25,.5])
    args=p.parse_args()
    records=json.loads((args.dataset/'manifest.json').read_text())
    records=[r for r in records if r['split']==args.split]
    if args.split=='train':
        records=[r for r in records if r['origin']=='factory' and r['repeat']==0]
    grouped=defaultdict(lambda:defaultdict(lambda:defaultdict(int)))
    per_image=[]
    for record,predicted,scores in iter_predictions(records,args.model,args.imgsz,min(args.conf_values)):
        shape=cv2.imread(record['image']).shape[:2]
        targets=read_instances(record['label'],shape)
        assert all(mask.shape==shape for mask in predicted)
        group=record.get('scenario',record['origin'])
        values={}
        for conf in args.conf_values:
            for threshold in (.3,.5):
                key=f'conf{conf:g}_iou{threshold:g}'
                result=compare(targets,predicted,scores,conf,threshold)
                for name,value in result.items():
                    grouped[group][key][name]+=value
                values[key]=result
        per_image.append({'source':record['source'],'image':record['image'],'group':group,'values':values})
    summary={}
    for group,variants in grouped.items():
        summary[group]={}
        for key,result in variants.items():
            result=dict(result)
            result['precision']=result['tp']/max(1,result['tp']+result['fp'])
            result['recall']=result['tp']/max(1,result['gt'])
            result['full_length_recall']=result['full_length']/max(1,result['gt'])
            summary[group][key]=result
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'model':str(args.model.resolve()),'imgsz':args.imgsz,
        'conf_values':args.conf_values,'nms_iou':.8,
        'decoder':'ONNX native RGB' if args.model.suffix.lower()=='.onnx' else 'Ultralytics native RGB',
        'summary':summary,'records':per_image,
        'limitation':'HA labels mostly provisional, same-session validation; FP can be unlabelled true edges.'},indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
