"""Widen train-only detection envelopes while retaining actual thin masks.

This is a separate experiment, not a modification to Ultralytics installation.
Validation labels, input preprocessing and inference are unchanged.
"""
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.models.yolo.segment.train import SegmentationTrainer


def widen_thin_boxes(boxes, image_height, image_width, min_width=24):
    result = boxes.clone()
    if not len(result):
        return result
    widths = result[:,2]*image_width
    heights = result[:,3]*image_height
    thin = (widths<min_width)&(heights>image_height*.35)&(heights>widths*8)
    result[thin,2] = min_width/image_width
    # Clip envelope at image boundary without shifting the real mask.
    left = (result[:,0]-result[:,2]/2).clamp(0,1)
    right = (result[:,0]+result[:,2]/2).clamp(0,1)
    result[:,0] = (left+right)/2
    result[:,2] = right-left
    return result


class ThinBoxTrainer(SegmentationTrainer):
    def preprocess_batch(self,batch):
        batch = super().preprocess_batch(batch)
        height,width = batch['img'].shape[-2:]
        batch['bboxes'] = widen_thin_boxes(batch['bboxes'],height,width)
        # Never modify batch['masks']; detection envelope != segmentation strip.
        return batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--pretrained',type=Path,required=True)
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--name',required=True)
    p.add_argument('--epochs',type=int,default=10)
    args = p.parse_args()
    model = YOLO(str(args.pretrained.resolve()))
    model.train(trainer=ThinBoxTrainer,data=str(args.data.resolve()),epochs=args.epochs,
                imgsz=512,batch=4,workers=2,device=0,seed=930,deterministic=True,
                optimizer='AdamW',lr0=.0001,lrf=.05,weight_decay=.0005,patience=10,
                warmup_epochs=0,warmup_bias_lr=.0001,amp=True,
                overlap_mask=False,mask_ratio=2,mosaic=0,mixup=0,copy_paste=0,
                degrees=0,translate=.02,scale=.05,shear=0,perspective=0,
                flipud=0,fliplr=0,hsv_h=.005,hsv_s=.1,hsv_v=.12,bgr=0,
                project=str(args.project.resolve()),name=args.name,exist_ok=False,plots=True,verbose=False)


if __name__=='__main__':
    main()
