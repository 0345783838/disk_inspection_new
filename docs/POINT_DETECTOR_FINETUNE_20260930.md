# Point detector fine-tune — 2026-09-30

Historical training report. The candidate was subsequently deployed for testing
on October 1; see [runtime contract and guarded completion](POINT_DETECTOR_DEPLOY_20261001.md).

## Scope and artifacts

- Source old: `D:/huynhvc/OTHERS/disk_checking/disk_checking/train_yolo_detect/train_yolo_detect/datasets/disk`.
- Source factory: `D:/huynhvc/OTHERS/disk_checking/disk_checking/new_data/HA/Cam1` and `Cam2` White BMP. The source raw files were not modified. UV images were not used for White point detection.
- Prepared dataset: `.work/detect_finetune_20260930/dataset_v3` (143 images, 9,225 boxes). Manifest and per-image review overlays are inside that folder.
- Candidate checkpoint: `.work/detect_finetune_20260930/training_v3/finetune/weights/best.pt`.
- Candidate ONNX: `.work/detect_finetune_20260930/training_v3/finetune/weights/best.onnx` (SHA-256 `B04223C1CE295416E4C7E6F7BF6342499BE7B87A81689AC8CEEE06E19C413BBE`).
- Existing production `SERVICE/config/models/d_p_n.onnx` was **not changed**.

## Label policy and preprocessing contract

The factory images are raw portrait frames. Rotate Cam1 90° clockwise and Cam2 90° counter-clockwise. The canonical result is 1944×2592 px. Crop to `[x1,y1,x2,y2] = [0,0,1944,2200]` **before detector inference**. The crop removes the lower overlap row; without this step the model and evaluation are not equivalent to training. Downstream classification/segmentation may still require the full canonical frame; this crop applies to the detector input only, and detector coordinates remain in the same top-left origin.

Factory labels were **not copied from predictions of the old model**. The 25 slot centers in each of the three owned rows were placed from visually selected endpoints for each camera, interpolated along each row, and transferred between frames using phase-correlation registration. Each box is 60×100 px; representative overlays including validation frames and late frames of both cameras were visually checked. The old detector was run separately for auditing suggestions only. This is geometric manual-assisted annotation, not 1,500 independently hand-drawn boxes; further manual review is appropriate before production release.

The combined dataset has 87 old train + 36 old val + 16 factory train + 4 factory val images. Two byte-identical old images that appeared in both splits were removed from val. Old polygon-like labels were converted to detection rectangles; all output labels have five YOLO fields. No identical source-image hash appears twice in the combined dataset.

## Training and evaluation

Fine-tuned YOLO11n from the old `train77/weights/best.pt`, 80 epochs, 640 px, batch 8, AdamW, starting LR 0.001. The model is one-class (`point`). Validation after training: precision 1.000, recall 1.000, mAP50 0.995, mAP50-95 0.857 on the combined 40-image val set. This val score is **not an independent factory generalization estimate**: the four factory validation frames are from the same short acquisition sessions as factory training frames.

ONNX input/output schema matches the existing detector: `images [1,3,640,640]` → `output0 [1,5,8400]`. Evaluation used `SERVICE/src/inference/yolo_detect/onnx_detector.py` (its stretch resize and NMS), IoU threshold 0.1, confidence threshold 0.5. A point was matched at box IoU ≥ 0.3. Full results are in `.work/detect_finetune_20260930/evaluation_all_factory_conf_0_5.json`.

| Set | Production ONNX | Candidate ONNX |
| --- | ---: | ---: |
| Old val, 36 images | 2325/2325 matched, 27 unmatched predictions | 2325/2325 matched, 43 unmatched predictions |
| Cam1 factory, 10 images | 744/750 matched, 6/10 complete 3×25 | 750/750 matched, 10/10 complete, 0 unmatched |
| Cam2 factory, 10 images | 737/750 matched, 3/10 complete 3×25, 14 unmatched | 750/750 matched, 10/10 complete, 0 unmatched |

The 20-frame factory result includes 16 training images, so it is an in-sample consistency check, not a production accuracy claim. On the four held-out factory frames alone the candidate matched 300/300 and had zero unmatched predictions. The old set contains partial-view images whose unlabeled visible rows can be predicted; the 43 unmatched predictions must be manually reviewed before treating all as operational false positives. The new candidate is intentionally **not deployed** yet.

## Reproduce

Run with the Python 3.10 environment containing PyTorch, Ultralytics, OpenCV and ONNX Runtime:

```powershell
python SERVICE/training/build_point_dataset.py --old-dataset D:/huynhvc/OTHERS/disk_checking/disk_checking/train_yolo_detect/train_yolo_detect/datasets/disk --factory D:/huynhvc/OTHERS/disk_checking/disk_checking/new_data/HA --output .work/detect_finetune_20260930/dataset_rebuild
python SERVICE/training/train_point_detector.py --dataset .work/detect_finetune_20260930/dataset_rebuild --checkpoint D:/huynhvc/OTHERS/disk_checking/disk_checking/train_yolo_detect/train_yolo_detect/runs/detect/train77/weights/best.pt --output .work/detect_finetune_20260930/training_rebuild
python SERVICE/training/evaluate_point_detector.py --dataset .work/detect_finetune_20260930/dataset_v3 --candidate .work/detect_finetune_20260930/training_v3/finetune/weights/best.onnx --production SERVICE/config/models/d_p_n.onnx --output .work/detect_finetune_20260930/evaluation_rebuild.json --conf 0.5 --all-factory
```

Before deployment: integrate the exact rotation/crop contract in APP/SERVICE; collect independent factory sessions covering displacement, illumination, disk states, and camera adjustment; visually audit all new labels and unmatched old-set predictions; then evaluate 3×25 per camera and full downstream inspection, not only box mAP.
