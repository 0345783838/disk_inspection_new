# Classifier fine-tuning — HA / fault priority — 2026-10-05

Completed training and offline evaluation. **NOT deployed**. Production classifier, runtime configuration,
camera/PLC settings and old training dataset remain unchanged.

Subsequent corrected-label retraining and the preserved pre-retrain backup are documented in
[CLASSIFIER_RETRAIN_REVIEWED_20261005.md](CLASSIFIER_RETRAIN_REVIEWED_20261005.md).
The original-label historical metrics below are retained, not silently replaced.

## Human labels and split

The user confirmed all 2,000 native HA crops; 175 labels changed from old-model proposals.
Final labels: ng=442, no_disk=321, ok=1,237. Zero unreviewed/uncertain/excluded items.
Review decisions SHA256: `3fddf7951a092b4391eed78bb8ff048bc8dbd2feaf9d84f9a7332b8bd304ec50`.
Snapshots are kept inside each generated dataset. Initial pseudo-labels are NOT used as ground truth.

Preserve user semantics: one disk in TARGET groove=ok; >one=ng; zero=no_disk.
Neighbor grooves / two rims of one disk are not extra target disks. 25 refers to grooves, not disks.
Inputs retain deployed width 1.8, height/vertical shift and top-face 180-degree rotation.

Split whole raw source frames, separately for each camera in capture-time order:

- Train indices 0,1,2,4,5,7: 12 frames / 1,200 crops (ng=218, no_disk=156, ok=826).
- Validation indices 3,6: four frames / 400 crops (ng=96, no_disk=67, ok=237).
- Test indices 8,9: four later frames / 400 crops (ng=128, no_disk=98, ok=174).

No per-crop random split or augmented validation/test images. Exact-byte cross-split duplicates are checked.
This is a **same-session chronological holdout**, NOT an independent factory/capture-session test.
Physical tray/background similarity across frames can still make results optimistic.

## Training contract

Old checkpoint: `D:/huynhvc/WORKING/TRAINING/training_cls/DISK_3/disk_22_05_nano/weights/best.pt`.
Old ONNX matches deployed `c_p.onnx` byte-for-byte:
`0be56f7a3442745e8a31c82601f84d04c80eb6a495d48729f729768e504c8200`.
Class order remains `[ng, no_disk, ok]`.

The installed library's default classification preprocessing preserves aspect ratio then center-crops;
the real service stretches the entire native crop with cv2.INTER_LINEAR to 224x224. Training/validation
now use **pixel-exact runtime preprocessing**: BGR→RGB, cv2 resize, float32 /255, CHW, no extra normalization.
No 224x224 crop dataset is exported. The PT's generic PIL transform approximates stretch; use the ONNX
with the service's cv2 preprocessing as the canonical inference contract, not default YOLO center-crop inference.

Online augmentations affect TRAIN only: horizontal flip 0.5; contrast 0.9..1.1; exposure 0.85..1.15;
gamma 0.9..1.1; occasional mild lighting gradient, noise and 3x3 Gaussian blur. No vertical flip,
random crop, translations, erasing, disk removal/addition, mixup/cutmix or AI-generated unverified labels.
The target-groove occupancy label is preserved. Class-aware sampling emphasizes fault classes.

Runtime/library: Ultralytics 8.3.200 / Torch 2.5.0+cu118; RTX 3050 OEM 8GB;
AdamW, batch 64, 224px, seed 1005, deterministic, workers 2, AMP off, cosine LR, patience 20.

## Experiments and selection

Artifacts: `.work/classifier_finetune_20261005/`.

| Trial | Training mix | HA val accuracy | Legacy val accuracy | Legacy true NG missed |
| --- | --- | --- | --- | --- |
| Original model | Existing checkpoint | 91.25% | 98.34% | 21/821 |
| HA-only | Fine-tune old checkpoint on reviewed HA | 99.50% | 98.34% | 24/821 |
| Fault replay | HA 85% sampling + 550 old TRAIN anchors | 99.25% | 95.49% | 21/821 |
| Dual-validation replay — selected | HA 75% sampling + 1,000 old TRAIN anchors | 98.50% | 98.19% | 19/821 |

Fault replay uses 400 ng / 100 no_disk / 50 ok legacy examples; rejected because legacy empty→OK
rose from 2 to 51. HA-only improves direct ng→ok (16→14), but increases ng→no_disk (5→10), so it
is NOT the safest final choice despite its higher aggregate accuracy.

Selected replay: 600 ng / 300 no_disk / 100 ok old TRAIN images, half hard examples and half seeded diverse
examples per class. Mining uses the HA-only candidate on old TRAIN only. Validation/test images are never
mined into training. 354 duplicate / validation-overlap legacy images were excluded from the replay pool;
clean pool ng=1,607, no_disk=483, ok=3,136. Old source files are untouched.
Legacy train class sampling fractions are 60% ng / 30% no_disk / 10% ok within its 25% sampling budget.
Native unique TRAIN counts are 2,200 total: ng=818, no_disk=456, ok=926.

Selected run starts from HA-only best, LR 0.0001, dropout 0.05; 28 epochs ran, best checkpoint at epoch 8.
HA-only uses LR 0.0003/dropout 0.1 and best epoch 6 (26 total); first fault-replay run used the old base
checkpoint, LR 0.0003/dropout 0.1 and 22 total epochs. Validation never augments images.
Dual run validates on 400 HA + all 2,770 legacy val images each epoch, separated by domain in metrics.

Production policy evidence: `MainController.cs` explicitly preserves Warning→PLC OK; service no_disk
predictions can yield WARNING. Therefore both **ng→ok and ng→no_disk count as defect misses**.
No production policy change was made. Original training selection favored direct ng→ok; before opening
test, the final selector was corrected to include both misses. The dual run's saved epoch 8 is also
the best eligible epoch under this corrected risk on its recorded validation results.

Final shipping selection uses NG-miss risk weighted HA 75% / legacy 25%, with retention gates:
legacy NG misses ≤ original 21; legacy empty→OK ≤ original 2; HA OK recall ≥90%; legacy OK recall ≥95%;
legacy overall accuracy ≥97%. Remaining tie-breaks consider direct ng→ok and macro-F1.
Selection is frozen in `selected/selection.json` **before** HA test inference; model/hash/dataset checks
enforce that the same chosen model is tested. No retraining or re-selection followed test results.
Legacy val participates in model selection, so it is a retention benchmark, not an independent test.
Old checkpoint training already used the old dataset; its old train/val duplication further limits independence.

## Locked final results (runtime ONNX, unfiltered argmax)

| Population | Accuracy old→new | True NG missed old→new | NG recall old→new | Empty→OK old→new |
| --- | --- | --- | --- | --- |
| HA validation, 400 crops | 91.25%→98.50% | 21/96→1/96 | 78.13%→98.96% | 4→0 |
| HA test, 400 crops | 89.25%→97.75% | 22/128→2/128 | 82.81%→98.44% | 3→0 |
| Legacy val, 2,770 crops | 98.34%→98.19% | 21/821→19/821 | 97.44%→97.69% | 2→2 |

Test residual NG misses: one ng→ok and one ng→no_disk. Legacy residuals: 15 ng→ok and four ng→no_disk.
Legacy OK falsely labeled ng/no_disk rises 18→25 (false reject tradeoff); HA test OK wrongly labeled
ng/no_disk falls 14→6. These are **per-crop classification errors**, not whole-tray PLC false-PASS rates.
HA TRAIN accuracy 99.75% is in-sample and is not the headline metric.
No threshold was lowered or runtime filtering modified to improve reported scores.

47 regression tests pass, including exact preprocessing, frame splitting, fault-ranking, warning-miss
gating and domain sampling tests. ONNX checker + PyTorch/ONNX probability and argmax parity are verified
for batches 1,2,32; see `final_evaluation/report.json` for exact probability tolerance measurements.

Model is improved, **not zero-miss / production-qualified**. Fresh sessions, lighting changes and independent
NG trays are still required for a factory safety conclusion. Do not train on this test set and keep calling
it an unseen test. If review reveals incorrect test labels, version the correction and disclose the changed protocol.

## Delivered files / next step

- `selected/c_p_ha_fault_priority_20261005.onnx` — canonical runtime candidate.
- `selected/c_p_ha_fault_priority_20261005.pt` — portable fine-tuned checkpoint.
- ONNX SHA256 `a3fa7c15a23c02d9fe3b1d7fa0ea175bc436ebab5c86b78ba99e4486e001b0a4`.
- PT SHA256 `210ca39fb60d7c53fbcb2f22ce8e785f8e531c1a95a6dbf781395899644d7002`.
- `final_evaluation/report.json`, `predictions.csv`, `per_image.json` — full metrics, GT and all probabilities.
- Read-only error gallery: `http://127.0.0.1:8777/`, default true NG missed by chosen model.
- `dataset_dual_safety/manifest.json` — exact train/val/test provenance and frozen review snapshots.

The next authorized step would be an offline full-flow trial or explicit deployment request, with backups.
This task does not replace `SERVICE/config/models/c_p.onnx` or packaged plugin `c_p.onnx` and does not issue
camera/PLC commands. Existing unrelated edits in the repository are preserved.

## Post-evaluation human label corrections — 2026-10-05

The user identified possible mistakes in the original human labels and authorized editing disagreement
images. Port 8777 now serves `training/serve_classifier_result_review.py` and
`training/classifier_result_review.html`. The original static result gallery is retained at `/index.html`.
The new UI includes the original disagreements of either model (default: the selected model's errors),
dataset/status/GT filters, confidence low-to-high/high-to-low sorting, and 48 native images per page.
Click ng/ok/no_disk to save or confirm the original label; cards remain visible after saving.

Corrections are separate in `final_evaluation/label_review/decisions.json`, bound by SHA256 to both
`per_image.json` and `report.json`. Each edit/reset keeps original GT, prior decision, new label, notes,
UTC timestamp and revision history. Stale concurrent tabs are rejected rather than overwriting edits.
Reload resumes persisted decisions. This endpoint does not alter original review snapshots, datasets,
legacy source images, predictions, selected model, selection lock, or original metrics, and never trains.

Metrics shown above and in the UI remain the **original-label evaluation**, pending a separately versioned
correction audit. Disagreement-only, prediction-visible review is not a blind full-population relabeling;
any later corrected metrics must disclose this scope. Corrected HA test records remain test-only and
must not silently enter training or checkpoint selection. User completion/confirmation is still needed
before applying the corrections to a new dataset or recomputing metrics.

Start from workspace root with:

```powershell
SERVICE/.venv/Scripts/python.exe -B SERVICE/training/serve_classifier_result_review.py --evaluation .work/classifier_finetune_20261005/final_evaluation --port 8777
```
