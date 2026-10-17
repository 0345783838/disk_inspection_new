# Classifier retraining after label correction — 2026-10-05

Training and evaluation are complete. **New candidate NOT deployed. Recommend keeping the previous
fine-tuned classifier:** no reduction in true NG misses; reviewed HA holdout accuracy is slightly lower.
The pre-retrain model is preserved both in its original selected folder and a separate verified backup.

Subsequent user-authorized full-HA learning (including former HA validation/test) is documented in
[CLASSIFIER_FULL_HA_20261005.md](CLASSIFIER_FULL_HA_20261005.md). Historical results below remain intact.

## Human correction snapshot / provenance

Original evaluation: `.work/classifier_finetune_20261005/final_evaluation/`.
Correction revision 92: 86 disagreement images reviewed, 37 labels changed.
All 68 original disagreements of the previous selected model were reviewed; 157 old-model-only
disagreements remain at their original labels. This is a **prediction-visible disagreement-only audit**,
not a blind full-dataset label audit. Unchanged original annotations are retained, not automatically relabeled.

- HA train: 4 corrections, all ok→ng. These are the ONLY corrected records used for training.
- HA validation: 2 corrections.
- HA test: 5 corrections.
- Original legacy validation: 26 corrections.

Source evaluation binding SHA256:
`2084abc47e3398ec9098abe94ff826590127f9eb954b323c57eae892539e803f`.
Frozen correction snapshot SHA256:
`150894da1b19dddd22b649278ed19bec7c1f4f4441eb07c33ccbe6f871ee6991`.

New root: `.work/classifier_retrain_reviewed_20261005/`. `dataset/manifest.json` and
`dataset/correction_audit.json` preserve prior labels, every correction, unchanged image hashes and splits.
Source crops, original datasets, initial review decisions, old evaluation and old model are not overwritten.
Legacy validation corrections are label overrides bound to original source-file checksums; the external
legacy folders remain unchanged. Legacy TRAIN replay membership remains the same as the previous run.

Same split: train=1,200 HA + 1,000 legacy TRAIN anchors; validation=400 HA + all 2,770 legacy val;
test=400 HA. Legacy anchors retain 600 NG / 300 no_disk / 100 OK and NG-heavy within-domain sampling.
HA train now ng=222, no_disk=156, ok=822; HA validation ng=98, no_disk=67, ok=235;
HA test ng=131, no_disk=99, ok=170. Corrected legacy val ng=814, no_disk=520, ok=1,436.
There are no split changes or exact-byte cross-split duplicates. Test images never enter training.

The HA test set has ALREADY been viewed and its disagreements reviewed. This round is a
**re-evaluation of a reviewed same-session holdout**, NOT a new blind or independent factory test.
Checkpoint selection for the new trials still uses validation only, and the candidate was locked
before its holdout inference. `selection_locked_before_ha_test=false` is deliberate; the narrower
`selection_locked_before_new_candidate_ha_test=true` records the actual protocol. No further training,
checkpoint switching or re-selection followed the new candidate's test results.

## Training and selection

Both trials start from the previously selected fine-tuned PT, not the original production classifier.
Exact cv2 runtime preprocessing and safe augmentation are unchanged: full native crop → RGB stretch
224x224, float32/255; TRAIN-only mild flip/contrast/exposure/gamma/noise/blur. No occupancy-changing
crop/erasing/mixup or unreviewed AI-generated labels. Native PNGs are retained; 224 only exists as tensors.
AdamW, batch64, workers2, RTX3050 OEM, FP32, seed1005, deterministic, cosine LR, patience20.

| Trial | Sampling HA/legacy | LR / dropout | Epochs completed | Validation decision |
| --- | --- | --- | --- | --- |
| `corrected_low_lr` | 75% / 25% | 0.00005 / 0.05 | 21 | Rejected: legacy NG missed=6, above limit5 |
| `corrected_ha_focus` | 95% / 5% | 0.00001 / 0 | 21 | Selected by corrected validation, saved checkpoint from epoch2 |

Retention gates were frozen from the previous model on SAME corrected labels: legacy NG misses≤5;
legacy no_disk→OK≤4; legacy accuracy≥98.7031%; HA OK recall≥90%; legacy OK recall≥95%.
Primary selection risk remains HA75% / legacy25% **NG→non-NG (ok OR no_disk)**, then direct NG→OK,
empty→OK and macro-F1. Sampling percentages and risk weights are separate concepts.

Ultralytics rounds trainer-returned fitness to five decimals; tiny tie-break terms can disappear there.
Its early-stop log names epoch1, but equal-best checkpoint saving also wrote epoch2; the saved PT contains
two epochs of results and its final validated/ONNX metrics match epoch2. The external candidate selector
uses unrounded metric-derived risk and compares the retained previous model, the rejected trial, and
the HA-focus trial before holdout inference. No claimed epoch is inferred solely from the early-stop log.

## SAME corrected labels — before / after new training

"Before" is the previous fine-tuned model, not the original production model or pre-correction metrics.
Frozen previous predictions were rescored first, then independently checked on corrected validation.

| Population | Accuracy before→after | True NG missed before→after | no_disk→OK before→after |
| --- | --- | --- | --- |
| HA validation, 400 | 99.00%→99.25% | 1/98→1/98 | 0→0 |
| Reviewed HA holdout, 400 | 99.00%→98.75% | 1/131→1/131 | 0→0 |
| Corrected legacy val, 2,770 | 98.95%→98.88% | 5/814→5/814 | 4→3 |

The new candidate introduces one additional false reject on HA holdout. Legacy OK false rejects
rise16→20; legacy no_disk errors fall8→6. HA TRAIN in-sample accuracy goes99.9167%→99.8333%,
NG misses1/222 unchanged; this is NOT evidence of generalization.

Original production model on corrected labels: HA holdout90.00%, NG misses22/131;
legacy val99.17%, NG misses7/814. The previous fine-tune remains a substantial HA improvement.
Original pre-correction numbers must not be mixed with these corrected-label comparisons.

NG miss counts are crop-level classification errors, not whole-tray PLC false-PASS rates.
APP's Warning→PLC OK policy is untouched; ng→no_disk remains a defect miss. Confidence thresholds
were not lowered and evaluations use unfiltered runtime argmax.

## Artifacts / preservation

- Previous model backup: `.work/classifier_model_backups/pre_relabel_retrain_20261005/`.
  Contains byte-identical ONNX/PT/selection, original evaluation, corrected-label rescore, correction snapshot,
  and recovery README. SHA256 matches the original selected files.
- New candidate: `selected/c_p_ha_reviewed_20261005.onnx` and `.pt`.
- ONNX SHA256: `728bcf1d12fdb110e27cc829924d5e1125e149ff93f83f95cf7f4b0288ea5ad0`.
- PT SHA256: `c88b596d61bfb0e644e7d7809ca052e2f7e4c739454c1a7baf4f8a11cf42d687`.
- `selected/selection.json`: validation-only lock, corrected-label provenance and training args.
- `final_evaluation/`: unmodified new inference outputs and ONNX checker / PT parity results.
- `comparison/`: separate three-model report, all per-image confidences, CSV and native error gallery.
- Read-only before/after gallery: `http://127.0.0.1:8778/`. Prior label review8777 remains intact.

PT/ONNX parity: batches1,2,32, max probability error3.3676624298095703e-06, zero argmax mismatches.
`delivery_decision.json` records **do not promote** and preference for the preserved previous model;
this does not rewrite the pre-holdout selection or switch any runtime model.

Production `SERVICE/config/models/c_p.onnx`, APP packaged classifier, config, camera and PLC are unchanged.
No new training is justified solely by reviewed holdout results without fresh independent data.
