# Classifier — learn all corrected real HA — 2026-10-05

Protocol changed at the user's explicit request: formerly held-out HA examples must be learned by
the final model. Historical runs and the pre-retrain ONNX/PT backup remain untouched. No deployment.
Training/evaluation are complete; the selected artifact and completed metrics are recorded below.

## Data and meaning of evaluation

Frozen source: `.work/classifier_retrain_reviewed_20261005/dataset/`, correction revision92,
SHA256 `150894da1b19dddd22b649278ed19bec7c1f4f4441eb07c33ccbe6f871ee6991`.
All 2,000 native HA crops / 20 raw source frames: NG451, no_disk322, OK1,227.
Nineteen hard IDs = nine genuine residual errors of the preserved model on corrected labels, union
eleven HA relabelled IDs. Their union is19, not20 because one record belongs to both categories.
Predictions are NOT used to change GT; all labels remain the original human review plus confirmed corrections.

No HA crop is permanently excluded from final training. Former train/validation/test designations
are preserved as `previous_split`, but **the final model's full HA metrics are in-sample fitting**.
They must never be reported as unseen-test accuracy. The old HA test protocol is explicitly abandoned
for this new final model, rather than silently leaking test data into an unchanged benchmark.

Three camera-block cross-validation folds, each grouping complete raw frames and all four faces/25 slots:
per camera sorted indices0..2 /3..5 /6..9, giving6/6/8 validation frames and600/600/800 HA validation crops.
Every HA crop is held out once per recipe. CV models each start from the ORIGINAL pre-HA classifier PT,
SHA256 `98fd8825e9c16f5e4ddd5ed96b6370ce09bbd182b0d43663ba819712dde91164`.
No prior HA-trained weights are used to initialize those CV models. Folds never share trained checkpoints.

This is still **same-session, previously reviewed, cross-validation used for checkpoint/recipe selection**,
not a blind independent factory test. Tray/background similarity across images and prediction-visible
label review limit conclusions about future captures. OOF predictions are a model-selection estimate.

Legacy replay: same clean original TRAIN anchors600NG/300no_disk/100OK, no old val samples in training.
All2,770 corrected legacy val records remain a retention benchmark. The original pretrained model's
old train/val overlap is known, so this is not an independent legacy test either. The same legacy images
evaluated under three CV models are not8,310 independent examples; aggregate gates scale by three.

## Recipes / augmentation / priorities

Both recipes: pre-HA checkpoint, AdamW LR0.0001, dropout0.05, batch64, deterministic seed1005,
workers2, AMPoff, cosine LR, TRAIN-only exact-runtime preprocessing and mild safe augmentation.
No random crop/erasing, occupancy-changing generation, rotation/translation, unreviewed synthetic GT,
lowered confidence, or runtime/PLC policy changes.

- Balanced: HA90% / legacy10% sampling, HA class factors[1.35,1.15,1.0], no hard-ID multiplier.
- Hard-focus: HA95% / legacy5%, HA factors[1.75,1.15,1.0], multiplier3 for hard IDs in TRAIN only.

Class order[NG,no_disk,OK]. Hard weighting does not add held-out IDs to a fold's training set.
Within legacy's budget:60%NG/30%no_disk/10%OK. Native crops retained;224 appears only in tensors.

Rank prioritizes actual HA NG misses, then legacy NG misses, then HA and legacy total errors.
Guard HA OK recall≥97%, legacy OK recall≥95%, legacy accuracy≥98.7%, empty→OK≤4;
legacy NG limit≤7 per CV model (original production on corrected GT),≤5 for final fitting candidates
(preserved previous fine-tune). Integer radix avoids disappearing priorities at five-decimal fitness rounding.

## Final fitting / preservation

Select recipe from grouped CV, then train two all-HA candidates: original pre-HA weights at LR0.0001,
and the preserved good HA checkpoint at LR0.00003. Final fitting includes all2,000 HA plus1,000 legacy anchors.
A deliberately duplicated `ha_fit` view of the2,000 HA is used only for fitting diagnostics/model choice,
along with2,770 untouched legacy validation images. This overlap is explicit and allowed ONLY for this
in-sample view; train/validation frame/hash overlap is forbidden in CV and for legacy val.

Compare both final candidates with the preserved previous model on identical corrected HA/legacy labels;
choose an eligible candidate without claiming in-sample metrics are generalization. If neither improves
the fault/fitting criteria while maintaining retention, preserve the previous model and report honestly.

Artifacts: `.work/classifier_fullha_20261005/`; protocol, per-fold datasets/logs/ONNX/PT,
OOF predictions, checked CV recipe choice, full-fit reports and release decision are versioned separately.
The user's backup `.work/classifier_model_backups/pre_relabel_retrain_20261005/` remains untouched.
`SERVICE/config/models/c_p.onnx`, APP packaged classifier, camera/PLC config are not changed.

Reproduce with `SERVICE/training/train_classifier_fullha.py --phase prepare`, then `--phase cv`, then
`--phase final`; all output trees fail rather than overwrite earlier work. This requires the global
Python3.10 Torch/Ultralytics CUDA environment; inference contract remains canonical ONNX/cv2 CPU runtime.

## Completed results / model chosen

Grouped CV: balanced99.05% (19 errors; NG missed4/451), hard-focus97.35% (53 errors; NG missed28/451).
All individual CV models satisfy their original legacy retention gates; heavy hard-ID sampling performed
worse on HA and is NOT used in the final model. Balanced OOF confusion rows[NG,no_disk,OK]:
`[[447,0,4],[1,319,2],[11,1,1215]]`.

Six CV runs were completed from the pre-HA checkpoint. Both final full-HA candidates use balanced
sampling. `full_from_original` completed28 epochs, saved best epoch16; `full_from_backup` completed14,
saved best epoch2. Original full-fit attained zero HA errors but failed legacy NG/empty retention.
The warm-start full-fit improved real HA NG misses3→1 while retaining legacy NG misses5.

Two additional **single-network weight/float-buffer interpolations** of those full-HA candidates were
tested (25% and50% cold-parent weight). This is not a two-model prediction ensemble, score manipulation,
threshold reduction or relabeling. Each candidate exports one normal YOLO11n-cls ONNX of the same topology.
Source files remain unchanged; each derived checkpoint records both parent hashes and fraction.
Best eligible artifact is **50% full-from-original /50% full-from-backup**. Interpolation choices are
selected on known HA fitting plus legacy retention, NOT independently validated by the CV estimate.
The CV99.05% describes recipe/checkpoint selection from pre-HA starts, not the exact final interpolated model.

| Candidate | Known HA fitting accuracy | Known HA NG misses | Legacy val accuracy | Legacy NG misses | Eligible |
| --- | --- | --- | --- | --- | --- |
| Preserved previous model | 99.55% | 3/451 | 98.95% | 5/814 | Yes |
| Full from original | 100.00% | 0/451 | 99.21% | 8/814 | No: NG/empty gates |
| Full from backup | 99.75% | 1/451 | 98.99% | 5/814 | Yes |
|25% cold weight interpolation | 99.85% | 0/451 | 99.10% | 5/814 | Yes |
|50% cold weight interpolation — selected | **99.95%** | **0/451** | **99.28%** | **5/814** | Yes |

Selected HA fitting confusion: `[[451,0,0],[0,322,0],[1,0,1226]]`.
Only residual HA error is true OK→NG:
`Cam1_Image_20260918130502793__row1_bottom__slot01`, confidence0.9130040407180786.
Selected legacy confusion: `[[809,0,5],[2,514,4],[8,1,1427]]`.
Legacy total errors29→20, NG misses remain5 and empty→OK remains4.
No zero-miss claim is made for unseen factory captures; full HA fitting100% NG recall is not generalization.

Artifact: `.work/classifier_fullha_20261005/selected_v2/c_p_ha_all_reviewed_20261005.onnx` and `.pt`.
ONNX SHA256 `ce0a150f00a56b78bbc4a07bbf3ad28862e7de7edd78a77bd0ee9f2d6a0d72bd`.
PT SHA256 `d8b82f801195cb9a3f24587f7cc5dac4e36200a1e6a336964bebf855a1835091`.
`release_decision_v2.json` retains all five candidate comparisons; initial `release_decision.json` and
initial `selected/` remain intact. The pre-retrain good model remains untouched in its separate backup.

Native comparison gallery: `http://127.0.0.1:8779/`, default19 hard/relabelled HA crops. Also supports
residual errors, NG misses, newly fixed and regressions; includes old/new confidences and separate
CV vs in-sample tables. `report/report.json`, `per_image.json` and `predictions.csv` retain full provenance.
Earlier review8777 and comparison8778 remain intact.

ONNX checker / direct PT parity: batches1,2,32; maximum probability difference2.4586915969848633e-07,
zero argmax mismatches for selected model. **60 regression tests pass**, including whole-frame grouping,
sampling budget/hard multiplier, explicit full-fit overlap, old-val exclusion and interpolation isolation.
`cv_role_audit.json` verifies zero actual CV hash overlap and shows old historical holdout roles replaced
by the new grouping; it supersedes the early metadata boolean generated while the long CV job was running.
Aggregate CV legacy gate checks are retained in `cv_selection_checked.json`, scaled for three repeated
evaluations of the same legacy population; original `cv_selection.json` is preserved.

At completion of training, production SERVICE/APP `c_p.onnx` still hashed to
`0be56f7a3442745e8a31c82601f84d04c80eb6a495d48729f729768e504c8200` and had not been deployed.
The user subsequently authorized deployment: see [CLASSIFIER_DEPLOY_LOCAL_DEBUG_20261005.md](CLASSIFIER_DEPLOY_LOCAL_DEBUG_20261005.md).
