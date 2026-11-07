# Native dense-mask experiments — 24 October 2026

> Subsequent user-authorized activation: native_union_hard_best is now the primary SERVICE/APP model. See [deployment and rollback](SEGMENTOR_MAIN_20261024.md). The no-activation statements below describe the completed experiment before this later deployment.

COMPLETE. Evaluation candidate **native_union_blend75**. Recommendation gates **PASSED**. No runtime activation.

The user's prior trained version was first saved into SERVICE/config/models as **s_p_edge_white_maskhead_20261024_092708.pt/.onnx**, exact hashes 57cdfb24ba31faa5b8eadd983541e7ddeaaaad890a201ddfd2616a9c80702531 / 6fedd421b0efdfc20b2bf1f8012a203c50e2bc87edaad7ef2b1d2c4aaac1be58. Metadata records its two-parent lineage. Installed APP/SERVICE configuration still selects the original baseline; none of these saves activates a model.

## Methods and preserved truth

Two native-grid fits from that saved parent compare full reviewed-union loss with/without train-only hard sampling. Proto/cv4 mask branches only; all other weights and BN statistics fixed. Dense loss: calibrated ROI pixel BCE + .1 Dice on binary 512px union, after bilinear logit upsample; max probability union matches native thresholded OR. Original instance loss remains; historical masks do not receive new union loss. Plain 2,364 samples/epoch; hard 2,592 references/epoch, adding 4 copies of each of the top57/228 original train crops ranked solely by archived training pixel error. No validation mining, label editing, mask geometry changes or split changes.

Same 261 validation crops: 96 new, 48 HA, 117 demo. Original new split stays 19/8 frames by capture group, both cameras. All 8 new backgrounds remain train-only. Checkpoint fitness now prioritizes mean new-camera strict pixel F1 (80%); old HA/demo/background retention contributes the remainder. All 9 previous installed-parent gates remain, plus the frozen archived-parent gates in protocol.json. The five inspected train files never enter selection fitness. Validation is same-session development data repeatedly reused, not independent acceptance testing; old labels retain partial review limitations.

Fits: [{"name": "native_union", "epochs": 11, "best_epoch": 1, "best_selection_fitness": 0.9541441900643478}, {"name": "native_union_hard", "epochs": 19, "best_epoch": 9, "best_selection_fitness": 0.9523974936994283}].

## Strict pixel F1

| Validation | Installed | Saved parent | New candidate |
|---|---:|---:|---:|
| new_Cam1 | 91.39% | 93.96% | 95.58% |
| new_Cam2 | 91.67% | 94.02% | 95.79% |
| new | 91.53% | 93.99% | 95.68% |
| old_ha | 84.65% | 85.00% | 84.61% |
| old_demo | 86.84% | 87.01% | 86.95% |


New uncovered reviewed regions: installed 15 → archived 13 → candidate 13. New full-region coverage: 538 → 538 /558. Validation mask F1 decreases on 0/96 new crops versus archived; inspect validation_masks.html. HA TP/FP/FN: 213/2/3 → 213/2/3. Demo: 482/36/12 → 481/37/13.

## Five inspected files — TRAIN diagnostics only

Each count cell is [band1, band2] using the exact Mid1/Mid2 caliper path. Labels remain the user-confirmed masks, not predictions. Original labels and runtime-cleaned labels give the same listed counts. These are mask-derived caliper counts, not independent physical disk count labels.

| File | Label mask caliper | Saved parent | New candidate |
|---|---:|---:|---:|
| Image_20261018112121604.bmp | [6, 4] | [6, 7] | [6, 6] |
| Image_20261018112241885.bmp | [6, 4] | [6, 7] | [6, 5] |
| Image_20261018112249401.bmp | [6, 4] | [6, 7] | [6, 5] |
| Image_20261018104018106.bmp | [23, 22] | [22, 21] | [22, 21] |
| Image_20261018112236010.bmp | [7, 7] | [7, 8] | [7, 7] |


Deduplicating the two identical Cam1 files, summed absolute caliper count error across 4 unique frames and 2 bands: 9 → 5. This is train diagnostic evidence, not generalization. See five_cases.html for raw / label / saved parent / new model.

## All experiments, technical stops and verification

The earlier balanced soft-prototype-grid exploratory method in training_dense_20261024 was stopped at epoch12. An independent 5px stripe fixture showed that objective favored an 8px prediction. That method and its best checkpoint/logs are preserved, not recommended; its planned second fit did not start. No historical metrics were overwritten.

The native fit initially failed on its first batch before any optimizer step because probability BCE rejects CUDA autocast. An exact mathematical rewrite to float32 logs was tested (including CUDA AMP), archived before/after code/protocol/logs, and documented in repairs/amp_bce_20261024/repair_receipt.json. Same labels/splits/gates/budgets. No checkpoint was lost or silently reset.

All 489 crops completed PT/ONNX raw parity; exact mask counts and minimum IoU are in parity.json. All 28 hardware-off White replays retained geometry/classifier and both-camera production/debug parity. Count changes versus archived are in replay/comparison.json. Frozen code/data/masks, installed runtime/config and classifier revision10 manifest/CSV hashes verified unchanged. Completion evidence is in completion_verification.json and delivery_verification.json.

## Artifacts

- release/: candidate PT/ONNX plus provenance. Recommendation passed.
- Service archive of improved candidate: s_p_edge_white_native_20261024_102721.
- five_cases.html, validation_masks.html, comparison.html: visual review.
- candidate_summary.json: every formal candidate and all gate results.
- protocol.json, authorization.json, hard_sampling.json, preflight_verification.json: frozen experiment and checks.

Dashboard http://127.0.0.1:8802/ . Completed; do not resume/repeat.


## All-candidate five-case diagnostics

| Candidate | Validation pixel F1 | Train-case pixel F1 | Train caliper absolute error | Validation gates |
|---|---:|---:|---:|---|
| native_union_best | 95.54% | 92.39% | 4 | True |
| native_union_blend25 | 94.71% | 89.97% | 7 | True |
| native_union_blend50 | 95.36% | 91.35% | 7 | True |
| native_union_blend75 | 95.68% | 92.61% | 5 | True |
| native_union_hard_best | 95.34% | 95.20% | 2 | True |
| native_union_hard_blend25 | 94.52% | 90.80% | 7 | True |
| native_union_hard_blend50 | 95.06% | 93.17% | 5 | True |
| native_union_hard_blend75 | 95.38% | 94.83% | 2 | True |

These 4 unique frames are training data; no candidate was selected using this table. Caliper reference comes from reviewed masks, not independent physical disk count labels. Full details: five_case_candidate_summary.json.


## Hard-example alternative — independently verified

The protocol-selected validation winner remains native_union_blend75, with new validation pixel F1 95.68%. The alternative **native_union_hard_best** passes the same 14 validation gates and has new validation pixel F1 95.34% (Cam1 95.24%, Cam2 95.44%). It is retained for the user's hard-case review; it does not replace the frozen validation selection.

On the 8 bands / 4 unique TRAIN frames inspected as 5 files, reviewed-mask pixel F1: archived parent 88.86% → validation winner 92.61% → hard alternative 95.20%. Summed absolute primary caliper count difference against masks: 9 → 5 → 2. This is fitted-data diagnostic improvement, not independent testing.

| File | Label [band1,band2] | Saved parent | Validation winner | Hard alternative |
|---|---:|---:|---:|---:|
| Image_20261018112121604.bmp | [6, 4] | [6, 7] | [6, 6] | [6, 4] |
| Image_20261018112241885.bmp | [6, 4] | [6, 7] | [6, 5] | [6, 4] |
| Image_20261018112249401.bmp | [6, 4] | [6, 7] | [6, 5] | [6, 4] |
| Image_20261018104018106.bmp | [23, 22] | [22, 21] | [22, 21] | [22, 22] |
| Image_20261018112236010.bmp | [7, 7] | [7, 8] | [7, 7] | [7, 6] |


The hard alternative fixes all three displayed Cam1 band2 counts and Cam2 10:40 band2. It still misses one in Cam2 10:40 band1 (22 versus23) and undershoots Cam2 11:22:36 band2 (6 versus7, compared with the parent's8). It is not a perfect mask solution.

Hard alternative passed its own full489-crop PT/ONNX parity and28-frame White replay, geometry/classifier retention, both-camera production/debug parity and runtime hash audit. Evidence: hard_focus_review/verification.json, parity.json and replay/candidate.json. Saved in SERVICE/config/models as **s_p_edge_white_native_hard_20261024_102450** with PT/ONNX/JSON. No activation. Visuals: hard_focus_review/index.html.

New fully empty training crops: parent2/8 FP; validation winner 2/8 FP (820 pixels), hard alternative 0/8 FP (0 pixels). These 8 crops were trained on and do not establish new-background generalization.

## Display-asset recovery

After the validation winner passed489-crop parity, the legacy error gallery expected archived-parent mask NPZ files at predictions/baseline. They had not been copied by preparation. Exactly261 parent snapshots were copied and hash-verified; the already-passed parity and all fits/candidates/metrics were retained. Only error gallery and remaining White replay were continued. No source/protocol/label/split/checkpoint/gate change. Full prior status and receipt: repairs/missing_parent_display_masks_20261024/. The source prepare script now copies these display assets for future runs; the experiment's frozen code remains preserved.
