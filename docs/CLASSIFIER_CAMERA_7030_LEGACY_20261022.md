# Training revision 10: both cameras, 70/30 and NG-heavy legacy

## Complete

Training stopped at epoch 210 with best at 110 after 100 epochs without strict
improvement. Evaluation, export parity, full-validation parity and replay are
complete. No new training or deployment is running; the heartbeat is paused.

| Fixed validation | Count | Baseline errors | Best errors | Baseline NG misses | Best NG misses |
| --- | ---: | ---: | ---: | ---: | ---: |
| HA | 570 | 0 | 24 | 0 | 22 |
| ANH_18_10 | 854 | 27 | 17 | 0 | 4 |
| Legacy | 181 | 1 | 10 | 1 | 5 |
| Total | 1,605 | 28 | 51 | 1 | 31 |

New-data false NG improved 14 -> 3 and no_disk-as-OK 11 -> 8, but increased
missed NG failed the predeclared recommendation gates. Installed baseline is
retained. HA Cam1 has 4 errors / 4 NG misses and HA Cam2 has 20 / 18. Training
has 1 error / 3,778 examples and is fitted-data diagnostics, not validation.

PT/ONNX has zero argmax mismatches across all 1,605 validation crops; maximum
probability discrepancy 9.48e-6. Batch 1/2/32 checks also passed. All 28 White
frames replayed with hardware off, unchanged geometry/counts and both-camera
production/debug parity. Protected runtime hashes and frozen input/code hashes
were verified unchanged. Canonical revision remains 10. Labels are still partially
reviewed, and the split shares frame context; this is not independent qualification.

Evidence: run `RESULTS.md`, `completion_verification.json`,
`artifacts/full_validation_export_parity.json`, `white_replay.json`.
Reports: <http://127.0.0.1:8794/comparison.html>,
<http://127.0.0.1:8794/error_review.html>.

User authorization: "rồi giờ train và đánh giá cho tao". Receipt is bound to
canonical revision 10, acknowledging remaining partial label-consistency review.
No deployment is authorized.

Run: `.work/classifier_camera_7030_legacy_20261022/`.
Dashboard: <http://127.0.0.1:8794/>.
Entrypoint: `SERVICE/training/train_classifier_canonical_holdout.py` with policy
`camera_target_slot_legacy_ng_v1`. Old revision-9 and robust campaigns remain
completed, with historical inputs/metrics unchanged.

The exact canonical manifest, hash, 5,383 crop copies and code are frozen in the
run. Dataset preparation does not recreate or randomize splits: 3,778 training
and 1,605 validation crops remain in their assigned groups. Legacy is 600 crops
(420 NG, 90 no_disk, 90 OK), with 419/181 in train/val. All HA/new cameras/classes
appear on both sides. Legacy camera metadata remains unknown.

Official YOLO11n-cls ImageNet initialization, SHA-256
`c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7`.
All parameters trainable; 234 shared tensors verified identical on startup.
AdamW lr 3e-4, cosine schedule, batch 64, dropout 0.03, seed 1022,
max 300 epochs, strict-improvement patience 100. Resume restores the saved
stopping clock and requires checkpoint/state epoch agreement. Use best, not last.

Sampling: HA 40%, new 40%, legacy 20%; HA/new cameras each receive half their
source budget. HA/new classes NG/no_disk/OK 30/30/40%; legacy 70/15/15% to
prioritize NG. Verified total training draw probabilities are 38/27/35% by class.
No validation row enters the sampler. Runtime preprocessing is RGB/OpenCV
stretch 224, float32 /255; target-preserving augmentation only.

Best fitness is validation macro-F1 weighted HA 40%, new 40%, legacy 20%.
Validation only selects checkpoints; fitted training rows do not enter fitness.
Recommendations require strict weighted macro-F1 improvement with no increase
in NG misses, no_disk-as-OK, false NG or total errors for each source and each
HA/new camera. Fixed gates are declared in protocol before training. Installed
baseline is kept regardless of results until a separate deployment instruction.

Evaluation includes frozen baseline and best PT/ONNX predictions by train/val,
source and camera; confusion matrices; error gallery; parity; 28 White source
replay frames with hardware disabled; protected runtime hashes. The former
missing-evaluation-directory error is fixed and regression tested, including
the entire three-source/four-camera report path. 133 tests passed before launch;
additional reporting and pointer regressions passed afterward without changing
frozen code.

Limitations: target-slot histories and decoded/flip/180 aliases stay together,
but all 48 frames share context across splits. This is spatial same-session
development validation, not independent frame/tray/temporal testing. The dataset
and split changed together, so comparisons with revision 9 do not isolate the
causal benefit of legacy. Old HA model has prior exposure to HA. Labels are the
authorized snapshot, not guaranteed factory truth; consistency review is partial.

Operational state is in `.work/ACTIVE_CLASSIFIER_TRAINING.json` and `status.json`.
The same-chat heartbeat is quiet during normal progress and reports completion
or actionable failure. Inspect parent/child processes before resuming; do not
restart if STOP_REQUESTED exists. Resume only the new run, never old campaigns.
