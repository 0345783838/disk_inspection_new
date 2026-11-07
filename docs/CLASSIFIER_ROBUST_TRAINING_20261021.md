# Autonomous training after partial consistency review

The user explicitly authorized training/evaluation on 21 October despite not
having time to finish the similarity audit. Do not describe this as a completed
manual review. Authorization is recorded in the canonical manifest and bound
to revision 8, which includes eight newly corrected records.

Active run: `.work/classifier_robust_20261021/`.
Dashboard: <http://127.0.0.1:8792/>.
Operational pointer: `.work/ACTIVE_CLASSIFIER_TRAINING.json`.
Heartbeat: `theo-d-i-train-classifier-qua-m`, every 30 minutes, quiet during
normal progress, to verify completion or repair failures without duplicate jobs.

The canonical dataset remains `DATASET_CLASSIFIER/`. The run contains immutable
input copies and an exact source manifest/hash as evidence, not another editable
dataset. Its 8,569 records comprise 5,799 training-pool crops and 2,770 legacy
validation crops. Later canonical edits do not affect this already frozen run.

## Predeclared methods

Each method runs three grouped new-capture development folds and a final fit,
up to 300 epochs with patience 100. Best checkpoints are used, not last epochs.

1. HA initialization, all trainable weights, frozen BatchNorm running statistics,
   learning rate 1e-5, distillation strength 0.5 on old examples whose teacher
   agrees with the label at confidence >=0.6.
2. Official YOLO11n-cls ImageNet initialization, all weights trainable, AdamW
   learning rate 3e-4, balanced domain/class sampling.
3. HA initialization, classification head only, frozen BatchNorm statistics,
   learning rate 1e-4, balanced replay.
4. Official ImageNet initialization, learning rate 1e-4, 80% CE + 20% generalized
   cross entropy (q=0.7), old-domain distillation strength 0.3. Unresolved close
   conflicting examples receive sampling weight 0.6 inside their class/domain;
   labels are unchanged and they remain in validation metrics.

All methods sample domains new/HA/legacy-train at 35%/30%/35%, and classes
NG/no_disk/OK at 30%/30%/40% within each domain. Thus legacy OK receives 14% of
training draws, versus 1% in the previous fresh-pretrained run. The hypothesis
is that the former recipe underexposed legacy OK; this experiment tests that
alongside initialization, stability and noise handling, not as a proven sole
cause. Explicit new corrections receive a modest 1.5 boost within fixed budgets.
Runtime RGB/stretch 224 preprocessing and target-preserving augmentation remain.

Warm-start methods compare predeclared 0.25/0.50/0.75/1.00 weight interpolation
fractions with the same HA parent in grouped development CV. A single fraction
per method is locked before final fitting. This produces single-network ONNX
artifacts, with no runtime threshold changes or prediction ensembles.

## Selection and limitations

Checkpoint fitness uses held-out new groups and legacy validation only, with
continuous safety/accuracy penalties and a feasibility bonus. Fitted HA/new
examples do not enter checkpoint selection. Train/validation exact hash overlap
is rejected; new capture groups are separated. Raw labels and exclusions are not
rewritten to improve results.

Final recommendation requires no increase in true-NG misses or no_disk-as-OK in
new/HA/legacy; no increase in false NG in new or legacy; bounded HA/legacy error
retention; and improvement in pooled grouped new-development metrics. The exact
gates are in `classifier_robust_core.retention_failures`. Method ranking uses
development OOF metrics, not final fitted-data accuracy. If no method passes,
the baseline is retained and reported honestly. Gates are never relaxed after
seeing outcomes.

Grouped CV here is development validation, not blind independent testing: the
same collection has already been viewed, revised and used for method development.
Legacy validation is checkpoint-selection data. HA and final new/legacy-train
metrics measure fitting/retention. Some labels remain unresolved; all metrics
measure agreement with the authorized snapshot, not guaranteed factory truth.

The campaign automatically generates the final comparison, error images,
portable PT/ONNX artifacts, export parity, and source White replay on 28 images
with hardware disabled. Integration replay does not establish whole-tray
accuracy without tray-level ground truth. Installed models and packaged backend
are protected by hashes; no deployment is authorized by this run.

## Operation

Python runtime: `C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe`.
The entry point is `SERVICE/training/train_classifier_robust.py`.
On a genuine failure, first inspect parent/child processes and logs. Resume only
after confirming no training process remains:

```powershell
python -u -B SERVICE/training/train_classifier_robust.py --phase campaign --root .work/classifier_robust_20261021 --resume
```

A `STOP_REQUESTED` file in the run stops the active child and prohibits restart.
Never restart a user-stopped run automatically. Record any repair and preserve
the original run inputs and predeclared selection gates. A normal resume skips
completed artifacts and resumes an unfinished training job from `last.pt`.

At startup, all 112 service tests passed, initialization matched 236 parent
tensors, legacy OK sampling was verified at 14%, the GPU job passed its first
eight epochs, and installed hashes were unchanged. See `start_verification.json`.

## Windows coordination repair, 21 October

The first heartbeat found a Windows `PermissionError` during status publication.
The old monitor advanced after that error without reaping its GPU child, leaving
two trainers running concurrently. The parent and the two verified child trees
were stopped for a controlled repair after backing up readable optimizer-bearing
checkpoints. No user stop marker was created and no labels or selection gates
were changed.

Atomic writes now retry transient Windows sharing/access violations with bounded
backoff. Child cleanup runs in a `finally` block so status failures cannot orphan
a trainer before the next recipe. Error logs retain full tracebacks. Resume also
restores the early-stopping clock, which the installed Ultralytics version does
not restore itself; historical best epochs were recovered from validation CSV
fitness, and strict stopper state is now persisted going forward.

All 116 service tests passed, including reader-lock retry, child cleanup and
resume-patience regressions. A real Windows held-reader test reproduced error
code 5 three times and succeeded through retries. Both current checkpoints kept
their optimizer state. The HA job resumed at epoch 148 with best epoch 126,
patience 100 and maximum 300; the ImageNet job's saved continuation starts at
epoch 117 with best epoch 91. After restart only one GPU trainer was present,
with the HA CSV advancing through epoch 153. All 8,569 frozen crop hashes and
installed runtime hashes were rechecked.

Evidence and post-repair source snapshots are retained under
`.work/classifier_robust_20261021/repairs/20261021_windows_status_lock/`.
The original input manifest, protocol and original code snapshot remain intact.

## Completed result, 22 October (Asia/Saigon)

All 16 fits, best-checkpoint exports, final evaluations and White replay finished
at 01:58 local time. No method passed all predeclared gates, so the selected
artifact is the retained installed HA baseline. `selected/best.pt` and
`selected/best.onnx` are copies of that baseline, not a newly approved model.

| Model | New-development CV false NG | New CV NG misses | New CV errors | Final legacy validation errors |
| --- | ---: | ---: | ---: | ---: |
| HA baseline | 38 | 4 | 91 | 19 |
| HA retention | 31 | 5 | 64 | 22 |
| ImageNet balanced | 11 | 8 | 48 | 210 |
| HA head | 58 | 3 | 107 | 21 |
| ImageNet robust | 21 | 7 | 74 | 264 |

New CV contains 2,799 crops, legacy validation 2,770. The baseline comparison
uses the same revision-8 labels. Eight recent corrections change some historical
counts; do not overwrite earlier reports with these new truths. Both fresh
ImageNet final fits match all training crops, while legacy validation degrades
substantially. This does not establish an independent factory accuracy, and
remaining label inconsistencies are not ruled out.

Completed epochs for CV0/CV1/CV2/final:

- HA retention: 226 / 141 / 101 / 101.
- ImageNet balanced: 253 / 142 / 174 / 263.
- HA head: 236 / 103 / 101 / 102.
- ImageNet robust: 166 / 191 / 175 / 295.

All 16 early-stopping clocks were checked against strict best-improvement epochs;
all stopped after 100 epochs without improvement. Saved best checkpoints can
occur later at equal fitness, so their epoch need not equal the first strict
best-improvement epoch.

Final verification checked 137,104 derived records and all 8,569 frozen input
hashes, zero exact train/validation hash overlap, label/role/class-directory
consistency, and recomputed all four models' OOF/final metrics from snapshot
labels. PT/ONNX parity for each final experimental model passed batches 1/2/32,
zero argmax differences, maximum probability errors below 6e-6. The retained HA
artifact also passed export parity. All installed runtime hashes are unchanged.
White replay covered 28 images, hardware off, with zero verdict changes relative
to the geometry-corrected baseline and debug/production agreement on both cameras.

Reports:

- <http://127.0.0.1:8792/comparison.html>: conclusion, all cohorts and trial artifacts.
- <http://127.0.0.1:8792/error_review.html>: 1,026 error/correction comparisons
  across the four methods' new CV and final legacy validation, with source crops,
  context where available, class scores and links to the one canonical editor.
- `.work/classifier_robust_20261021/completion_verification_20261022.json`:
  detailed integrity, metric, stopping, parity and report verification.

The original generated comparison and baseline-only error report are retained as
`comparison.generated.html` and `baseline_error_review.html`; their release
decision and metrics were not rewritten. The improved completion pages distinguish
the retained baseline from failed experimental artifacts. No further campaign
is queued; the overnight monitoring heartbeat is paused after final verification.
