# Classifier dataset authority

These rules apply to classifier labels and training. Detector and segmentation
datasets have separate scopes and must not be merged into this classifier pool.

- The **only active classifier dataset** is `DATASET_CLASSIFIER/`, registered by
  `.work/ACTIVE_CLASSIFIER_DATASET.json`. Read the registry and current manifest
  at the start of work; do not infer the active dataset from folder dates.
- The canonical manifest's current records and actual class folders are the
  label authority. Edit through `SERVICE/training/classifier_dataset.py`
  (`CanonicalDataset.save`) or the editor at `http://127.0.0.1:8791/`.
  An edit must update the manifest, class folders, catalog and CSV together.
  Do not create a separate decisions overlay or another active prepared pool.
- Old `.work/` decisions, prepared pools, training datasets, code snapshots and
  reports are historical evidence. Never apply them back over newer canonical
  labels. The October 21 reconciliation found 2,224 latest included decisions
  matching the source pool and one correctly excluded image; details are in
  `docs/CLASSIFIER_DATASET_CANONICAL.md`.
- New images should be incorporated into this canonical dataset with stable IDs,
  hashes and provenance, preserving current corrections. Do not rebuild current
  labels from an older model or decisions snapshot.
- On 21 October the user explicitly authorized autonomous training/evaluation
  despite not finishing consistency review, with inspection planned tomorrow.
  Use the current canonical labels, including eight saved corrections at that
  authorization. Record incomplete review honestly; never mark it fully reviewed.
  `partial_review_training_authorized` requires an explicit receipt bound to the
  current canonical revision. A new label edit revokes that live authorization;
  an already frozen run may finish against its documented input snapshot.
- After authorization, training must call `active_dataset` and capture the
  exact canonical manifest and hash in its run before deriving training splits.
  Derived run datasets are immutable records of that run, not new label sources.
  Recheck split leakage and dataset integrity after changes. Old evaluation
  truth/metrics stay tied to their original run, not rewritten as current truth.
- Do not start the retired standalone classifier decision editors. Port 8784
  and the previous dataset port 8790 redirect to 8791. Use the canonical server:
  `python -u -B SERVICE/training/serve_classifier_dataset.py --port 8791 --legacy-ports 8784,8790`.

See `docs/CLASSIFIER_DATASET_CANONICAL.md` for reconciliation evidence and layout.

## Current: camera-balanced 70/30 with NG-heavy legacy, revision 10

The latest user instruction supersedes the no-legacy and whole-camera holdout
policies below: add a little legacy, prioritize NG in both train and val, balance
both cameras in both splits, and use **70/30**, not 80/20. Canonical revision 10
preserves all 4,783 existing labels/hashes and adds 600 unique legacy crops from
the actual class folders in the same selected final_all_new source only:
420 NG, 90 no_disk, 90 OK. Never restore the 16 user-deleted unsuitable crops.

There are 5,383 active crops: 3,778 train / 1,605 val (70.18% / 29.82% with intact
groups). HA Cam1: 704/290; HA Cam2: 717/280; new Cam1: 1,030/464; new Cam2:
908/390; legacy: 419/181. Both splits contain all three classes for each
HA/new camera and legacy. Legacy camera/frame metadata is unknown; do not invent it.

Policy `camera_target_slot_legacy_ng_v1` groups all repeated views of each
origin/camera/region/slot and connects decoded/flip/180 aliases. These groups
never cross splits. Frames and sessions ARE shared as context, deliberately:
this is spatial same-session validation, not independent frame/tray/temporal
testing. The old 120-second whole-burst grouping no longer defines this split.
The audit explicitly checks target provenance, group/hash isolation and camera
class coverage while reporting shared context frames honestly.

Evidence: `.work/classifier_rebalance_7030_20261022/`. The editor at 8791 has
source, camera and train/val filters. Previous revisions are inactive recovery
history, never label sources. Current labels/CSV/catalog/class folders agree.
No new training was requested by this repartition; revision-9 training is complete
and its authorization/metrics do not transfer. New revision has no live training
authorization, and review remains partial. Existing trainers reject incompatible
split policies; future training must honor the current assignments and legacy.

The user subsequently explicitly authorized revision-10 training: "rồi giờ train
và đánh giá cho tao". The live run is
`.work/classifier_camera_7030_legacy_20261022/`, dashboard 8794, using
`SERVICE/training/train_classifier_canonical_holdout.py --phase campaign --root
.work/classifier_camera_7030_legacy_20261022 --policy camera_target_slot_legacy_ng_v1`.
Its receipt is bound to revision 10 with partial review acknowledged. One fresh
official ImageNet YOLO11n-cls fit, max 300 epochs/patience 100. Exact canonical
split/labels are frozen; never rebuild splits or restore excluded crops. Sampling
is HA/new/legacy 40/40/20%, both cameras equally within HA/new. HA/new classes
NG/no_disk/OK are 30/30/40%, legacy classes 70/15/15%. Best fitness is validation
macro-F1 weighted HA/new/legacy 40/40/20%. Report per source and per HA/new camera;
training rows never enter fitness. No deployment. Old runs remain complete.
The same-chat heartbeat monitors only this new run; check real parent/child state
and STOP_REQUESTED before any resume. The pipeline writes comparison/error review,
PT/ONNX parity, hardware-off 28-White replay and protected runtime checks.

Revision-10 training/evaluation is now COMPLETE: 210 epochs, best 110, patience
100. No restart/resume. On the fixed validation, baseline -> best errors are
HA 0 -> 24, new 27 -> 17, legacy 1 -> 10; total 28 -> 51. NG misses rise 1 -> 31,
so recommendation gates failed and the installed baseline is retained. HA Cam1
has 4 errors / 4 NG misses; HA Cam2 has 20 / 18. PT/ONNX argmax matches on all
1,605 validation crops (max probability error 9.48e-6); 28 hardware-off White
frames completed with geometry/counts unchanged, both-camera debug parity and
protected runtime hashes unchanged. The heartbeat is paused. See `RESULTS.md`
and `completion_verification.json` in the run. Canonical remains revision 10;
training completion does not confirm remaining label consistency.

The user next explicitly authorized "thử finetune đi". Current live run is
`.work/classifier_finetune_ha_7030_20261022/`, dashboard 8795. Its revision-10
receipt freezes exactly the same 5,383 rows and split as the completed ImageNet
run. Initialize from installed HA PT (verified against runtime ONNX), preserve
the three-class head, train all weights with lr=1e-5 and frozen BatchNorm running
statistics. Max 300 epochs/patience 100; sampling and best-selection fitness are
unchanged. Compare best against both installed HA and previous ImageNet best.
HA validation is retention testing because the parent previously saw HA, not
independent generalization. No label/split/gate changes and no deployment.
The same-chat heartbeat now monitors only this fine-tune. Check status and real
parent/child processes before any recovery, and never resume a completed old run.

This HA fine-tune is now COMPLETE: 211 epochs, best 111, patience 100. Do not
resume/repeat. On the same 1,605 validation crops, baseline/ImageNet/fine-tune
errors are 28/51/15; false NG 14/6/2; missed NG 1/31/7; no_disk-as-OK 11/8/0.
Fine-tune has 5 HA errors (all Cam2 NG misses), 9 new errors (1 NG miss) and
1 legacy error (1 NG miss). Recommendation gates failed on HA and new NG
retention. Installed HA retained; no deployment. Full-validation PT/ONNX
argmax parity passed (max probability error 7.51e-6), 28 hardware-off White
frames passed integration parity, canonical manifest/code/input/runtime hashes
verified unchanged. Completion audit passed; heartbeat paused. Read run
`RESULTS.md`, `completion_verification.json`, `val_best_errors.html` and
`docs/CLASSIFIER_FINETUNE_HA_7030_20261022.md` before proposing follow-up work.

## New autonomous NG-priority campaign, 22 October 2026

The latest user explicitly authorizes autonomous augmentation/AI generation,
training and optimization, prioritizing missed NG first and few false alarms,
with review tomorrow. New run: `.work/classifier_safety_campaign_20261022/`,
dashboard 8797, entrypoint `SERVICE/training/train_classifier_safety_campaign.py`.
Do not resume any completed earlier campaign. Read its protocol, status,
`docs/CLASSIFIER_SAFETY_CAMPAIGN_20261022.md` and active training pointer.

The current canonical revision-10 labels/splits remain unchanged. A new explicit
receipt is bound to that revision; review remains partial. Four HA-initialized
methods (final linear only, last two blocks, full weights, full weights with
AI-derived lighting), then seed repeat for the top two by predeclared safety
fitness. Six full fits max, 300 epochs/patience100; all BN running stats frozen.
Teacher agreement KD, NG-weighted supervised loss, train-only hard-example
sampling and geometry-preserving augmentation are tested. AI outputs provide
only bounded smooth illumination fields applied to original train pixels;
raw AI geometry/unconfirmed synthetic labels do not enter the classifier pool.

Checkpoint fitness prioritizes NG regression/misses before false NG. Existing
recommendation gates are unchanged. Fixed single-network HA weight interpolation
fractions are in the protocol. All candidates/failures must be reported, with
per-source/camera results. No deployment; preserve runtime hashes. Complete
PT/ONNX parity on all 1,605 val and 28 White hardware-off integration replay.
Validation is repeatedly used development data, not independent testing.

Heartbeat `theo-d-i-train-classifier-qua-m` now monitors only this run. Be quiet
during normal progress. On failure inspect real Python parent/child processes,
logs, immutable code/protocol hashes and STOP_REQUESTED. Resume only after no
parent/child fit remains and no stop request, using global Python and the new
entrypoint with `--phase campaign --root .work/classifier_safety_campaign_20261022
--resume`. Never spawn duplicate trainers. Frozen code/protocol changes require
an explicit documented, tested repair; labels/splits/gates cannot be changed to
improve metrics. On completion verify release, full parity, White replay,
completion audit and runtime hashes; report honest results and pause heartbeat.

## Completed NG-priority campaign, 22 October 2026

Completion update for the NG-priority campaign: `.work/classifier_safety_campaign_20261022/`
is COMPLETE on 22 October at 18:48 Asia/Saigon. All six fits and 29 fixed
interpolation candidates finished; exactly one passed unchanged gates:
`full_anchor_a025` (75% HA parent + 25% full-anchor best, a single network).
Its source fit ran 105 epochs, best 5. On 1,605 development validation crops,
baseline -> candidate: errors 28 -> 26, NG misses 1 -> 0, false NG 14 -> 12,
no_disk-as-OK 11 -> 11, OK-as-no_disk 2 -> 3. HA remains 0 errors; new
27 -> 26; legacy 1 -> 0. New Cam1 errors 19 -> 18 / false NG 10 -> 8;
new Cam2 errors 8 -> 8 / false NG 4 -> 4. Improvement is modest and this
is not independent validation or a production zero-miss guarantee.

All 1,605 PT/ONNX argmax predictions match (max probability error 4.59e-6),
28 hardware-off White frames passed geometry/counts and both-camera debug
parity. Canonical integrity/current manifest and all input/code/protocol/runtime
hashes were verified unchanged. AI branch was evaluated but not selected.
No deployment; installed HA remains. Do not resume or repeat this completed
campaign. Read RESULTS.md, fit_summary.json, completion_verification.json,
integrity_after_monitor.json, comparison.html and val_best_errors.html. The
completed run's heartbeat is removed; leave report/editor servers available.

## Historical dataset selection, earlier on 22 October 2026

The user explicitly selected the actual class folders in
`.work/classifier_yolo_pretrained_reviewed_20261021/datasets/final_all_new`
as the replacement label source, confirmed the 16 absent crops were intentionally
deleted as unsuitable, and then requested removal of all legacy data. Adopt only
the present `old_ha` and `new_factory` crops from that source. Never restore the
deleted crops or reapply revision-8 labels over the newly selected folder labels.

Canonical revision 9 contains 4,783 crops (1,991 HA + 2,792 ANH_18_10), no legacy.
The authoritative ImageFolder layout is `train/{ng,no_disk,ok}` (3,386) and
`val/{ng,no_disk,ok}` (1,397). Validation has 997 HA + 400 new crops, no legacy.
Preserve manifest `split`, `role_directories`, `capture_group`, and `split_group`.
Keep whole capture bursts and decoded/flip/180 aliases on one side. A trainer must
honor these canonical assignments; retired legacy-validation recipes reject this
split policy. Historical runs and their metrics remain unchanged.

Evidence is `.work/classifier_source_adoption_20261022/`. The previous canonical
revision is inactive recovery history there, not another label source. No new
training was started by this adoption. Review remains partial and revision-8
training authorization does not authorize revision 9. The diagnostic predictions
remain fitted-data review cues and do not evaluate a new model on this split.

Later on 22 October the user explicitly requested: "thử train bản này xong đánh giá
thử xem sao". A new revision-9 receipt now authorizes one fresh ImageNet YOLO11n-cls
fit and evaluation on the exact canonical split, maximum 300 epochs/patience 100,
without deployment. Run: `.work/classifier_ha_new_holdout_20261022/`, dashboard
`http://127.0.0.1:8793/`, entrypoint
`SERVICE/training/train_classifier_canonical_holdout.py`. Read its protocol and
the active training pointer. Best uses mean validation macro-F1 across HA/new;
training rows never enter checkpoint fitness. Do not resume the old robust run.
The current run automatically evaluates best, compares baseline by train/val and
origin, checks PT/ONNX parity and performs hardware-off 28-White replay. Labels
remain unchanged, partial review remains acknowledged, and runtime is hash-protected.

This HA/new holdout run is now complete: 217 epochs, best at 117 (patience 100).
Evaluation initially failed because evaluations/ did not exist; the directory was
created, a regression-tested source fix added, and evaluation/replay resumed with
unchanged frozen code/checkpoints/data/gates. Validation errors: HA 0 -> 57
(51 NG misses), new 10 -> 4 (false NG 5 -> 0). Recommendation gates failed on HA.
Keep installed baseline; no deployment. PT/ONNX parity and 28-frame hardware-off
White replay passed; runtime hashes unchanged. Monitor is paused. Do not restart
training. See `RESULTS.md` and `completion_verification.json` in the run directory.

The authorized overnight campaign `.work/classifier_robust_20261021/` completed
on 22 October (Asia/Saigon): all 16 fits and evaluation/replay are complete.
None of the four methods passed all predeclared gates; the installed HA model
was retained. Do not resume or silently repeat this completed campaign. Read
`docs/CLASSIFIER_ROBUST_TRAINING_20261021.md` and the completion report before
proposing follow-up work. That run used revision 8 and consistency
review was still partial; training completion does not confirm those labels.

Earlier on 22 October the user requested full-dataset review and explicitly chose all
HA + new images for diagnostic error review, without changing train/validation
splits. `.work/ACTIVE_CLASSIFIER_EVALUATION.json` registers that evaluation scope
(`old_ha`, `new_factory` only). It is not a label source. Current labels always
come from the canonical dataset. These are fitted-data diagnostics, never claim
independent validation. Legacy remains in full review and original roles remain.
The editor defaults to `?view=all`; `?view=evaluation` shows HA/new diagnostics.

## Current YOLO11s-cls trial, 29 October 2026

The user explicitly requested increasing classifier size n to s and clarified
“không lọt NG với tốt nhất không được lỗi ảo nhiều”. One new fit/evaluation is
at `.work/classifier_size_s_20261029/`, dashboard 8798, entrypoint
`SERVICE/training/train_classifier_size_s.py`. See
`docs/CLASSIFIER_SIZE_S_20261029.md` and the run protocol/status before acting.
Canonical revision-10 labels/hashes/splits remain fixed; a new receipt records
partial review. Official ImageNet YOLO11s-cls initializes this different-sized
architecture; do not try to load HA-n weights directly into s. Max300/patience100,
NG-weighted CE 2/1/1 and NG-priority checkpoint fitness. This is not a size-only
ablation relative to older macro-F1-selected ImageNet-n. Compare all prior n
baselines honestly. Existing per-source/camera gates are unchanged; additionally
require 0 validation NG misses and false NG no higher than selected n-safety in
each source/camera. No deployment. Full validation parity, current-baseline
28-White hardware-off replay and current runtime/integrity hashes are required.
Do not restart completed campaigns. Before recovering this run inspect actual
parent/child processes and STOP_REQUESTED; never duplicate trainers. This trial
has one fit only, with no automatic repeated seeds or model search.

Completion update for the YOLO11s-cls trial: `.work/classifier_size_s_20261029/`
is COMPLETE on 29 October at 11:36 Asia/Saigon. Do not resume/repeat. One fit,
181 epochs, best 81, patience100. On the same 1,605 development val, s has
44 errors / 27 missed NG / 10 false NG / 1 no_disk-as-OK, versus selected
n-safety 26 / 0 / 12 / 11 and installed HA-n 28 / 1 / 14 / 11. Missed NG are
HA19 (Cam1 5, Cam2 14), new5 (Cam1 1, Cam2 4), legacy3. Gates failed; installed
HA-n retained, no deployment. Full-val PT/ONNX argmax matches (max6.17e-6),
28 current-baseline hardware-off White replay and both-camera debug parity pass;
canonical/frozen code/input/protocol/runtime hashes unchanged. CPU batch2 median
preprocess+inference n7.68ms / s13.75ms, not whole-tray latency. All trainer
processes exited. No heartbeat exists for this trial; leave dashboard8798 up.
Read run RESULTS/completion_verification and docs/CLASSIFIER_SIZE_S_20261029.md.

## Latest constraint and research, 29 October 2026

The user rejected direct DINO inference after learning its size; DINO training
was cancelled before any fit/receipt/new active training run. Only the official
checkpoint download and frozen forward smoke were done. Do not start that DINO
trial without a new instruction. The user accepts slowdown up to 2x for the
whole processing flow, not a 2x parameter budget or classifier-only timer.
Read `.work/classifier_lightweight_options_20261029/RESULTS.md` for a read-only
selective two-nano proposal: safety-n always, preserve every safety NG, invoke
fine-tune-n on safety non-NG in row1_bottom/row3_top only. Post-hoc development
val: 18 errors / 0 missed NG / 12 false NG / 0 empty-as-OK. Existing gate
comparison passes descriptively; this is not a predeclared old-run candidate.
28 paired current-source White frames: baseline median1.184s vs selective1.334s,
median paired ratio1.126x, max1.188x; no hardware calls, geometry/counts/debug
parity and runtime/canonical hashes passed. No deployment or new training.
The benchmark face-call counter is simulation only; production would need an
explicit region argument. Do not claim independent validation or zero-miss
production guarantees. Active training pointer remains completed s trial.

## Current frozen-parent spatial NG trial, 29 October 2026

The user requested saving the good fine-tune to SERVICE/config/models and then
trying the proposed spatial NG branch. Original fine-tune best111 is retained as
`c_p_finetune_ha_r10_best111_20261029.{pt,onnx,json,provenance.json}`; hashes match
its original release. This storage is not deployment; active c_p/config unchanged.
Current run `.work/classifier_spatial_ng_20261029/`, dashboard8800, entrypoint
`SERVICE/training/train_classifier_spatial_ng.py`. Read its protocol/status and
`docs/CLASSIFIER_SPATIAL_NG_20261029.md` before acting. Canonical revision10 exact
labels/hashes/splits frozen, partial review acknowledged with a new receipt;
seven diagnostic NG remain val. One fit max300/patience100. Parent fine-tune
backbone/head/BN frozen completely; only zero-initialized 5,120-weight spatial
NG correction branch learns. Train-only augmented feature cache and original-
train-only feature normalization. No DINO, extra fits, deeper fine-tune, label
edits or deployment. Full-val PT/ONNX/cache parity, current-baseline28-White
hardware-off replay, <=2x whole-software-flow latency and integrity/runtime
hashes required. Keep old gates plus zero NG miss and false NG <=fine-tune per
source/camera. Inspect actual parent/child processes and STOP_REQUESTED before
any recovery; never duplicate trainers or resume completed/user-stopped runs.

Completion update for spatial NG: `.work/classifier_spatial_ng_20261029/` is
COMPLETE on29October14:57Asia/Saigon. Do not resume/repeat. 100 learned epochs,
no improvement over epoch0. Selected epoch0 = unchanged fine-tune (15 errors /
7 NG misses /2 false NG); best learned epoch3 is16/8/2. This is a negative
result for the frozen-parent linear spatial correction, not an improved model.
All parent weights/head/BN unchanged; full1605 PT/ONNX/cache parity passed
(max8.23e-6), 28 hardware-off White and both-camera debug parity passed;
paired whole-White ratios all<=1.154x. Latency P95 varied under machine load;
no speed-improvement claim. Canonical/frozen/runtime/retained-model hashes pass.
All trainer processes exited, no heartbeat. Retained original fine-tune files
remain SERVICE/config/models/c_p_finetune_ha_r10_best111_20261029.{pt,onnx,json,
provenance.json}; active model/config/plugin unchanged. Read run RESULTS and
completion_verification before proposing follow-up. No automatic deeper fit.

## Same-slot AI image preview, 29 October 2026

The user requested synthetic two-disks-in-one-groove variations, including
reflection ghosts and overlap shadows. Preview `.work/classifier_ai_same_slot_pilot_20261029/`
contains ten built-in ImageGen reference edits, with four shadow/reflection
focused prompts. Gallery8801 compares original train crop/context and raw AI
output. This newer request authorizes this geometry preview, not a change to
the completed safety campaign's historical illumination-only protocol.
All three real reference crops are current revision10 train NG; no val crops
or the seven fine-tune validation misses were supplied as generation inputs.
Canonical manifest SHA256 remains763f4ba07723adfa5d89eb0ea41e17a62ec48b8641128258f59a7d71cf78eb5f.
All generated labels/counts/groove membership remain unconfirmed; no canonical
inclusion, label changes, training, deployment or new active prepared pool.
Fine-tune calls A03/E01 OK; safety calls A03 OK and E01 NG. These are screening
predictions, not confirmed NG misses or evidence of accuracy improvement.
Read pilot README, summary, provenance and full prompt set before follow-up.
Adoption must use canonical stable IDs/hashes/provenance and same train family;
do not mix generated descendants into val or alter existing labels/splits.

## Accepted AI additions and supplementary nano fit, 29 October 2026

The user subsequently approved gallery8801 with "ok cho vào train bổ sung đi".
This supersedes the preview-only restriction for these ten crops. Canonical
revision11 now has5393 rows:3788train/1605val, adding10NG train only through
CanonicalDataset.save(additions=...). All5383 previous records and exclusions
are identical; original1605val and seven diagnostic NG remain val. IDs start
ai_same_slot_20261029__; is_synthetic and full parent/hash/prompt/user acceptance
are explicit. Source camera/frame/context refer to the real train reference,
not a new camera capture. Keep all descendants with the original train group.
Evidence `.work/classifier_ai_same_slot_adoption_20261029/`; review stays partial.

One new fit authorized: `.work/classifier_same_slot_ai_20261029/`, dashboard8802,
entrypoint SERVICE/training/train_classifier_same_slot_ai.py. Read its protocol,
status and docs/CLASSIFIER_SAME_SLOT_AI_20261029.md. Start from saved fine-tune
best111 YOLO11n-cls, preserve head; full weights lr3e-6, BN statistics frozen,
max300/patience100, NG-weighted CE2/1/1. Existing40/40/20 source/camera/class
budgets retained; AI takes15% within three NG buckets,2.7% total sampler.
NG-first fitness unchanged; compare learned best to parent epoch0 and retain
parent on tie/regression, reporting both. Old gates plus zeroNG and falseNG
<=saved fine-tune by source/camera; <=2x whole software White. No deployment.
Revision11 receipt acknowledges partial review; exact canonical snapshot/hash,
full1605 PT/ONNX parity,28 hardware-off White, runtime/retained hashes required.
Do not resume old campaigns or create extra fits. Before recovery inspect real
parent/child processes and STOP_REQUESTED; never duplicate or resume complete.

Completion update for supplementary AI: `.work/classifier_same_slot_ai_20261029/`
is COMPLETE on30October after evaluation recovery. Fit132epochs; first strict
best32, saved equal-fitness checkpoint42. Initial evaluation hash guard failed;
full5393 snapshot/current hashes subsequently matched, exact guard could not
reproduce failure. Evaluation/replay/finalize only were resumed on30October;
no new fit, image edits, label/split/gate changes. Saved original fine-tune→AI:
errors15→15, missedNG7→3, falseNG2→8, empty-as-OK0→1. Four oldNG recovered,
no newNG missed. Six new falseNG all in new Cam1 row3_top. Ten AI train crops
all calledNG are fitted diagnostics. Gates FAILED; installed HA retained.
Full1605 PT/ONNX argmax parity max5.81e-6,28hardware-off White geometry/counts
and both-camera debug parity passed. Whole-software median ratio1.024x,
max1.282x (<2x); canonical/input/code/protocol/runtime/retained hashes pass.
No deployment, no heartbeat. Canonical remains revision11 and partial review.
Read ASSESSMENT_20261030.md, RESULTS.md, completion_verification.json and
case_change_analysis.json. Do not resume/repeat this completed run.

## Coverage repartition and fresh nano fit, 30 October 2026

User explicitly requested repartition train/val for distribution coverage and
train anew. This supersedes frozen-split restrictions only for this new run.
Canonical revision12:5393rows,3778train/1615val;2350records switch split through
CanonicalDataset.save(repartition=...). All labels/hashes/IDs/provenance and
exclusions preserved,600legacy and10accepted AI retained. Same target/alias
groups never cross; AI/root three target groups forcedtrain. Partial review.
Evidence `.work/classifier_coverage_split_20261030/`: before had2/51region-class
cells and21/131appearance clusters missing a side; after all51/51 and131/131
cover both. Design uses labels/source/camera/region + frozen official ImageNet
features/photometric clusters, never inspection predictions, errorIDs or scores.
MILP feasible after90s,30.88%optimality gap: not proven globally optimal.
Frames/sessions still shared context; legacy camera unknown. No independent
temporal/tray claim. Grouping policyid remainscamera_target_slot_legacy_ng_v1,
new partition_strategy region_visual_group_milp_v2. Read docs/CLASSIFIER_COVERAGE_20261030.md.

One new fit authorized: `.work/classifier_coverage_train_20261030/`, dashboard8803,
entrypoint SERVICE/training/train_classifier_coverage.py. Fresh official ImageNet
YOLO11n-cls1000→new3-class head; no inspection checkpoint reuse. All weights/BN
learn,lr3e-4,300max/patience100,NG CE2/1/1, runtime224 and safe augmentation.
Each epoch every train crop once +3778balanced extra draws (7556total): extras
source/camera/class/realregion/visual/target balanced; extraAI2.7%, effective
epoch AI~1.48%. Extra budgets differ from full mixture; report actual sampling.
Exposure is guaranteed, learning/zero misses is not. Sampler counts train only.
NG-first fitness and old gates kept, zeroNG/new-val falseNG<=retained fine-tune
on same new cohort, <=2x whole-software White. All old model comparisons may
have prior exposure; old1605val partly train now, diagnostic only/no fitness.
No deployment. Exact revision12 receipt/snapshot, full1615 PT/ONNX parity,
28hardware-off White and canonical/input/code/protocol/runtime hashes required.
Before recovery inspect actual processes/STOP_REQUESTED, no duplicates or extra
fits; never resume completed runs. Parent must publish final complete status.

Completion update: `.work/classifier_coverage_train_20261030/` is COMPLETE.
260epochs, first strict best160/saved equal-fitness167. New1615val40errors/
17missedNG/14falseNG/4empty-as-OK; train3778 has0errors. HA9NG misses/new3/
legacy5; seven are HA Cam1 row2_bottom slot02 views. Gates FAILED, no deployment.
All1615 PT/ONNX argmax matches(max6.35e-6),28hardware-off White geometry/counts
and both-camera debug parity pass; paired ratio median0.979x/max1.149x, not a
speed-improvement claim. Canonical/input/code/protocol/runtime hashes pass.
Canonical remains revision12, partial review. Do not resume/repeat this run.

User next requested alternative saved checkpoint evaluation from this same fit.
Read `.work/classifier_coverage_checkpoint_review_20261030/ASSESSMENT.md`,
summary.json and review_verification.json; dashboard8804.13saved files reviewed
on same1615val, no training/label/split/gate change or official release override.
Best167 is also lowest saved-model errors40/NG17/highest weighted macroF1.
Epoch76 falseNG13 instead of14 costs27NGmisses/53errors; last26044/20/14.
All original gates fail. CSV114(38errors/22NG/9false),120(39/20/10) have no
saved weights and were not re-tested/reconstructed. AnchorPT matches original
ONNX1615argmax; all checkpoint hashes unchanged. This remains post-hoc
development selection, not independent generalization. No extra fits authorized.

## SERVICE cleanup and authorized field-test classifier, 30 October 2026

Latest user requested removing poor/extra trained models from SERVICE and one
best classifier for real-machine testing. This explicitly authorizes SERVICE
field-test activation despite earlier no-deployment instructions/gate failures.
SERVICE/config/models/c_p.onnx is NOW the fine-tune+AI checkpoint42 from
.work/classifier_same_slot_ai_20261029/artifacts/best.onnx, SHA256
89a7ef199e575cdb77a7bcc8edb0d1250affa41bb9f804cadbdc24b1fbccc434.
This follows NG-first preference: original1605val3NG/8false vs fine-tune7/2;
all5383real diagnostics3NG/8false/15errors. Source gates remain failed and
are not rewritten; temporary field test is not independent validation or a
production-zero-miss claim. Current SERVICE config/crop width1.8 unchanged.

Runtime folder now has3ONNX only: c_p.onnx, active d_p_ha_20260930.onnx and
active s_p_edge_white_native_hard_20261024_102450.onnx, plus MODEL_SELECTION.json
and __init__.py.16unused model/PT/metadata files removed after SHA256-verified
backup to .work/service_models_field_test_20261030/backup/models/. This includes
the old SERVICE retained fine-tune copies; original .work run artifacts remain.
APP plugin, SERVICE source/config, current detector/segmenter and canonical
revision12 unchanged.28hardware-off White replay using actual SERVICE config
passed geometry/counts/debug parity, ratio median1.0013x/max1.0876x(<2x).
Fresh SERVICE instance loads new c_p and predicts successfully. No live server
or PLC launched/restarted; restart SERVICE before real testing if already open.
Read docs/SERVICE_FIELD_TEST_MODEL_20261030.md and cleanup receipts. Old training
protocol/runtime hashes describe their original state; do not restore removed
files or old c_p merely to satisfy guards for completed campaigns. No new fits.

User correction immediately after cleanup: "ô phiên bản finetune từ HA tốt nhất
mày cũng xóa đi rồi à". Restore and KEEP the original good HA fine-tune best111
in SERVICE/config/models/c_p_finetune_ha_r10_best111_20261029.{onnx,pt,json,
provenance.json}. All four restored byte-for-byte from cleanup backup; ONNX
775872fc29f8e00ab9934747cfaa543ad966e2e905d5e53740717d4114b8363c,
PT51e002c68b4c64b64fd74d8f0987c6df987b4fc3eec7a779bb6464cc516be252.
This retained comparison model is an explicit exception to one-classifier-file
cleanup; do not remove it as poor/extra. Active c_p remains fine-tune+AI89a7ef...
and config unchanged; user did not ask to switch active model. MODEL_SELECTION
now records retained_alternates. Restoration receipt is in cleanup folder's
restore_ha_best111_20261030/restoration_verification.json. No training/PLC calls.

Read-only side-by-side missedNG/falseNG review requested next: dashboard8805,
.work/classifier_missed_vs_false_review_20261030/. Default current+AI3miss/8false,
switch best1117/2 or union15cases. Scope is original1605val comparison cohort,
not new1615val. Predictions fixed/model-hash bound; labels/assets read current
canonical live with crop-hash validation. Context/full images, zoom and display
brightness/contrast; no POST/label overlay/model changes/training.43assets
verified. Keep current canonical revision12 labels as authority; review is partial.

## Target-aware raw-context pilot, 30 October 2026

User explicitly authorized the proposed local-detail + neighboring-slot context
experiment: "vậy thử đi bro, trông cậy vào mày, tao ngu bất lực rồi". New run
`.work/classifier_context_20261030/`, dashboard8806, entrypoint
SERVICE/training/train_classifier_context.py. Read protocol, active pointer and
docs/CLASSIFIER_CONTEXT_20261030.md. Receipt bound to revision12; partial review.
No canonical labels/splits change, no deployment. Current SERVICE+AI and retained
HA best111 are protected and must remain available for the user's field test.

Frozen best111 local YOLO11n-cls (all weights/BN fixed) + raw five-slot context
with explicit query mask + small residual head. 122966 trainable extra parameters,
1657913 total. Exactly two matched fits: mask-only control and real-context,
seed1031,300max/patience100. No extra full fits/threshold sweep/interpolation.
All4783 real HA/new local crops reconstructed pixel-exact from raw with saved
geometry; historical labels never imported. Legacy600/AI10 have no raw context
and retain exact parent predictions via zero-mask fallback.

Capture-separated adaptation:2231 existing train rows fit;1117 same-held-capture
train rows quarantined. Primary480 existing val rows (HA Cam2 300/new180), held
bursts HA01/new01,05,07 chosen from labels/capture metadata only. New primaryNG
support is only3. Parent has historical exposure, so never call this blind
independent validation. Other canonical-val rows are secondary development.
NG-first selection and comparison to matched control; exact epoch0 parent is
eligible. Require full1615 PT/ONNX parity and actual-config28White hardware-off
replay, geometry/count/debug checks, whole-flow<=2x and protected hashes.

Initial cache root-resolution failure occurred before fits; tested path/preview
repair recorded in repairs/frozen_root_and_preview. No model/loss/gate/data
changes. Before any recovery inspect real parent/child, logs, STOP_REQUESTED and
hashes. Never duplicate fits or resume a complete trial. Report failures honestly.

Context pilot is now COMPLETE: both fits101epochs/best1. Primary480 NG/false:
parent2/0, SERVICE0/0, mask-only0/1, context1/0. Current1615val: parent3/0/4errors,
SERVICE1/2/3errors, control1/3/5errors, context2/1/4errors. Context fails NG-first
gates and is not selected/deployed. Historical1605 diagnostic: context4NG/5false
vs SERVICE3/8; never treat those old images as independent testing/fitness.
Both full1615 PT/ONNX parity and independent all5393 cached-vs-actual-ONNX
argmax checks pass. GPU TF32 probability rounding is documented, no argmax or
metric changes. White28 median1.005x/max1.060x (<2x), debug/geometry parity and
all input/context/raw/code/protocol/runtime hashes passed. Canonical revision12
labels/splits and partial review unchanged. SERVICE+AI and retained HA best111
unchanged. No resume or extra fits. Read run ASSESSMENT.md, RESULTS.md,
errors_context.html, additional_diagnostics.json and completion_verification.json.

## Current groove-entry ownership trial, 4 November 2026

The user explicitly authorized agent auxiliary labeling, training/evaluation and
human review afterward: "xong mày thử tự label theo ý của mày rồi train thử và
đánh giá giải pháp mới xem thế nào nhé, mai tao kiểm tra, có gì ko ổn tao sẽ sửa
sau", then resumed unfinished work on4November. New run
`.work/classifier_groove_ownership_20261104/`, dashboard8808, entrypoint
`SERVICE/training/train_classifier_groove_ownership.py`. Read protocol/status and
`docs/CLASSIFIER_GROOVE_OWNERSHIP_20261104.md` before acting. Never resume an old
completed campaign. A new revision12 receipt acknowledges partial classifier and
auxiliary review. All5,393canonical labels/IDs/hashes/splits/exclusions remain
identical;3,778train/1,615val, legacy600/AI10 retained. Only authorization headers
changed. Frozen manifest SHA256 is
`c815244b24deb76d7ed174e774d2f5668d28db3b17b7965db50c8edda83d7bce`.

The user clarified that the disk receiving groove is the concave interval
between adjacent protruding support peaks. Account for reflection-obscured
grooves; inferred geometry never proves disk occupancy.48real train references
have agent-drafted peak/entry/trace/ownership/visibility labels,80peak points,
31entry points,19complete occupancy drafts and29unknown counts. They are NOT
human-verified. Aux SHA256
`5832f0a3054eabc7a9a8585755f383598e0f4297fb5bd9f447efe287c209eef1`.
Estimated/unknown coordinates have no point loss; partial review is not complete.
Case03legacy was skipped only in discussion, not removed from canonical/metrics.
No val auxiliary labels train; case05 remains diagnostic. Geometry annotations
are a separate auxiliary task, not another classifier label pool/overlay.

Exactly two matched fits: control and auxiliary, same current+AI nano parent,
local RGB/stretch224 input, architecture, seed1104 and sampler; only auxiliary
loss weights differ. Last2backbone blocks+head train at3e-6; geometry branch
at3e-4; early backbone and all BN running stats frozen. Max300/patience100,
epoch0eligible, NG-first checkpoint rank and unchanged source/camera gates.
Each epoch3778original train+3778balanced extras+192aux-reference draws;
all7748 are train. Actual source/camera/class/synthetic exposure is reported.
This local pilot does not implement raw-context lattice recovery when an entire
groove is hidden. Validation is reused spatial development with parent exposure,
not independent generalization. No deployment or zero-miss production claim.

Full1615PT/ONNX argmax parity,28current-config White hardware-off replay,
both-camera debug/geometry/count parity,<=2x whole-software-flow latency and
frozen input/code/protocol/runtime/retained-model hashes are required. Current
SERVICE+AI and original HA best111 remain protected. The campaign parent must
publish final COMPLETE only after evaluation/replay/audit. Before any recovery
inspect actual parent/child processes, logs, hashes and STOP_REQUESTED; never
duplicate trainers or resume completed/user-stopped runs. Only unfinished work
may use global Python and this new entrypoint with --phase campaign --resume.
Frozen code repair must be explicit, documented and tested; no label/split/gate
changes to improve metrics. Leave report and auxiliary review servers available.
Same-chat heartbeat `theo-d-i-train-u-a-khe` monitors only this new run every
15minutes, quiet during normal progress. Verify final parent COMPLETE, all
required artifacts/hash/parity/replay checks, report honest results, then pause
that heartbeat. Inspect real parent/child/STOP before any recovery; never spawn
a duplicate or resume a completed/user-stopped run.

Recovery note04November: control stopped after committed epoch106 on a White input hash guard. Independent repeated reads and full5393crop/canonical/code/protocol/input/runtime/reference checks matched the original hashes; root cause was not reproduced and no expected hash/code/data/gate was changed. The same unfinished campaign resumed from106, completed control107epochs/best7 and advanced to auxiliary. Evidence: run recoveries/20261104_unreproduced_white_hash/. Separate bounded15000draw frozen augmentation diagnostic reproduced no warning/nonfinite output; original epoch67 NumPy warning remains unexplained. Never restart control or create extra fits.

## Groove-entry ownership completion, 4 November 2026

Run `.work/classifier_groove_ownership_20261104/` is COMPLETE at10:10Asia/Saigon. No resume/repeat or extra fits. Control107epochs/best7; auxiliary100epochs/selected0(parent), learned best1 tied parent rank. Current1615development val SERVICE/control/aux errors3/2/3, missedNG1/1/1, falseNG2/1/2, empty-asOK0/0/0. Both selectedmodels: HA Cam1/Cam2 and new Cam2 zeroerrors; new Cam1 row3_top slot11 has2falseNG SERVICE/aux,1control; legacy1NGmiss has unknown camera/slot. Best111 remains4errors/3NG/0false. Auxiliary did not improve classification; unchanged gates FAIL; no deployment. Do not present epoch0 fallback as a newly improved model.

Both selected checkpoints have full1615PT/ONNX argmax parity(max2.62e-6control,3.10e-6aux). White28actual-config geometry/count/debug bothcams pass, median1.043x/max1.119x; replay covers selectedaux epoch0 only, not learned1/control latency. Full5393frozen crop/canonical/code/protocol/input/runtime/retained hash checks and independent metric recomputation passed. Parent COMPLETE and no trainer remains. Canonical revision12labels/splits unchanged, reviewpartial;48agent geometry labels awaithumanreview. Reused val includes prior parent exposure; no independent/production zero-miss claim.

Read run ASSESSMENT.md, RESULTS.md, independent_completion_verification.json, completion_verification.json and error_review.html. Evaluation marker binds prereplay release: removing only whole_white_latency and Windows serialization reproduces original receipt SHA exactly; independent verification binds finalrelease too. Recovery was resume from106after unreproduced Whitehash guard, no data/code/gate repair; original NumPy warning unreproduced on15000readonly draws. Evidence retained in recoveries/. Completed heartbeat `theo-d-i-train-u-a-khe` was deleted after verified completion; leave dashboard8808 and auxiliaryreview available. Current SERVICE+AI and retained best111 remain unchanged.

Post-hoc geometry check04November authorized by user "ừ check lại xem nào": run geometry_diagnostic/ contains inference gallery48aux-train+39val and all1615val predictions. No newfit/label/split/selection/deploy. Auxepoch1 peakmedian31.61px/entry38.08px against unverified train drafts; last100learns146/146supervised points within5px, count19/19, but val2NG/1false (new HA Cam2row1_bottomslot19miss) vs parent1/2. Zero-residual ablation at100worsens NG2to3, showing oneNG rescue relative to adapted head but overall parent remains safer. No geometry truth onval; no generalization claim. Read geometry_diagnostic/ASSESSMENT.md, summary.json, residual_ablation.json and verification.json. Completed run/weights/gates remain unchanged.

## User paused classifier improvement,4November2026

Latest explicit instruction: "thôi được rồi tạm thời dừng cải tiến và cho tao bản tốt nhất hiện tại để đi hiện trường đánh giá thêm". Pause improvement/training/optimization until a new user request. No automatic newfit/recovery of completed runs. Current field-test classifier remains HA fine-tune+AI checkpoint42 at SERVICE/config/models/c_p.onnx SHA89a7ef199e575cdb77a7bcc8edb0d1250affa41bb9f804cadbdc24b1fbccc434. Retained original best111 kept unchanged. Portable classifier package `.work/field_test_best_classifier_20261104/classifier_field_test_20261104.zip` includes current ONNX, runtime parameters and reference config; not the full SERVICE application. Hash/ZIP/CPU ONNXbatch1/2 verified, runtime/config/canonical unchanged, no SERVICE/PLC started. Read package README_VI.md, RELEASE.json and folder verification.json. `.work/CLASSIFIER_IMPROVEMENT_PAUSE.json` records pause and field package. Current model already installed; restart SERVICE before field test if running.

## Latest user-selected field classifier: original HA best111,4November2026

User explicitly rejected checkpoint42 after viewing8falseNG and instructed "thế không được rồi, vẫn ưu tiển bản checkpoint best 111 đi". This supersedes the previous +AI field selection. Active SERVICE/config/models/c_p.onnx is now byte-identical to original HA fine-tune best111, SHA775872fc29f8e00ab9934747cfaa543ad966e2e905d5e53740717d4114b8363c. Preserve all four original c_p_finetune_ha_r10_best111_20261029.* files. Prior +AI c_p and selection metadata SHA-verified backup in .work/field_test_best111_20261104/backup/. No training, config, dataset, source, detector or segmenter changes. Improvement remains paused.

User-directed activation verified fresh SERVICE load and28actual-config hardware-off White paired replay vs+AI, geometry/counts/debug bothcams pass, whole-software ratios recorded in white_replay.json. No live SERVICE/PLC started; restart SERVICE before field testing. Best111old1605val15errors/7missedNG/2falseNG vs+AI15/3/8; current1615val4/3/0. Historical gates remain failed and prior exposure acknowledged. Package .work/field_test_best111_20261104/classifier_best111_field_20261104.zip; active pointer .work/ACTIVE_CLASSIFIER_FIELD_TEST.json; verification.json binds new runtime. The older checkpoint42field ZIP is superseded history. Completed campaigns keep their original protected-runtime hashes; do not restore checkpoint42 just to pass old guards. Do not resume/retrain.

## Full field application, 7 November 2026

User requested program-flow review before the factory visit and explicitly asked
for one complete application in `APP/DiskInspection/bin/Debug`. That folder now
contains the current WPF x64 build and newly compiled standalone SERVICE;
`plugin/config/models/c_p.onnx` matches the selected original HA best111 SHA
`775872fc29f8e00ab9934747cfaa543ad966e2e905d5e53740717d4114b8363c`.
All retained best111 files are preserved. SERVICE source/config/models and
canonical revision12 manifest SHA `c815244b24deb76d7ed174e774d2f5668d28db3b17b7965db50c8edda83d7bce`
remain unchanged; improvement/training remains paused.

Five APP source files received bounded flow repairs: owned modal Settings/Debug,
EXE-relative working directory, consistent startup waiting and retry of failed
camera opens. Package-only SavePath is `Images`; other defaults are unchanged.
Source runtime tests77, APP flow18, four real CameraManager paths with fake SDK
(old source fails), fake Modbus checks, actual packaged28White, eight concurrent
camera White/UV responses and nine actual APP merge cases passed. Packaged
source parity and all model/source/ZIP hashes pass. Isolated test servers exited;
no camera/PLC/light operations or production APP launch occurred. License was
validated on this machine only. Factory SN/exposure/IP/license/SDK and PLC bit
reset/stop behavior still require physical checks. Warning still sends PLC OK.

Read Debug `README_HIEN_TRUONG.md`, `RELEASE.json`, `RELEASE_VERIFICATION.json`
and `SHA256SUMS.txt`. Complete ZIP:
`.work/field_readiness_20261107/DiskInspection_factory_20261107.zip`;
evidence/completion and original Debug backup are in that run folder.
BroSight and earlier classifier-only ZIPs are not this complete application.
