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

## Excessive disk crossing by paired caliper spacing, 9 November 2026

User requested comparing the same adjacent disks on the two calipers in each
White segmentation band and authorized implementation. After reviewing the
21.5px examples at20px, latest user request sets **30px** in
`MAX_CALIPER_GAP_DELTA_PX`, both `SERVICE/config/config.env` and the installed
`APP/DiskInspection/bin/Debug/plugin/config/config.env`. Strictly greater is NG;
equality passes. Restart SERVICE after config edits. Compare calipers1/2 and3/4
within their respective bands; retain existing rectified centre-to-centre
spacing. Match disk identities through segmentation components, not nearest
raw X (a uniformly tilted row can shift more than half a pitch). Only intervals
adjacent on both scans are compared. Skipped intervals are reported as PARTIAL;
no valid interval produces measurement NG. Production/Debug share the gate,
`CaliperGapChecks` and red overlays; existing NG reasons/PLC error code remain.

Evidence `.work/caliper_gap_delta_20261109/`; docs
`docs/CALIPER_GAP_DELTA_20261109.md`.97runtime tests,28source White and28new
standalone White passed, both-camera Debug/concurrent parity and actual APP
transport/9merge cases passed. At30px no overall result changes on those28
stored frames (4OK/15NG/9Warning); a30.5px Cam2 interval is on an already-NG
frame.20px preview instead changed2Warning toNG at21.5px. Not factory accuracy
truth. Models including active best111, retained models and canonical manifest
are unchanged; no training or physical camera/PLC/light calls. Debug backend
was rebuilt/installed with backup `backup_plugin`; test backend exited. A
pre-existing invalid XML namespace in Debug `DiskInspection.exe.config` was
repaired without changing parsed XML/settings values; source App.config and
APP binary unchanged. Receipts bind the installed files.07November ZIP is
historical and does not include this gate; use current Debug folder.

## Factory segmentation supplement,17November2026

The user completed344/344band review in `.work/segmentation_factory_20261117/`
and explicitly authorized checking/training, unrestricted object overlap into
one union mask, and balanced old/new training with old-model comparison.
Current new run `.work/segmentation_factory_train_20261117/`, dashboard8811,
frozen entrypoint `code/train_segment_factory.py --phase campaign`; read
`docs/SEGMENTOR_FACTORY_TRAIN_20261117.md`, protocol/status and active segment
training pointer before acting. Review editor8810 stays available. No classifier
training, label changes or deployment; classifier canonical revision12 remains
separate and unchanged, current best111/detector/segmenter are protected.

All172new White frames/344bands/2064tiles retained, including correct/empty
data. Shared Canvas renderer applies62selected-instance erasures, then ORs all
object masks; overlap is allowed, fragment IDs are not physical disk counts.
Exact frozen NPZ union pixels are truth; txt paths are bbox proxies only.
Grouped70/30 has122frames/1464tiles train and50frames/600tiles val, each new
camera732train/300val; new empty50train/26val. Paired capture groups,120s,
decoded flip aliases and212near-appearance candidates are conservatively
connected. Groups are heuristic, not verified physical tray identities.

All2364old train/261old val retain labels/splits. Every combined3828train
record appears once each epoch plus900new train-only hard draws,450per camera,
giving2364old/2364new references(4728total). Val861=261old+600new. One fresh
native-hard fit only from current hard parent PT10bcc17c.../ONNX13394589...,
mask proto/cv4 only, all BN/non-mask state frozen, lr1e-4/max40/patience10,
batch8, seed1117. No extra fits or val hard mining. Same calibrated native512
union objective as previous successful hard fit plus retained mask/background
loss. Fit checkpoint fitness/gates and fixed25/50/75% blends predeclared.

Before recovery inspect actual Python parent/child, lock, active_child, logs,
frozen protocol/code/input hashes and STOP_REQUESTED; never duplicate trainers,
silently restart unfinished fits or resume a completed/user-stopped run.
Parent must finish all candidates, full2325crop PT/ONNX parity,200paired White
hardware-off replays(28old+172new),344full-band union comparisons, both-camera
debug/geometry/classifier checks, whole-software<=2x and protected hash audit
before publishing COMPLETE. No automatic monitoring automation was created.
Additional user requirements/CUDA preflight are bound by preflight_addendum.json.
Development validation and parent exposure on old val must be acknowledged.
No deployment; report all failures/candidates and retain installed parent if
unchanged gates fail. Leave editor/report servers available.

Technical loader repair17November: first attempt failed94/591batches into
epoch1, no committed epoch/checkpoint. Ultralytics deduplicated same-bbox
records despite differing valid overlapping masks. Restore original proxy
records in the exact-mask loader; all4728train/861val counts and affected-case
CUDA AMP forward/backward tests pass. No labels/mask pixels/splits/sampling/
loss/gates/parent/runtime changed. Failed attempt and original code/protocol
are retained in run repairs/duplicate_bbox_loader_20261117/. One valid fit is
retried from parent because no optimizer checkpoint exists; not an extra search
candidate. Current protocol SHA40f06549dac594f3849dcfac1524952cd0465a81e96ff403858b25fa1209dca8.

User next authorized autonomous completion/evaluation, inspection tomorrow.
Run finish_review.py --wait is a separate postprocessor, NOT a trainer: after
primary completion it runs historical5filename/4unique-frame/8band TRAIN
diagnostics on parent/all candidates and independently recomputes all861saved
baseline/selected union metrics plus hashes. It owns postprocess.lock and
writes postprocess_status.json/independent_completion_verification.json/
ASSESSMENT.md. Do not start a duplicate; inspect its actual process/status on
failure. It never changes fitness/gates or launches fits. Full user-requested
completion includes both primary and postprocessing audits, with no deployment.

## Completed factory segmentation supplement, 17 November 2026

`.work/segmentation_factory_train_20261117/` is COMPLETE:21 learned epochs,
best11, patience10. Do not resume/repeat. One completed fit plus the preserved
failed94-batch loader attempt. All2064 new tiles and2625 retained old records
were used across immutable train/val; val600new+261old. Training exposure50/50
old/new, new cameras balanced, all exact overlaps/62erasures preserved.

Installed parent -> learned best new-val union F1:90.62% ->95.25%
(Cam1 90.57->95.38; Cam2 90.68->95.11). New uncovered mask regions121->129;
October retained val F1 95.34->91.26. All four fixed candidates failed gates;
installed parent retained, no deployment. Historical5filenames/4unique frames/
8bands are TRAIN diagnostics:union F1 95.15->91.48%, mask-derived primary-
caliper absolute error2->3. Priority case Image_20261018112236010.bmp Cam2
band1 has mask/parent/learned caliper counts7/7/6; not physical count truth.

Both selected parent and actual learned best passed separate full2325-crop
PT/ONNX and200paired hardware-off White replays. Learned has2318exact union
masks/minIoU0.999371; raw outputs allclose. Learned whole-White median
1.021x/max1.200x, all<=2x; geometry/classifier and both-camera
production/debug parity pass. Full344band learned comparison is under
learned_diagnostic/, not the original parent-fallback galleries. release/
s_p_factory_20261117.* intentionally contains the retained parent; actual new
weights remain fits/native_hard/weights/best.*. Never confuse these or deploy
the learned model without a new instruction.

Original training input/code/protocol/runtime/canonical hashes remain exact.
Report audit recovered from one precisely verified library UI-notice transition:
Ultralytics exporter.py393 changes local openvino_msg true->false and CRLF->LF.
Exact prior bytes/hash reproduced; receipt and negative guard tests preserved
in repairs/library_notice_settings_20261117/. Original protocol/addendum kept;
no model/label/split/gate changes or resumed training for this repair.

Read DANH_GIA.md, ASSESSMENT.md, delivery_verification.json,
independent_completion_verification.json, learned_diagnostic/completion.json
and old_hard_cases.html. Final independent recomputation checks all861 learned
union predictions/group F1. Validation is development data; old hard cases are
fitted diagnostics, not independent physical-tray acceptance. Human factory
review remains pending. Preserve current SERVICE/APP/canonical revision12 and
retained best111 files. Leave review8810/report8811 available; no automation.

## Two-model segmentation review preference, 18 November 2026

The user only wants to compare the installed segmenter and newly trained best.
Public dashboard8811 and old_hard_cases.html now show only installed_parent
(active s_p_edge_white_native_hard_20261024_102450.onnx) and native_hard_best
(run fits/native_hard/weights/best.onnx, epoch11). Hide blend candidates in the
review presentation; retain their original metrics/checkpoints as historical
evidence. No model rename/deployment/training/label change. Prior full HTML is
archived in presentation_history/two_models_20261118/. See
two_model_presentation_verification.json.

## Pending legacy-scope inspection, 18 November 2026

User requested another segmentation fit excluding legacy, then explicitly asked
to see the legacy images to verify the intended exclusion first. No new fit or
exclusion has started. Read-only gallery8812 is registered in
.work/ACTIVE_SEGMENT_LEGACY_REVIEW.json, artifacts
.work/segmentation_legacy_review_20261118/. Proposed scope is origin==old only:
246 crops from an/code_train/datasets/disk,129train/117val in the17November
manifest. HA816, HA-derived523, reviewedOctober1040 and November2064 are
separate, not yet proposed for exclusion. User confirmation of which source
means legacy is pending; do not start dependent training/exclusion before that
clarification. Labels, source pools, prior splits/models/runtime remain intact.

## Authorized no-legacy segmentation repartition, 18 November 2026

User confirmed the246-crop origin=old legacy gallery, then authorized exclusion and repartition. User subsequently asked about union loss; exact17November method is retained. Planned October union-loss widening was withdrawn before any fit; see method_unchanged_receipt.json. No deployment/classifier work.

Run .work/segmentation_no_legacy_20261118/, dashboard8813, code/train_no_legacy.py. Protocol SHAfad326d1be3ef5e38ed534fa3f17b972ad9c8c80f6d176c8c59bc2b0510c6798. Never resume a completed run or duplicate parent/child. Before recovery inspect real processes, campaign.lock, active_child.json, logs, STOP_REQUESTED and frozen hashes. Completed17November run is not resumed.

Exactly246 legacy IDs match the user-reviewed gallery; sources/labels remain untouched.4443 retained records partition3172train/1271val pool;768unaugmented real val select checkpoints:600November,96October,72HA.503held-out derivatives diagnostic only.1535records switch split;192of former600November val remain val. Whole raw source/derived families, previousNovember capture/near-appearance groups and decoded/flip/180aliases never cross splits. MILP bounded1%relative gap, actualgap0.0097115, not exact global optimality. Classifier canonical revision12 unchanged.

Each3172train record once plus244new train-only hard extras,122per camera;1708old/1708new references per epoch. Same previous25%hard-pool algorithm, initial installed parent only, not the17November learned best. Same NativeUnionLoss source bytes, proto/cv4-only, all BN running stats frozen, AdamW lr1e-4, max40/patience10/batch8/seed1117, bounded HSV and no geometry. Split seed1118. Exact-mask loader count and CUDA AMP forward/backward tests pass; no optimizer step in preflight. Original dense_reviewed flags true/true/false/false are retained. No widening to October.

Same relative nonlegacy fitness weights(.8new-camera strict unionF1+.1HA-cameraF2+.05October unionF1)/.95. Original14nonlegacy tests unchanged;2legacy-specific gates not applicable, never silently passed. One fit, no blends/search. Only installed vs new model shown. Artifact s_p_edge_no_legacy_20261118.onnx is evaluation-only; activeSERVICE stays current hard segmenter. All object masks overlap and OR,62erasures/labels untouched.

Parent automatically evaluates fixed768val, verifies fit non-mask/BN exact state, exports new model, full2232crop PT/ONNX parity, evaluates held-out503derivatives separately, generates val errors, replays200current-config White hardware-off with344new-band union comparison and both-camera production/debug/geometry/classifier parity, then reviews4old hard frames/8bands with current split labels. Final union arithmetic independently recomputed from saved masks and all hashes rechecked before COMPLETE. Reports must acknowledge changed cohorts, parent HA/October exposure and development-only validation. Cannot attribute changes solely to legacy removal because partition changed simultaneously. Timing excludes camera/PLC/network/file/JPEG prep; <=2x whole-software limit.

The new no-legacy fit is now running: parent5620/child22300 at launch, exec
session91183, dashboard8813. Use actual process state and run status before
recovery; never duplicate. Initialization is the installed parent which had
historical legacy exposure; no-legacy means no246legacy images in this new
fit/val, not erasure of the parent's prior learning. No new automation.

## NEW_IMAGES_2 classifier fast review, 18 November 2026

The user requested full preliminary labeling of NEW_IMAGES_2, lowest confidence
first, then explicitly requested interactive review/label changes and fast UX.
Review is now http://127.0.0.1:8814/, artifacts
.work/classifier_factory_20261118/. Port8813 belongs to the concurrent
segmentation_no_legacy_20261118 dashboard; the old classifier-only server on
that port was stopped without touching the trainer.

All17200crop proposals from172White frames/86capture cycles are retained.
review_store.py writes user decisions ONLY through CanonicalDataset.save.
New confirmed real crops have origin=factory_20261118, stable normalized IDs,
source/crop/proposal hashes and camera/frame/slot/geometry provenance. Valid
labels go into canonical reference/{ng,no_disk,ok}, role=reference_only,
split=null; uncertain/excluded go into canonical excluded/. No train/val
assignment or training authorization is implied. Existing5393labels/splits
must remain unchanged except an explicit user edit to the same canonical
alias. No separate decisions overlay/prepared training pool. Live revision
comes from the current manifest, not the initial revision12 preparation hash.

UX:1NG/2no_disk/3OK, Enter keep proposal, U uncertain, X exclude, arrows skip,
Ctrl+Z undo, auto-save then auto-next, grid multi-select, progress/status filters.
Undo first confirmation returns that canonical record to unreviewed; it does
not erase provenance. Only navigation/last crop is stored in the browser.
All automated label-writing tests used the marked ux_fixture/ or temporary
fixtures, never the actual dataset; these are not training sources. Read
README_VI.md and ui_verification_v2.json. No new fit/model activation/runtime
source/config/model change by this UX task. User saves revoke old classifier
training authorization and legitimately change the live manifest hash; do not
restore old labels to satisfy historical/completed-run integrity guards.

Concurrent-review guard amendment: the active segmentation run now freezes the
exact prior classifier manifest/catalog/CSV bytes under
.work/segmentation_no_legacy_20261118/repairs/concurrent_classifier_review_20261118/.
Only those three protected annotation paths point to their frozen copies.
Protocol original and original authorization receipt are preserved; receipt.json
links old SHA fad326d1be3ef5e38ed534fa3f17b972ad9c8c80f6d176c8c59bc2b0510c6798
and amended SHA c0b2280526abd039bac5b7aaebed8c18c35da375b78f6369b503fc3ad5b30577.
All segmentation training/code/input/split/loss/sampling/gates and other runtime
hashes are identical; fit was not stopped/restarted. Positive live-label-change
and negative runtime/snapshot guard tests passed, and all actual hashes passed.
This scope repair implements the latest explicit concurrent label-edit request.
Do not claim the LIVE classifier manifest stayed unchanged after user review;
the inherited final_report wording must be read with/corrected by this receipt.

## Edge completeness supersedes area matching, 18 November 2026

The no-legacy fit is COMPLETE:15epochs, best5. Do not resume. The user now
explicitly prioritizes catching every edge, not union-pixel area overlap.
Reassess existing best5 and last15 only at
.work/segmentation_edge_coverage_review_20261118/, dashboard8815, entrypoint
evaluate.py (read-only; no fit). Classifier review owns8814; do not interfere.
Primary:1-to-1 missed labeled edge crossings at two real caliper rows; separate
raw-mask misses from production cleanup/caliper misses. Fixed8raw-pixel center
tolerance;4/8/12sensitivity, never threshold-search. All-row centerline crossing
recall measures continuity, not area IoU. Reference is reviewed mask runs,
not independent physical disk counts. Rank best/last lexicographically by
val caliper misses, raw-mask misses, caliper extras; only116valbands select,
282train bands diagnostic. No extra training/weight/threshold/runtime changes
or deployment. Public view only installed vs selected new. Preserve previous
F1 results/gate failures as history; new metric is post-hoc development and
not retroactive passing of old gates. Last15 export/state check and12-image
PT/ONNX smoke passed; only best5 has prior full2232parity/200White replay.
Read .work/ACTIVE_SEGMENT_COVERAGE_REVIEW.json and protocol/status before
recovery; no duplicate evaluation. See no-legacy SCOPE_NOTE.md for authorized
concurrent classifier label edits and frozen classifier snapshot guard repair.
Never claim the live classifier annotations stayed revision12/unchanged.

## Completed edge-first assessment, 18 November 2026

Coverage review8815 is COMPLETE;398reviewed bands evaluated,116current val
select only between saved best5/last15. No extra fit. Selected new is best5,
copy s_p_edge_coverage_20261118.onnx under review artifacts; bytes match the
no-legacy best5, not a newly trained network. Installed model retained.
Val5246labeled edge crossings on two production scan lines: installed/new
raw-mask misses104/107, actual cleanup+caliper misses134/162, extras2/2,
bands with no caliper miss35/116 vs25/116. Last15 has105raw misses,179caliper
misses,3extras; internal only.4px/8px/12px sensitivity preserves ordering;
8px and12px totals identical. These are crossings, not unique physical disks.
Old8hard bands: caliper misses10/10 on162reference crossings; raw misses7/8.

Diagnostic decomposition of val actual-caliper misses: installed99already
missing in raw mask +11lost by area cleanup +24present after cleanup but
caliper-unread. New102+6+54;47of new54unread stripes are1–3px wide. No threshold
or runtime change. See caliper_miss_decomposition.json. Model is still not
better under the user's edge-completeness goal; area F1 must not override it.

Public index/old_hard_cases.html show only installed vs selected new with red
missing-position markers. Results and original training gate failures remain
unchanged historical evidence. delivery_verification.json and
access_and_arithmetic_verification.json pass; no evaluator remains active.
Segmentation annotation snapshot still matches current review. Concurrent
classifier labels may advance; do not restore them or claim live revision12
unchanged. Current deployment/model/config unchanged. Read review protocol,
RESULTS.md and latest classifier review section before further work.

Any future authorized fit should freeze missed-edge/continuity-first checkpoint
selection before training, rather than reuse the old area-heavy fitness as
primary. This does not authorize an extra fit or rewrite a completed run.
Preserve exact old loss/method unless the user authorizes a method change.


## YOLO11s pretrained segmentation, 18 November 2026

User authorized one new trial from generic YOLO weights and all non-demo data.
Run .work/segmentation_pretrained_s_20261118/, dashboard8816. Initial weights
are the 80-class yolo11s-seg.pt used on30September; train the 1-class network
including backbone, box/confidence, mask and BN, from the fixed no-legacy split.
4443 retained records (3172 train,1271 val pool), 3416 epoch references, 246
legacy demo excluded. No classifier training or live label interaction.
One fit max40/patience10,512px/batch8/lr5e-4, previous exact-mask loss. Selection
uses missed crossings at current caliper first, extras second, raw mask misses
third across the116 reviewed val bands. Offline thin-trace comparison only.
An Ultralytics grayscale singleton loader repair passed all116 bands/696 tiles
prior to starting any fit; logs and exact protocol change kept in preflight_fix.
Do not duplicate/restart a fit without checking actual trainer, lock, logs and
STOP_REQUESTED. Evaluate actual best, paired parity/replay and the historical
8 bands. Keep deployment separate.

The caliper investigation for the preceding fine-tune reproduced current results
on398 bands. Offline reading of thin mask traces cut val misses134->120 for
installed and162->116 for best5, with2 extras for both. Train diagnostic extras
increased21->25 and4->5. No runtime change. Report on8815/caliper_recheck.html.


## Updated YOLO11s training horizon, 18 November 2026

The user superseded max40/patience10 with max200/patience60. Resumed from
the optimizer-bearing epoch4 checkpoint, restored EMA and best epoch3/fitness.
The original launcher was stopped at an epoch boundary and its epoch5 output
archived. Epoch5 is repeated under the revised linear schedule, retaining the
first four epochs. Actual resume verification records 290 optimizer states
and 369 EMA updates. Process22320 at launch. Read status.json and new protocol,
train.stdout.log and schedule_transition_200e. No change to data/model size,
loss or ranking, and no deployment.


## Caliper checkpoint selection at 3 px, 18 November 2026

User requested evaluation below the 4px minimum peak separation. Choose best
at 3px on the same116 val bands by misses/extras/raw-mask misses; report2px and
4px separately. Matching tolerance, profile thickness, scan length, cleanup,
model loss, deployed SERVICE config, max200 and patience60 remain unchanged.
Older4px fitness cannot rank under3px. Preserved best epoch6 and last epoch26
were re-scored at2/3/4px and epoch6 seeded the new best. Earlier unsaved epochs
cannot be re-scored. Epoch27 repeats from optimizer-bearing epoch26 checkpoint
with EMA preserved. Patience counts from epoch26 or a subsequent improvement.
See caliper_transition_3px receipts/logs and caliper_resume_verification.json.
Final report scores both models under all three same settings. Hardware-off
White replay remains under the as-configured SERVICE 4px settings.


## Completed YOLO11s pretrained segmentation, 18 November 2026

Run segmentation_pretrained_s_20261118 complete, epoch86 early stop, best6
rescored under3px from the preserved checkpoints. Do not resume/repeat.
116 val bands, model installed->pretrained misses/extras by gap: 3px120/3
->122/4; 2px116/4->73/5; 4px134/2->156/3. Old difficult8bands at3px misses8->6.
PT/ONNX2232 tiles passed raw tolerances,2228 exact unions. 200 White runtime
replays pass with SERVICE 4px, median0.980x/max1.442x. 2px is paired offline
development, not a field acceptance result. No deployment. See dashboard8816
and completion.json. Active segmentation pointer recorded complete.


## White mask postprocessing audit at 3 px, 18 November 2026

User asked to improve mask shape before caliper, then test at3px and separate
caliper misses from segmentation misses/false edges. Reusable prototype is
SERVICE/src/tools/white_edge_mask.py. Offline evaluation and gallery8817 are
under .work/segmentation_white_mask_postprocess_20261118/. No deployment or fit.
Width5 with no gap bridging showed the best measured tradeoff among tried
variants at3px on116 development-val bands. Installed misses/extras120/3 to
110/4, YOLO11s pretrained122/4 to87/4. Caliper misses despite mask on scanline
21->12 and73->41 respectively. Separate experiments (filter first + bridge7 +
width7; add only raw-supported pairs) are retained with outcomes. The mask
queue is for review of frozen val only, not direct train additions. Original
mask local absence/excess under +/-7 row audit is50/6 installed and33/7 new.
16 White hardware-off frames,32 paired replay comparisons at3px, runtime
geometry/classifier and both-camera normal/debug pass. Reports and tests only;
existing SERVICE startup/config/active models were not changed.


## One-column label seam repair and re-training, 18 November 2026

User flagged screenshot showing first November Cam1 band2 with label fragments
t4p0 and t5p3 at core boundary1555. Verified missing column1554 for68 rows.
Audit398reviewed bands, repaired369 (326Nov/43Oct), added24533 pixels;
1035train/480val records receive linked exact-mask lineage. Ambiguous pairs260
including11 protected intentional erasures stay as supplied. Source review
and completed campaigns immutable. See new snapshot and before/after link.
Run .work/segmentation_label_seam_repair_20261118/ dashboard8818 PID1548 at
launch. Generic pretrained YOLO11s-seg, same split/ordered sampler all4443
records excluding246legacy, max200/patience60, exact mask loss. Launch once.
Selection3px, secondary2/4px across same116bands with corrected5175 crossings.
Previous pretrained control now2px43misses/46extras under corrected labels
versus prior73/5 under old labels. Final compare both prior fits and installed
on same corrected truth; model improvements are separate from re-labeling.
Complete parity/replay automatically. No production postprocess or deployment;
user specifically wants label repair/train first, fragmentation work later.

## NEW_IMAGES_2 user-confirmed intake and supplementary classifier fit, 18 November

User said every NG/empty suggestion was reviewed and explicitly asked to assume
remaining OK correct, preserve current corrections, and train/evaluate. Snapshot
revision219 has190 individually saved decisions (171 changed), all retained.
Revision220 adopts17,200 real November crops through CanonicalDataset.save:
190 saved,1,911 unchanged NG/empty accepted by category,15,099 assumed OK.
Only the 190 are marked individually reviewed. Existing5,393 rows and old
splits preserved exactly. No crops omitted. New grouped12,000 train/5,200 val;
combined15,778/6,815. Both cameras per capture, burst, apparent duplicates and
decoded/flip aliases stay together; validation is development, not verified
independent physical trays. Read .work/classifier_factory_train_20261118/
user_review_receipt.json, review_decisions_snapshot.json, split_report.json and
intake_audit.json. Do not restore older canonical labels or TRAIN from proposals.

One new YOLO11n-cls fit/evaluation, dashboard8821, run
.work/classifier_factory_train_20261118/ and train.py. Exact active_dataset
revision220 manifest/hash and crop bytes frozen before fit. Parent HA best111,
frozen BN statistics, AdamW lr1e-5, NG CE2:1:1, side-aware cycling source/camera/
class sampler from train only,8192 exposures/epoch, max20/patience6/min7.
NG-first rank retains old cells and guards reject-all. Compare learned best vs
installed HA best111 by source/camera and label provenance, full6815 PT/ONNX
parity and 36 offline White frames. No deployment. Check real process, lock,
logs and STOP_REQUESTED before any recovery; do not duplicate or resume a
completed run. Later live label edits do not change this immutable snapshot.


## Broad 1–3 px label bridging and re-training, 18 November 2026

User explicitly requested joining near mask fragments without a direction
condition, with 1–3 px fill and autonomous re-training/evaluation. Old
seam-only run COMPLETE, 80epochs/best20, PT/ONNX2232 and White replay200
passed; do not resume. New run .work/segmentation_label_near_gap_20261118/,
dashboard8820, train.py campaign PID23592 at launch. New exact label
manifest bridges bracketed blank horizontal/vertical runs 1–3px once,
merges linked mask lineage and protects manual erasures. 18,000 pixels on
393/398 bands; 1,225 train and602val records regenerated. Verified all
four previously skipped screenshot points. Original manual editors 8810
and 8819 stay intact; new target revision is frozen in this run. Same
4,443 nonlegacy rows, ordered3,416 epoch exposures and split. New validation
has5,153 crossings on116bands. Re-score prior seam-only and installed
against exact same new labels before model comparison. Generic YOLO11s-seg
pretrain, max200/patience60, select3px and report2/4px, old hard cases,
PT/ONNX2232, White replay200, per-point gallery with length experiment.
No runtime changes or deployment. Before recovery check process, lock,
STOP_REQUESTED and status; never start duplicate or restart completed fit.

Completion update: classifier_factory_train_20261118 is COMPLETE. One fit,
13 epochs, best7, all15,778 training rows seen. Initial attempt failed before
optimizer/checkpoint on Windows int32 CE targets; verified explicit torch.long
repair, archived original code/protocol/logs; no recipe or snapshot changes.
Frozen 6815val installed -> learned ONNX errors51->44, NG misses7->5,
falseNG12->14, empty-as-OK14->12. New5200val 47->39 errors, NG4->3,
falseNG12->14, empty-as-OK14->9. New Cam1 NG1->0, Cam2 3->3.
53 individually saved new val errors47->29; 4534 assumed-OK val 0->9 label
disagreements (5 NG and 4 no_disk), labels assumed so ground truth unverified.
HA2NG->0, October empty-as-OK0->3, legacyNG1->2. Gates fail. Keep best111 installed.
36 White offline pairs pass geometry/caliper and both-camera debug, median0.992x
max1.422x. Original GPU TF32 PT vs CPU ONNX:1 argmax mismatch/max0.004984;
independent CPU PT/CPU ONNX and TF32-off GPU PT/CPU ONNX match all6815 argmax
(max1.05e-5 and5.66e-6). Original gate retained, no export/weights modified.
Active training pointer complete, no deployment. See dashboard8821,
RESULTS.md, completion.json, parity_diagnostic.json and errors.html.

## Segmentation 1–3 px label bridge complete, 19 November 2026

Run `.work/segmentation_label_near_gap_20261118/` is COMPLETE. One fit,
86 epochs, best26, patience60. Do not resume/repeat. The first White replay
stopped on image86 because two unchanged November masks had manifest paths
in the preceding run and no local replay copies. Evaluation-only recovery
verified every November mask digest and materialized two byte-identical copies.
Original code, protocol, labels, split, weights and initial diagnostics are
unchanged. See `evaluation_recovery/repair_receipt.json` and `completion.json`.

On the same 116 development-val bands, installed to new misses/extras are
25/6 to18/2 at2px, 28/4 to21/1 at3px, 41/2 to40/1 at4px. Historical hard
held-out two bands worsen from3 to6 misses at3px. PT/ONNX2232 raw tensors
pass, 2227 unions exact and the other five differ by one pixel each. All200
hardware-off White pairs and both-camera debug checks pass; median paired
ratio1.023x, max1.558x. Runtime hashes unchanged and no deployment.
Dashboard8820 and `final_count_review/` are available.

## Classifier archive and one no-legacy fit, 19 November 2026

User requested saving the learned with-legacy model as the 19_11 version and
one new classifier fit without legacy. Byte-exact checkpoint epoch7 from
classifier_factory_train_20261118 is archived at SERVICE/config/models/
c_p_19_11_2026_co_legacy.{onnx,pt}. Active c_p.onnx remains HA best111.
Archive receipt: .work/classifier_no_legacy_20261119/archived_with_legacy.json.

New run .work/classifier_no_legacy_20261119/, dashboard8822, train_no_legacy.py.
Receipt acknowledges partial review on exact canonical revision228. Preserves
all crop IDs, hashes, labels and splits in canonical; archived input snapshot
has exact manifest, only nonlegacy crops on disk. Excludes420 train and180
val origin=old_legacy_train. Remaining15358 train/6635 val, same split groups.
All retained training labels match previous fit; four revised validation labels
are used to score all three models. Keep old eval truths and results historical.
One fit from same installed HA best111 PT, old parent has legacy exposure.
Reuse crop preprocessing, AdamW, fixed BN, NG CE2:1:1, seed, flip, 8192 draws,
max20/min7/patience6. Reassign prior 5% legacy draw budget equally to HA and
October, November stays65%. Checkpoint ranking drops legacy scopes and scores
with TF32 off for ONNX consistency; train precision unchanged. All models
compared under fixed revision228 truth. Review still partial and development-only.
No deployment. On completion verify full6635 PT/ONNX and36 paired White offline,
runtime hashes and process exit. Do not launch a duplicate or resume a completed
fit. Check real process, campaign.lock and STOP_REQUESTED before any recovery.

The short no-legacy run is COMPLETE: 12 epochs/best6, PT/ONNX6635 exact,
36 White pairs pass (median0.975x/max1.187x). Same 6635val HA best111 / saved
with legacy / short no legacy: errors54/38/39, missed NG8/3/3, false NG12/12/14,
empty-as-OK16/12/12. All three live models compare on revision228. Short run
is historical and not to be repeated. No activation.

User superseded the short horizon with max200 / early stop after60 consecutive
epochs without strict improvement. New long run
.work/classifier_no_legacy_200e_20261119/, dashboard8823, train_no_legacy.py,
PID32808 at launch. Fresh optimizer from same HA best111 since prior weights
do not contain optimizer state. Exact same revision228 records, exclusions,
train15358/val6635, class/source budgets and precision policy. Only max horizon
and patience change (cosine schedule stretches accordingly). Leave short run
8822 and saved with-legacy version available. New run auto-evaluates against
both fixed models and runs full parity/replay verification. No deployment.
Never duplicate, resume a complete run, or touch another trainer. Check its
actual processes, campaign.lock, logs, STOP_REQUESTED before recovery.

Long no-legacy run .work/classifier_no_legacy_200e_20261119/ is COMPLETE:
77 epochs, best17, exactly60 epochs without improvement, cap200. No resume
or repeat. Revision228 shared6635val best111 / saved with legacy / long no
legacy errors54/38/30, missedNG8/3/3, falseNG12/12/7, empty-as-OK16/12/13.
With and without legacy miss the same3 November Cam2 NG. New Cam1 errors
9->7, Cam2 26->21; Cam2 falseNG9->4 and empty-as-OK9->10. November5200
validation errors35->28. Old short no-legacy run had39errors/3NG/14falseNG.
Development validation, 4529assumed-OK retained; excludes legacy from new
fit/val but same parent had prior legacy exposure. Different train horizon
precludes attributing the result solely to legacy removal. PT/ONNX6635
exact argmax (max1.63e-5), 36 White pairs geometry/counts/debug pass,
median1.016x/max1.250x. Gates fail, no activation. Archived 19_11 with-legacy
files and active HA best111 hashes verified. Trainer exited. Dashboard8823,
RESULTS.md, comparison_three.json and errors_three.html remain available.

## User-selected SERVICE classifier, 19 November 2026

Latest user explicitly selected the long no-legacy best17 checkpoint as current
best and requested replacing SERVICE's classifier. SERVICE/config/models/c_p.onnx
now has SHA256 79f95f774a9bd4fed783eec3b7d577a5d6d4ec347ec4bb70e4b05997a418b38e.
Selection metadata and ACTIVE_CLASSIFIER_FIELD_TEST point to the new model.
Old best111 was hash-verified and backed up with previous selection metadata
under .work/service_classifier_no_legacy_20261119/backup/. The named best111
source and named 19_11 with-legacy archive are retained.

Activation receipt .work/service_classifier_no_legacy_20261119/activation.json.
Fresh SERVICE initializer using actual config verified 9 crops and replayed
one stored November White image per camera without hardware. Config/canonical/
detector/segmenter/APP Debug plugin preserved. This covers SERVICE's configured
model path; APP Debug uses its own separate plugin. No live SERVICE listener
was present and no server or PLC was started. Prior run metrics and recommendation
failures stay historical, including three validation NG misses, and the user
override authorizes this selection. No additional fit or deployment implied.


## Segmentation train-only tile shifts, 19 November 2026

User explicitly requested re-training with shifted image/mask crops and their
two recently corrected Cam2 labels. Run .work/segmentation_shift_tiles_20261119/
dashboard8824, PID35744 at launch. Exact editor Canvas mask export for revision
5 of Cam2_ef6bf23e5d83a522_band2 and revision7 of Cam2_8f6253eda4258da9_band1.
Both stay validation; three tile targets changed. Preserve previous bridges
away from edited strokes, use latest manual pixels around edits. 116 val
bands, 5,151 crossings. Prior epoch26 re-scored on same new labels has
2px16/2 and3px19/1 missing/extra; SERVICE segmenter2px24/7 and3px27/5.
All4,443 nonlegacy originals, same groups/split and3,416 baseline epoch
references; add1,692 train-only horizontal shifts +/-16/32px across both
cameras, 5,108 total references per epoch. Source/image/decoded-flip groups
stay within the split, exact aligned raster masks OR. No geometry changes
at evaluation. Fresh generic YOLO11s-seg, same200/60 and loss, choose3px
misses first, report2/3/4px, two edited bands, old hard cases, parity2232
and hardware-offWhite200. Inspect gallery for both 2/3px. Current protected
runtime captures pre-existing user-selected classifier epoch17; segmenter
unchanged. No runtime edits/deployment in this campaign. Do not duplicate
or resume a completed run. Check process/lock/STOP_REQUESTED before recovery.

Shifted-tile run stopped partway through epoch30 on a dense binary-target
assertion. The exact failing value did not reproduce from 3,528 frozen source
masks. Archived original code/protocol/logs/best25/last29 under
`recoveries/epoch30_binary_target/`. Resize and collate dense occupancy as
checked uint8 before GPU float conversion. All source unions produce the same
nearest-resampled values and the CUDA loss/gradients match on all shape classes.
Resume PID36328 from epoch29 with saved EMA/AdamW and original best25/patience60;
no new fit or label/split/selection change. Separate SERVICE White/UV changes
already on disk were recorded for current-config replay; the validation mask
cleanup code is identical. Inactive classifier files were verified in their
existing archive. Do not restore them into SERVICE. The new receipt, protocol
and `resume_verification.json` document recovery. If a later failure occurs,
check last epoch, lock, real processes and STOP_REQUESTED before acting.

Shifted-tile fit/evaluation is COMPLETE, 91 epochs, best31, patience60. The
binary-target recovery passed epoch30 and remaining training. Do not resume or
repeat. On the same 116 val bands including the two edits, epoch26 to epoch31
ONNX missing/extra is 16/2 to26/5 at2px, 19/1 to28/3 at3px, 38/1 to54/3 at4px.
At3px, 6 prior misses recovered and 15 new misses appeared (9 November Cam1).
The two old hard val bands improved 6 to5 misses at3px. Epoch26 still leads
overall. The 2px validation count from ONNX is26/5; live PT was26/6.
PT/ONNX2232 numerical parity passed (2229 exact unions). White replay200 and
both-camera debug parity passed, median0.978x/max1.385x. Frozen data and
protected current runtime checks passed, process and lock gone, no deployment.
Dashboard8824, `final_count_review/` and `completion.json` are available.

User requested retaining the previous near-gap checkpoint26 in SERVICE with
a separate name. Archived ONNX is `SERVICE/config/models/s_p_edge_yolo11s_ep26_20261118.onnx`,
SHA256 2f8dff4bb355919a8bf4a0e21df331e40ba24a62a41ffeac5b8e626d529b48e2.
Storage only, explicitly preserve it. Active SERVICE models and selection remain
as before. Training PT stays at the original near-gap run.

Read-only follow-up `.work/segmentation_epoch26_refinement_20261119/`, dashboard
8824 `epoch26_refinement/`. On fixed 116 val with two updated Cam2 labels, base
3px19miss/1extra, one-pass thin/small-area trace recovery15/1, three named
2px masks and one227.5px² cleanup loss rescued. Two shifted ±32px passes with
agreement/full two-scan trace yield14/1 at ~3.02x segmentation inference only;
old hard val misses6 unchanged. x1444 Cam2 band2 remains a merged13px stripe;
generic brightness split14/30 overcounts. Development trials only, no new fit,
runtime code edit, config change or activation.

Epoch26 guarded recovery offline `.work/segmentation_epoch26_guarded_mask_20261120/`.
Retain 0.50 standard caliper pairs, widen supported raw traces to5px, then
optionally add candidate-mask pairs from the same YOLO forward at0.05 if raw
trace spans both scans, local brightness supports it, and separation >8px.
On 116 reused development val, 4px original38/1, guarded widening17/1,
widening plus weak-confirmed13/1. Independent end-to-end offline verification
agrees; median paired band ratio1.127x, not whole SERVICE latency. At3px19/1
to11/1. Unrestricted0.05 gives26/3 and0.01 gives24/9 at4px; guarded0.01
gives12/3. Short endpoint bridges produced no additional gain on this val.
Nine points lacked raw support near the original scan; highest tested cutoffs
restoring an associated mask were0.15 (3),0.05 (2),0.03 (1),0.01 (1), absent
through0.01 (2). See assessment.json and candidate_verification.json.
Preserve active runtime and E26 archive; no deployment or additional fit.

## SERVICE epoch 26 activation and White scan endpoints, 20 November 2026

The user selected epoch 26 for SERVICE. `SERVICE/config/config.env` now points
to `s_p_edge_yolo11s_ep26_20261118.onnx` with minimum caliper peak distance 3 px,
segmentation confidence 0.10 and scan length 0.95. The prior SERVICE ONNX was
renamed to `s_p_edge_white_native_hard_20261024_102450_old.onnx` with the same
SHA256. See MODEL_SELECTION.json and
`.work/segmentation_epoch26_guarded_mask_20261120/service_activation/`.

White caliper now extends its sample by up to 8 px per side when an original
endpoint is inside a foreground stripe. It adds only a complete crossing pair
that does not overlap an original pair. Case 06 (`14112026_164345_9268961_Cam2_White_Raw.png`,
band2 x1815 y214) recovered its right boundary. On the same 116 development
bands, epoch 26 at 3 px/0.10 went from 14 misses / 2 extras to 13 / 2, with no
other pair changes. Both stored White frames pass production/debug parity in
fresh SERVICE with PLC methods blocked. Remaining points are in
`replay_3px_010/after_boundary/` under the evaluation directory.

## Nano segmentation comparison, 20 November 2026

User authorized one YOLO11n-seg training and evaluation. The active run is
`.work/segmentation_nano_20261120/`, dashboard 8827, entrypoint `train.py campaign`.
Initialize from local official COCO nano weights. Use the same 4,443 nonlegacy
tile records and 3,416 sampling references per epoch as epoch 26, with the latest
two Cam2 validation edits and exact split groups. Keep the same NativeUnionLoss
and 200 epoch cap / patience 60. Rank checkpoints by missed crossings, then extra
crossings, then raw mask misses at 3 px / confidence 0.10 using the current White
caliper. One fit. Epoch 26 in SERVICE stays the active model.

Campaign exports ONNX and evaluates both models with and without the selective
flip on the same 116 validation bands. Also report the eight historical hard
bands by original split, PT/ONNX mask parity and 16 hardware-off paired White
CPU timing frames. The same validation bands are development data. Read the
protocol, real process status, campaign.lock and STOP_REQUESTED before any
recovery. Never duplicate or restart a completed campaign.

A machine reset interrupted the nano fit in epoch 3. Both Ultralytics
checkpoints lacked their ZIP directory and could not be loaded. The interrupted
fit, logs, stale lock and original protocol are retained under
`recoveries/machine_reset_epoch03/` in the nano run. Restarted from the same
pretrained weights, frozen data and protocol, with PID 29096 at launch and
dashboard 8827. The only trainer change adds validated, flushed atomic copies
`last.durable.pt` and `best.durable.pt` after saves. In a future interruption,
verify epoch/optimizer/EMA and fit state from the durable copies before any
resume. Do not restart a live or completed fit.

At epoch 77 validation, Windows briefly denied the atomic status file replace.
The previous trainer exited before saving epoch 77. Archive and recovery proof
are in `recoveries/status_lock_epoch77/`. The only repair adds a bounded retry
for Windows file-share errors. The retry was tested under a temporary read
lock. Resumed at epoch 77 from saved epoch 76 with optimizer/EMA verified, PID
30292 at launch, and original best epoch 28 and patience retained.

Nano run COMPLETE: 88 epochs, best 28. On 116 development validation bands
at 3 px / confidence 0.10, epoch 26 plain/gated scored 13/2 and 7/2 missing/extra;
nano plain/gated scored 10/3 and 8/3. On the two held-out historical hard bands,
missing counts were 5/3 and 4/4 respectively. The other six historical bands
were train diagnostics. PT/ONNX nano union minimum IoU was 0.99810 across all
116 validation bands. On 16 paired hardware-off White frames with four ONNX CPU
threads, median whole-flow timings were 1.217/1.397 s for epoch 26 and
0.787/0.931 s for nano. No hardware timing or independent test claim. ONNX is
`artifacts/s_p_edge_yolo11n_20261120.onnx` in the run. Dashboard 8827 includes
`report.html`, `edge_evaluation.json`, `timing.json` and `completion.json`.
SERVICE still selects epoch 26. Do not resume or repeat this completed fit.

## SERVICE nano activation with White edge recovery, 20 November 2026

User selected nano best epoch 28 as the SERVICE default and requested the
tested White recovery in the main flow. `SERVICE/config/config.env` selects
`config/models/s_p_edge_yolo11n_20261120.onnx` with 3 px / confidence 0.10.
Epoch 26 stays in `SERVICE/config/models/s_p_edge_yolo11s_ep26_20261118.onnx`
as the hash-verified alternative. Switch `DISK_SEGMENT_MODEL_PATH` and restart
SERVICE to compare. `WHITE_EDGE_RECOVERY_ENABLED = True` enables the tested
same-forward 0.05 mask filter and selective horizontal tile retries. False
enables one pass. White and default Debug share the same implementation.

Verification `.work/service_nano_white_20261120/`: production positions matched
the offline trial on all 232 scanlines for each model, all 217 tests passed,
fresh SERVICE configuration loaded nano, and both models passed hardware-off
White/Debug parity on four stored frames across Cam1 and Cam2. Previous config
and model selection metadata are backed up under that verification directory.
