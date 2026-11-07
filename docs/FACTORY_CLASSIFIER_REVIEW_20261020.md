# Factory images 18 October: classifier review and point correction

> **Current authority, 21 October:** The active dataset is now
> `DATASET_CLASSIFIER/`, registered in `.work/ACTIVE_CLASSIFIER_DATASET.json`.
> Use <http://127.0.0.1:8791/> to edit labels directly. Old prepared datasets and
> decisions below describe historical workflow; do not rebuild over current
> corrections from them. Known local saved reviews were reconciled without
> missing decisions. Training is stopped while the user reviews consistency.
> See [CLASSIFIER_DATASET_CANONICAL.md](CLASSIFIER_DATASET_CANONICAL.md).

The user deferred UV/mixing work and requested classifier false-reject and point
detection corrections. After completing NG review and inspecting the prepared
dataset, the user explicitly authorized autonomous training and evaluation on
20 October 2026. Training outputs are separate from the installed runtime.

## Review

URL: <http://127.0.0.1:8784/>. Output:
`.work/classifier_factory_20261020/review/`.

There are 2,800 native lossless PNG crops from 28 White images, with all four
production faces and 25 target slots per face. The default filter shows the 113
model NG proposals first. These predictions are not confirmed ground truth.
Each crop has neighboring context, the target crop boundary, full-frame context,
all three original scores and a stable ID/hash. Top faces retain production's
180-degree rotation. All 2,700 proposals from previously usable images reproduce
the audit's native-input classifier scores with zero mismatches. The other 100
come from the formerly rejected point geometry after the correction below.

The existing review UI saves choices separately to `decisions.json`, validates
dataset hashes/IDs, uses atomic writes, and resumes after refresh/restart. Old
reviews and original proposals are not modified by label edits. No label starts
confirmed. Model confidence is not a measured correctness probability.

- `ok`: exactly one disk in the target groove.
- `ng`: more than one disk in the target groove.
- `no_disk`: no disk in the target groove.
- `uncertain` / `exclude`: unresolved / unsuitable; exclude from training.

Neighboring grooves and two rims/reflections of one disk do not mean target NG.
The user can change the model-class filter to review no_disk and OK proposals as
well. Only explicitly reviewed labels may enter supplementary training; no
implicit confirmation of the other 2,800 proposals is allowed.

Restart from the repository root:

```powershell
& ./SERVICE/.venv/Scripts/python.exe -B SERVICE/training/serve_classifier_review.py --output .work/classifier_factory_20261020/review --port 8784
```

Rebuild a separate review with `python -m training.prepare_factory_classifier_review`
from SERVICE, supplying `--audit`, `--output`; existing reviews are never overwritten.

## Point correction

Cam2 `Image_20261018103035152.bmp` had 78 raw detections. Selected rows were
25/26/25; an extra box beyond the final groove caused geometry rejection.
`filter_off_grid_extras` now uses two complete, regularly spaced reference rows
to remove an unambiguous off-grid extra before existing slot completion.
It requires exactly one candidate per target groove and a consistent projective
fit. Duplicate/ambiguous candidates, missing anchors or multiple overfull rows
still reject. It does not select the highest-scoring 25 boxes or force disk counts.

The real image now reaches classification and segmentation with 25/25/25 point
rows. This fixes geometry failure; its two NG and two empty classifier proposals
remain subject to human review. Source production/debug geometry and final
render match. Other 27 frames retain their original point geometry/class scores.

Validation: 94 SERVICE tests, 82 passed / 12 skipped for optional training
libraries. Review HTTP/assets and JavaScript syntax/default filter checked;
automated browser rendering was unavailable due to its Windows helper failure.
The installed backend has not been rebuilt/deployed with this point correction.

## Training continuation / retained artifacts

The current classifier remains the full-HA blend50 deployed on 5 October:
`ce0a150f00a56b78bbc4a07bbf3ad28862e7de7edd78a77bd0ee9f2d6a0d72bd`.
Matching PT/ONNX are retained in
`.work/classifier_fullha_20261005/selected_v2/`.
The new review contains a byte-identical ONNX snapshot.

All 2,000 human-reviewed old HA crops and corrections remain in
`.work/classifier_retrain_reviewed_20261005/dataset/`; the previous good model
backup remains in `.work/classifier_model_backups/pre_relabel_retrain_20261005/`.
Earlier full-HA/CV runs and evaluation reports are retained.

The user reviewed all 113 original NG proposals: 75 NG, 26 OK, 12 no_disk.
The other 2,687 model OK/no_disk proposals were not manually reviewed. The user
explicitly requested that those also be included in the prepared dataset;
their label provenance must remain model proposals, not human confirmation.
The effective new pool is 2,800 crops: 75 NG, 1,966 OK and 759 no_disk.

Prepared pool: `.work/classifier_factory_20261020/prepared_dataset_v1/`.
`manifest.json` and `labels.csv` preserve effective labels, original predictions,
human decisions and source checksums. `snapshots/` freezes the exact old manifest
and new review inputs. It includes 2,000 old HA + 1,000 legacy train + 2,800 new
crops in `train_pool/`; 2,770 legacy validation crops remain in `evaluation/`.
The old duplicate `ha_fit` view is not copied as additional training examples.
New splits are unassigned until the dataset is approved and grouped by capture.
The read-only catalog supports origin, class and label-provenance filters.
It is served at <http://127.0.0.1:8785/?origin=old_ha>, initially showing the old
human-reviewed HA dataset; select old legacy train or the new factory data in
the source filter. The actual old training source is
`.work/classifier_fullha_20261005/datasets/full_balanced/`, whose training part
contains 2,000 HA and 1,000 legacy anchors. The legacy originals come from
`D:/huynhvc/WORKING/TRAINING/training_cls/dataset/disk_3/`.

Prepared label counts:

| Population | NG | no_disk | OK | Total |
| --- | ---: | ---: | ---: | ---: |
| Old HA | 451 | 322 | 1,227 | 2,000 |
| Old legacy train | 600 | 300 | 100 | 1,000 |
| New factory, effective labels | 75 | 759 | 1,966 | 2,800 |
| Training pool | 1,126 | 1,381 | 3,293 | 5,800 |
| Legacy validation, excluded from training | 814 | 520 | 1,436 | 2,770 |

The new Cam1 frames `Image_20261018112241885` and `Image_20261018112249401`
produce 100 byte-identical crop pairs. They are retained in this preparation,
with exact-duplicate groups recorded; their frames must remain together when
splitting data. There are no conflicting labels or exact train/evaluation
duplicates. All old labels and crop hashes are preserved.

The user has now verified the preparation and authorized training. Use the frozen
labels (excluding uncertain/excluded items and retaining
the user-requested proposal provenance), and train from the retained classifier
with old labeled HA/legacy replay; group new splits by real capture cycle and
near-duplicate frames, never by crop; compare false NG, true NG misses and empty
errors against the current model on identical confirmed labels and old retention
benchmarks. Do not report fitted/reviewed data as independent test accuracy.
Store candidates separately, validate ONNX/runtime parity, and package the selected
classifier together with the point correction when ready for deployment.

The authorized run is `.work/classifier_supplementary_20261020/`, implemented by
`SERVICE/training/train_classifier_supplementary.py`. Its protocol freezes the
current blend50 PT/ONNX, source manifest hash, human decisions and a predeclared
three-fold new-capture grouping / adaptation recipe. Consecutive captures within
120 seconds (both cameras) stay together; exact-duplicate frame pairs cannot
cross folds. CV chooses the single-network update fraction using reviewed new
NG misses / false NG and old-domain retention. The final model learns all new
crops plus old replay. Its new-data metrics are explicitly fitted-data results,
while grouped CV is a reviewed collection model-selection estimate. The 2,687
model labels are not presented as independent human ground truth. No deployment
has been authorized or performed in this run.

## Completed short runs and requested longer training (21 October)

The first comparison completed 12 training jobs (three recipes, each with three
grouped CV fits and one all-new-data fit). Their limits were CV 30 epochs with
patience 8, and final 40 epochs with patience 10. Actual completed epochs:

| Recipe | CV folds 0 / 1 / 2 | Final fit |
| --- | --- | ---: |
| All weights and BatchNorm | 13 / 22 / 9 | 34 |
| Frozen backbone, frozen BatchNorm | 9 / 9 / 9 | 11 |
| All weights, frozen BatchNorm | 12 / 11 / 9 | 28 |

Small-update selection reports are retained in each recipe's
`release_decision_v2.json`. The head-only 0.15 update passed the frozen retention
gates: reviewed false NG 38 -> 36, no misses among the 75 reviewed NG, unchanged
old HA and legacy confusion matrices. Its grouped OOF false NG is also 36 with
zero reviewed NG misses. This is a modest improvement, not a resolution of the
false NG problem. All 28 White tray verdicts remained unchanged in the actual
source integration replay. Geometry was unchanged relative to the point-fixed
reference; debug/production parity was checked on both cameras.

The all-weights/frozen-BatchNorm 0.10 update reduced fitted false NG to 32 (OOF
34), but was rejected because legacy no_disk-as-OK rose from 4 to 5. The full
network recipe's grouped selection retained the baseline. No retention limits
were relaxed. All previous checkpoints, reports and fitted models remain intact.

PT/ONNX parity for the head-only candidate also passed on the 32 lowest-margin
newly reviewed crops, at batch sizes 1, 2 and 32: maximum probability difference
9.33e-6, zero argmax mismatches. The complete SERVICE suite passed all 96 tests
using the training Python environment plus the SERVICE virtual environment's
runtime dependencies. The SERVICE-only environment lacks optional PyYAML/training
packages; the complete test run used the combined installed dependency paths.

The user then explicitly requested **200 epochs and early stopping after 100
consecutive epochs without improvement**. This supersedes the short-run limits.
`train_classifier_supplementary.py` now propagates `--epochs` and `--patience`
through the run phase to every CV and final training subprocess. The stopping
monitor remains validation safety fitness; Ultralytics 8.3.200 resets patience
only on a strict fitness improvement. The best checkpoint remains the candidate.

The new campaign is `.work/classifier_supplementary_200e_20261021/`, served at
<http://127.0.0.1:8788/>. It is managed by
`SERVICE/training/run_classifier_supplementary_campaign.py` and runs serially on
one GPU: head-only, all-weights/frozen-BatchNorm, then full network. Each recipe
starts from the same retained production baseline and uses the same frozen
5,800-crop pool and grouped folds. All 12 fits have a 200-epoch limit and patience
100. The update grid 0/.05/.10/.15/.20/.25/.50/.75/1 is recorded before the new
campaign starts. Existing short-run outputs are not overwritten.

The campaign records source snapshots, all 8,570 input crop checksums, the pool
manifest hash and installed runtime hashes. It automatically performs grouped
evaluation, final retention gates, selection by grouped CV, PT/ONNX parity and
actual White replay for a qualifying selected update. `status.json` reports
running/completed/failed states and actual epoch counts; `final_selection.json`
and `comparison.html` are produced after completion. Training and evaluation are
authorized; installed models and packaged backend remain unchanged.

### Training stopped for additional human review

On 21 October the user explicitly stopped the new campaign to review the model
OK/no_disk labels in ANH_18_10. The controller, training subprocess and its data
workers were terminated; no supplementary training process remains. The first
head-only CV fold reached epoch 64. Both `best.pt` (epoch 1) and `last.pt` (epoch
64) remain readable, with checksums in `checkpoint_state_at_user_stop.json`.
`status.json` now says `stopped`; `STOP_REQUESTED.json` records that automatic
restart is disallowed and further training requires user authorization.

The editable review is <http://127.0.0.1:8784/?label=ok_no_disk&status=unreviewed>.
It filters the original model OK/no_disk proposals: 1,940 OK and 747 no_disk,
2,687 initially unreviewed crops. The previous 113 NG-proposal decisions remain
in the same decisions file and have a backup at
`review/snapshots/decisions_before_ok_no_disk_review_20261021.json`. The immutable
`items.json` and original predictions are unchanged. Selecting a label saves a
separate human decision; page confirmation applies only to the reviewed page.
The frozen prepared_dataset_v1 and stopped training outputs are historical;
after review, prepare a new dataset version incorporating the new decisions.

The user subsequently requested correction-only review. The active editor now
preselects each model label, with existing human overrides taking priority. It
shows all OK/no_disk proposals by default so corrections do not shift pagination.
No per-image/page acceptance is required. Only edits create saved decisions;
unchanged defaults keep model provenance until review completion is declared.
The prior 113 decisions were preserved, with no synthetic human confirmations.
Editor: <http://127.0.0.1:8784/?label=ok_no_disk&status=&mode=corrections>.
Training remains stopped pending further user authorization.

## Review completed; fresh YOLO pretrained run authorized

Later on 21 October the user declared the full correction-only review complete
and authorized training again, recommending official YOLO pretrained weights and
200–300 epochs, evaluating the best checkpoint. The new run uses a 300-epoch
limit and retains the requested patience 100. The stopped HA-initialized campaign
is preserved and is not resumed.

`review/review_completion_20261021.json` binds that declaration to the exact
immutable items and current decisions checksums. There are 169 saved decisions:
168 included decisions plus one excluded crop. All 2,631 remaining defaults are
accepted under the user's declared completion, with provenance
`new_human_reviewed_model_default`; individual clicks are not fabricated. The
new accepted class counts are 79 NG / 787 no_disk / 1,933 OK (2,799 total).
The new review discovered four true NG originally predicted OK and 39 empty
slots originally predicted OK. One original OK proposal is excluded.

The frozen pool is
`.work/classifier_factory_20261020/prepared_dataset_v2_reviewed_20261021/`, served
at <http://127.0.0.1:8790/?origin=all_pool>. It contains 5,799 training-pool crops
(1,130 NG / 1,409 no_disk / 3,260 OK) and 2,770 legacy validation-only crops.
Old labels and hashes are unchanged; the review completion and explicit decisions
are snapshotted. `validation_reviewed_pool.json` records the verification.

The active run is `.work/classifier_yolo_pretrained_reviewed_20261021/`, served at
<http://127.0.0.1:8789/> and implemented by
`SERVICE/training/train_classifier_yolo_pretrained.py`. Its official source is
<https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-cls.pt>,
SHA256 `c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7`.
The downloaded model has the expected ImageNet 1,000-class head and YOLO11n-cls
architecture. The initialization callback verified 234 transferred tensors
against the official weights, correct target class order, and all parameters
trainable. No HA checkpoint or weight blending initializes the new model.

The three CV fits hold out whole new capture groups, yielding 900 / 900 / 999
new validation crops. Exact duplicates remain in the same fold. Old HA and
legacy-train examples stay in train. Their fitted retention is evaluated
separately after checkpoint selection. The final fit trains the whole 5,799
pool; **only legacy validation selects its best checkpoint**, without new/HA
resubstitution views in validation. New grouped OOF estimates and final fitted
metrics remain distinct. Validation selects best and is not an independent
blind test. Newly reviewed labels were reviewed with original predictions visible.

All four fits use official pretrained initialization, full-network training,
AdamW lr0 .001, cosine schedule, 300 epochs and patience 100, batch 64, RGB
runtime stretch 224 and the existing mild augmentations. Save best/last and
periodic checkpoints. The process runs serially on the GPU, exports best to
ONNX, evaluates per-crop/old-domain errors, checks PT/ONNX parity on ambiguous
new crops, and replays all 28 White images with hardware disabled. Source
snapshots, input hashes and runtime hashes are retained. Models remain separate
from production; deployment has not been authorized.

Re-evaluating the old HA model on the completed new review gives 38 false NG,
4 missed NG, 39 no_disk-as-OK, and 92 total errors among 2,799 accepted crops.
Old HA retention remains 1 total error / 0 missed NG; legacy validation remains
20 total errors / 5 missed NG / 4 no_disk-as-OK. These are the locked comparison
numbers for the new run. The complete SERVICE suite passed all 99 tests,
including new checks for grouped CV separation, final validation isolation and
rejection of legacy empty-slot regression.

## Completed evaluation of official-pretrained run

The four fits completed with early stopping: CV 115 / 120 / 115 epochs,
final 112 epochs. The best epochs were 15 / 20 / 15 and final 12. Thus the
300-epoch cap and patience 100 were honored; the final best is not epoch 112.
Automated export, per-crop evaluation and all 28 White replays completed.

**Reject this candidate for deployment.** On the new grouped OOF predictions,
false NG rose from 38 to 46 and missed NG rose from 4 to 5; empty-as-OK fell
39 -> 23. Total OOF errors were 91 versus baseline 92, so overall agreement
conceals the worse NG behavior. Final fitted new-data errors fell 92 -> 39
(false NG 38 -> 19, missed NG 4 -> 1, empty-as-OK 39 -> 2), but those are fitted
diagnostics. Old HA retention errors rose 1 -> 19. Legacy validation errors
rose 20 -> 509: false NG 10 -> 429, missed NG 5 -> 14, empty-as-OK 4 -> 5;
accuracy fell 99.278% -> 81.625%. All three CV candidates also failed the
frozen retention gates. No safety limits were relaxed.

The final best was chosen by configured safety fitness, which prioritizes
missed NG. Even the highest legacy accuracy logged in the final run was only
94.801% at epoch 13, with 55 missed NG, so accuracy-only checkpoint selection
would still fail the old-domain requirements.

Additional PT/ONNX parity on 16 lowest-margin legacy crops plus 16 highest-
confidence legacy false NG passed at batch sizes 1, 2 and 32: zero argmax
mismatches, maximum probability difference 6.35e-6. The original new-crop
parity also passed (4.23e-6). These checks found no export discrepancy.

White replay geometry/count fields stayed unchanged and debug/production
matched on both cameras, with hardware disabled. Seven tray verdicts changed:
two NG -> OK and five NG -> WARNING. The two NG -> OK frames have 100 reviewed
OK crops and 100 model OK predictions. Two WARNING transitions have 100
reviewed OK crops but one false no_disk prediction each. Crop annotations
do not supply full spacing/segmentation truth, so this is not whole-flow
accuracy. One fitted true-NG crop is predicted OK with 0.9968 confidence:
Cam2_Image_20261018104018106 / row1_bottom / slot11.

A concrete configuration concern is that fresh training reused only the
1,000 legacy replay anchors (600 NG / 300 empty / 100 OK). With 10% sampling
allocated to legacy and 10% of that to OK, legacy OK receives only about 1%
of sampled examples. This is a plausible contributor to old-domain degradation,
not established causality. A next experiment should examine the full legacy
training set, exclude validation duplicates, and rebalance domain/class sampling
before another fresh-pretrained run. No additional experiment is started by
this evaluation, and the deployed runtime/model remain byte-identical.

Read-only evaluation report and gallery:
<http://127.0.0.1:8789/evaluation_review_20261021/>.
It contains 814 old/new error-or-correction cases, native crop/context assets,
fold/epoch evidence, White changes and summary JSON. The final ONNX SHA256 is
`e382eeeabf299e12a3f51e37c7ff2fc85bce169b178cca341dc567dbec790f6f`;
the candidate remains stored separately and is not recommended for deployment.

## Label consistency audit requested after evaluation

The user asked whether training labels were mixed up. A read-only audit checked
all 8,569 pool images against source/review snapshots, every ImageFolder sample
and checksum in all four actual datasets (34,276 records), class-directory
membership, exclusions, frozen/live review differences, and PT/ONNX name order.
No ingestion, index permutation, crop mapping, missing correction or hash error
was found. The class order is consistently ng=0 / no_disk=1 / ok=2.
All 198 differences from old source directory labels are accounted for by saved
human reviews (172 old HA and 26 legacy validation), not conversion mistakes.
No identical decoded-pixel/180-degree/flip-equivalent images carry conflicting
labels or cross the training-pool/legacy-validation boundary.

However, similarity search found near-identical images carrying opposing labels.
Features use mean-centered 32x32 grayscale cosine similarity >=.99, followed by
96x96 raw grayscale RMSE <.03. This yields 230 candidate pairs / 287 unique
images, including 167 new/new, 20 HA/HA, 17 legacy-train/legacy-val, seven
legacy-train/train and 19 legacy-val/val pairs. These numbers are **candidate
pairs, not proven label-error counts**: a thin disk rim can create a legitimate
class change at low global RMSE. Four priority pairs were visually inspected
with native crops and available context and are shown first in the report.

One specific new-data concern is Cam2 row3_top slot25 at
Image_20261018112110963 (explicitly saved OK) versus
Image_20261018112236010 (accepted default no_disk): both contexts look very
similar. Old HA row1_bottom slot24 at Image_20260918133449979 (OK) and
Image_20260918133502603 (NG) is another suspicious pair. Two priority legacy
train-OK / validation-no_disk pairs are also extremely similar; one validation
label was explicitly changed from OK to no_disk in the previous review.
These require consistent interpretation of the target groove; no label was
automatically overwritten based on appearance or model predictions.

The audit also evaluated both classifiers on the actual 1,000 legacy train
anchors. Old HA matches 998/1,000; fresh best matches 943/1,000. Of 100 legacy
OK examples, fresh best predicts 23 NG and nine no_disk, whereas old HA predicts
all 100 OK. Thus label consistency and weak learning/sampling are separate
concerns; label suspicion does not by itself explain the whole regression.
Two legacy NG samples predicted OK by both models were visually inspected but
remain ambiguous, not declared mislabels.

The audit report is <http://127.0.0.1:8789/label_audit_20261021/>.
`integrity.json`, `legacy_training_predictions.json`, `comparison_pairs.json`
and `similarity_protocol.json` retain the evidence. The report is read-only,
with links to the existing new-image editor when a user wants to correct a
specific new crop. Training inputs, saved decisions and installed models are
unchanged by this audit. No further training was started.
