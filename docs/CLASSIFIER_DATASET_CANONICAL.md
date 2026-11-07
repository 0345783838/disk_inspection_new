# One active classifier dataset

## Current revision 12: region and visual coverage 70/30

On30October the user explicitly requested repartition and fresh training.
There are5,393 unchanged labels/IDs/hashes:3,778train/1,615val. Target-slot
histories, decoded/flip/180aliases stay together;10AI and their real source
target groups remain train. Split design uses51source/camera/region/class
cells plus131fixed ImageNet/photometric appearance clusters, all now covered
on both sides. Previous split missed2cells and21clusters. No inspection model
predictions or old errors were used to choose groups. Feasible MILP solution,
not proven globally optimal. Evidence `.work/classifier_coverage_split_20261030/`.

`CanonicalDataset.save(repartition=...)` synchronizes folders/manifest/catalog/
CSV with recoverable journaled moves.2,350records change split; labels, hashes,
IDs, other provenance and exclusions are preserved. Partial review remains.
Same-session shared-frame context remains; no independent temporal/tray test.
Historical metrics stay tied to old runs; original1,605val is now mixed with
train and serves diagnostics only. See docs/CLASSIFIER_COVERAGE_20261030.md.

## Previous revision 11: ten accepted AI same-slot train NG crops

On 29 October, after reviewing the ten ImageGen candidates at gallery8801, the
user authorized "ok cho vào train bổ sung đi". Revision11 adds exactly ten NG
train crops, preserving all5,383 revision10 records, exclusions and validation
assignments. There are5,393 active rows:3,788train/1,605val. Train classes are
680NG/849no_disk/2,259OK; validation remains270/350/985. This is train-only
augmentation of the current70/30 policy, without repartitioning existing groups.

Five AI descendants inherit HA Cam2 train targets; three inherit new Cam1 and
two inherit new Cam2. All original frames/cameras/contexts identify their source
reference, not a new synthetic capture. Stable IDs begin
`ai_same_slot_20261029__`; hashes, decoded/flip fingerprints, source train group,
full prompt, raw generated ancestry and explicit user acceptance are recorded in
each canonical record. E01 retains generated parent A03. None of the seven
fine-tune missed-NG validation crops were used as generation references or added
to train. Whole-pool consistency review remains partial.

Adoption uses `CanonicalDataset.save(..., additions=...)`, synchronizing actual
folders, manifest, catalog and CSV. Evidence and before-manifest backup are at
`.work/classifier_ai_same_slot_adoption_20261029/`; all old rows are byte-for-byte
equal as JSON records, and integrity/group isolation passed. The gallery's old
screening predictions stay tied to its revision10 generation evidence. Current
labels remain exclusively in the canonical manifest and class folders.

One new supplementary nano fine-tune is authorized on revision11, from saved
best111 with low learning rate, frozen BN statistics and bounded synthetic
sampling. See `docs/CLASSIFIER_SAME_SLOT_AI_20261029.md` and
`.work/classifier_same_slot_ai_20261029/`. Old completed runs remain complete;
their metrics retain their original frozen inputs. No deployment is authorized.

## Previous revision 10: 70/30, both cameras, selected NG-heavy legacy

The latest user instruction supersedes revision 9's no-legacy / whole-camera
holdout. Preserve existing canonical labels and add a little legacy prioritizing
NG; both train and val must represent both cameras. The user explicitly changed
the proposed 80/20 ratio to **70/30**.

| Source / camera | Train | Validation |
| --- | ---: | ---: |
| HA Cam1 | 704 | 290 |
| HA Cam2 | 717 | 280 |
| ANH_18_10 Cam1 | 1,030 | 464 |
| ANH_18_10 Cam2 | 908 | 390 |
| Legacy (camera unknown) | 419 | 181 |
| Total | 3,778 | 1,605 |

All existing 4,783 IDs, labels and hashes are preserved, and all 16 user-deleted
unsuitable crops remain absent. The 600 new legacy crops come only from the
actual source train class folders in the previously selected final_all_new:
420 NG, 90 no_disk, 90 OK, deduplicated by decoded image including flips/180.
Existing canonical corrections win for existing records. No historical decision
overlay or other fragmented pool was merged. Legacy frame/camera metadata is
unavailable and remains unknown. Its current split follows this new partition,
not the meaning of its historical origin name `old_legacy_train`.

Actual fraction is 70.18%/29.82% to preserve groups. Train classes are 670 NG,
849 no_disk, 2,259 OK; validation classes 270 NG, 350 no_disk, 985 OK. Legacy
train has 291/65/63 and legacy validation 129/25/27 in that class order.

Policy `camera_target_slot_legacy_ng_v1` groups every repeated view of each
source/camera/region/slot, preserving changing labels over time within that
physical target. Decoded exact/flip/180 aliases join groups. Ten-fold stratified
group candidates were searched over 16 fixed seeds; three folds are held out,
choosing only by source/camera/class balance, without model predictions or scores.
Every source/camera/class appears on both sides. Hash/decoded/target/split groups
do not cross. **All 48 HA/new frames are shared as context across different
target slots**; the audit reports this rather than treating it as frame holdout.
This validation is same-session spatial validation, not independent tray/frame/
temporal testing. More independent captures are still needed for that claim.

Active editor supports source/camera/split filters, e.g.
<http://127.0.0.1:8791/?view=all&split=val> and
<http://127.0.0.1:8791/?view=all&origin=old_legacy_train&label=ng&split=val>.
Plan, integrity and source-preservation evidence are in
`.work/classifier_rebalance_7030_20261022/`. Previous canonical revision 9 is
inactive recovery history there. Existing model/evaluation reports remain tied
to their original snapshots, never reused as validation scores on this split.
No new training has started; partial review and revoked training authorization
are recorded. Future trainers must handle the new policy and actual legacy rows.

## Historical revision 9: user-selected source, 22 October 2026

This section supersedes the earlier scopes and layouts recorded below.
The user selected the actual folders in
`.work/classifier_yolo_pretrained_reviewed_20261021/datasets/final_all_new`,
confirmed that 16 missing crops were intentionally deleted as unsuitable,
and requested that all legacy data be omitted. No other crop/label pool was
merged. The source was inventoried by stable ID and SHA-256; actual class folders
override its stale manifest. Only present HA/new crops were adopted into
`DATASET_CLASSIFIER` revision 9. The chosen source itself remains unchanged as
historical evidence. Previous canonical data is inactive recovery history.

| Split | HA | ANH_18_10 | Total | ng | no_disk | ok |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 994 | 2,392 | 3,386 | 311 | 892 | 2,183 |
| val | 997 | 400 | 1,397 | 209 | 217 | 971 |
| all | 1,991 | 2,792 | 4,783 | 520 | 1,109 | 3,154 |

All 4,783 present HA/new source crops are represented with identical labels and
hashes; none are missing or added from elsewhere. The 1,000 legacy-train and 2,770
legacy-val crops are absent from the active dataset. Four surviving HA/new labels
differ from revision 8; the explicitly selected source wins. Those differences
are recorded, not silently reconciled from older decisions. The 16 deleted crops
remain excluded; a separate original exclusion also remains historical metadata.

Canonical folders are now standard `train/{ng,no_disk,ok}` and
`val/{ng,no_disk,ok}`. The editor writes manifest, folders, catalog and CSV together
and preserves split on label edits/exclude/restore. Read validation at
<http://127.0.0.1:8791/?view=all&split=val> and training at
<http://127.0.0.1:8791/?view=all&split=train>.

Split selection used only source labels/group coverage, never model scores.
Consecutive frames within 120 seconds form capture bursts. Decoded exact copies,
including horizontal/vertical flips and 180-degree rotations, connect groups.
Whole groups stay on one side; SHA, decoded hash, frame and capture/split group
overlap are all zero. Validation holds one HA burst and two new bursts. HA has
only two bursts, so the feasible grouped split is 29.2% validation rather than
20%. New-data validation has only eight NG crops, limiting its precision for
that error rate. These images have appeared in previous training/development;
old-model diagnostics are not an independent test of this new split.

Evidence: `.work/classifier_source_adoption_20261022/adoption_plan.json`,
`source_coverage.json`, `staging_integrity.json`, and `integrity_after.json`.
The former canonical revision 8 is recoverable in that directory but is not an
active source. Historical training/evaluation reports retain their old inputs.
Legacy-validation training entrypoints explicitly reject the new split policy;
future training must honor the current canonical train/val assignments. No new
training has started, consistency review remains partial, and old authorization
does not transfer to the new revision.

As of 21 October 2026, the user requested one consistent dataset with direct
label updates. The active folder is:

`D:\huynhvc\OTHERS\4.DISK_INSPECTION\DATASET_CLASSIFIER`

Editor: <http://127.0.0.1:8791/>. The initial view contains 230 candidate pairs
covering 287 unique images. Similar appearance is a review cue, not proof of a
wrong label. The same editor has a full-dataset view and current CSV downloads.
Defaults are **current dataset labels**, not fresh model predictions.

## Historical review scope, earlier on 22 October 2026

The user requested another full-dataset review and explicitly chose: “Dùng toàn
bộ HA + ảnh mới để rà lỗi, chưa chia lại train/val”. The editor now defaults to
all 8,569 canonical crops, with a separate diagnostic view covering all 2,000
`old_ha` and 2,799 `new_factory` crops. Legacy images remain available in full
review. Existing train/validation roles are unchanged: 5,799 training-pool crops
and 2,770 historical legacy-validation crops. No new training was started.

- Full label review: <http://127.0.0.1:8791/?view=all>.
- HA/new diagnostic scope: <http://127.0.0.1:8791/?view=evaluation>.
- HA/new disagreements: <http://127.0.0.1:8791/?view=evaluation&errorOnly=different>.

`.work/ACTIVE_CLASSIFIER_EVALUATION.json` registers this diagnostic scope and an
immutable prediction snapshot, not label decisions or another prepared dataset.
The installed HA classifier was run on all 4,799 crops with production RGB/stretch
224 preprocessing. At revision 8 it disagrees with 4 HA labels and 91 new labels.
These are review cues, not automatic corrections. The full-review view does not
display predictions. The diagnostic view compares fixed image/model-hash-bound
predictions with the **current** canonical labels, updating counts after saves;
exclusions leave the diagnostic cohort. Changed images/model invalidate their
cached predictions. Frozen generation reports retain their original revision.

All HA/new examples may have participated in past training/development. These
numbers are fitted-data diagnostics, not independent holdout accuracy. Consistency
review remains partial. Selecting a label or explicitly confirming its current
value uses the same canonical save transaction and revokes live training
authorization; it does not modify past runs.

The read-only audit `SERVICE/training/audit_classifier_dataset.py` verified all
8,569 crop hashes and 13,416 unique crop/context paths, folder roles/classes,
catalog/CSV agreement, and zero train/validation hash or frame overlap. It found
100 exact-duplicate groups within roles and no conflicting identical-image labels.
The one historical exclusion remains excluded; it has provenance but no crop in
the active pool. File integrity does not establish visual label correctness.
Evidence: `.work/classifier_dataset_review_20261022/integrity_before.json` and
`ha_new_diagnostics/summary_at_generation.json` in that directory.

Regenerate diagnostics after an image/model change into a **new** output folder:

```powershell
python -B SERVICE/training/classifier_diagnostic_review.py --output .work/<new-diagnostic-run>
```

The previously completed robust training campaign remains completed. Its
selection gates, snapshots and reports are historical and are not changed here.

## What was reconciled

The known saved classifier review sources were compared by crop identity, taking
the latest decision for overlapping IDs:

| Saved source | Decisions |
| --- | ---: |
| `.work/classifier_label_review_20261005/decisions.json` | 2,000 |
| `.work/classifier_finetune_20261005/final_evaluation/label_review/decisions.json` | 86 |
| `.work/classifier_factory_20261020/review/decisions.json` | 169 |

This yields 2,225 latest unique decisions: 2,224 match included dataset labels,
and one image was correctly excluded. No missing or conflicting ingestion of
those saved decisions was found. This is an audit of known local saved sources,
not a claim about unsaved actions or unavailable files from every past chat.

There were multiple frozen prepared datasets and run copies. That made the
current authority unclear, even though the known corrections reached the most
recent pool. The most recent pool was **moved**, not copied, from
`.work/classifier_factory_20261020/prepared_dataset_v2_reviewed_20261021/`
to `DATASET_CLASSIFIER/`. No labels changed in this migration.

Initial canonical contents: 5,799 training-pool crops plus 2,770 legacy
validation crops, 8,569 total. All 8,569 image hashes were verified and all
13,416 referenced unique image/context assets were present.

Evidence:

- `.work/classifier_yolo_pretrained_reviewed_20261021/label_audit_20261021/review_source_reconciliation.json`
- `.work/classifier_yolo_pretrained_reviewed_20261021/label_audit_20261021/canonical_migration_verification.json`
- `.work/classifier_yolo_pretrained_reviewed_20261021/source_manifest_at_training.json`
  preserves the exact old manifest associated with the completed training run.

## Direct updates

`.work/ACTIVE_CLASSIFIER_DATASET.json` is only a pointer to the active folder and
review candidates. It contains no label overrides.

- `manifest.json`: current labels, stable IDs, image hashes, train/validation
  roles, provenance and per-record label history.
- `train_pool/{ng,no_disk,ok}/`: actual current training images.
- `evaluation/legacy_val/{ng,no_disk,ok}/`: current validation images.
- `catalog.json` and `labels.csv`: current derived views, updated on every save.
- `excluded/{uncertain,exclude}/`: images removed during the current review;
  restoring a class restores their original train/validation role.

Selecting a different label autosaves directly, moving the actual image to its
new class directory and updating all label views in one recoverable transaction.
The manifest revision is the commit marker. An interrupted write is recovered
on restart; a stale tab cannot overwrite a newer revision. Exact byte-identical
copies receive the same label. Similar images remain independently labeled.
No new `decisions.json` is used as a separate layer.

The old standalone editor is disabled when the registry exists. Ports 8784 and
8790 redirect to 8791 and reject old write endpoints. The old dataset viewer on
8785 links to the canonical editor. Historical reports remain evidence of the
labels and models used at that time; they are not live label views.

## Training state

No training or deployment was started by the migration. Later on 21 October,
the user explicitly authorized autonomous training/evaluation with the current
labels despite not completing the consistency audit. This is recorded as
`partial_review_training_authorized`, never as a completed manual review.
The receipt is bound to the canonical revision and revoked by a later label edit.
An already started run uses its immutable snapshot, including the eight saved
corrections present at authorization. Current campaign status is registered in
`.work/ACTIVE_CLASSIFIER_TRAINING.json`; report URL is <http://127.0.0.1:8792/>.
Future training must resolve the registered canonical folder and authorization.
The supported training entry points reject retired dataset paths or an
unfinished review without an explicit revision-bound user override. Each new run must preserve its exact source manifest/hash;
run copies remain historical, while all future corrections stay in this folder.

Restart the editor from the project root, using a Python runtime available on
the machine:

```powershell
python -u -B SERVICE/training/serve_classifier_dataset.py --port 8791 --legacy-ports 8784,8790
```
