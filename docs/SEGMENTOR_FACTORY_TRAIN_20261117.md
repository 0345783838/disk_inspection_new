# Factory segmentation supplementary fit —17November2026

The user completed review and explicitly authorized checking/training, then
clarified unrestricted object overlap into a single whole-image union and
balanced old/new training followed by comparison against the old model.

Run `.work/segmentation_factory_train_20261117/`, dashboard8811. This is a new
segmentation run; never resume/repeat a completed October campaign. Source
review server8810 and classifier canonical dataset are separate. All344new
bands are reviewed with geometry checked;62selected-instance erasure regions
are honored by the shared Canvas renderer. Labels/geometry are frozen in
`input/`; no new images or artificial geometry/labels are generated.

All172White frames produce2,064tiles, including correct images and76empty
tiles. New split:122frames/1,464tiles train and50frames/600tiles val
(70.93%/29.07%), with paired cameras sharing the split. Each camera has732train
and300val tiles. New backgrounds50train/26val. All repeated capture groups,
exact decoded/flip aliases, and212similar-appearance candidate pairs are
conservatively connected before split optimization. Grouped distribution MILP
completed with0gap. These are heuristic groups, not verified physical tray IDs;
no claim of independent tray/session acceptance testing.

The original October dataset is retained:2,364train/261val, with labels/splits
unchanged. Combined3,828unique/materialized train records,861val. Each epoch
visits every train record once, then900new train-only hard references,450per
new camera, yielding exactly2,364old/2,364new references(4,728total). The hard
pool uses only top25% training pixel errors within each new camera. Validation
never enters hard mining. Historical materialized photometric/synthetic data
keep their original provenance; new data contains no AI augmentation.

One fresh fit from current `native_union_hard_best`, PT SHA
`10bcc17c0c68be3d54e5049ca651f38ff60348d115bf9deb5bd72501ff1e8d29`,
ONNX SHA `133945894b62f6dd9bc46baf80070947c4854dd488e64b181c28d035e8ead2c0`.
Only proto/cv4 mask branches learn, all other parameters and all BN running
statistics frozen. Same native512/RGB/letterbox/core union supervision as the
previous successful hard fit: calibrated binary BCE+.1Dice, weight.5, plus
retained instance/background loss. Overlap is allowed; exact NPZ union pixels
are truth, proxy polygon paths never define loss pixels or physical disk counts.
LR1e-4, max40epochs/patience10, batch8, seed1117; bounded HSV only, no geometry
augmentation. CUDA AMP forward/backward preflight passed with new positive,
new empty, old HA and October-reviewed examples, without an optimizer step.

Validation fitness weights mean new-camera strict union F1(.80), mean HA-camera
instance F2(.10), October-reviewed union F1(.05), old demo F2(.025), old-demo
background specificity(.025). Original9gates and native-parent retention gates
remain, with October-camera union retention/new-background checks added before
baseline metrics. Evaluate learned best and fixed25/50/75% parent blends. No
extra fits or threshold search. If no candidate passes, installed parent is
retained and the best learned diagnostic is still reported.

Full PT/ONNX parity covers all2,064new tiles plus261old val. Paired hardware-off
replay covers28historical and172new White frames, current actual SERVICE
settings, both-camera production/debug geometry/classifier checks, full344-band
union comparisons, and whole-software-flow<=2x. Count changes are diagnostics,
not independently confirmed physical counts. No deployment; preserve active
classifier HA best111, all retained models, detector, source/config/plugin and
canonical revision12. Historical protected hashes are not restored/rewritten.

Entrypoint frozen in run `code/train_segment_factory.py --phase campaign`;
`SEGMENT_SUPPLEMENT_ROOT` points to this run. Before recovery inspect actual
parent/child processes, logs, `campaign.lock`, `active_child.json`, frozen hashes
and `STOP_REQUESTED`. Never duplicate trainers or silently restart an unfinished
fit. A completed run cannot resume. Parent publishes complete only after fit,
all candidates, parity, replay and final hash audit. No monitoring automation
has been created. Review/report servers may remain available.

The first headless Edge attempt failed before exporting masks due sandbox GPU
startup; the same shared renderer was run through the in-app browser and all344
exports completed. No labels/pixels changed. CUDA preflight and subsequent
union/balancing user instructions are bound in `preflight_addendum.json`; its
hashes must be checked with final audit. Frozen protocol SHA
`356393a9f74ffb65d92478eefb1d13c79e9e951c29a80d17f1612ca6d2ceac94`.

## Tested technical loader repair

The first attempt failed after94/591batches of epoch1, with no committed epoch
or checkpoint. Ultralytics had removed same-class/same-bbox records even when
their exact overlapping masks differ. The formatter rejected the missing
instances instead of silently changing truth. Failed fit/logs/status/protocol
are preserved in `repairs/duplicate_bbox_loader_20261117/` and are not quality
results or extra search candidates.

The exact-mask dataset loader now restores the original proxy records for
NPZ-backed rows if verification deduplicates their boxes. Full4728train and861
val loader counts match exact masks, and CUDA AMP forward/backward passes on
the affected train/val cases. No image, annotation, mask/union pixel, split,
sampler, loss, gate, parent or installed runtime hash changed. With no optimizer
checkpoint available, the same intended fit is technically retried from its
parent with the same seed; one valid fit remains planned. Receipt binds the
tested code repair and all unchanged hashes. Current protocol SHA
`40f06549dac594f3849dcfac1524952cd0465a81e96ff403858b25fa1209dca8`.

The user additionally asked to review historical hard cases after training,
then authorized autonomous completion/evaluation with human inspection tomorrow.
Five historically inspected filenames represent four unique frames; these are
training diagnostics, not additional validation fitness. Their reviewed union
masks/old-vs-new full-band and count comparisons must be reported separately.

## Additional learned-model assessment

All four fixed candidates failed at least one unchanged gate. The primary
release therefore retains the installed parent. Its parity/replay is a safety
fallback check; it cannot demonstrate the learned checkpoint's behavior.
`additional_diagnostic_protocol.json` consequently records a separate assessment
of the best learned candidate under the original fitness. It adds full2,325-crop
PT/ONNX parity, saved861-val predictions, a separate error gallery, and200paired
current-config hardware-off White replays/full344-band comparisons under
`learned_diagnostic/`. It changes no fit, label, split, threshold, gate, model
selection or runtime. Primary evidence remains unchanged. These diagnostics
do not turn a failed candidate into a recommended release.

## Postprocessing settings guard repair

The primary train/evaluation/replay completed without a runtime/input change.
The additional report audit then stopped on run-local Ultralytics settings.
The only semantic change was `openvino_msg: true -> false`, written by the
installed exporter after its one-time OpenVINO suggestion (exporter.py:393);
the library also changed CRLF to LF. Reconstructing the original JSON with the
original newlines reproduces its exact preflight SHA256. Both before/after
bytes, library source hash, failure status and diagnostic script versions are
preserved in `repairs/library_notice_settings_20261117/`.

No settings were restored to force an old guard to pass. A diagnostics-only
guard accepts exactly this receipt-bound transition and rejects unrelated
runtime/hash drift and an invalid prior settings hash. All original training
data, code, protocol, runtime and other addendum hashes remain exact; original
protocol/addendum hashes are unchanged. Only report postprocessing was rerun;
no training resumed. This exception does not authorize a label, training,
threshold, gate or model-selection change.

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
