The user explicitly authorized autonomous auxiliary geometry labeling, training
and evaluation, with human correction afterward:

> xong mày thử tự label theo ý của mày rồi train thử và đánh giá giải pháp mới xem thế nào nhé, mai tao kiểm tra, có gì ko ổn tao sẽ sửa sau

On04November the user resumed the unfinished work. This is a fresh trial at
`.work/classifier_groove_ownership_20261104/`, not a restart of a completed run.
Entrypoint: `SERVICE/training/train_classifier_groove_ownership.py`.
Dashboard: <http://127.0.0.1:8808/> once the campaign starts.

Canonical revision12 remains the classifier label authority:5,393rows,
3,778train/1,615val, unchanged labels/IDs/hashes/splits/exclusions. A new explicit
receipt records partial review and freezes the exact canonical manifest/hash
after `authorize_training` and `active_dataset`. Derived run snapshots are
immutable evidence, not label sources. Existing SOURCE/CSV/catalog/class folders
remain synchronized. Review is not complete.

The physical rule confirmed by the user is a receiving recess **between two
adjacent protruding support peaks**. Two physical disk ends entering one recess
are a double-disk case; a rim crossing a crop may belong to another recess.
The user specifically requires reflection-obscured/partially visible grooves to
be considered. Inferred position is not disk-presence truth. Missing boundaries
must not become no_disk automatically. Legacy camera/slot provenance stays
unknown. The user's case03 skip applies only to the observability discussion;
that crop remains in canonical validation and full evaluation.

48 existing real train crops received agent-drafted auxiliary labels,16per
canonical class,41target groups/30frames. There are80peak coordinates,
31insertion points and19complete occupancy drafts (8zero,5one,6two-or-more).
29occupancies are unknown. Coordinates and counts remain
`agent_autolabel_user_authorized`, `user_verified:false`, pending review.
No validation images are used for geometry loss. Ca05 remains diagnostic val.
No class-to-count automatic mapping or new classifier pool was created.

Source: `.work/classifier_accuracy_research_20261103/groove_ownership_preparation/auto_annotations_merged.json`.
Its original manifest hash is bound to the exact same revision12 records;
authorization changes manifest headers only. The run freezes its own auxiliary
copy/hash. Later human correction does not silently alter an already frozen fit.

Two matched fits only, `control` and `auxiliary`, each max300epochs/patience100,
seed1104, batch64. Both initialize from the current +AI nano PT, SHA256
`8dd29530326ac68295c5acf8ac7aee45441cac39513f2f4eb5d9e1a590603492`.
Both have the same architecture, initialization, RGB/stretch224 input, last two
trainable backbone blocks plus classifier, frozen early backbone and frozen BN
running statistics. Parent LR3e-6, geometry LR3e-4. A zero-initialized residual
preserves exact parent behavior at epoch0; epoch0 remains eligible.

The small branch combines28x28early features and upsampled trainable late
features, predicting L/R peaks, two ordered disk entry/trace pairs, visibility,
ownership and0/1/2+occupancy. Its predictions/embedding feed the class residual;
no annotation truth enters inference. This initial trial uses the same local
crop and does not implement raw-context lattice reconstruction. Fully hidden
geometry may remain unresolved. The model has approximately50kextra parameters.

Both fits see every3,778train crop once each epoch, plus3,778balanced extra
draws and192aux-reference draws,7,748total. Extra budgets are HA/new/legacy
40/40/20, equal cameras within HA/new, classes30/30/40 and legacy70/15/15.
The full-epoch mixture differs from those extra budgets; actual sampling counts
are recorded. Accepted AI10and legacy600 remain in canonical classification
training. Mild photometric/horizontal flips preserve target-slot class; point,
instance and neighbor labels transform together.

Both methods use NG-weighted CE2/1/1 and teacher KD only when the transformed
parent prediction is correct and confident. Auxiliary adds masked normalized
Gaussian point CE0.3, visibility0.1, occupancy0.2, ownership0.1. Clear agent point
weight0.5, tentative0.25; estimated/unknown coordinates have zero point loss.
Unknown auxiliary targets yield zero auxiliary loss/gradient while canonical
classification rows remain in training/evaluation.

Checkpoint selection is a strict tuple: per-source/camera NG regressions,
all missedNG, falseNG, empty-asOK, errors. No train row enters checkpoint
fitness. References are the current SERVICE +AI and retained HA best111.
Release recommendation requires zeroNG and unchanged source/camera retention
gates, falseNG/empty-asOK/errors no worse than either reference, and auxiliary
strictly better than matched control. No threshold sweep/interpolation/extra fit.
Report selected and best-learned results even if epoch0 is kept.

The1,615val is repeatedly used spatial development with shared frame/session
context and parent prior exposure (1,180current-val rows were in the +AI parent's
training split). It is not independent testing or a production zero-miss claim.
Full-val PT/ONNX argmax parity, metadata-bound exports,28paired current-config
White hardware-off replay, both-camera debug/geometry/count parity and whole
software flow<=2x are required. Replay excludes camera/PLC/network/file load.

SERVICE +AI, retained HA best111, detector, segmenter, source/config/plugin and
input hashes are protected. No deployment. No old runtime restoration to satisfy
historical guards. The real campaign parent must publish final COMPLETE only
after fits, evaluation, parity, White replay and completion audit finish.

Before any recovery inspect actual parent/child Python processes, status/logs,
frozen hashes and `STOP_REQUESTED`. Do not duplicate trainers or resume a
completed/user-stopped run. Use the global Python interpreter and this fresh
entrypoint with `--phase campaign --root .work/classifier_groove_ownership_20261104 --resume`
only when no parent/child remains and no stop request exists. Frozen repair
requires a documented tested repair; labels/splits/gates cannot be changed to
improve metrics. Leave the report server available for human review.

## Groove-entry ownership completion, 4 November 2026

Run `.work/classifier_groove_ownership_20261104/` is COMPLETE at10:10Asia/Saigon. No resume/repeat or extra fits. Control107epochs/best7; auxiliary100epochs/selected0(parent), learned best1 tied parent rank. Current1615development val SERVICE/control/aux errors3/2/3, missedNG1/1/1, falseNG2/1/2, empty-asOK0/0/0. Both selectedmodels: HA Cam1/Cam2 and new Cam2 zeroerrors; new Cam1 row3_top slot11 has2falseNG SERVICE/aux,1control; legacy1NGmiss has unknown camera/slot. Best111 remains4errors/3NG/0false. Auxiliary did not improve classification; unchanged gates FAIL; no deployment. Do not present epoch0 fallback as a newly improved model.

Both selected checkpoints have full1615PT/ONNX argmax parity(max2.62e-6control,3.10e-6aux). White28actual-config geometry/count/debug bothcams pass, median1.043x/max1.119x; replay covers selectedaux epoch0 only, not learned1/control latency. Full5393frozen crop/canonical/code/protocol/input/runtime/retained hash checks and independent metric recomputation passed. Parent COMPLETE and no trainer remains. Canonical revision12labels/splits unchanged, reviewpartial;48agent geometry labels awaithumanreview. Reused val includes prior parent exposure; no independent/production zero-miss claim.

Read run ASSESSMENT.md, RESULTS.md, independent_completion_verification.json, completion_verification.json and error_review.html. Evaluation marker binds prereplay release: removing only whole_white_latency and Windows serialization reproduces original receipt SHA exactly; independent verification binds finalrelease too. Recovery was resume from106after unreproduced Whitehash guard, no data/code/gate repair; original NumPy warning unreproduced on15000readonly draws. Evidence retained in recoveries/. Completed heartbeat `theo-d-i-train-u-a-khe` was deleted after verified completion; leave dashboard8808 and auxiliaryreview available. Current SERVICE+AI and retained best111 remain unchanged.
