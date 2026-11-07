# Fresh classifier on canonical HA/new holdout

## Completed result

Training stopped at epoch 217 after 100 epochs without improvement; best is
epoch 117. Evaluation initially failed on a missing output directory. The
directory was created and evaluation/replay resumed without retraining or
changing frozen inputs/code/checkpoints/fitness/gates. The source runner now
creates output parents and has a regression test; all 128 service tests passed.

HA validation: 57/997 errors, including 51 missed NG (47 as OK, 4 as no_disk),
versus 0 errors for the installed HA model. New validation: 4/400 errors versus
10; false NG 0 versus 5; missed NG unchanged at 1; no_disk-as-OK 3 versus 4.
Training crops matched 100%, while HA holdout generalization remained poor.
All HA errors were in the held-out Cam2 burst; HA training contains Cam1.
The installed HA model previously learned HA, so this is not a blind symmetric
model comparison. Validation is checkpoint-selection data, not an independent test.

The fixed recommendation gates failed on HA. Installed baseline retained,
no deployment. PT/ONNX argmax mismatches 0 on batches 1/2/32; maximum probability
error 3.46e-6. All 28 White replay frames completed with unchanged geometry/counts,
both-camera production/debug parity and no hardware calls. Protected runtime
hashes stayed unchanged. Tray-level truth is unavailable, so replay is not an
accuracy measurement. The heartbeat is paused; do not resume training.

Reports: <http://127.0.0.1:8793/comparison.html> and
<http://127.0.0.1:8793/error_review.html>. Run evidence: `RESULTS.md`,
`completion_verification.json`, `evaluation_directory_repair.json`.

The user requested training and evaluation after selecting the curated source and
removing legacy. The exact revision-9 authorization is saved in the canonical
manifest, with incomplete consistency review acknowledged. No deployment is
authorized. The old robust campaign remains completed and unchanged.

Run: `.work/classifier_ha_new_holdout_20261022/`.
Dashboard: <http://127.0.0.1:8793/>.
Entrypoint: `SERVICE/training/train_classifier_canonical_holdout.py`.

Inputs are the exact 3,386 train and 1,397 validation crops in `DATASET_CLASSIFIER`.
The manifest and its SHA-256, crop copies, grouping metadata, code and model
references are frozen inside the run. They are immutable run evidence, never an
editable label source. No legacy, other fragmented pools or user-deleted crops
are added. Changes later made in the canonical editor do not affect the run.

Initialization is official ImageNet YOLO11n-cls (1,000 original classes), SHA-256
`c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7`.
Initialization verification found 234 shared tensors identical and all parameters
trainable. AdamW lr 3e-4, cosine schedule, dropout 0.03, batch 64, seed 1022,
maximum 300 epochs, early stopping after 100 epochs without strict fitness
improvement. The best checkpoint, not last, is exported and evaluated. Resume
restores the saved early-stopping clock and requires checkpoint/state agreement.

Sampling uses HA/new at 50%/50%, with NG/no_disk/OK 30%/30%/40% in each domain.
Only training examples enter the sampler. Image preprocessing matches runtime:
OpenCV BGR-to-RGB, stretch 224, float32 /255. Augmentation preserves the target
slot: horizontal flip, mild exposure/contrast/gamma/noise/blur; no spatial crop,
erasing, rotation or mixing. Validation has no augmentation.

Best selection is the arithmetic mean of per-origin validation macro-F1, balancing
NG precision and recall and preventing the larger HA burst from masking new data.
No fitted training rows enter fitness. Recommendation for further review requires
no increase in NG misses, no_disk-as-OK, false NG or total errors in either
validation origin and a strict improvement in mean macro-F1. Gates stay fixed.
Runtime baseline is retained in the installed service regardless of results.

The pipeline compares frozen baseline/candidate predictions on validation and
training separately, creates error review, checks PT/ONNX probability/argmax
parity and runs 28 source White frames with hardware disabled. Crop/frame/group
leakage checks and crop hashes passed before training; 127 service tests passed.
All runtime model/config/binary hashes are protected and checked after replay.

Limitations: validation is checkpoint-selection data, not an independent test.
The collection was used in prior development. HA has only two capture bursts;
the holdout contains one camera burst. New-data validation has only eight NG
examples. Some label consistency review remains incomplete. Report errors and
uncertainty honestly, without relabeling or relaxing gates to improve metrics.

The existing same-chat heartbeat monitors this new run quietly and pauses when
complete or user-stopped. Inspect real parent/child processes before any resume.
Never restart if `STOP_REQUESTED` exists. The old campaign must not be resumed.
