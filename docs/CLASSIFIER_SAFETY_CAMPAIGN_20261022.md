# NG-priority optimization authorized on 22 October 2026

The user explicitly authorized autonomous data augmentation, AI generation,
training and optimization, prioritizing missed defects first and false alarms
second, with review planned tomorrow. This authorizes a new bounded campaign,
not restart of any completed run and not deployment.

Run: `.work/classifier_safety_campaign_20261022/`.
Dashboard: <http://127.0.0.1:8797/>.
Entrypoint: `SERVICE/training/train_classifier_safety_campaign.py`.
The receipt is bound to the current canonical revision. CanonicalDataset's
authorization API records the receipt; labels/splits are unchanged. Training
must call active_dataset, audit canonical integrity and freeze the exact
manifest/hash and images before fitting. Old reports retain their old truth.

## Predeclared methods and objective

All starts use the verified HA parent with the three-class head preserved and
frozen BatchNorm running statistics. Four methods, each max 300 epochs with
patience 100:

- Final linear layer only, learning rate 3e-5, old-domain KD strength 4.
- Last two model blocks, learning rate 5e-6, old-domain KD strength 4.
- All weights, learning rate 3e-6, old-domain KD strength 4.
- Same full-model method plus AI-derived photometric augmentation.

Two methods with highest NG-priority validation fitness are repeated at seed
2027; first fits use seed 1022. Six full fits maximum, after short implementation
smoke checks. A stopped job never resumes automatically. Parent processes reap
children in finally blocks before launching another job.

Checkpoint fitness is fixed before training:

`-1e10 * NG_excess -1e7 * NG_misses -1000 * false_NG -no_disk_as_OK -.001 * errors`.

NG_excess sums regressions against the installed HA baseline across the three
sources and four HA/new camera subsets. Other terms sum the three disjoint
sources. This changes best selection from the earlier macro-F1 recipe; it does
not loosen recommendation gates. Gate requires no increase in NG misses,
no_disk-as-OK, false NG or total errors in any source/camera and strictly better
weighted macro-F1. Eligible models rank by missed NG, false NG, no_disk-as-OK,
errors and weighted F1. If none pass, the installed HA model is retained.

Sampling is HA/new/legacy 40/45/15%. HA/new cameras each receive half their
source budget. Classes NG/no_disk/OK: HA 35/25/40%, new 40/25/35%, legacy
70/15/15%. Hard-example weights of at most 4 are computed only from train rows
using installed HA and the previous fine-tune, normalized inside each stratum.
Supervised CE gives NG weight 2. KD is masked to teacher agreement with the
canonical label at confidence >=0.6. For new data it only applies to true NG,
with strength 1; old domains use strength 4. Labels never come from the teacher.

Each fitted best is compared at fixed HA-parent weight interpolation fractions
0.25/0.5/0.75/1.0. The previous fine-tune is also compared at fractions
0.1/0.25/0.5/0.75/1.0. These are single-network artifacts, no runtime routing or
threshold change. Fractions are fixed before outcomes and cannot be extended
to chase validation metrics. All candidates, including failures, are reported.

## AI and geometry preservation

Six earlier ImageGen previews are frozen with source train IDs/hashes and exact
prompts. Raw generated scenes have unconfirmed label/geometry fidelity and do
not enter training as labeled examples. Instead the AI method extracts a smooth
8x8 illumination-ratio field, clipped to 0.8–1.2, and applies the field only as
per-pixel multiplication to the canonical train image, on the fly. There is no
warping, new edge, copy-paste object, crop, inpainting, synthetic ground truth,
separate label overlay or additional active dataset. The AI field is applied
at probability 0.35 and randomized strength 0.3–1.0.

All methods use mild geometry-preserving runtime augmentation: horizontal flip,
bounded brightness/contrast/gamma, small sensor noise and occasional mild blur.
No spatial crop/translation/rotation/erasing/Mosaic/MixUp is used. Augmentation
realism and generalization remain empirical questions; geometry-preserving
transforms do not prove factory performance.

## Verification and operation

Frozen code and campaign/job protocols are SHA-256 checked. Job image folders
are immutable hard links to frozen run copies, never canonical label sources.
Resume restores the strict improvement clock and requires matching checkpoint
epoch. Frozen weights and BN buffers are checked each save. Final best receives
PT/ONNX parity on all 1,605 real validation crops, hardware-off White replay on
28 frames, frozen input/code and installed runtime hash checks.

Only one GPU fit runs at a time. A heartbeat checks real parent/child state,
status/logs and STOP_REQUESTED, stays quiet during normal progress, reports
completion or actionable failure and pauses after completion/user stop. Recovery:

```powershell
C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe -u -B SERVICE/training/train_classifier_safety_campaign.py --phase campaign --root .work/classifier_safety_campaign_20261022 --resume
```

Resume only after verifying no parent/child trainer remains, no user stop, and
no STOP_REQUESTED. Never resume completed historical runs. No deployment.

HA validation is retention because the parent previously learned HA. The fixed
70/30 split shares frame context across different target groups. Validation has
been viewed and used for repeated development; it is not independent testing.
Label consistency review remains partial. Report exact snapshot agreement,
NG misses and false alarms per source and camera, not a production zero-miss
guarantee or a claim that training confirms labels.

## Completion — 22 October 2026, 18:48 Asia/Saigon

All six fits and 29 predeclared single-network interpolation candidates finished.
One candidate passed the unchanged gates: `full_anchor_a025`, 75% HA parent
and 25% full-anchor checkpoint. The full-anchor source fit ran 105 epochs with
best at 5. No additional fractions, fits or threshold searches were added.

| Source | Count | Errors, HA → candidate | Missed NG | False NG | no_disk → OK |
| --- | ---: | ---: | ---: | ---: | ---: |
| HA | 570 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| ANH_18_10 | 854 | 27 → 26 | 0 → 0 | 14 → 12 | 11 → 11 |
| Legacy | 181 | 1 → 0 | 1 → 0 | 0 → 0 | 0 → 0 |
| Total | 1,605 | 28 → 26 | 1 → 0 | 14 → 12 | 11 → 11 |

OK-as-no_disk rises 2 → 3; this is included in the total errors. Both HA cameras
retain zero errors. New Cam1 errors 19 → 18, false NG 10 → 8; new Cam2 errors
8 → 8, false NG 4 → 4. Improvement is modest. The AI branch was evaluated but
not chosen. Current canonical labels remain revision 10 with partial review.

Fit epoch/best pairs: fc_anchor 102/2; late_anchor 114/15; full_anchor 105/5;
full_anchor_ai 101/1; late_anchor_seed2027 113/13; full_anchor_seed2027 104/4.
The late-anchor strict improvement clock was 14; a tied best checkpoint saved
at 15, both preserved in the completion metadata.

Final verification passed: PT/ONNX argmax matches all 1,605 real validation
crops (max probability error 4.59e-6); White integration replay covers 28 frames
with hardware off, geometry/counts unchanged and both-camera debug parity.
Frozen input/code/protocol/AI-source and installed runtime hashes match.
The post-run canonical integrity audit checks 5,383 records and passed.
No model was deployed; installed HA remains. The completed heartbeat is removed.
Do not resume this completed campaign.

- Comparison: <http://127.0.0.1:8797/comparison.html>
- Candidate-only validation errors: <http://127.0.0.1:8797/val_best_errors.html>
- Epoch summary: <http://127.0.0.1:8797/fit_summary.json>
- Verification: <http://127.0.0.1:8797/completion_verification.json>
