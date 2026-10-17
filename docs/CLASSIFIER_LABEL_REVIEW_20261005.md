# Classifier HA label proposals — 2026-10-05

Subsequent status: user completed all 2,000 labels and authorized training. See
[CLASSIFIER_FINETUNE_20261005.md](CLASSIFIER_FINETUNE_20261005.md) for frozen labels,
training/legacy retention evaluation and the non-deployed candidate. The remaining sections record
the original pseudo-label preparation state, not current training status.

Status: labeling/review ONLY. No classifier training, deployment or modification of old training data.
User must confirm corrected labels before any training. Confidence is a model score, not measured accuracy.

## Provenance

Old training script: `D:/huynhvc/WORKING/TRAINING/training_cls/train_disk_3.py`.
Experiment: `DISK_3/disk_22_05_nano`; `weights/best_cls.onnx` is byte-identical to
`SERVICE/config/models/c_p.onnx` (SHA256 `0be56f7a3442745e8a31c82601f84d04c80eb6a495d48729f729768e504c8200`).
ONNX metadata and `best.pt` agree on class order: `ng`, `no_disk`, `ok`.
Old `dataset/disk_3` counts: train ng=1691, no_disk=753, ok=3136;
val ng=821, no_disk=512, ok=1437. The old tree is read-only and unchanged.

Inputs: 20 real HA White frames / 2,000 native crops from the current pipeline,
using the retrained point detector and crop width 1.8. Top-face crops are rotated 180 degrees.
Every crop was reconstructed from raw HA rotation + saved rectification matrix and verified pixel-exact.
No 224x224 crops are exported. Model preprocessing alone uses runtime RGB stretch-resize to 224x224,
float32 /255, CHW, batch size 2. Fresh inference matched all original crop-export predictions.
No confidence threshold drops inputs: all 2,000 are retained for review.

Label rules: `ok`=one disk in TARGET groove; `ng`=more than one; `no_disk`=zero.
Neighbor groove disks and two rims/highlights of one disk must not be counted as multiple target disks.
25 is the groove count, not a forced disk count or label.

## Output / review

Root: `.work/classifier_label_review_20261005/`.

- `proposed/ok`: 1,273 native PNGs.
- `proposed/ng`: 383 native PNGs.
- `proposed/no_disk`: 344 native PNGs.
- Filename prefix = model confidence to 6 decimals; alphabetical filename order is low-to-high.
- `proposals.csv`: initial model label + full confidence, sorted low-to-high.
- `items.json`: all three class probabilities, stable crop IDs, provenance/hashes, geometry, model score.
- `context/`: neighboring area with target crop outlined in yellow; previews ONLY, never training images.
- `decisions.json`: human decisions written by the review server, separate from immutable proposals.

Range: 0.364638..1.0; scores <0.5: 25; <0.7: 96; <0.9: 202.
High score does not certify a correct label, especially with wider crops and new camera distribution.
All proposals start unreviewed; there is no automatic confirmation or train/val split.

Review URL: `http://127.0.0.1:8776/`. Default: unreviewed, confidence ascending.
Filter by proposed class, camera, confidence ceiling, frame/region. Click a crop for neighborhood/full-frame context.
Choose correct class; `uncertain` retains an unresolved item, `exclude` marks unusable images.
Human label corrections do NOT change the original prediction or its confidence.
In detail view: keys 1=ok, 2=ng, 3=no_disk, U=uncertain, X=exclude; saved labels advance to next image.
Page confirmation explicitly confirms ONLY unreviewed images on that visible page after user review.
No button authorizes training. Tell the assistant when review is complete to audit remaining uncertainty
and obtain final dataset confirmation. Use the UI to save corrections, rather than moving proposal files.
Training later must exclude unreviewed/uncertain/excluded items and group splits by source frame/session
to avoid leakage. HA has no independent real NG session; quality must not be overstated.

Restart server from workspace root:

```powershell
& ./SERVICE/.venv/Scripts/python.exe -B SERVICE/training/serve_classifier_review.py --output .work/classifier_label_review_20261005 --port 8776
```

Server binds loopback only, validates dataset digest and IDs/classes, checks Origin,
uses locked atomic saves, and resumes decisions on restart. Tests use isolated fixtures, not real review labels.
40 service regression tests ran successfully (4 optional torch/ultralytics tests skipped under production venv).
Actual browser automation could not initialize (Windows sandbox helper error on both attempts);
HTTP serving and JS syntax/fixture behavior are tested, but automated visual browser QA is not claimed.
