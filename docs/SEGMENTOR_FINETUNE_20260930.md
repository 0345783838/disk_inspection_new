# HA disk-edge segmentation candidate (2026-09-30)

Follow-up: [512/native/core refinement on 2026-10-01](SEGMENTOR_REFINE_20261001.md)
describes the newer candidate and its current output galleries. This document
retains the September 30 candidate's provenance and historical evaluations.

This is a **candidate**, not a production model replacement. The active
`SERVICE/config/models/s_p.onnx` was not changed.

## Applied human review

The 21-item review is complete: 20 accepted, one corrected (R15 using eight
user-drawn centerline points). R01 and R15 were disjoint proposals for the
same physical edge in Cam2 `Image_20260918133434256`, band23; they were merged
along that path, not treated as two disks. Thickness is five pixels, inferred
from the original fragments; only the centerline was manually specified.
The correction changes 1,684 mask pixels and no unrelated edge pixels.

`apply_segment_human_review.py` validates queue/source binding, reconstructs
the original masks exactly, and writes a new version instead of overwriting:

- Masks/provenance: `.work/segmentation_finetune_20260930/curated_v3_reviewed`
- Baseline dataset: `.work/segmentation_finetune_20260930/dataset_reviewed_v1`
- Scenario dataset: `.work/segmentation_finetune_20260930/dataset_reviewed_synthetic_v1`
- Held-out scenarios: `.work/segmentation_finetune_20260930/challenge_reviewed_v1`
- Visual correction: `curated_v3_reviewed/R15_raw_before_after.png`
  (raw / previous mask / corrected mask, left to right).

The edited source is in val, not train. SHA256 comparison found all 1,338
training images AND labels identical to `dataset_synthetic_v3`; val images
are also unchanged. Only two HA val tile labels changed (tile3/tile4).
The curved edge enters a tile-overlap boundary, so val now contains 216
clipped polygon instances rather than 215; this is not a physical disk count.
Existing checkpoints were therefore re-evaluated, not redundantly retrained.
The challenge was regenerated from the revised val masks. Review snapshots
and hashes are retained in `curated_v3_reviewed/application_report.json`.

This confirms only the selected edges, not every edge in every band. Dataset
metadata keeps that partial-review scope explicit; the remaining labels are
still provisional. Original review images and decisions are retained.

## Data and label provenance

- Source: 20 raw HA captures (10 Cam1, 10 Cam2). The point dataset v3 supplies
  rotation and outer-point geometry. `build_segment_crops.py` reproduces the
  service rectification, extracts band12/band23, and uses six 5%-overlap tiles
  per band: 40 bands, 240 tiles.
- The old ONNX segmentor supplies proposals only. `curate_segment_masks.py`
  keeps long connected edge proposals and extends missing rows along local
  image ridges. It does **not** impose 25 edges: originally 885 accepted
  proposals across 40 bands, with 18–25 per band; 884 edges after the reviewed
  fragment merge. Original visual contact sheets are in
  `.work/segmentation_finetune_20260930/curated_v2/review`.
- These are **semi-automatic provisional labels**, not pixel-perfect human
  ground truth. 67 extensions cover more than 40 rows; 21 of those have weak
  median ridge evidence (<10), and eight exceed 100 rows. An edge wholly missed
  by the old model may still be missing from the labels.
- The 21 weak long extensions were collected as zoomed sheets and a JSON
  review queue in `.work/segmentation_finetune_20260930/label_risk_review`.
  They have now been reviewed as described above; this does not establish
  that entirely missed edges are absent from those bands.
- Source-grouped split: 16 HA frames train, four HA frames val (two per
  camera). Train has 192 base HA tiles; val has 48. Old demo-heavy data is
  limited to three unique crops per source frame, 387 train and 117 val.
  No source group or byte-identical old image crosses train/val.
- HA train tiles have four variants total (original, mild lower light, lower
  edge contrast, mild blur/sensor noise): 768 images. Val images are
  unaugmented. Polygon conversion preserves at least 96.3% and on average
  98.8% of curated source-mask pixels on the 240 base tiles.

## Additional disk configurations

`synthesize_segment_scenarios.py` creates 183 train-only samples from
camera-derived edge residuals: 62 missing-visible-edge, 64 near-overlap,
57 crossing. Edge offsets and faintness vary. Removal uses local inpainting;
added edges and their instance masks undergo the same geometric remap.
The current reviewed dataset is
`.work/segmentation_finetune_20260930/dataset_reviewed_synthetic_v1`. The
baseline without these cases is `dataset_reviewed_v1`. The actual training
runs used their train-identical predecessors `dataset_synthetic_v3` and
`dataset_v1`.

An image-generation edit was also piloted for near-overlap. It redrew existing
edges and changed image resolution/geometry, so its old mask could not be
trusted. It was excluded from labeled training. The controlled synthetic
cases above are approximations of defects, not proof that real empty,
overlapping, or crossed disks look exactly the same.

`build_synthetic_challenge.py` separately makes 135 scenario tiles
(45 per type) from the four held-out HA frames. None are in train.

## Training and results

Both YOLO11s-seg runs started from
`D:/huynhvc/OTHERS/an/code_train/yolo11s-seg.pt`, used 40 epochs,
320-pixel input, batch four, AdamW, small photometric/translation/scale
augmentations, no mosaic/flips, mask ratio two, and identical seeds/config.
The only data difference is the 183 scenario images.

| Val set / mask metric | Baseline | With scenarios |
| --- | ---: | ---: |
| HA original, mAP50 | 0.846 | **0.929** |
| HA original, mAP50:95 | 0.314 | **0.379** |
| Old subset, mAP50 | 0.918 | **0.940** |
| Held-out synthetic empty-edge, mAP50 | 0.874 | **0.973** |
| Held-out synthetic near-overlap, mAP50 | 0.621 | **0.846** |
| Held-out synthetic crossing, mAP50 | 0.536 | **0.739** |
| HA edge full-length recall (IoU >= .30 and both end regions hit) | 188/216 (87.0%) | **189/216 (87.5%)** |

Detailed metrics:
`.work/segmentation_finetune_20260930/eval_reviewed_baseline` and
`.work/segmentation_finetune_20260930/eval_reviewed_candidate`. These metrics
supersede the earlier provisional-label evaluations. Full-length is a tile
criterion (intersection in the first/last 12 target rows), not verified
full-band continuity or correct tray counts.

The new candidate is:

- PyTorch: `.work/segmentation_finetune_20260930/runs/with_synthetic/weights/best.pt`
- ONNX: `.work/segmentation_finetune_20260930/runs/with_synthetic/weights/best.onnx`

The ONNX output schema matches production: input `[1,3,320,320]`,
outputs `[1,37,2100]` and `[1,32,80,80]`. It ran through the service's
six-tile `OnnxSegmentor` decoder. At production confidence .5, its mean
full-band pixel IoU to the provisional masks was .538 on HA val; the old
model gets .929 **because those labels began as old-model proposals**.
This comparison is biased toward the old model and thin-mask pixel IoU is
sensitive to a few pixels of lateral shift. It must not be used alone to
select deployment. Lowering the candidate threshold to .1 raised provisional
mask recall from .697 to .751 with mean precision about .697; no runtime
threshold change was made.

An opt-in `preprocess_mode="letterbox"` was added to `OnnxSegmentor` because
Ultralytics validation uses aspect-preserving padding while the existing
service stretches every tile. The default remains `stretch` with BGR input;
the old production model produced exactly the same audit metrics after the
change. For the new ONNX, provisional-mask mean IoU/recall was .538/.697
with current stretch at confidence .5, .583/.753 with letterbox at .5, and
.593/.769 with letterbox at .2. An experimental ridge-guided edge completer in the evaluation
script raised these to .602/.787, but it may hallucinate weak continuations
and was **not** enabled in production. The optional decoder and threshold
must be validated against corrected real masks before rollout.

After the partial human review, candidate letterbox/BGR at confidence .2
has mean full-band pixel IoU .593, recall .770, and precision .718. These
remain predominantly provisional-label scores, not independent human-gold
accuracy. Runtime production settings were not changed.

## Release gate

Do not replace `s_p.onnx` yet. The HA val frames come from the same short
capture session and their masks are provisional. The synthetic challenge
cannot substitute for real defects. Before production rollout, manually
complete the remaining 40-band label audit, especially entirely missed faint
edges beyond the now-reviewed 21-item queue, then test actual
empty/extra/overlapping/crossing trays from a different session. The current
full-length criterion remains only 87.5%.

All `.work` artifacts are local and gitignored. The scripts in
`SERVICE/training` reproduce the dataset, training, and evaluation.

## Human review UI

`SERVICE/training/serve_segment_label_review.py` serves the 21 weak
extensions at `http://127.0.0.1:8768`. Each R01–R21 item has an unmodified
crop, an overlay highlighting only the selected edge, and full-band context.
Yellow marks the original proposal; cyan marks the inferred continuation.
Brightness and zoom controls affect display only.

Select accept, fix, reject, or uncertain. For fix, optionally click centerline
points on the original crop and leave a note about truncation or thickness.
Responses are saved automatically to
`.work/segmentation_finetune_20260930/human_review/decisions.json`, including
full-band point coordinates and a hash of the source review queue. Reviewing
alone does not alter the masks; the application script described above applies
the saved decisions into a separate version. The UI also exports a backup JSON.

Start it again after closing the process with:

```powershell
& 'C:/Users/Admin/AppData/Local/Programs/Python/Python310/python.exe' -u -B SERVICE/training/serve_segment_label_review.py --point-dataset .work/detect_finetune_20260930/dataset_v3 --curated .work/segmentation_finetune_20260930/curated_v2 --queue .work/segmentation_finetune_20260930/label_risk_review/queue.json --output .work/segmentation_finetune_20260930/human_review --port 8768
```
