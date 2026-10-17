# Faint-edge follow-up: corrected labels and a limited AI pilot

At the time of this experiment, production ONNX/config were unchanged.
The user subsequently requested a temporary deployment for factory testing:
see [the deployment record](SEGMENTOR_TEST_DEPLOY_20261001.md).
This follows [the first October 1 refinement](SEGMENTOR_REFINE_20261001.md).

## User-reported cases and diagnosis

- Cam1_Image_20260918130452232_band12: validation source. The existing label
  contains the thin rim around x=1133. The baseline predicts a partial-height
  proposal at confidence 0.196 and a lower-score full-height proposal. At 0.5,
  neither is accepted. Not caused by core tile fusion or area/length filtering.
- Cam2_Image_20260918133424033_band23
- Cam2_Image_20260918133409702_band23
- Cam2_Image_20260918133355656_band23
- Cam2_Image_20260918133346095_band23

All four Cam2 sources are train. Their faint central rim around x=737..741
was entirely absent from curated labels and had no matching baseline proposal
even at confidence 0.01. The corresponding visible rim is labeled in the other
Cam2 frames. Inconsistent supervision is a concrete training defect here.
The max/core comparison did not recover these targeted missing rims.

## Data changes

Root: `.work/segmentation_faint_20261001/`

- `curated_v4_faint_train`: preserve old masks; add exactly one visible rim to
  each of the four reported Cam2 train bands. No fixed disk count is imposed.
- Agent inspected the real images and supplied 8 centerline anchors per rim,
  with PCHIP interpolation and a 3-pixel strip. These are visual annotations,
  not old-model pseudo-labels, and are NOT independently user-confirmed.
- 28 existing training tile labels changed, including photometric repeats,
  shifted real crops, and unchanged regions of constructed scenarios.
- Add 192 real crop/photometric variants from 16 train-only central tiles:
  crop offsets -60/0/+60, original / mild low-light / contrast reduction /
  mild blur and sensor noise. Labels are cropped from corrected full-band masks.
- One built-in image_gen edit from TRAIN Cam2_Image_20260918133424033_band23,
  reference ROI x=600..1000. Move a faint rim to a different lateral position
  while retaining dark camera contrast, blur, imperfect highlights and texture.
- The AI edit changed geometry. Original labels were NOT reused. Only three
  interior crop windows with two identifiable rims were accepted; their
  geometry was visually re-annotated. Outer ambiguous margins were excluded.
- 12 AI-derived training tiles (one generation, three windows, four variants),
  only 0.845% of 1420 training images. Not 12 independent generated photos.
- All 165 validation images AND label files remain byte-identical, including
  48 HA tiles from four held-out raw frames. No validation image was supplied
  to image_gen or used for a training augmentation.
- One old empty-label crop plus 16 constructed erased-edge negatives; these
  are not proof of real empty-tray or real-NG accuracy.

AI assets: `ai/faint_reposition_v1.png`, `ai/usable_roi_label_review.png`,
`ai/annotation.json`, and the exact built-in prompt in `ai/prompt.md`.
Image generation was built-in, not a separate CLI/API runner.

Dataset: `dataset_faint_v2`. Counts: old train129/val117, HA train768/val48,
constructed scenarios183, real hard shifts120, constructed negatives16,
new real faint variants192, AI12. Source-group leakage assertions pass.
Geometry audit: 408 original/shifted real tiles, minimum pixel recall 0.962660,
mean recall 0.987711, mean IoU 0.968274 against source masks. This checks
polygon conversion, not whether provisional original masks are human-gold.

## Training and evaluation protocol

Warm start the previous refined512 best.pt, 20 epochs, 512 input, batch4,
AdamW lr0=0.00015, seed930, mask_ratio2, no mosaic, no flips. HA-only validation
selects the checkpoint. Production model/config are not replaced.

Compare the five targeted rims at confidence0.5, NMS IoU0.8, letterbox/RGB,
native mask decoding, six tiles/core fusion, with no edge completion or area/
length post-filter. Four cases are TRAIN; their recovery is a training
regression check, not independent generalization evidence. Cam1 is VAL but
user-directed iteration on that image also limits its independence.

Evaluate all HA/old val tiles and controlled synthetic negatives/scenarios,
using one-to-one mask matches and full-length checks. New data, corrected
labels and fine-tuning are combined changes; without ablation, an improvement
cannot be attributed specifically to AI generation. There is still no real
independent NG or another-session test set.

## Results

The first 20-epoch v2 run completed and exported static ONNX512. At the original
six-tile/core geometry and confidence0.5, four Cam2 target rims recovered with
confidence 0.754/0.710/0.778/0.746, covering 285..287 of 287 visible rows. Cam1
still failed at0.5. This is not yet a complete solution to all five reports.

### Cam1 grid-position sensitivity and second training experiment

Using the SAME v2 ONNX and unchanged image pixels, slide tile3's source window
by just four pixels. Base x=918, width341. Target Cam1 rim near x1133:

| Window shift | Best matching confidence | Visible predicted rows |
| --- | ---: | ---: |
| -12 | 0.8755 | 286 |
| -8 | 0.8685 | 287 |
| -4 | 0.8657 | 287 |
| 0 | 0.4371 | 160 |
| +4 | 0.8778 | 288 |
| +8 | 0.8703 | 287 |
| +12 | 0.8640 | 287 |

This demonstrates strong localization/grid-phase sensitivity, not simply an
invisible edge. Local Ultralytics code selects eligible anchors inside GT
boxes and uses finite-range DFL distances. Repeated roughly160-row proposals
are consistent with a fine-stride head struggling with extremely tall, narrow
objects. This is a mechanistic hypothesis, not a proof that all misses share
one cause. A fixed crop shift is not applied to production: it could move the
same failure to other rims.

Follow-up `runs/thinbox512_v3`: 10 epochs from v2, AdamWlr0=0.0001, no warmup.
`train_segmentor_thin_boxes.py` widens TRAIN detection envelopes to at least24
input pixels for tall/thin objects (height>35% image, aspect>8). Actual binary
instance MASKS remain unchanged, and validation retains original labels.
This trains box support around the rim; it does not paint thicker masks, force
disk counts, stretch predicted edges, or change inference thresholds.
Custom trainer is project-local; no installed Ultralytics file is modified.

### Final candidate and targeted recovery

Both runs completed: v2 20 epochs in0.191h, v3 10 epochs in approximately0.093h.
Selected checkpoint: `runs/thinbox512_v3/weights/best.pt` and `best.onnx`.
ONNX SHA256: `bd6f958aaa3cc1c606e91f709b0099a8920b569eed16aef155bb19ca753da534`.
Static input512, opset12. Exported model uses standard YOLO segmentation;
the custom trainer does not add an inference dependency or crop retry.

The five TARGETED rims, not every possible edge in those images, recovered at
the ORIGINAL crop geometry and confidence0.5. No inference-time crop shift,
mask extension or length filter was used.

| Case suffix | Split | Confidence | Predicted visible rows |
| --- | --- | ---: | ---: |
| Cam1_130452232_band12 | val | 0.933583 | 288/288 |
| Cam2_133424033_band23 | train | 0.922846 | 287/287 |
| Cam2_133409702_band23 | train | 0.914616 | 287/287 |
| Cam2_133355656_band23 | train | 0.899378 | 287/287 |
| Cam2_133346095_band23 | train | 0.900313 | 287/287 |

All target row coverage is1.0 with a two-pixel horizontal matching tolerance.
Mask IoU versus their provisional/agent visual labels: .770/.895/.765/.949/.836.
New masks remain thin even though their predicted detection envelopes are wider.

### Validation and precision trade-offs

Fixed confidence0.5, NMS IoU0.8, one-to-one MASK IoU match0.3. These counts are
clipped tile instances, NOT physical disk counts or end-to-end accuracy.

| HA48 val tiles /216 instances | Before refined512 | Data-only v2 | Thin-box v3 |
| --- | ---: | ---: | ---: |
| TP / FP / FN | 213 /3 /3 | 211 /4 /5 | 213 /2 /3 |
| Precision | .986111 | .981395 | .990698 |
| Recall | .986111 | .976852 | .986111 |
| Full-length matched instances | 201 | 197 | 202 |

At stricter mask IoU0.5, v3 TP207/FP8/FN9, versus before TP207/FP9/FN9.
Standard mask mAP50/95 on HA: .937144/.448037, versus before
.928941/.429791. Old-demo117 val mask mAP50/95 .934453/.468303 versus before
.940761/.470556: slight regression, not a uniform improvement on old data.
Old fixed IoU0.3 TP477/FP32/FN17 before -> TP482/FP36/FN12 v3; more recall
but also more false matches under provisional labels.

Full-band ONNX/core/RGB union metrics on8 HA val bands: before mean IoU
.732935, pixel recall .857135, precision .834485; v3 mean IoU .732901,
recall .868879, precision .823114. More pixel coverage, less pixel precision,
essentially unchanged union IoU. Tile instance metrics do not replace this view.

Controlled held-out scenarios at conf0.5 / maskIoU0.3:

| Scenario | Before TP/FP/FN | V3 TP/FP/FN | Full-length before -> v3 |
| --- | --- | --- | --- |
| Removed slot | 155/0/5 | 155/1/5 | 143/160 ->147/160 |
| Near overlap | 235/11/15 | 242/16/8 | 214/250 ->223/250 |
| Crossing | 217/16/33 | 225/20/25 | 183/250 ->193/250 |

The16 constructed erased-edge negative tiles worsen from2 FP to4 FP at0.5.
These are synthetic erasures with possible residual ghost gradients, not real
empty trays. This is a meaningful warning, not evidence of real-NG reliability.

Calibration (HA counts matched at maskIoU0.3):

| Confidence | HA TP/FP/FN | Full-length | Erased-background FP |
| --- | --- | ---: | ---: |
| .5 | 213/2/3 | 202/216 | 4 |
| .6 | 211/2/5 | 202/216 | 2 |
| .7 | 208/1/8 | 201/216 | 1 |
| .8 | 207/0/9 | 200/216 | 1 |

All five target rims have scores>.89, so none needs a low operating threshold.
Keep0.5 for the same-threshold review gallery and recall-oriented comparison.
0.6 is a possible more conservative candidate, but loses two HA tile matches;
0.7/0.8 lose more. No threshold is promoted to production without review and
real independent negative/anomaly data. Negative FP0 is NOT established.

### Review outputs, tests and provenance

- `.work/HA_faint_20261001_v3_conf05_core/index.html`: all20 White frames and
  40 band comparisons; raw overlays, original bands, predicted masks.
- `.work/HA_faint_20261001_v3_before_after/index.html`: original / before /
  after at identical confidence0.5, 512 input, native RGB, core fusion.
- Geometry is derived from PREPARED point labels, not detector predictions.
  This is not an end-to-end detector-plus-segmentor evaluation.
- `eval_after_v3/metrics.json`: five source-bound targeted checks.
- `eval_v3/fixed.json`, `negative.json`, `challenge.json`, calibration files,
  `runtime_core_conf05/metrics.json` and `standard/metrics.json`: exact metrics.
- 28 tests pass: 9 decoder, 3 thin-box training, 16 inspection flow.
- Pixel-mask preservation is explicitly tested in the custom batch preprocessing.
- Production `s_p.onnx` SHA remains `f341c3e6054bfd764103ce27149edaec8d6f5fef03d4a37586e357cc6bd368a8`;
  config.env remains `7619ba977a1779e7d1f32b6ef0ec8d1761c18fcecc340ad3ba1216af06868c3f`.
- No commits, production-model replacement, or installation/library edits.
- AMP auxiliary download is preserved as `aux_amp_check_yolo11n.pt` in this work root.

AI QA brightness, resized to reference size: camera mean20.04/std10.07,
AI mean20.70/std10.52; median gray16/255 for both, gray p5/p95 camera11/39
versus AI12/41. These summaries support exposure similarity, not proof of
physical realism. The exact prompt and input/output hashes remain in the AI
artifact directory. No independent ablation isolates AI's contribution.
