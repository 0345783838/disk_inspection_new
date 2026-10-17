# HA segmentation refinement (2026-10-01)

Follow-up candidate with corrected missing labels, a limited AI pilot and
thin-box training: [faint-edge results](SEGMENTOR_FAINT_EDGES_20261001.md).
The models/results below are retained as the before baseline.

Candidate only. Production `SERVICE/config/models/s_p.onnx` and `config.env`
were not replaced. This follows the partially human-reviewed September 30 run.

## Selected candidate and inference

- Weights: `.work/segmentation_refine_20261001/runs/refine512_hard/weights/best.pt`
- ONNX: same directory, `best.onnx`
- Input `[1,3,512,512]`; outputs `[1,37,5376]` and `[1,32,128,128]`.
- Letterbox, RGB, native mask decoding, six 5%-overlap tiles with **core** fusion.
- Confidence **0.5**, NMS IoU **0.8**, mask logit threshold zero (probability 0.5).
- No predicted-edge extrapolation, minimum length, or fixed disk count.

```python
OnnxSegmentor(new_model, ["disk"], conf_thres=.5, iou_thres=.8,
              preprocess_mode="letterbox", rgb_input=True,
              mask_decode="native", tile_merge="core")
```

Defaults remain stretch/BGR/legacy/max for the existing production model.
The 512 candidate must not be used with those legacy settings.

All 20 White HA frames (train and val) were inferred. UV1/UV2 are outside this
segmentation branch. Crop/rectification geometry uses prepared point labels,
not end-to-end detector predictions. The overlays are therefore segmentation
inspection, not proof of whole-system counting accuracy.

- Final gallery: `.work/HA_refined_20261001_conf05_core/index.html`
- Raw-coordinate overlays: that directory's `raw_overlay/`
- Before/after: `.work/HA_refined_20261001_core_before_after/index.html`
  (original / previous 320 candidate at confidence .1 / refined candidate at .5).
- Intermediate max-fusion output is preserved in `.work/HA_refined_20261001_conf05`.

## Diagnosed causes

The user's screenshot exactly matched Cam1 `Image_20260918130347016`, band12.
Its edge near x=990 already had a training label, but the 320 ONNX/BGR decoder
received scores only .021–.042. The refined 512/RGB model detects it at .8368,
with 287 mask rows out of a 288-row band. This source is train, not independent
validation. Edge x=1129 also exposed a mask decoding error: cropping prototype
cells first and stretching that crop distorted/truncated the thin mask.

Native decoding instead resizes full mask logits, removes letterbox padding,
then clips to the floating-point box and binarizes. A test compares it directly
with the installed training library's native implementation. No torch dependency
was added to production inference; the reference test is optional in runtimes
without training dependencies.

The thin extra branches near x=1240 and x=1540 were from boundary-clipped
predictions in adjacent overlap tiles, with scores .901 and .949. Max fusion
kept those pixels alongside a better full-edge prediction in the neighboring
tile. Per-tile NMS cannot remove duplicates across tiles. Core fusion uses the
overlap as context but writes only each tile's owned central columns. It removed
those two branches without deleting the recovered x=990 edge. This is not
general area/length filtering: genuine short or narrow masks in an owned region
are not automatically rejected.

## Data and training

Dataset: `.work/segmentation_refine_20261001/dataset_hard_v1`.
Hard mining used only the 192 original HA train tiles at confidence .25, with
the old candidate's native predictions. 64 tiles had low IoU or row coverage;
120 additional crops shift the real window left/right by about 5.5%, preserving
the source mask geometry, with mild gain/noise variations. Boundary shifts that
would produce an unchanged crop are skipped.

Training totals: 1,216 images = 768 HA photometric samples, 120 hard real crops,
183 prior controlled missing/near-overlap/crossing samples, 16 constructed
edge-erased negatives, and 129 old-demo images (down from 387). All labels remain
partially reviewed/model-assisted; transformed geometry is not new human gold.

Validation remains 48 HA tiles from four held-out raw frames and 117 old images.
No val source was used for hard mining or train augmentation. Standard best
checkpoint selection used HA validation only; old validation was evaluated
separately. The augmentation audit covers 360 original/shifted real tiles:
minimum mask-to-polygon pixel recall .96266, mean .98765. Source-group leakage
checks include the synthetic/hard variants, not just each origin separately.

Fine-tuned from the September 30 scenario candidate for 25 epochs, input 512,
batch four, AdamW lr .0003, mask ratio two, seed 930, no mosaic/flips. The best
checkpoint was epoch 20 before stripping optimizer state. Training completed;
both PyTorch and ONNX exports were evaluated.

## Validation results and definitions

Fixed-threshold results use one-to-one matching at mask IoU >= .30. Full length
additionally requires pixel intersection in >=90% of target rows and both end
regions. These are clipped tile instances, not physical disk counts. This is
stricter than the earlier end-hit-only metric and must not be mixed with it.

| HA val, confidence .5 | Previous 320, native RGB | Refined 512, native RGB |
| --- | ---: | ---: |
| Matched / GT | 204/216 | 213/216 |
| Unmatched predictions | 2 | 3 |
| Recall | .9444 | .9861 |
| Precision | .9903 | .9861 |
| Strict full-length recall (PyTorch) | 185/216 (.8565) | 200/216 (.9259) |

Actual ONNX/native evaluation gives the same matched/unmatched counts and
201/216 full-length masks; tiny numerical/boundary differences affect the strict
row-coverage threshold. These tile metrics precede band fusion.

Confidence .6 gives 209 matches and two unmatched predictions; .7 gives 206 and
two. Chosen .5 retains four more matches than .6 for one additional unmatched
prediction, without resorting to .01–.1 operation. Threshold selection is on
validation, not an independent test or statistical calibration guarantee.

Full-band ONNX/RGB pixel metrics versus mostly provisional masks:

| Fusion, confidence .5 | Mean IoU | Mean precision | Mean recall |
| --- | ---: | ---: | ---: |
| Max | .73154 | .82888 | .86118 |
| Core (selected) | .73294 | .83448 | .85713 |

Core modestly trades recall for boundary-artifact suppression. It may expose a
seam miss if an owning tile fails where its neighbor succeeds; inspect seams in
the gallery. The default production fusion was not changed.

Ultralytics validation mask mAP50 / mAP50:95: HA .92894/.42979 (previous
.92864/.37888), old subset .94076/.47056 (previous .93953/.43260).
Detailed results are in `eval_final`, `final_fixed.json`, `final_onnx_fixed.json`
and `runtime_native_core_conf05` under `.work/segmentation_refine_20261001`.

## Synthetic checks and remaining limits

At confidence .5 / matching IoU .30, controlled missing-edge scenarios improve
151 to 155 matches with unmatched predictions dropping 3 to 0; near-overlap
improves 222 to 235 matches but unmatched predictions rise 8 to 11; crossing
improves 211 to 217 matches but unmatched predictions rise 8 to 16. More recall
does not mean fewer false positives in every scenario. The challenge has 135
tiles derived from val sources, not actual NG captures.

Scenario mask mAP50: missing-edge .9363 (previous .9729, a regression),
near-overlap .8626 (previous .8459), crossing .7711 (previous .7392). All three
improve mAP50:95. Full metrics and fixed-threshold counts are retained; these
definitions measure different trade-offs and should not be cherry-picked.

The 16 held-out constructed negative crops still produce two predictions at
confidence .5 (previous candidate one). Erasure/inpainting is only a stress
test; it does not establish real empty-tray accuracy. Old val includes actual
background-labelled demo crops, but there is still no separate-session real NG
set. Partially reviewed labels can omit genuine edges, making an unmatched
prediction not necessarily a proven false edge.

Before production replacement, inspect the full gallery and remaining source
labels, then check actual missing/extra/overlapping/crossed trays from another
session and end-to-end detector/counter behavior. Do not claim all faint edges
or future defect configurations are solved. Nine decoder/preprocessing tests
and 16 inspection-flow tests pass; old default decoder audit remained identical.
