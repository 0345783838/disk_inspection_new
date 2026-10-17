# HA White edge retest — 2026-10-13

User reports missing White edges and requests retesting existing HA data. Scope is
diagnostics only: no new thresholds/models, production code changes or build.

## Reproduction

-20 raw White BMPs (10 per camera) from HA/Cam1 and HA/Cam2; UV excluded.
- Camera rotation/ROI follows the installed APP config (Cam1CW90, Cam2CCW90 defaults).
- The actual source White pipeline uses the installed plugin config, not the
  SERVICE config. Source confidence is0.1; plugin confidence remains0.5.
- Detector-based slot completion/rectification, classifier, segmentor native RGB
  letterbox/core merge, area filter and four caliper positions are exercised.
- Inputs undergo an OpenCV JPEG roundtrip to approximate APP transport; encoding
  is not claimed byte-identical to Emgu ToJpegData().
- All20 inputs also run through the actual packaged main.exe White HTTP endpoint
  on isolated port8387. Verdict/error/geometry/Mid1/Mid2/min-max distances match
  the source diagnostic run. No PLC/camera endpoints are called; the temporary
  backend is stopped. `packaged_parity.json` records verification.

Segmentor SHA256:
`bd6f958aaa3cc1c606e91f709b0099a8920b569eed16aeF155bb19ca753da534`
(source/plugin files identical). All runtime config and backend hashes match their
pre-test values recorded in the report.

## Observations across40 bands /80 scanlines

- Conf0.5→0.1 adds only97 raw mask pixels in aggregate; no caliper counts change.
- Area filtering removes280 pixels total; removing this filter changes zero
  caliper counts. It is not the observed main reason for missing counts here.
- Min-edge4→1 changes38/80 counts, sometimes22→37 or23→39. Those extra pairs can
  be fragments/duplicate boundaries; larger count is not evidence of better recall.
- Scan length0.95→1.00 adds one pair on8 Cam2 band2 lower scanlines. Full-width
  overlays show a far-right rim outside the original95% scan segment being counted.
- Missing internal structures can already be absent from the aggregated segmentor
  mask before area filtering. This implicates segmentation (including decode/tile
  assembly), not merely the area filter or caliper. Individual-model vs tile-merge
  causes have not been isolated here; do not claim a training-only root cause.
- Production visualization paints mask only around the0.25 and0.75 scanlines with
  a40%-band-height strip. No paint outside those strips need not mean absent mask.

Examples: `Cam1_Image_20260918130452232` has counts22/22 and20/21; conf0.1,
no-area and full-width tests do not restore its missing internal predictions.
`Cam2_Image_20260918133424033` band2 changes23/22→23/23 with full-width scanning.
Do not force any count to25:25 is the slot grid, not confirmed disk ground truth.

## Deliverables

`.work/white_retest_20261013/`: rotated originals, production render, original bands,
raw/filtered/removed masks, overlays with actual95% scan segments, conf0.1 and100%
scan ablations, full JSON and packaged API responses. All381 report links verified.
Report served read-only at `http://127.0.0.1:8781/`.

`SERVICE/training/diagnose_white_edges.py` generates these diagnostics;
`redraw_white_diagnostics.py` regenerates overlays from saved masks without models.
Mask generation and counting are separate from full-tray verdict and classifier
accuracy. No edge-level labels were projected to this detector geometry, so this
is not a new recall/precision evaluation or independent test-set result.

Next correction requires separate authorization: validate a full-width counting
change for edge rims, and investigate missed internal mask predictions without
blindly lowering confidence/min-edge or applying the UV-specific tracker to White.
