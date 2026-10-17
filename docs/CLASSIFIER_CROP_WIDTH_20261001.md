# Classifier crop width trial — 2026-10-01

User-defined per-slot labels:

- `ok`: exactly one disk in the target slot.
- `ng`: more than one disk in the same target slot.
- `no_disk`: no disk in the target slot.

Neighbor-slot disks visible as context must not contribute to the target label.
Do not mistake two visible rims/highlights of one disk for two disks. There is
no fixed disk-count labeling assumption; 25 refers to tray slots only.

## Crop adjustment

`POINT_CLASSIFY_CROP_WIDTH_RATIO=1.8` replaces the earlier 1.5 width factor.
The factor multiplies the rectified point-box width. Horizontal center, height
factor 1.5, vertical shift 0.5 and top-face 180-degree rotation are unchanged.
Typical native width on HA grows from 90 to 108 px, roughly 9 extra px per side.
Pixel rounding can introduce a half-pixel difference in the crop's discrete center.
Image bounds still limit the real crop; no pixels are invented beyond the FOV.

The setting is shared by production, debug, and crop exports. Without the new
key, default width remains 1.5 for backward compatibility; an explicit helper
argument overrides the configured width. Invalid nonpositive/nonfinite ratios
are rejected. Detector model/ROI, slot completion, segmentation and classifier
weights are not changed in this trial.

A preflight audit of 2,000 HA crops found no adjacent slot center inside the
proposed wider rectangle (3,840 adjacent-center checks). This does NOT prove
absence of neighboring disk pixels: disks may lean or overlap into the crop.
Manual per-slot review and suitable multi-disk/empty-slot examples are still needed.

## Review / verification

New native crops: `.work/HA_classify_wide_20261001/`.
Before/after review: `.work/HA_classify_wide_comparison_20261001/`.
The comparison checks identical point boxes and Y bounds and verifies that every
old native crop is a pixel-exact subimage of its wider replacement, accounting
for top-face rotation. Labels on sheets are predictions, not confirmed truth.
No 224x224 image export is requested or generated.

Final audit: all 2,000 old crops were retained pixel-exact inside the new crops;
median width 90 -> 108 px; zero adjacent slot centers inside the new rectangles
on 3,840 checks. The unchanged classifier switched labels on 128/2,000 crops;
this is prediction drift, not an accuracy metric. All 20 packaged API responses
matched source exactly, still with NG/ERROR_001 and zero inferred slots.
Native crop gallery: localhost port 8774. Before/after gallery: localhost port 8775.

38 regression tests passed, including width-only geometry changes, legacy defaults,
explicit override, invalid ratios and production/debug parity. Packaged API
verification is saved separately in `.work/classifier_crop_wide_20261001/api_smoke_all_HA.json`.
Classifier weights have not been retrained for the wider input distribution;
changes in its predicted labels cannot establish an accuracy improvement without
reviewed labels / independent NG scenes. Future classifier training should use
the approved crop contract and these explicit per-slot labeling rules.

## Deployment / rollback

Actual backend: `APP/DiskInspection/bin/BroSight_Disinspector/plugin/main.exe`.
Original plugin and source configuration/code are backed up in
`.work/classifier_crop_wide_20261001/rollback/`; deployment hashes are in
`.work/classifier_crop_wide_20261001/deployment.json`.
APP EXE, camera/PLC/user settings and license are preserved. Verification does
not issue camera or PLC commands.

For width-only rollback, set `POINT_CLASSIFY_CROP_WIDTH_RATIO=1.5` in BOTH
`SERVICE/config/config.env` and the deployed plugin config and restart backend.
No rebuild is needed for subsequent width adjustments. A complete plugin
snapshot is also available; review any settings changed after backup before restoring.
