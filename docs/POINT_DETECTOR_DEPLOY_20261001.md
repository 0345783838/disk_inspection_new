# Detector fine-tune + guarded slot completion — 2026-10-01

Subsequent classifier crop-width adjustment: see
[the 1.8-width review contract](CLASSIFIER_CROP_WIDTH_20261001.md).

Corrects the earlier segmentation-only trial: that trial still used the old
`d_p_n.onnx` detector. Its classifier-crop gallery on port 8771 is historical.
The actual APP plugin now uses the September 30 detector candidate,
with its exact detector preprocessing contract, not the old full-frame input.

## Runtime contract

- Point model: `config/models/d_p_ha_20260930.onnx`.
- SHA256: `B04223C1CE295416E4C7E6F7BF6342499BE7B87A81689AC8CEEE06E19C413BBE`.
- Detection confidence 0.5; NMS IoU 0.1; existing stretch/RGB preprocessing.
- APP normalizes Cam1 clockwise and Cam2 counterclockwise to 1944x2592.
- Only detector input is cropped to x=0,y=0,width=1944,height=2200. This removes
  the overlap exactly as in training. Classifier/segmentation/UV still use the
  full normalized frame; detector boxes are returned in full-frame coordinates.
- Wrong frame dimensions or an invalid detector ROI are rejected, not silently resized/cropped.
- New independent `DISK_POINT_SLOT_COUNT=25` represents tray slots, not disk count.
  `NUM_DISK`/inspection parameters retain their disk-count business meaning.
- Segmentation candidate/settings and old classifier are unchanged.

## Development order / evidence

1. Switched source to the trained detector and correct ROI before any completion
   logic was used. All 20 HA frames returned exactly 75 boxes / 25 per row at
   confidence 0.5; 1500/1500 matched existing factory labels at IoU >=0.3.
   Lowest accepted confidence across HA was about 0.842.
2. Developed slot completion separately. It requires two complete, well-spaced
   reference rows and at least 20 observed boxes in the deficient row. Reference
   rows establish column identities, including when end slots are missing.
3. A deterministic 1-D projective RANSAC fits horizontal slot coordinates;
   a quadratic fit models mild vertical row curvature. At most five slots in
   one row may be inferred. Original accepted detector boxes are not moved.
4. Reject ambiguous index assignments, duplicate/irregular reference rows,
   unstable fits, insufficient anchors, multiple deficient rows, and inferred
   boxes outside the actual image. Do not create image pixels beyond the FOV.
5. Each inferred box is tagged `geometry_inferred`, with no detector confidence.
   White responses expose `InferredSlots` in normalized-frame coordinates and
   `PointGeometryStatus`. Debug DetectImg marks inferred boxes yellow.
   Classifier remains responsible for disk presence/OK/NG; inferred slots do
   not imply disk presence or force an OK verdict.

35 regression tests passed in the training-capable Python environment. Production
venv passed 31 tests and skipped four optional torch/Ultralytics reference tests.
Controlled deletion tests on HA covered 480 cases, removing 1–5 original model
boxes from every row, including endpoints: 480 recovered; maximum center error
5.588 px, 95th percentile of case maxima 4.710 px. This compares against withheld
model detections, NOT independent hand-measured coordinates or real NG scenes.
All 20 complete HA frames pass unchanged through geometry, with zero inferred slots.

Artifacts: `.work/detector_deploy_20261001/model_only_HA.json`,
`geometry_validation/report.json`, `geometry_validation/index.html`.
The HA set is still largely in-sample / same-session. Neither 100% count nor
synthetic deletion tests establish real-world classifier/NG reliability.

## Trial package and rollback

Run `APP/DiskInspection/bin/BroSight_Disinspector/DiskInspection.exe` from its
own directory. Only the backend/plugin is rebuilt for this update. Existing
APP EXE, camera/PLC/user settings and license files are preserved.
Deployment file hashes are recorded in `.work/detector_deploy_20261001/deployment.json`.
No camera or PLC command is used by the verification scripts.

Final packaged-API check: all response fields (including rendered images) matched
source on all 20 HA frames; every response has `PointGeometryStatus=detected` and
zero inferred slots. All 20 still return NG (`ERROR_001`), so the detector count
fix does not qualify the full OK/NG pipeline. Classifier was not retrained here.
Evidence: `.work/detector_deploy_20261001/api_smoke_all_HA.json`.

New native-size crop gallery: `.work/HA_classify_detector_new_20261001/index.html`
(localhost port 8772): 20 production frames, 2000 crops, no diagnostic-only frames,
no 224x224 exports. Controlled-deletion overlays are on localhost port 8773.

Pre-update plugin and source files are backed up under
`.work/detector_deploy_20261001/rollback/`. Stop APP/backend before restoring
the old plugin snapshot and `service_config.env`; the old `d_p_n.onnx` remains
present. Review any operator settings changed since backup before restoring.
To disable only completion, set `DISK_POINT_SLOT_COMPLETION=False` in BOTH source
and deployed plugin configs and restart the backend; the new detector remains active.
