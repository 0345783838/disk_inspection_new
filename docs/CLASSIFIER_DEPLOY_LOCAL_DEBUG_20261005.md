# Classifier deployment and local RAW White UI test — 2026-10-05

User authorized replacing the running-flow classifier after full-HA fine-tuning, and testing the main UI
by loading one local image instead of capturing a camera frame.

## Deployed artifacts

Both `SERVICE/config/models/c_p.onnx` and
`APP/DiskInspection/bin/BroSight_Disinspector/plugin/config/models/c_p.onnx` now contain the selected
single-network full-HA blend50 model from `.work/classifier_fullha_20261005/selected_v2/`.

SHA256: `ce0a150f00a56b78bbc4a07bbf3ad28862e7de7edd78a77bd0ee9f2d6a0d72bd`.
Class order remains `ng,no_disk,ok`; existing RGB/stretch224 preprocessing, native crop width1.8,
point detector, segmentor, confidence/geometry thresholds and production PLC policy are unchanged.
The backend executable did not require rebuilding; its ONNX file is loaded at process startup.

The updated WPF EXE/PDB have been built and copied into
`APP/DiskInspection/bin/BroSight_Disinspector/`. Existing machine configuration, license and user settings
were preserved, not replaced by build defaults. Restart/open this APP to use the new model/UI.

## Backups / rollback

`.work/classifier_deploy_20261005/` contains:

- `c_p_before_SERVICE.onnx`, `c_p_before_APP.onnx`: models actually deployed before this change,
  both SHA256 `0be56f7a3442745e8a31c82601f84d04c80eb6a495d48729f729768e504c8200`.
- `before_DiskInspection.exe`, `before_DiskInspection.pdb`, `before_DiskInspection.exe.config`:
  application snapshot before adding the main-UI local test.
- `deployment.json`: exact target/backup paths and before/after hashes.
- `smoke/`: integration overlays and Python HTTP smoke metadata.

The separate good-model backup requested before retraining remains untouched:
`.work/classifier_model_backups/pre_relabel_retrain_20261005/`, ONNX SHA256
`a3fa7c15a23c02d9fe3b1d7fa0ea175bc436ebab5c86b78ba99e4486e001b0a4`.
It is distinct from the older model that was actually deployed.

To roll back, stop the APP/backend, restore the appropriate ONNX files (and EXE/PDB if rolling back UI),
then restart. Do not overwrite ONNX while an existing backend is holding the previous model in memory.
No rollback has been performed.

## Test on main UI with one local image

1. Open `APP/DiskInspection/bin/BroSight_Disinspector/DiskInspection.exe` normally. License/AI startup
   remain required; camera and PLC startup are not required for this menu.
2. Do **not** press Start. If production inspection is running, Stop it first.
3. Choose **Tools > Test local RAW White (no camera / PLC) > Camera 1 / Camera 2 - RAW White...**.
   Log in with the existing authorized role, then select one raw full-frame White image.
4. Select the actual source camera: Cam1 rotates CW90; Cam2 rotates CCW90 (or App.config overrides).
   For HA use `HA/Cam1/*.bmp` or `HA/Cam2/*.bmp`. Do not select UV, already-rotated images or band crops.
   The filename need not contain `White`, `Cam1` or `Cam2` because the camera is explicitly selected.
5. The selected camera's main panels show normalized origin and White result; the image viewer contains
   RAW, normalized and result frames. Existing image-click viewer behavior is retained.

The orange menu notice says **LOCAL CAMx WHITE ONLY - UV / PLC NOT TESTED**. Per-camera status is
White-only; aggregate tray status stays UNKNOWN, UV count says Not tested, and the other camera is cleared.
No production counters, automatic image saving, light commands, PLC polling, triggers or result signals
are issued by this local execution. Closing a session that only used this new local menu also skips PLC
light shutdown; cleanup is retained when production or legacy settings/debug windows have used hardware.
The notice disappears only when a real production inspection cycle begins, not merely on failed Start.
Do not use a local White-only result as a completed White+UV/two-camera production verdict.

## Debug points

`APP/DiskInspection/Controllers/MainController.cs`:

- `InspectLocalWhiteAsync`: rejects concurrent inspection/start/stop; its task is canceled/joined on shutdown.
- `RunLocalWhiteAsync`: `new Bitmap(path)` is the local-file replacement for camera capture; inspect `raw`
  and `normalized`. No commenting out PLC/camera guards is necessary.
- `InspectWhiteFrameAsync`: shared by local test and real dual-camera flow, calls the production White API.
- `UpdateUIWithWhiteResults`: shared result-image/min-max UI update.

`SERVICE/src/service/check_disk_service_yolo.py`:

- `check_disk_white` / `_check_white`: production detect + 25-slot completion + native crops + classification
  + segmentation/caliper and geometry/result decision.
- `point_classification_model.predict_batch(crops)`: inspect native crops, labels and confidences.

For source debugging, launch with a working directory containing `plugin/main.exe` (the packaged APP
directory), since the existing service launcher resolves `plugin` relatively. Debugging Python source
requires starting the SERVICE source backend instead of the compiled plugin; do not run both on8386.

## Verification / limitations

- WPF Release build succeeded; 11 C# flow tests and63 Python regression tests passed.
- Packaged backend loaded the newly deployed model and returned HTTP200 for one HA raw frame per camera.
- `APP/Tests/LocalWhiteApiSmoke.csproj` verifies the real APP normalizer, Emgu image encoder,
  RestSharp production API call, JSON deserializer and geometry using the deployed APP assembly.
  Both cameras passed; raw2592x1944 becomes normalized1944x2592.
- Both smoke samples return White NG (`ERROR_001`) under the current full-pipeline criteria; successful
  integration is not evidence that the overall tray verdict is correct. The classifier's crop-level
  fitting/legacy metrics are documented separately and do not measure whole-tray false NG/false pass.
- Python smoke helper `SERVICE/training/smoke_local_white.py` reproduces loopback White calls and writes
  overlays/metadata. Its OpenCV JPEG encoding is not claimed byte-identical to Emgu; the C# test also ran.
- Backend warning about unavailable CUDA provider falls back to CPU, consistent with deployed CPU usage.
- GUI was compiled but not interactively exercised here; camera/PLC/UV and hardware cycle timing remain
  for the user's on-machine test. The temporary backend used for smoke is stopped after verification;
  normal APP startup launches a fresh backend with the deployed model.

These tests do not change labels, add training images, lower confidence thresholds or retrain models.
