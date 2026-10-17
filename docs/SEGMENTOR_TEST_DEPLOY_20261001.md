# Temporary factory-test deployment — 2026-10-01

This records the earlier segmentation-only trial. The subsequent
[detector update](POINT_DETECTOR_DEPLOY_20261001.md) supersedes its old detector settings.

User authorized updating the actual inspection flow to test the new segmentation
candidate. This is a trial deployment, not qualification for production acceptance.

## Run this package

`D:/huynhvc/OTHERS/4.DISK_INSPECTION/APP/DiskInspection/bin/BroSight_Disinspector/DiskInspection.exe`

Start it from its own folder, or use a shortcut whose **Start in** is that folder.
The APP starts `plugin/main.exe` with the plugin working directory. Do not use
the old ZIP or another previously copied EXE; those were not updated.

The APP was rebuilt Release/x64 from the current source, including the existing
camera-normalization and inspection-flow work. Three required runtime DLLs were
added. Backend was rebuilt with Nuitka from the SERVICE virtual environment.
The standalone backend and its models were copied into the actual plugin folder.
`DiskInspection.exe.config`, `user.ini`, `NLog.config`, and existing license files
were preserved. Camera/PLC addresses and other user settings were not changed.
No APP/camera/PLC operation was triggered during deployment verification.

## Active segmentation

- Model: `config/models/s_p_ha_thinbox512_20261001.onnx` in both SERVICE and plugin.
- SHA256: `bd6f958aaa3cc1c606e91f709b0099a8920b569eed16aef155bb19ca753da534`.
- Input: 512x512; letterbox, RGB, native mask decoding, core tile ownership.
- Confidence 0.5; NMS IoU 0.8; existing `MIN_DISK_AREA=250` retained.
- Point detector, point classifier, caliper, disk-count and UV settings unchanged.
- The old `s_p.onnx` remains present; it was not overwritten.
- Legacy model compatibility defaults are kept when the new decoder flags are absent.

The existing normalization defaults apply even though the preserved APP config
does not contain the newer appSettings: Cam1 clockwise 90 degrees, Cam2
counterclockwise 90 degrees, full rotated frame. No unapproved ROI was introduced.

## Verification and limitations

- 28 SERVICE regression tests and 11 APP flow tests passed; Release/x64 build succeeded.
- Source service masks matched the candidate gallery pixel-for-pixel on all 40
  prepared HA bands. All five reported target rims survived actual area cleanup.
- The final deployed `plugin/main.exe` was started hidden and served port 8386.
  Two raw HA images, one per camera, were normalized and submitted through the
  real `/ai/check_disk_debug` endpoint. All response fields, including rendered
  segmentation images, exactly matched the source service.
- These two full-flow responses are still NG: Cam1 `ERROR_001` (abnormal tray),
  Cam2 `ERROR_002` (abnormal slot count). Segmentation was available in both.
  API/source agreement proves packaging consistency, **not** correct OK/NG
  classification. Full-frame geometry differs from the prepared-band gallery.
- No independent real NG test set exists. The candidate recovers the reported
  faint rims, but synthetic negative tests showed increased false positives;
  inspect both faint-edge recall and spurious/split edges during camera testing.

Evidence lives in `.work/segmentation_deploy_20261001/`:
`source_smoke.json`, `api_smoke.json`, and `deployment.json` (hash-verified
53 plugin files, five APP files, and hashes of unchanged user configuration).
Training caveats and metrics remain in `SEGMENTOR_FAINT_EDGES_20261001.md`.

## Backup / rollback

Original runtime snapshot:
`.work/segmentation_deploy_20261001/rollback/plugin/`.
Original APP EXE/PDB and configuration copies:
`.work/segmentation_deploy_20261001/rollback/app_files/`.
Original source configuration/code copies: `service_config.env`, `base_service.py`.

To roll back the model only, close the APP and its backend first, restore
`service_config.env` to `SERVICE/config/config.env` and `plugin_config.env` to
the deployed plugin's `config/config.env`, then restart. The new binary supports
the old model/decoder defaults and the old ONNX is still present.
For a full runtime rollback, additionally restore the backed-up plugin contents
and original APP EXE/PDB. Do not overwrite camera/PLC/user/license changes made
after this trial without reviewing them first. The three newly added DLLs and
the versioned new model can remain unused; no deletion is required.
