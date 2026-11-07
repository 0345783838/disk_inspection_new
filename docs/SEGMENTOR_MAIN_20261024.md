# Main disk-edge segmentor — 24 October 2026

The user explicitly selected **native_union_hard_best** as the primary runtime model after viewing its complete 261-crop normal-validation comparison: "oke giữ lại bản này làm bản chạy chính đi".

Active model: `config/models/s_p_edge_white_native_hard_20261024_102450.onnx` in both SERVICE and APP's packaged plugin.

ONNX SHA256: `133945894b62f6dd9bc46baf80070947c4854dd488e64b181c28d035e8ead2c0`.

Only `DISK_SEGMENT_MODEL_PATH` changed in each existing config. Exact before/after files and byte-level checks preserve all other user changes. Previous model files, detector/classifier weights, runtime executable, APP settings and classifier dataset were retained unchanged. Input512, conf0.5, NMS0.8, letterbox/RGB/native/core remain unchanged. The dated PT/ONNX and provenance are retained in SERVICE/config/models. The historical archive JSON describes the save before activation; current activation authority is `.work/ACTIVE_SEGMENT_MODEL.json` and the two live config files.

The formal experiment winner remains **native_union_blend75**. This activation is the user's subsequent choice of the hard alternative; no candidate selection, training data, validation metrics or recommendation gates were rewritten. Same-session reused development-validation Pixel F1: new93.99% ->95.34%, old HA85.00% ->84.48%, old demo87.01% ->86.68%, all89.09% ->89.30%. The hard variant improves the reviewed train cases and new masks but does not improve all old masks.

## Actual runtime verification

- Current source imports its real config and loads the exact selected model hash without an override.
- All28 White frames were submitted to the actual installed `plugin/main.exe` over its production HTTP API. An isolated run of the same executable with the original Oct1 model supplied28 counterfactual responses.
- Packaged detector geometry is exactly unchanged on all28 frames. 27/28 frames reach segmentation and match the frozen hard-model replay in geometry, primary caliper counts, result and error code.
- Both cameras have exact current-source/packaged production and debug response parity on the two selected hard-case frames. Production/debug geometry, caliper points and decisions also agree.
- All protected prior model/data/classifier/executable hashes pass. No PLC API, hardware connection or camera acquisition was used.

## Existing packaged/source discrepancy

1 frame(s) are rejected by the installed executable before segmentation, while the current source accepts them after filtering an off-grid point. The same rejection and error happen with the original model, verified through the isolated old-model packaged run. This activation keeps the installed executable and geometry algorithm; it does not claim those rejected frames exercised the new segmentor.

- `Image_20261018103035152.bmp`: rejected: Too many missing slots or extra detections

## Evidence and rollback

Deployment directory: `.work/segmentation_deploy_20261024_113440`.

- `activation.json`: explicit authorization, exact model and protected hashes, both config backups and SHA256.
- `runtime_verification.json`:28 current packaged responses, old-model comparisons and both-camera parity.
- `baseline_api_cache.json`:28 original-model packaged responses on isolated port8387; the temporary process is stopped afterwards.
- `geometry_difference_initial.json`, `verification_initial.py`, `verification_execution_adjustment.json`: preserved initial strict source/package mismatch and verification adjustments. No training or evaluation truth changed.
- `rollback.py`: restores both exact pre-activation configs if they have not changed since. Restart the backend afterwards to load the old model; dated model files remain available.

The packaged primary backend is left running on port8386 with the hard configuration. APP remains unopened and no hardware is connected by this workflow. The frozen campaign's runtime hashes describe the pre-activation baseline; keep them as historical evidence and never rewrite them to match this authorized deployment. Completed campaigns must not resume.
