# Segmentor concurrency correction — 2026-10-17

The shared ONNX segmentor previously stored letterbox geometry on the model instance. Two camera requests could overwrite it between preprocessing and decoding, changing White distances and Mid1 points supplied to UV.

The correction removes per-image mutable instance state. `_letterbox_transform(img_h, img_w)` derives the same ratio and rounded padding from the current image dimensions for preprocessing and decoding. The public method signatures, tensor pixels, model files, thresholds, tiling, mask algorithms, UV policy and APP aggregation are unchanged. Camera requests remain parallel; no global lock, additional model instance or model load is introduced.

## Validation

- The new concurrent regression fails on the pre-fix source in all four letterbox combinations (production/debug methods and native/legacy masks) and passes after the correction. It compares all box/score/class/mask arrays with serial results while forcing inference overlap. Stretch behavior is also exercised.
- Full SERVICE suite: 90 tests, 78 passed, 12 skipped for optional training libraries. The already installed global PyYAML package was explicitly loaded only by the test runner for training-tool fixtures; production dependencies were not changed.
- Existing APP flow tests: 11 passed.
- Twenty HA White frames (10 per camera): all structured outputs and encoded PNG hashes from the corrected concurrent source exactly match the pre-fix serial source.
- Staged backend: 20 White -> UV cycles with both cameras submitted concurrently; every structured field matches the same runtime's serial baseline. Packaged/source/reference White/UV parity also passes for both camera samples (the existing Cam2 White rendering has the documented 5-pixel rasterization difference).
- Deployed backend: another 20 concurrent two-camera White -> UV cycles pass with no structured output differences. Actual APP binary, Emgu JPEG encoding, RestSharp transport/deserialization and BuildResult are checked against staged and deployed backends; nine merge/error cases pass.

Reports, logs, source snapshot, build output and rollback are under `.work/segmentor_concurrency_fix_20261017/`.

## Deployment

Only `APP/DiskInspection/bin/BroSight_Disinspector/plugin/main.exe` is replaced. All other package files are hash-verified unchanged, including the APP, config, models, dependencies and license. All 45 non-config dependency files in the rebuilt standalone runtime match the installed runtime.

- Previous backend SHA256: `c4e6304410692e2aa89c2299f9206dec48d9426e267adc068a20bbe9ef18e65a`
- Corrected backend SHA256: `16c85ed5e971859651d8df19788b86193f9aa8e9f00bf7ed5ff706ab5e83f855`

The release archive contains the complete application folder, startup notes and a hash-guarded rollback script/previous backend. The rollback backend retains the known concurrency defect and is provided for diagnosis.

## Factory run

Extract the entire release and open `BroSight_Disinspector/DiskInspection.exe`. Retain the directory structure. Verify both camera serial numbers/exposure, PLC IP/port and save directory in Settings on the destination machine: per-user Windows settings do not travel with the ZIP. Package defaults currently have an empty Cam2Sn and PLC IP 127.0.0.1.

Before running a lot, verify one real White -> UV -> result/PLC cycle using known OK, NG and mixing trays. These checks have used saved, unpaired White/UV images and do not establish real camera/light/PLC timing or inspection accuracy. No camera/PLC control routes were called during verification.

Existing policy remains: Warning sends OK to PLC, mixing has priority among NG signals, and production UV uses the legacy HSV/caliper algorithm. UV_RIM settings remain experimental.
