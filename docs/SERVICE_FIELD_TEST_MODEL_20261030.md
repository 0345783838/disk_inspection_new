# SERVICE model selection for user-requested field testing

On30October the user requested deleting poor/extra trained models from SERVICE and retaining one best classifier to test on the real machine. The chosen classifier is the single-network YOLO11n-cls fine-tune with ten accepted AI same-slot NG crops, saved checkpoint42 from `.work/classifier_same_slot_ai_20261029/artifacts/best.onnx`.

It is now installed at `SERVICE/config/models/c_p.onnx`, SHA256 `89a7ef199e575cdb77a7bcc8edb0d1250affa41bb9f804cadbdc24b1fbccc434`. Labels NG/no_disk/OK, runtime RGB/stretch224, batch2. The current SERVICE config remains unchanged, including classifier crop-width1.8. Fresh SERVICE instance loading the configured path and predicting successfully was verified. A running SERVICE process must be restarted to load replacement weights.

Selection follows the user's missed-NG-first priority. On the original1,605validation, retained fine-tune→chosen model is7→3missedNG,2→8falseNG,15→15errors. Across the full5,383real diagnostic crops (fitted/reused, not independent), chosen model has3missedNG/8falseNG/15errors vs retained fine-tune7/2/18. The newly trained fresh model has17/14/40 on new1,615val and was not installed. On that changed new cohort the chosen model has1missedNG/2falseNG/3errors, with prior exposure acknowledged. The original recommendation gates remain failed in the source report; the newer explicit user instruction authorizes temporary field testing, not a zero-miss production claim. No historical release recommendation was rewritten.

The runtime folder retains exactly three ONNX models, one per necessary stage:

- Classifier: `c_p.onnx` (selected fine-tune+AI).
- Existing detector: `d_p_ha_20260930.onnx`, hash `b04223c1ce295416e4c7e6f7bf6342499be7b87a81689ac8ceee06e19c413bbe`.
- Existing segmentation: `s_p_edge_white_native_hard_20261024_102450.onnx`, hash `133945894b62f6dd9bc46baf80070947c4854dd488e64b181c28d035e8ead2c0`.

`MODEL_SELECTION.json` records the active file hashes/source and rollback location; `__init__.py` remains. Sixteen other ONNX/PT/metadata files were removed from SERVICE only after every file had a SHA256-verified backup under `.work/service_models_field_test_20261030/backup/models/`. Training/history files in `.work` were not deleted. The old installed HA `c_p.onnx` is preserved in that backup. Config backup is `backup/config.env`.

Before installation,28White frames were replayed with the actual current SERVICE config, comparing previous configured HA with the candidate in memory. No hardware calls; geometry/counts unchanged and both-camera debug/production parity passed. Median paired software latency1.0013x, maximum1.0876x, all below2x. This checks integration, not ground-truth field accuracy. After cleanup a fresh SERVICE instance confirmed the selected configured ONNX loads; no live server/PLC was launched or restarted.

SERVICE source/config, canonical classifier revision12, APP plugin and active detector/segmenter hashes are unchanged. Older training protocols intentionally retain their historical protected-runtime hashes; do not restore retired SERVICE files or the old classifier just to make completed-run guards match. Read the field-test selection receipt instead. No additional training or dataset/label changes occurred.

Evidence: `.work/service_models_field_test_20261030/{cleanup_plan,preflight,cleanup_complete,verification}.json`. The manage.py helper's prepare/preflight/apply/verify phases document checked paths and nonrecursive file deletion. The user decides field-test outcomes before any further model work.

User correction after cleanup: the good original HA fine-tune best111 must also be retained. Its four ONNX/PT/metadata files were restored byte-for-byte under `SERVICE/config/models/c_p_finetune_ha_r10_best111_20261029.*`; hashes match the original saved release. It is an explicit retained comparison model, not an extra poor checkpoint to delete. `MODEL_SELECTION.json` lists it under retained_alternates. Active c_p is still fine-tune+AI and config is unchanged; no active-model switch was requested. Evidence: cleanup folder `restore_ha_best111_20261030/restoration_verification.json`.

## User override04November2026
Active classifier is now original HA fine-tune best111 SHA775872fc29f8e00ab9934747cfaa543ad966e2e905d5e53740717d4114b8363c, following explicit user choice after reviewing8falseNG. Previous +AI field choice above is historical. See .work/ACTIVE_CLASSIFIER_FIELD_TEST.json and .work/field_test_best111_20261104/verification.json/package. Improvement paused; no live SERVICE/PLC launch.
