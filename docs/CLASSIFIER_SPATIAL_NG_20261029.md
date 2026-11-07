# Frozen nano spatial NG correction — 29 October 2026

The user requested retaining the relatively good fine-tune in SERVICE/config/models, then explicitly authorized trying the proposed spatial NG correction. The retained original is `c_p_finetune_ha_r10_best111_20261029.{pt,onnx,json,provenance.json}`. PT and ONNX hashes match the completed fine-tune release. Active `c_p.onnx`, runtime plugin and configuration remain unchanged.

New run: `.work/classifier_spatial_ng_20261029/`; dashboard http://127.0.0.1:8800/; entrypoint `SERVICE/training/train_classifier_spatial_ng.py`.

One fit only: parent is the retained YOLO11n-cls fine-tune best111, all parent backbone/head/BatchNorm frozen. A zero-initialized linear branch reads 2x2 spatial pooling (4x1280 features) and corrects only the NG logit. Exactly 5,120 trainable parameters, +0.33% over 1,534,947 parent parameters. Output at initialization matches the parent exactly; OK/no_disk relative logits remain unchanged. No deeper fine-tuning, DINO, repeated seeds or deployment.

Canonical revision10 remains 5,383 rows, 3,778 train /1,605 val. A new explicit receipt acknowledges partial review; all IDs/labels/hashes/splits are unchanged. Seven previously viewed missed-NG remain in val. Train rows only provide original plus three deterministic geometry-preserving safe-augmentation views. Validation has original view only. Frozen features/logits are immutable run evidence, not label sources. Feature standardization uses only original train rows.

Max300 epochs /patience100, AdamW on branch only, lr1e-3 cosine to5%, batch128, weight decay1e-4, seed1022. Sampler uses existing source/camera/class balanced40/40/20 recipe. Loss: CE on combined logits with NG/no_disk/OK weights2/1/1 plus0.5 auxiliary NG BCE on the branch with positive weight2. Auxiliary supervision supplies gradients even when the frozen parent already fits train with saturated confidence. No validation labels or examples enter updates/statistics/sampling.

Checkpoint selection uses existing NG-priority source/camera fitness, with epoch0 unchanged fine-tune an eligible baseline. Original installed-baseline gates remain; additionally zero val missed NG, false NG no higher than retained fine-tune per source/camera, and whole software White processing <=2x current installed baseline. Report all tradeoffs; this is development validation, HA retention, not independent testing.

Before training, four focused model tests and ten canonical regression tests passed. Tests cover exact zero-residual output identity, frozen BN/weights, gradients only in the new branch, portable checkpoint roundtrip and ONNX batch1/2/3 parity. GPU smoke on three actual train crops passed without saving a fit checkpoint. Frozen parent SHA256 over state tensors is recorded in the protocol and checked after fitting.

The pipeline freezes canonical/code/protocol/runtime hashes, caches parent features, fits branch, exports ONNX, checks full1,605-val PT/ONNX/cached-feature argmax, compares parent/n-safety/installed HA, reviews seven original missed NG, replays28 current-source White frames with PLC blocked, measures paired White latency and audits integrity/runtime hashes. No deployment. The saved original fine-tune is protected by hashes.

Before any recovery inspect real Python parent/child processes and STOP_REQUESTED. Resume only an unfinished recorded phase with unchanged frozen inputs/protocol/code, never duplicate fits and never restart a completed/user-stopped run. Do not unfreeze old weights or alter labels/splits/gates to improve results.

## Completion — 29 October 2026, 14:57 Asia/Saigon

COMPLETE. Do not resume/repeat. 100 learned epochs without improvement over epoch0. Selected epoch0 is exactly the unchanged fine-tune; no trained spatial checkpoint improved the predeclared safety fitness. Best learned checkpoint by recorded CSV fitness is epoch3:16 errors /8 missed NG /2 false NG /0 empty-as-OK, versus parent15/7/2/0. The zero residual is selected, not a successful trained spatial model. Last trained epochs had11 missed NG. This is a negative result for the linear spatial branch with the entire encoder frozen in this trial; do not silently unfreeze or start another fit.

All parent weights, old head and BN buffers remain bit-identical. Full1,605-val PT/ONNX/cached-feature argmax matches, max probability error8.23e-6. Nonzero-residual dynamic ONNX unit test also passed. All28 current-source hardware-off White frames and both-camera debug parity passed; all measured paired latency ratios <=1.154x. Raw latency had large P95 variation under machine load, so do not claim a speed improvement. Canonical revision10/current manifest, frozen code/protocol/features/images and runtime/protected retained-model hashes verified unchanged. All trainer parent/child processes exited; no heartbeat was created.

The retained original remains `SERVICE/config/models/c_p_finetune_ha_r10_best111_20261029.pt` and `.onnx`, with JSON provenance. Active c_p.onnx and runtime config/plugin unchanged; no deployment. Read run RESULTS.md and completion_verification.json for honest epoch0/best-learned distinction. Dashboard8800 remains available.
