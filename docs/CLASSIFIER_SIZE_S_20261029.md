# YOLO11s-cls trial — 29 October 2026

The user authorized increasing the classifier from n to s, then clarified: “không lọt NG với tốt nhất không được lỗi ảo nhiều”. This authorizes one new YOLO11s-cls fit and evaluation, not deployment or restarting completed runs.

- Run: `.work/classifier_size_s_20261029/`; report: http://127.0.0.1:8798/.
- Entrypoint: `SERVICE/training/train_classifier_size_s.py --phase campaign --root .work/classifier_size_s_20261029`.
- Exact canonical revision-10 crops, labels, hashes and 70/30 assignments are frozen after a new revision-bound receipt; review remains partial. No new crops or labels. Previous run snapshots remain unchanged.
- Official Ultralytics v8.3.0 `yolo11s-cls.pt`, SHA256 `e2b605d1c8c212b434a75a32759a6f7adf1d2b29c35f76bdccd4c794cb653cf2`, 1,000-class ImageNet initialization. Architecture s cannot directly inherit the HA-n weights. Three-class s has 5,446,851 parameters.
- One fit, max 300 epochs / patience 100; full weights, seed 1022, batch 64, RGB/stretch 224, AdamW lr 3e-4, BN statistics trainable, same balanced 40/40/20 source sampler as the prior ImageNet-n experiment. Preserve all geometry during augmentation, using the existing safe transform.
- NG/no_disk/OK CE weights 2/1/1. Checkpoint fitness is the existing NG-priority formula: source/camera NG regression, NG misses, false NG, no_disk-as-OK, errors. No distillation, synthetic data, threshold search, ensemble or weight interpolation. This changes objective/loss as well as model size relative to the older ImageNet-n fit; report the limitation.
- Compare to installed HA-n, earlier ImageNet-n, fine-tune-n and selected NG-priority-n. Cached old predictions are reused only after verifying model hashes and exact current ID/label/hash/split signature; they are not label sources.
- Preserve existing recommendation gates by source and camera. Additionally require 0 NG misses on development val and false NG no greater than selected NG-priority-n in every source/camera. This does not guarantee zero misses outside the evaluation data.
- Export PT/ONNX; check full 1,605-val parity, 28-frame White hardware-off replay, current canonical integrity and runtime hashes. Replay runs current installed baseline immediately before candidate on each frame, protecting against unrelated geometry changes since the older replay. No camera capture or PLC actions.
- Runtime hashes are captured from current installed files, current referenced models and source inference dependencies. Old runtime-hash sets are historical; do not restore old binaries/configs to match them.
- Before any recovery check actual Python parent/child state and STOP_REQUESTED; never spawn duplicate fits and never resume a completed run. The script resumes only unfinished recorded phases against frozen code/protocol. It does not automatically resume user-stopped jobs.

At startup, 15 focused tests passed (five NG-safety tests plus ten canonical-trainer regressions). A three-train-crop GPU forward/backward smoke passed with a 3-class output, finite loss/gradients and no saved fit checkpoint. All later progress and final results belong in the run, not in older completed reports.
## Completion — 29 October 2026, 11:36 Asia/Saigon

COMPLETE. Do not resume or repeat this run. The parent and fit/evaluation child processes have exited. No deployment occurred; installed HA-n remains.

181 epochs, best at 81, stopped after 100 epochs without safety-fitness improvement. On fixed development validation:

| Model | Errors | Missed NG | False NG | no_disk → OK |
| --- | ---: | ---: | ---: | ---: |
| Installed HA-n | 28 | 1 | 14 | 11 |
| Earlier ImageNet-n | 51 | 31 | 6 | 8 |
| Fine-tune-n | 15 | 7 | 2 | 0 |
| Selected NG-priority-n | 26 | 0 | 12 | 11 |
| New YOLO11s-cls | 44 | 27 | 10 | 1 |

The s model misses 19 HA NG (Cam1 5, Cam2 14), 5 new NG (Cam1 1, Cam2 4) and 3 legacy NG. False NG are HA 5, new 1, legacy 4. All recommendation gates remain unchanged; the new zero-miss and n-safety false-alarm requirements also failed. Do not recommend this s checkpoint for production. This trial changed both model size and the objective/loss relative to older ImageNet-n, so it does not isolate a causal effect of size alone.

PT/ONNX argmax matches on all 1,605 val, maximum probability difference 6.1691e-6. All 28 current-source White replays passed baseline/candidate geometry/count comparisons and debug parity on both cameras, with PLC connection/writes blocked. All frames reached detected/filtered geometry (none bypassed the classifier through failed point geometry). Canonical integrity, exact source manifest and frozen input/code/protocol/runtime hashes passed.

CPU batch-2 median for preprocess + classifier inference: installed HA-n 7.68 ms, s 13.75 ms; p95 8.94/15.14 ms, 30 iterations after warm-up. This is about 1.79× slower for the classifier batch, not a whole-tray latency measurement.

Reports: run `RESULTS.md`, `release_decision.json`, `completion_verification.json`, `artifacts/cpu_latency.json`, `comparison.html` and `val_best_errors.html`. Dashboard remains http://127.0.0.1:8798/. No heartbeat was created for this trial.
