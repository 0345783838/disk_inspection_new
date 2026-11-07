# Supplementary disk-edge segmentation — completed 23 October 2026

The authorized White-only, both-camera supplementary training is complete. Full report: `.work/segmentation_supplementary_20261023/RESULTS.md`.

Selected candidate: `training_mask_head/candidates/mask_head_blend75`, packaged as `training_mask_head/release/s_p_edge_white_20261023.pt` / `.onnx`; all 9 unchanged recommendation tests passed. Installed APP/SERVICE segment model remains unchanged; no deployment. Review/dataset and classifier labels were not modified by training.

New strict pixel F1 91.53% → 93.99%; reviewed regions under 80% tolerant coverage 15 → 13. Both cameras improve. Old HA/demo TP/FP/FN unchanged. 489-crop PT/ONNX parity and 28 hardware-off White replay completed. Five source-file caliper counts changed without independent count truth. Same-session development validation, old partial labels, and train-only new background are explicit limitations.

Source masks: `.work/segmentation_supplementary_20261023/review/annotations.json` (54 bands user-reviewed at frozen snapshot); 324 original crops, including 8 true empty crops. Combined immutable run dataset: 2,364 train / 261 val. Complete evidence: `training_mask_head/completion_verification.json`, `release/release_manifest.json`, `primary_errors.html`, `comparison.html`, `background.html`; all candidate results in the outer `campaign_summary.json`.

Do not resume any completed run. `training/` is an invalid export predecessor stopped at 3 epochs; `training_v2/` has two completed whole-network fits and 8 candidates that failed gates; `training_mask_head/` has the completed mask-only fit and 4 passing candidates. Preserve `training_v2/` because the mask-only run manifest points to its immutable pixel/mask dataset. For blends, trust the release manifest's two-parent lineage rather than inherited baseline train_args metadata.

Dashboard 8799 and review editor 8798 may remain available. No classifier campaign was started or resumed; classifier canonical revision 10 is unchanged.
