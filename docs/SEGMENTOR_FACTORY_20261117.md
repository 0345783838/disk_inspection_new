# Factory White segmentation preparation — 17 November 2026

The user requested improving segmentation first and explicitly including every
new factory image, even where the current model is already correct. The new
source is `.work/NEW_IMAGES_2/`; this work does not alter classifier labels,
weights, runtime settings or the completed historical campaigns.

Run: `.work/segmentation_factory_20261117/`. Entrypoint:
`SERVICE/training/prepare_factory_segment_review.py`.

The source contains 86 saved capture sets, each with both cameras, White/UV,
Raw PNG and Result JPG: 688 files, including 172 White Raw frames. File names
identify captures on 8 and 14 November; the folder `Anh Vision` is a source
batch name, not a different camera or a label.

All White Raw sources are inventoried and processed regardless of the model's
OK/NG result. UV and saved Result images remain provenance/context; they are
not segmentation pixel truth. White is rotated exactly once using the current
Debug APP rotation/ROI configuration. A default OpenCV JPEG roundtrip models the
APP transport, without claiming byte identity with Emgu. The clean, normalized
transport image and rectified bands are retained for repeatable review.

The baseline is the configured `native_union_hard_best` ONNX, SHA256
`133945894b62f6dd9bc46baf80070947c4854dd488e64b181c28d035e8ead2c0`:
512 input, RGB/letterbox/native/core, confidence 0.5, NMS 0.8. Band geometry
comes from the actual current SERVICE White flow, including point completion.
The proposal capture wraps the real tile inference and asserts exact equality
between the captured union and the untouched runtime mask on every band.
Raw proposal masks are shown before runtime area cleaning; the whole-flow
replay retains the actual area and 30px paired-caliper settings.

The editor reuses the previous user-reviewed polygon/line drawing and save
semantics. Original instance raster masks are also retained so polygon
approximation is explicit and auditable. Predictions, empty-looking bands and
geometry are unverified. No band is automatically marked reviewed because a
model prediction or whole-tray result looks correct. No disk count is imposed,
and no missing edge is extrapolated into a label. Geometry failures stay in the
inventory/gallery for correction rather than being silently discarded.

`grouping_draft.json` connects paired cameras, temporally adjacent captures
(120s), and exact decoded aliases. It separately lists similar-appearance pairs
for review. These constraints do not establish physical tray identity. No new
split is assigned yet; review cross-group near-duplicates before freezing a
training/selection/final-test allocation. Every tile and augmentation of a
source family must stay on the same side. Existing old validation truth and
historical metrics remain tied to their original datasets/runs.

The preparation is hardware-off: real PLC methods are replaced in-process with
functions that raise if called. No camera capture, PLC/light action, live
SERVICE launch, training or deployment occurs. Original source files and
current SERVICE source/config/models, retained best111 and canonical classifier
manifest are hash-checked before and after. SERVICE's existing venv is used for
runtime compatibility; the global Python lacks `pymodbus`. No dependencies were
installed or changed.

Next steps after review are to freeze corrected masks and grouped splits,
export every tile including genuine backgrounds and model-correct crops, mix
with the preserved old segmentation data, and define a fresh supplementary
fit/evaluation protocol. Keep old-data retention, both-camera edge continuity,
pixel/false-edge and actual caliper/count outcomes visible. Future validation
must use confirmed labels; agreeing with baseline pseudo-labels is not evidence
of accuracy. Model export parity and current-config whole-flow replay remain
required before recommending a replacement.

Artifacts: `inventory.json`, `grouping_draft.json`, `protected_before.json`,
`protected_after.json`, `status.json`, `review/items.json`,
`review/proposal_summary.json`, `review/index.html`, `review/overview.html`.
`review/annotations.json` is created by actual saves in the editor, not seeded
with false human-review claims. Earlier smoke roots are preparation tests only,
never alternative datasets or training runs.

Preparation completed: all 172 unique White frames, 344 bands, 2,064 prospective
tiles, 0 geometry failures. Review server 8810 provides editor `/` and gallery
`/overview.html`. No labels are automatically reviewed and no training started.
Two audited proposal-serialization repairs retain 66 bands' 1,688 one-pixel
boundary pixels/76 union fragments and preserve holes in 376 polygons. Original
runtime PNGs, inputs and inference results are unchanged. All 8,751 editable
instances reconstruct their original OpenCV pixels exactly; browser Canvas
antialiasing remains part of the separate reviewed-mask export semantics.
Before/after snapshots and receipts are in run `repairs/`.
