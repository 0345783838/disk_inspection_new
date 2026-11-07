# Target-aware context pilot, 30 October 2026

The user authorized trying the proposed local-detail plus neighboring-slot
context approach. Run: `.work/classifier_context_20261030/`; dashboard:
<http://127.0.0.1:8806/>. This experiment leaves SERVICE's field-test classifier,
the retained HA best111 model, detector, segmenter and runtime configuration intact.

## Inputs and model

The authoritative registry selects canonical revision 12: 5,393 crops, 3,778
train and 1,615 validation. Review remains partial. The new explicit receipt is
bound to that revision. The trainer calls `active_dataset`, freezes the exact
manifest/hash and all crop bytes before deriving eligibility. No labels or
canonical split assignments change, and no historical labels are imported.

The local model is the retained YOLO11n-cls HA fine-tune best111, with all weights
and BatchNorm statistics frozen. A small context encoder reads a clean strip
spanning five neighboring slots and a separate fourth channel marking the
queried slot. Local features and context features produce a residual correction
to the three original logits. Its initial correction is zero. Total parameters:
1,657,913, including 122,966 trainable additions (about 8% over the parent).
Parameter count alone does not establish runtime latency.

Raw images and recorded geometric transforms reconstruct all 4,783 HA/new
canonical crops exactly, checked pixel by pixel. Context is extracted from those
same raw rectified frames, never from drawn review overlays. Historical geometry
is provenance only. The 600 legacy and 10 AI crops lack matching raw context;
their zero mask guarantees the original parent's exact predictions. They remain
in the immutable canonical snapshot and full-validation report.

## Trial design

Two fits only, using the same seed (1031), inputs, sampling and loss:

1. Control: context RGB is zero; only the target mask and frozen local features
   are available. This measures gains from another local decision head.
2. Context: the same architecture also receives the actual neighboring pixels.

Each fit has maximum 300 epochs and patience 100. Every eligible training crop
is seen once per epoch, plus an equally sized source/camera/class-balanced draw.
There are three cached views per eligible crop: original plus two synchronized,
bounded illumination/flip variations. NG-weighted CE, an auxiliary context loss
and agreement with correct confident parent predictions are fixed in protocol.
No threshold search, interpolation or extra fits are included.

Eligibility uses only canonical labels and capture metadata, not predictions or
known error IDs. Held bursts are `old_ha_burst_01` and new bursts 01, 05, 07.
Only 2,231 existing canonical-train rows outside those captures fit the added
branch. Another 1,117 canonical-train rows are quarantined. The primary selection
cohort contains 480 existing canonical-val rows from held captures: 300 HA Cam2
and 180 new-factory crops. HA Cam1 has no separate capture burst available for
this holdout. The primary new-factory cohort has only three NG examples, so its
NG evidence is limited. Other canonical validation rows are secondary development
diagnostics. The parent previously saw historical data: this is capture-separated
adaptation testing, not a blind independent evaluation.

Checkpoint selection first penalizes NG regression per represented source/camera,
then NG misses, false NG, empty-as-OK and total errors. Epoch zero (unchanged
parent) is eligible. Context must beat the mask-only control on the primary
cohort, have no missed NG, and avoid NG/false-NG regressions against both the
parent and current SERVICE model. Existing fallback errors may prevent those
gates passing; the experiment must report this honestly.

## Verification and recovery

Both selected networks require PT/ONNX parity on all 1,615 canonical validation
crops. Hardware-off replay uses the actual current SERVICE configuration on 28
White frames, checks geometry/counts and both-camera debug parity, and measures
the whole processing flow against current SERVICE. The user's maximum is 2x.
Runtime/model, input, context, raw-source, code and protocol hashes are protected.
No deployment is authorized by this pilot.

The initial cache phase failed before fitting because a frozen builder resolved
the project root with a fixed parent index. A tested, documented repair locates
the SERVICE anchor and separates RGB/mask in the display preview. It changes
neither labels, split, model, loss nor selection. Receipt and original bytes:
`repairs/frozen_root_and_preview/` inside the run. Seven context tests passed,
including nested snapshot resolution and ONNX batches 1/2/3.

Before recovery inspect the actual parent/child processes, immutable hashes,
logs and `STOP_REQUESTED`. Never duplicate training, resume a completed trial,
or restart any completed older campaign. Results and completion verification
must be read before claiming improvement.

## Completed

Both fits completed: 101 epochs each, best epoch 1. Primary480 missedNG/falseNG:
parent2/0, SERVICE0/0, control0/1, context1/0. Current1615val: parent3/0 (4errors),
SERVICE1/2 (3errors), control1/3 (5errors), context2/1 (4errors). Context did not
beat control under the fixed NG-first fitness and failed release gates. Retain
the current SERVICE field-test model and the separate HA best111 copy.

Full1615 PT/ONNX argmax parity passed for both models. Independent all5393
cached-vs-actual-ONNX argmax verification also passed; CUDA TF32 probability
rounding was identified and documented without changing any checkpoint or gate.
White28 replay passed, median1.005x/max1.060x. All protected hashes and canonical
labels/splits passed. No deployment, no resume or extra fits. Read run
`ASSESSMENT.md`, `RESULTS.md`, `errors_context.html`, `additional_diagnostics.json`
and `completion_verification.json`. The old1605 cohort is diagnostic only:
context4NG/5false vs SERVICE3/8, fewer false alarms at the cost of another NG miss.
