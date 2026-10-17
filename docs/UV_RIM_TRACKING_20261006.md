# UV rim counting experiment — 2026-10-06

User requested first retaining fluorescent-rim masks, then excluding curved
reflections of the lower/opposite face before counting. This is an OFFLINE
prototype, not a change to production HSV, White, UV policy or packaged backend.

## Method

`SERVICE/src/tools/uv_rim_counter.py`:

1. Permissive blue: OpenCV H100..135, S>=12, V>=35 with local horizontal brightness
   contrast. Adjacent low-saturation V>=110 cores can join a blue halo. Pure white
   metal without the blue halo is not accepted automatically.
2. Strong UV evidence: V>=85 and local contrast>=12. An absolute V floor alone
   is not sufficient: the track must have spatial persistence and bright support.
3. Estimate the long primary rim slope field from Hough segments. Opposite faces
   use opposite convergence directions. Align the fan-like curves using a
   count-independent coherence criterion, not a target25 count or White Mid count.
4. Find persistent ridge peaks across the whole band. Refine each trajectory over
   many rows, reject insufficient length, gaps, insufficient bright evidence and
   incompatible slope. Merge duplicate tracks of one ridge.
5. Require mask coverage>=.60, span>=.80 and strong evidence on>=.35 of band rows.
   A spanning weak candidate with some strong evidence but below the bright gate
   is marked uncertain, not silently counted as UV. Rejected candidates and
   their reasons remain in JSON and are shown as red crosses.

This does not use contour area (which can erase1px strips), minimum binary-strip
width, greedy boundary pairing, a connected-component disk count or a fixed disk
number. There is no49/50-pair cap. It does not synthesize missing UV disks to fit
the known slot grid.

## Real-image observations (not accuracy)

Report: `.work/uv_hsv_20261006/middle_regions/rim_tracking_v2/index.html`, served
at `http://127.0.0.1:8780/middle_regions/rim_tracking_v2/`.

| First White ROI | Legacy caliper | V60 caliper | New counted tracks | Uncertain |
| --- | --- | --- | --- | --- |
| Cam1 band1 | 8 | 13 | 21 | 0 |
| Cam1 band2 | 5 | 14 | 22 | 0 |
| Cam2 band1 | 15 | 18 | 21 | 0 |
| Cam2 band2 | 14 | 19 | 21 | 1 |

The Cam2 band2 extra weak trajectory lies near the illuminated metal support at
x≈191 in the band. Its mask covers about62% of rows but strong evidence only12%;
it is not counted, and is marked orange `?1` for human review. The initial looser
prototype counted this as a22nd edge; that historical run is retained as
`rim_tracking_v1`. The brighter main rim beside it remains a counted track.

Last-White reference crops return the same21/22 and21/21 counts. Adding/subtracting
20 from all BGR channels and a3x3 sigma0.6 blur on the first-reference bands also
retains those counted totals. These are sensitivity checks on the SAME TWO raw
UV images, not extra test captures or proof of recall/precision. Individual-track
membership stability is not claimed solely from equal totals.

## Validation and limits

Synthetic fixtures cover1px/thick strips, close distinct rims, counts0/1/2/4/29,
neutral white cores with/without blue halos, dark blue distractors, disconnected
mask pieces, short reflections, long opposed reflections, uniform blue background
and input nonmutation. They test mechanics only, not factory accuracy.

Both real UV captures are confirmed mixing, but exact UV ground truth per band is
still unavailable. White and UV are not same-cycle pairs. Human review of the
numbered rims and the weak Cam2 `?1` is required before claiming correct counts.
Bright parallel reflections with matching hue and direction, changed geometry,
very weak UV, severe occlusion and neutral blue-lit metal can remain ambiguous.
The program records `ground_truth_verified=False` and does not change the old
homogeneous-tray pass rule or promote zero detections to guaranteed no-UV.

No ONNX model/config edits, training, hardware commands or Nuitka builds were
performed in this experiment. The prior production backend is untouched.

Run from SERVICE with PYTHONPATH set to SERVICE:

    .venv/Scripts/python.exe -B training/evaluate_uv_rims.py --input-root ../.work/uv_hsv_20261006/middle_regions --output ../.work/uv_hsv_20261006/middle_regions/rim_tracking_v2
