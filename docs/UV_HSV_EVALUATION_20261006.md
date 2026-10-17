# UV HSV evaluation — 2026-10-06

This records the original OUTER-region experiment. The subsequent user-requested
middle-region change and packaged deployment are documented in
`docs/UV_MIDDLE_REGIONS_20261006.md`; the original measurements below are retained.

## Scope

User requests checking the current UV algorithm on the new standalone HA UV frames, borrowing crop
coordinates returned from a White image. White is temporarily accepted and must remain unchanged.
There is one UV BMP per camera and no same-cycle White pair. After the first evaluation the user
confirmed **both UV images are mixing**. Per-disk/per-band UV counts remain unconfirmed; the earlier
JSON evaluations are retained as historical records rather than rewritten.

Sources: `D:/huynhvc/OTHERS/disk_checking/disk_checking/new_data/HA/UV1/Image_20260918134008664.bmp`
and `UV2/Image_20260918133942315.bmp`. Cam1 uses CW90; Cam2 uses CCW90, identical to White normalization.

`SERVICE/training/evaluate_uv_hsv.py` invokes the actual source White service to retrieve CropBox,
UvBox1/2 and Mid1/2, then the actual UV service for each HSV candidate. It also records raw caliper
pair counts, mask pixels, locations and parameter ablations without changing shared caliper settings.
The raw UV files, White/model files and existing caliper edits are not modified.

Two White choices are exercised per camera:

- First White: Cam1 `Image_20260918130338631.bmp`, Cam2 `Image_20260918133316199.bmp`.
- Last/closest-in-time White: Cam1 `Image_20260918130514462.bmp`, Cam2 `Image_20260918133502603.bmp`.

These are different reference images, NOT additional UV samples or an independent validation set.
Their White disc counts are not UV ground truth; we do not force25 disks per row.

## Full-frame region overlays after mixing confirmation

`SERVICE/training/render_uv_regions.py` projects the actual integer-sliced Band1/Band2 boundaries
and caliper scanlines back through the inverse White perspective warp, then through the inverse
camera rotation onto the original2592x1944 UV frames. It verifies the rendered rectangles select
exactly the cached pixels used in the evaluation, not estimated boxes.

`.work/uv_hsv_20261006/regions/first/Cam1/` and `Cam2/` contain full RAW, normalized and rectified
overlays; `regions/last/` shows the alternate latest White reference. Cyan is the borrowed CropBox,
green is Band1, magenta is Band2, yellow is the actual95%-width scanline. Only the two bands are
evaluated, not the entire cyan crop or every visible disk. `regions/geometry_overlay.json` records
coordinates and the user's coarse mixing confirmation. Native UV originals are unchanged.
The HTML report now displays these full-frame raw overlays before the small band comparisons.

## Algorithm and findings

Current flow: perspective crop from White → two bands → BGR-to-HSV/inRange →3x3 closing → contour
area filter20 → one center scanline per band with thickness variants → greedy opposite-edge pairing.
Band1 uses min-edge5/max-pair50/thickness1,3; band2 inherits min-edge4/max-pair25/thickness3,5,7.
Scan length is95% of band width; maximum50 returned pairs can cap severe noise.

Baseline HSV is `(107,35,40)..(119,150,250)`. On the new images it catches real blue edges but also
dark bluish background, fragments and reflected/intersecting rims. Increasing Smax alone has no
observable count benefit in these samples. Allowing V255 retains bright edge pixels, but also does
not fix raw counting. Lowering S/V and widening H naively makes background noise worse.

An OFFLINE balanced test candidate is `(105,35,60)..(125,150,255)`:
broader blue hue, higher V floor to reject dark background, and no upper-V clipping.
It reduces some background but still miscounts; it is **not selected as production-ready or deployed**.
V80 cleans more background but loses thin/dim edge detections, especially Cam2.

Raw caliper pairs (not ground-truth UV counts):

| Camera/reference | Legacy band1 / band2 | V60 test band1 / band2 | V80 test band1 / band2 |
| --- | --- | --- | --- |
| Cam1 first White |21 /36 |21 /30 |21 /16 |
| Cam2 first White |20 /25 |20 /26 |18 /13 |
| Cam1 last White |22 /40 |21 /32 |21 /19 |
| Cam2 last White |19 /24 |18 /22 |15 /11 |

Visual overlays show multiple caliper markers on fragments of the same bright rim and obvious
blue bands with missing markers. Changes in reference geometry also change scanline intersections.
The first/last White counts differ (Cam1 first25/24, last20/20; Cam2 first25/25, last22/23), which is
another reason not to score borrowed-geometry API pass/fail as correct classification.

Thin-edge ablation: on the first reference, reducing UV min-edge to1px raises legacy raw counts
from21→23 (Cam1 band1) and20→40 (Cam2 band1); Cam1 band2 rises36→50, Cam2 band2 reaches50.
This demonstrates count sensitivity/overcounting, not a successful fix. Full-width scanning makes
Cam2 band1 legacy20→21, but does not solve the background/fragment problem.

Runtime UV decision currently passes when total UV count is0, or both UV band counts equal their
White counts. Accepted homogeneous trays deliberately report `CountUvDisk=0`, even with nonzero
detections. Thus zero reported count must not be read as zero detections, and pass/fail in this
unpaired experiment must not be read as accuracy. A missed-all UV mask could also look homogeneous
to that rule; changing that product policy is outside this threshold evaluation.

## Deliverables / status

- `.work/uv_hsv_20261006/index.html`: static visual report, native raw/mask/overlay links for4 bands,
  all18 HSV candidates, first/last reference counts and explicit limitations.
- `diagnostics_first/report.json`: real source UV responses and raw counts plus caliper ablations.
- `v_candidates_last/report.json`: same two UV images with latest White reference.
- `baseline_last/`: original nine-candidate baseline preserved.
- `hsv_candidate.env`: offline candidate values only; **not copied to either production config**.
- `SERVICE/training/render_uv_hsv_report.py`: regenerates the visual report.
- `SERVICE/tests/test_uv_hsv_evaluation.py`:3 synthetic helper tests for dark-blue rejection,
  non-blue rejection and shared-caliper nonmutation. These are not real-image accuracy tests.

Production SERVICE/APP HSV thresholds remain unchanged. No hardware, PLC commands, UV-policy
change, model training, White change or packaged-backend rebuild was performed.
The existing user edit to `caliper_advanced_new.py` (visualization line thickness) is preserved.
All66 SERVICE regression tests pass. The report's69 image/data links resolve and the loopback
report at `http://127.0.0.1:8780/` returns HTTP200. Config, source service and shared-caliper hashes
still match the evaluation baseline; the deployed White classifier hash is unchanged.

Conclusion: HSV color extraction remains useful, but the existing HSV+single-scanline count is not
demonstrably reliable on these native new images. No tested single HSV interval fixes all4 bands.
Do not deploy a threshold merely because its raw count approaches25 or the borrowed White count.
Next work should isolate a UV-only multi-scanline/continuous-rim grouping counter, keeping White
unchanged, and check paired same-tray White+UV with known all-UV/all-nonUV/mixing cases.

Run from SERVICE with `PYTHONPATH` set to SERVICE:

    .venv/Scripts/python.exe -B training/evaluate_uv_hsv.py --ha D:/huynhvc/OTHERS/disk_checking/disk_checking/new_data/HA --output ../.work/uv_hsv_20261006/diagnostics_first --white-reference first

The cached White geometry is tied to its source hash, but is not automatically refreshed after model,
config or service edits. Use a fresh output directory for a new baseline; stored report hashes identify
the code/config used in this evaluation. Production deployment requires a separate evidence check.
