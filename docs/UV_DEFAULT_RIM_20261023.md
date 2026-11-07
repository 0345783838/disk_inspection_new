# Default UV rim tracking and mixing audit — 23 October 2026

Current temporary config, explicitly requested later on 23 October:
`UV_RIM_MIN_SPAN=0.70` and `UV_RIM_MIN_COVERAGE=0.50`.
Replaying selected pairs 07/14 with these values returns UV 20/20 and 18/17,
matching their White counts; production/debug images and results agree.
Pair 07 still has one separate uncertain trajectory that is not included in count.
Results are in `.work/uv_default_new_20261023/threshold_070_050/`.
The initial audit below used the previous thresholds 0.80/0.60 and is historical.

The local source now uses the existing persistent rim tracker for both
`check_disk_uv` and `check_disk_uv_debug`. The user requested development from
the current local branch, a default-true switch, replay of the other White/UV
pairs, and an explanation for apparently matching counts reported as mixing.

## Configuration and runtime behavior

`SERVICE/config/config.env` contains `UV_USE_RIM_TRACKING = True`. A missing key
also defaults to true. Set false to explicitly use the previous HSV/caliper
counter. `BaseService` takes the switch and a frozen `RimSettings` snapshot from
the same active config used for models and other service settings. Restart the
source backend after editing its config. The new tracker ignores the legacy
HSV/min-area fields in a debug request; production and debug use the same
`UV_RIM_*` values. There is no silent fallback to the old algorithm on errors.

The rim algorithm and all 13 default rim thresholds are unchanged. Its count
does not use the expected White count or force 25 disks. White geometry,
classification, segmentation, models, classifier labels/splits and PLC result
policy are unchanged. The source changes have not been compiled into or deployed
over the installed `plugin/main.exe`; adding a key to an old executable's config
alone cannot make that executable run the new code.

Both UV response DTOs now include `UvAlgorithm`, `UvCounts`, `WhiteCounts`,
`UvUncertainCounts` and `UvRequiresReview`. `UvCounts` always contains the two
raw counts. The legacy `CountUvDisk` convention remains zero on PASS to preserve
APP compatibility. New rendered results state UV/White raw counts for each band;
logs include the algorithm, counts, uncertainty and verdict when info logging is enabled.

## Replay and findings

Artifacts: `.work/uv_default_new_20261023/index.html`, `report.json`, `pairs.csv`,
`mixing_investigation.json`, `legacy_fallback_parity.json` and `app_1143.json` /
`app_1053.json`. Original audit inputs and historical reports are preserved.

There are 28 White files and 15 UV files in ANH_18_10; two Cam1 UV files are
identical, so these are 14 unique UV contents. Same-camera nearest timestamps
also rank first by image-structure similarity. This supports the pair candidates
but does not prove shared trigger/tray IDs. All two-camera combinations remain
replay hypotheses rather than confirmed production cycles.

| Candidate pair | White counts | New UV counts | UV verdict |
| --- | --- | --- | --- |
| Cam1 White `Image_20261018114302478.bmp` → UV `Image_20261018114316119.bmp` | 25 / 25 | 25 / 25 | PASS |
| Cam2 White `Image_20261018114254260.bmp` → UV `Image_20261018114323775.bmp` | 25 / 25 | 0 / 25 | mixing |
| Cam1 White `Image_20261018105248090.bmp` → UV `Image_20261018105310231.bmp` | 20 / 20 | 20 / 18 | mixing; one uncertain trajectory |
| Cam2 White `Image_20261018105239606.bmp` → UV `Image_20261018105319152.bmp` | 18 / 17 | 18 / 15 | mixing |

All 15 production UV HTTP responses exactly match the debug verdict, raw
diagnostics and rendered result pixels. One passes and 14 have count mismatches;
this is a count-policy result, not measured accuracy against reviewed UV truth.
No real response with matching per-band counts returns mixing.

The application combines both cameras. At 11:43, Cam1 passes UV, but Cam2 fails
UV, so the aggregate policy reports mixing. The actual APP's Emgu JPEG encoder,
RestSharp transport, deserializer and private `BuildResult` were exercised against
the source backend at both 11:43 and 10:53. Both runs pass all nine existing merge
and error cases. At 11:43 Cam1 still has a separate White abnormal result; that is
not a UV mixing verdict. Actual historical factory behavior cannot be assigned
to one cause without its four input frames and raw responses.

Other sources of apparent agreement:

- Old yellow labels map each UV detection to its nearest White index. Indices
  can repeat or jump to 25 with fewer than 25 UV detections.
- The comparison is per band, not just total. UV 19/21 against White 20/20 has
  equal totals but correctly fails the existing policy.
- The previously installed/source API always used the old HSV/caliper branch;
  it had no selection switch. On the full Cam1 sample it counts only 17/20.

## Short-tray investigation

On Cam1 10:53, two bright lower-band trajectories at x approximately 1609 and
1747 are rejected by the new tracker's `min_span=0.80` gate. Their mask coverage /
strong coverage / span are approximately 0.785/0.780/0.789 and
0.722/0.713/0.742 respectively. Thus the comparator receives 18 rather than 20.
They are marked in `cam1_1053_two_rejected_bright_rims.png`.

A bounded diagnostic with only `min_span=0.70`, keeping the algorithm independent
of White counts, restores this sample to 20/20. It also changes counts in other
images, and Cam2 10:53 remains 18/16 against White 18/17. This is evidence for the
rejection mechanism, not a validated replacement threshold. The diagnostic does
not modify config, default thresholds, canonical data or production files.

Follow-up visual review of the user's selected pairs 07 and 14 confirms two
bright lower-band rims are omitted in each image. Pair 07 supports UV 20/20;
pair 14 supports UV 18/17. On Cam2, x approximately 1226 fails span (0.761),
and x approximately 1382 fails coverage (0.565). The fixed 11-pixel search
window contains mask support on 193/209 and 159/209 rows respectively, but
the polynomial residual filter retains only 155/209 and 118/209 rows. Thus
the failure involves tracking/fit support, not just absent fluorescence.

The marked images and diagnostics are in
`.work/uv_default_new_20261023/review_pairs_07_14/`. A separate diagnostic
`min_span=0.70, min_coverage=0.55` restores both selected pairs without using
White counts, but also changes other images. Merely widening the tracking
window causes losses or extra detections elsewhere. None of these diagnostic
variants changes the service algorithm or config defaults.

## Verification

- 33 UV selection/config/counter tests and 23 inspection-flow tests pass.
- Eight slot-geometry tests and the segmentor concurrency test pass (65 total).
- All 28 White HTTP responses and every rendered pixel match the saved
  pre-change source replay.
- All 15 UV production/debug results and images agree; six concurrent two-camera
  UV rounds match the corresponding serial structured outputs.
- Explicit false preserves all pre-change counts, verdicts and image pixels on
  all 15 real UV files.
- Two actual APP transport/merge replays pass, with 18 merge checks in total.
- All 55 protected input/canonical/runtime files retain their hashes.

No camera/PLC commands, training, dataset edits, model replacement or deployment
were performed. The report server is a loopback static review server only.
