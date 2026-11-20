## Completed edge-first assessment, 18 November 2026

Coverage review8815 is COMPLETE;398reviewed bands evaluated,116current val
select only between saved best5/last15. No extra fit. Selected new is best5,
copy s_p_edge_coverage_20261118.onnx under review artifacts; bytes match the
no-legacy best5, not a newly trained network. Installed model retained.
Val5246labeled edge crossings on two production scan lines: installed/new
raw-mask misses104/107, actual cleanup+caliper misses134/162, extras2/2,
bands with no caliper miss35/116 vs25/116. Last15 has105raw misses,179caliper
misses,3extras; internal only.4px/8px/12px sensitivity preserves ordering;
8px and12px totals identical. These are crossings, not unique physical disks.
Old8hard bands: caliper misses10/10 on162reference crossings; raw misses7/8.

Diagnostic decomposition of val actual-caliper misses: installed99already
missing in raw mask +11lost by area cleanup +24present after cleanup but
caliper-unread. New102+6+54;47of new54unread stripes are1–3px wide. No threshold
or runtime change. See caliper_miss_decomposition.json. Model is still not
better under the user's edge-completeness goal; area F1 must not override it.

Public index/old_hard_cases.html show only installed vs selected new with red
missing-position markers. Results and original training gate failures remain
unchanged historical evidence. delivery_verification.json and
access_and_arithmetic_verification.json pass; no evaluator remains active.
Segmentation annotation snapshot still matches current review. Concurrent
classifier labels may advance; do not restore them or claim live revision12
unchanged. Current deployment/model/config unchanged. Read review protocol,
RESULTS.md and latest classifier review section before further work.

Any future authorized fit should freeze missed-edge/continuity-first checkpoint
selection before training, rather than reuse the old area-heavy fitness as
primary. This does not authorize an extra fit or rewrite a completed run.
Preserve exact old loss/method unless the user authorizes a method change.
