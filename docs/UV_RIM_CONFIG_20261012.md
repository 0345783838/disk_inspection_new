# UV rim counter config — 2026-10-12

Edit the `UV_RIM_*` block in `SERVICE/config/config.env`. All13 user-facing rim
parameters, including white-core saturation/value limits, now load from this file.
The values initially match the October6 prototype exactly. Existing `UV_DISK_*`
keys still control the old production HSV/caliper flow and are not changed.

`load_rim_settings()` reads a fresh snapshot, defaulting to the SERVICE config
path derived from the module location rather than the current working directory.
It does not modify the global decouple config. Missing keys in older config files
retain defaults; a missing file, malformed value or invalid range raises an error
with the parameter name. Standard python-decouple environment overrides apply.
H uses OpenCV0..179; S/V use0..255; fractions use0..1 (e.g.0.60, not60); minimum
separation is an integer>=1px. Contrast must be finite and nonnegative.

`count_uv_rims()` and `uv_masks()` read config when no explicit settings are given.
`evaluate_uv_rims.py` reads it once per run and supplies the same settings to all
bands and perturbations. Its report records the actual settings, config path and
SHA256, and its HTML description is generated from the settings rather than old
hardcoded values. A different config can be selected with `--config ABSOLUTE_PATH`.

After editing, run the evaluator again to regenerate images. Existing HTML reports
do not automatically recompute. From SERVICE (set PYTHONPATH to SERVICE):

    .venv/Scripts/python.exe -B training/evaluate_uv_rims.py --input-root ../.work/uv_hsv_20261006/middle_regions --output ../.work/uv_hsv_20261006/middle_regions/rim_config_20261012

This does not integrate the experimental counter into production, copy new keys
to the packaged APP config, change the production HSV values or rebuild a backend.
The executable remains the existing version. Production integration is a separate
task after human UV-label review.
