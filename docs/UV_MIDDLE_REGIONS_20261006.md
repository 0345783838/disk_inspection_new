# UV: two faces of the middle support — 2026-10-06

The user requested moving the two UV inspection bands to the brighter rims beside
the central support (marked red on the Cam1 full-frame overlay).

`_check_white` now anchors both UV regions to the median transformed row-2 point box:
Band1 above row2, Band2 below row2 in the normalized/rectified frame. Each retains
height 2x the point box and the existing full crop width. The support itself is excluded.
Coordinates follow White geometry rather than fixed RAW pixels. Cam1/Cam2 rotations
are unchanged, so RAW left/right order differs between cameras.

White classification/segmentation/crop/verdict, Mid1/Mid2, models, production HSV,
caliper parameters and the existing UV decision policy are not changed by this edit.
The public API still returns CropBox/UvBox1/UvBox2/Mid1/Mid2. A subsequent White
inspection supplies the new UV geometry; old saved geometry must not be reused.

## Evidence

- Fresh evaluation: `.work/uv_hsv_20261006/middle_regions/`.
- Report: `http://127.0.0.1:8780/middle_regions/`.
- `regions/first/` and `regions/last/`: full RAW, normalized and rectified overlays
  for both cameras. Renderer verifies the rectangles select the actual cached band pixels.
- The original outer-region report remains available at port8780 root.
- 67 SERVICE regression tests pass, including the new middle-row anchoring test.

Raw caliper pair counts, not ground-truth disk counts:

| Camera/reference | Legacy HSV band1 / band2 | Offline V60 candidate |
| --- | --- | --- |
| Cam1 first | 8 / 5 | 13 / 14 |
| Cam2 first | 15 / 14 | 18 / 19 |
| Cam1 last | 10 / 6 | 13 / 16 |
| Cam2 last | 16 / 15 | 18 / 18 |

Both samples are confirmed mixing, but no exact UV count per band and no same-cycle
White/UV pair are available. Neither raw count nor borrowed-White API verdict is an
accuracy metric. The old HSV still misses visible rims in these brighter regions;
the offline candidate is NOT deployed simply because it returns a higher count.

## Rollback and packaged verification

Pre-change source/backend/config backup: `.work/uv_middle_regions_20261006/rollback/`.
Build and deployment evidence are kept under `.work/uv_middle_regions_20261006/`.
`SERVICE/training/smoke_uv_middle_backend.py` checks a supplied packaged runtime
against the fresh source evaluation via inspection HTTP routes only. It verifies
all non-image White/UV outputs against a fresh source call, plus rendered images for
both cameras, refuses an occupied test port, and stops only its own temporary
backend. It does not operate cameras, lights or PLC.

Packaged deployment completed: only
`APP/DiskInspection/bin/BroSight_Disinspector/plugin/main.exe` was replaced.
SHA256: `c4e6304410692e2aa89c2299f9206dec48d9426e267adc068a20bbe9ef18e65a`.
Config and every model asset match the pre-deployment staged copies, including
classifier `ce0a150f00a56b78bbc4a07bbf3ad28862e7de7edd78a77bd0ee9f2d6a0d72bd`.
`stage_smoke.json` and `deployed_smoke.json` record checks on both cameras.
UV render pixels match exactly. White Cam1 matches exactly; Cam2 differs in only
5 pixels at x619..620/y915..917 of a 1862x2051 rendered frame. This rasterization
difference is recorded (16-pixel visualization tolerance), not hidden; all structured
outputs must match, including verdict, geometry, mids and distance metrics.

The first Nuitka build crashed in python310.dll. The successful retry uses the
existing runtime/compiler with `--standalone --msvc=latest --jobs=4 --low-memory
--include-package=uvicorn`; it follows required source imports rather than forcing
every unused src module. Build report: `build-retry-report.xml`.

`SERVICE/main.py` allows `DISK_API_PORT` as an environment override solely to allow
isolated tests; production default stays8386. Smoke tests use8387. The existing
source backend on8386 (PID4652 at verification) was NOT interrupted. It must be
restarted to load the edited source/new UV geometry; each UV inspection should use
fresh geometry from the subsequent White response, not the old cached UvBox values.
