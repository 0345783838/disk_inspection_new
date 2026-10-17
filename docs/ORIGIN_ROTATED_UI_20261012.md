# Rotated White/UV Origin — 2026-10-12

User requested rotated Origin images to fit the existing portrait UI design.
The main panel already receives canonical/normalized frames from MainController;
the gallery also registered unrotated RAW frames as Origin thumbnails.

- Main and gallery Origin now consistently use the already normalized frame.
- Camera1: clockwise90; camera2: counter-clockwise90, for both White and UV.
- No extra WPF/image rotation is applied, avoiding a second90-degree transform.
- Main Origin images explicitly use Stretch=Uniform to preserve aspect ratio.
- Gallery entries are named White-Origin / UV-Origin (rotated). Unrotated RAW
  frames are no longer duplicated in the gallery as Origin entries.
- RAW caches and existing RAW/Normalized image-save snapshots remain intact.
- Local White testing uses the same rotated Origin as the camera path.
- AI inputs, normalization options/ROI, inference, models, UV thresholds and PLC
  flow are unchanged. A configured normalized ROI still applies as before.

Verification: C# rotation tests cover both White/UV cameras and RAW nonmutation.
UI compilation is separate from backend/model builds; no Python rebuild needed.

Release UI build and11 C# flow tests passed. The new EXE/PDB were copied to
`APP/DiskInspection/bin/BroSight_Disinspector/`; UI SHA256 is
`04599548a1720d32c18feb6e0b4def6829f30797c8f5687bb6495484eafd2dbc`.
Original UI/config backup: `.work/origin_ui_20261012/rollback/`.
The runtime config, backend and classifier hashes remain unchanged. The UI was
not launched interactively and no camera/PLC commands were issued during checks.
