ForgeGaffersVision - head-positioning patch v3

Apply from the root of the existing ForgeGaffersVision checkout on
feature/head-tracking. This patch overlays individual files; do NOT replace the
whole src folder.

What changed:
- Uses MediaPipe Pose head_box to crop the player's head.
- Makes a square crop with 1.5x padding and resizes it to 256x256 before Face Landmarker.
- Uses provisional clip-specific hysteresis thresholds: head-down <= 15 degrees,
  head-up > 20 degrees. These are based on the six crop samples already tested:
  manually-down timestamps produced ~4.5-10.9 degrees; sampled outside timestamps
  produced ~30-34 degrees. The full-clip diagnostic must validate these thresholds.
- Diagnostic now sweeps both lower-pitch and higher-pitch directions.
- Clarifies that transformation-matrix pitch is camera-relative and needs calibration.

Apply:
  unzip -o ~/Downloads/ForgeGaffersVision-head-positioning-patch-v3.zip -d .

Then run:
  python -m pytest -q
  python -m src.run_clip data/clips/tune/2026-11-02_classroom/20261102_134245.mp4 --refresh
  python -m src.diagnose_head_pose 20261102_134245
  python -m src.evaluate output/20261102_134245_stride1/report.json

Do not use the provisional thresholds as final until the full-clip diagnostic and
manual-label evaluation have been reviewed. Original labels and output files are
not included or overwritten by this patch.
