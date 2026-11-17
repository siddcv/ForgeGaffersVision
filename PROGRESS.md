# Ideas
- Frequency of the touches. Any time there is a disruption, we count it as a lost possession
- Any time that the ball is not on the screen, the user has lost control and is counted as a lost ball
- Starting baseline can just be that both frequencies of the touches per minute of the left and right foot (5% threshold maybe)
- If everything is working, test out a initial bounding box to test whether baseline can be dynamic
- A disruption is basically when the frequency changes and is not immediately repeated after

Ideas to explore:
- Screen positioning bounding box
- A frame by frame (maybe every 5th) breakdown for a more precise analysis
- Take the difference in frequency between every consecutive X frames (for each foot) and compare it against a set bound. If the difference is more negative than a set threshold (and only for a short while), then it should be a ball loss.
- If frequency increases, not a ball loss. If it decreases and stays down, not a ball loss. Only a ball loss if decreases for a few seconds/ frame (need to define this period), and jumps back up, then a ball loss.

# Currently implemented
- Spatial / miss lost-control already lives in `src/metrics.py` (full-frame miss or ball far from feet for `lost_control_min_s`; crop-only misses do not count).
- Touch detection by foot already lives in `src/metrics.py` (horizontal reversal near a foot).
- New standalone frequency detector in `src/ball-loss.py`:
  - Builds a baseline inter-touch interval after `baseline_s` of play.
  - A gap is "late" once it exceeds `baseline_interval / (1 - drop_frac)` (and at least `dip_min_s`), up to `dip_max_s`.
  - Emits `ball-loss` only when a late gap is followed by a return to near-baseline pace (recovery confirmation).
  - Another late gap instead of recovery updates the baseline (pace change); speed-ups never count as a loss.
  - Own `BallLossConfig` / `BallLossState` / `BallLossEvent`; not wired into `metrics.step` yet.
- Synthetic coverage in `tests/test_ball-loss.py` (steady play, brief gap, sustained slowdown, speed-up, pre-baseline gap, 30/60 fps).

# To-do
- Wire `ball-loss.step` into `metrics.step` (or `run_clip`) after each touch / frame, using real touch events.
- Move `BallLossConfig` fields into `src/config.py` and pass them through like other thresholds.
- Map `ball-loss` events to the existing `lost_control` label kind (or document a subtype) and de-dupe with spatial lost-control when both fire.
- Tune thresholds on labelled `lost-in-frame`, `lost-out-of-frame`, `baseline`, and `slow-fast` clips.
- Decide whether to keep combined rate only or add per-foot baselines.
- Optional later: dynamic baseline from an initial screen bounding box; offline every-Nth-frame analysis notebooks only.

# Open questions
- Combined touch rate for v1, or separate left/right rates with ~5% thresholds?
- Is one missed beat enough, or should a loss require a multi-touch gap?
- Should frequency and spatial detectors both emit `lost_control`, and how do we de-dupe close doubles?
- Exact dip window (`dip_min_s` / `dip_max_s`) and `drop_frac` once real clips are available.
- After a sustained slowdown, should the baseline always retarget to the new rate?
