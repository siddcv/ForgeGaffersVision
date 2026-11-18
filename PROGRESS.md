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
- New standalone frequency detector in `src/ball_loss.py`:
  - Builds a baseline inter-touch interval after `baseline_s` of play.
  - A gap is "late" once it exceeds `baseline_interval / (1 - drop_frac)` (and at least `dip_min_s`), up to `dip_max_s`.
  - Emits `ball-loss` only when a late gap is followed by a return to near-baseline pace (recovery confirmation).
  - Another late gap instead of recovery updates the baseline (pace change); speed-ups never count as a loss.
  - Own `BallLossConfig` / `BallLossState` / `BallLossEvent`; not wired into `metrics.step` yet.
- Synthetic coverage in `tests/test_ball_loss.py` (steady play, brief gap, sustained slowdown, speed-up, pre-baseline gap, 30/60 fps).

# To-do
- Wire `ball_loss.step` into `metrics.step` (or `run_clip`) after each touch / frame, using real touch events.
- Move `BallLossConfig` fields into `src/config.py` and pass them through like other thresholds.
- Map `ball-loss` events to the existing `lost_control` label kind (or document a subtype) and de-dupe with spatial lost-control when both fire.
- Tune thresholds on labelled `lost-in-frame`, `lost-out-of-frame`, `baseline`, and `slow-fast` clips.
- Decide whether to keep combined rate only or add per-foot baselines.
- Optional later: dynamic baseline from an initial screen bounding box; offline every-Nth-frame analysis notebooks only.

# Roadmap
Steps in order; the to-do items above map onto them.

- [x] **0. Rename** `ball-loss.py` -> `ball_loss.py` (and its test) so it imports normally.
- [ ] **1. Thresholds into `config.py`.** Add the `BallLossConfig` fields to `Config` with a `ball_loss_` prefix, make `ball_loss.step` take any object with those fields (like `metrics.py`), drop `BallLossConfig`. Stays stdlib-only.
- [ ] **2. Feed touch times, not a per-frame flag.** Change to `step(state, touch_t_s, cfg)`, called once per touch event using the event's `t_s`. `metrics` stamps a touch at the previous ball frame and reports it a frame later, so the current flag is off by a frame or more. Add a test with uneven frame times.
- [ ] **3. Decide the design questions** (see Open questions), in particular:
  - Event kind: new `rhythm_loss`, or `lost_control` plus a `source` field on `Event`.
  - Combined rate first, per-foot only if data demands it.
  - De-dupe: suppress a frequency loss whose gap overlaps a spatial `lost_control`/`reset` episode; reset the frequency baseline after a `reset`.
  - Missed-touch risk: one missed touch doubles a gap then returns to pace, which reads as a loss at `drop_frac=0.3`. Options: require a larger gap (~2.5x baseline) or a ball-position cue (ball moved away from the feet).
- [ ] **4. Wire into `metrics.step`.** Nest the ball-loss state in `metrics.State`, feed it touch events from `_step_ball`, apply the de-dupe rule. Integration tests in `test_metrics.py` with `tests/synth.py`: steady -> none, one long gap -> one loss, spatial loss -> no duplicate.
- [ ] **5. Surface in `run_clip`.** Count the new events in `summarize()` / `report.json`; optional overlay marker.
- [ ] **6. Ground truth.** Get the two manifest clips into `data/clips/`; label touches and `lost_control` for at least `20261101_183403` (likely loss ~8.2-10.6 s). Record `lost-in-frame`, `lost-out-of-frame`, `slow-fast`, `repeat-touches` takes.
- [ ] **7. Evaluation script.** Small `evaluate.py` matching predicted events to labels within a time tolerance: precision, recall, latency. Also serves touch metrics.
- [ ] **8. Tune and decide.** Tune `drop_frac`, dip window, `baseline_s` on train clips, check on test clips. Measure missed-touch false losses; add per-foot baselines only if justified.
- [ ] **9. Docs and commit.** Update this file and CLAUDE.md (Status, Architecture); commit and push only after review.

# Open questions
- Combined touch rate for v1, or separate left/right rates with ~5% thresholds?
- Is one missed beat enough, or should a loss require a multi-touch gap?
- Should frequency and spatial detectors both emit `lost_control`, and how do we de-dupe close doubles?
- Exact dip window (`dip_min_s` / `dip_max_s`) and `drop_frac` once real clips are available.
- After a sustained slowdown, should the baseline always retarget to the new rate?
