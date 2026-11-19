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
  - Builds a baseline inter-touch interval after `ball_loss_baseline_s` of play: the median of the last `ball_loss_history_intervals` gaps, so one long gap cannot move it, a sustained slowdown raises it and faster play brings it back down.
  - A gap is "late" once it exceeds `baseline_interval / (1 - ball_loss_drop_frac)` (and at least `ball_loss_dip_min_s`). No upper limit: any late gap can be a loss.
  - Emits `ball-loss` only when a late gap is followed by a return to near-baseline pace (recovery confirmation). If late gaps come in a row, the newest one is judged.
  - `ball_loss_pace_change_gaps` (3) late gaps in a row is a pace change, not a loss; speed-ups never count as a loss. Late gaps join the baseline history only once their run is resolved, and a gap confirmed as a loss never does.
  - Called once per touch: `step(state, touch_t_s, reported_t_s, cfg)`. Gaps use touch time; `reported_t_s` only stamps the event so latency is measurable.
  - Thresholds are `ball_loss_*` fields on `Config` in `src/config.py`.
  - Wired into `metrics.step` (`_step_rhythm`): each touch event is fed in, and a loss comes out as `lost_control` with `source="rhythm"` (spatial ones have `source="spatial"`). A spatial `lost_control` or `reset` calls `ball_loss.interrupt`, which drops the last touch and any pending loss but keeps the baseline, so one incident is reported once.
- Synthetic coverage in `tests/test_ball_loss.py` (steady play, brief gap, long gap, sustained slowdown, three late gaps, slightly-late gap before a fumble, speed-up, baseline back down after speed-up, long gap in warmup, pre-baseline gap, 30/60 fps, uneven report delay, event timestamps, interrupt) and end-to-end in `tests/test_metrics.py` (long steady play, one late touch, ball escaping not double-reported), using `with_hold` / `with_ball_away` in `tests/synth.py`.

# To-do
- ~~Wire `ball_loss.step` into `metrics.step` (or `run_clip`) after each touch / frame, using real touch events.~~ Done (roadmap step 4).
- ~~Move `BallLossConfig` fields into `src/config.py` and pass them through like other thresholds.~~ Done (roadmap step 1).
- ~~Map `ball-loss` events to the existing `lost_control` label kind (or document a subtype) and de-dupe with spatial lost-control when both fire.~~ Done (roadmap steps 3-4).
- Tune thresholds on labelled `lost-in-frame`, `lost-out-of-frame`, `baseline`, and `slow-fast` clips.
- Decide whether to keep combined rate only or add per-foot baselines.
- Optional later: dynamic baseline from an initial screen bounding box; offline every-Nth-frame analysis notebooks only.

# Roadmap
Steps in order; the to-do items above map onto them.

- [x] **0. Rename** `ball-loss.py` -> `ball_loss.py` (and its test) so it imports normally.
- [x] **1. Thresholds into `config.py`.** Add the `BallLossConfig` fields to `Config` with a `ball_loss_` prefix, make `ball_loss.step` take any object with those fields (like `metrics.py`), drop `BallLossConfig`. Stays stdlib-only.
- [x] **2. Feed touch times, not a per-frame flag.** Changed to `step(state, touch_t_s, reported_t_s, cfg)`, called once per touch event using the event's `t_s`. `metrics` stamps a touch at the previous ball frame and reports it a frame later, so the current flag is off by a frame or more. Add a test with uneven frame times.
- [x] **3. Decide the design questions.** Decisions:
  - **Decided - event kind:** reuse `lost_control` and add a `source` field to `Event` (`"spatial"` / `"rhythm"`). One concept in reports and labels; detectors stay distinguishable.
  - **Decided - rate:** one combined touch rate. Per-foot baselines only if real data demands it (step 8).
  - **Decided - de-dupe:** one incident is reported once. Suppress a rhythm loss whose gap overlaps a spatial `lost_control`/`reset` episode, and clear the rhythm detector's pending state after a `reset` so the gap across a spatial loss is never judged.
  - **Decided - strictness:** keep `drop_frac=0.3` as is. An uneven touch is the wrong rhythm and should be flagged, even if that also flags some touches the touch detector failed to see.
    - *Revisit later:* if false losses from undetected touches show up on real clips (step 8), go easier on the player and stricter on the software: require a larger gap (~2.5x baseline) or a ball-position cue (ball moved away from the feet) before reporting.
- [x] **4. Wire into `metrics.step`.** Nest the ball-loss state in `metrics.State`, feed it touch events from `_step_ball`, apply the de-dupe rule. Integration tests in `test_metrics.py` with `tests/synth.py`: steady -> none, one long gap -> one loss, spatial loss -> no duplicate.
- [x] **5. Surface in `run_clip`.** `report.json` summary keeps `lost_control` (total) and adds `lost_control_by_source` (`spatial` / `rhythm`), via `metrics.count_lost_control`. The overlay shows `lost sp:N rh:N` in the top line and a red `LOST CONTROL (<source>)` banner for `LOST_BANNER_S` (1 s) from the frame the loss was reported.
- [ ] **6. Ground truth.** Get the two manifest clips into `data/clips/`; label touches and `lost_control` for at least `20261101_183403` (likely loss ~8.2-10.6 s). Record `lost-in-frame`, `lost-out-of-frame`, `slow-fast`, `repeat-touches` takes.
- [ ] **7. Evaluation script.** Small `evaluate.py` matching predicted events to labels within a time tolerance: precision, recall, latency. Also serves touch metrics.
- [ ] **8. Tune and decide.** Tune `drop_frac`, dip window, `baseline_s` on train clips, check on test clips. Measure missed-touch false losses; add per-foot baselines only if justified.
- [ ] **9. Docs and commit.** Update this file and CLAUDE.md (Status, Architecture); commit and push only after review.

# Open questions
- ~~Combined touch rate for v1, or separate left/right rates with ~5% thresholds?~~ Combined (see step 3).
- ~~Is one missed beat enough, or should a loss require a multi-touch gap?~~ One is enough for now (see step 3).
- ~~Should frequency and spatial detectors both emit `lost_control`?~~ Yes, with a `source` field and de-dupe (see step 3).
- Exact `dip_min_s`, `drop_frac` and `pace_change_gaps` once real clips are available.
- ~~After a sustained slowdown, should the baseline always retarget to the new rate?~~ It follows the median of recent gaps (bug fix from clip `20261102_135033`, where one long gap had switched the detector off).
- Duplicate touches (0.1 s apart) pull the median baseline down, which can make normal gaps look late (e.g. `IMG_1707` at 9.85 s). Fix in touch detection rather than in `ball_loss.py`.

# To look at later
- **Pending loss at the end of the drill.** Ball loss is decided live, one frame at a time, in `metrics.step` (no post-recording pass). A rhythm loss is only confirmed by the touch after play resumes, so if recording stops while one is pending (late touch, no confirming touch yet) it is silently dropped. Decide whether an end-of-session "finish" step should report it, drop it, or flag it as unconfirmed.
- **3+ same-foot touches in a row as a rhythm loss.** The tick-tock alternates feet, so a run of 3 or more touches on one foot breaks the drill even if the pace stays steady. Idea: report it as `lost_control` with `source="rhythm"` (or a new source, e.g. `"same_foot"`). Before building it:
  - Fix touch detection first. On `20261101_184645`, some runs look like detector errors rather than play: one touch counted 2-3 times (e.g. right at 17.45 / 17.54 / 17.71 s; `touch_min_interval_s` is only 2 frames at 24 fps) and possible wrong-foot assignment (4 lefts in a row at 5.27-6.70 s). This rule would turn those errors into false losses.
  - Decide the run length (3? 4?) and whether one repeat (2 in a row) stays allowed, as CLAUDE.md's drill definition says. Confirm with the client.
  - Decide how it de-dupes with a pace-based rhythm loss over the same touches.
