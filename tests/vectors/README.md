# Portable test vectors

Inputs and expected outputs of the per-frame logic, as JSON, so the Swift (or Kotlin) port can be
checked against the Python code without running Python. Written by
`python -m scripts.export_test_vectors`; `tests/test_vectors.py` replays them in Python on every
test run, so they always match the current code.

**When the logic or config changes on purpose:** run the exporter, check the diff (one frame or
observation per line, so it shows what moved), and commit the new vectors with the code change.
The port is then expected to follow.

## Passing
A port passes a vector when it produces the same outputs:
- strings, booleans, integers, nulls and list lengths: exactly equal
- numbers: `|expected - actual| <= 1e-6 + 1e-9 * |expected|`

Do the arithmetic in 64-bit `Double`, in the same order as the Python. Tiny differences in `atan2`
or `hypot` are inside the tolerance; a missing or extra event is not.

Every file has `format_version` (now 1) and a `config` object holding every `Config` field plus
the derived `ball_diameter_m`. Build the port's config from it rather than from defaults, since
some vectors override fields. Times are seconds, positions are pixels in the upright, unmirrored
full frame, origin top-left. `null` means "none" (no ball, foot not visible, no pose).

## Files

| File | Tests | Python |
|---|---|---|
| `metrics/*.json` (25 scenarios) | the metric rules on scripted observations | `metrics.step`, `metrics.touch_rate` |
| `filters.json` | One Euro and Kalman, including a 6-frame gap with `peek` / `predict` | `filters.py` |
| `search_step.json` | the crop / full-frame search switch | `ball.needs_full_frame`, `ball.search_step` |
| `pipeline/IMG_1710.json` | everything after the models, on 10 s of a real clip (300 frames at 30 fps) | `pipeline.FramePipeline.step` |

Port and pass them in that order: each file isolates one layer, and the pipeline vector then
checks they fit together (smoothing, tracker, search switch and metrics).

### `metrics/<scenario>.json`
- `observations`: one `Observation` per frame: `t_s`, `ball_xy` `[x, y]`, `ball_diameter_px`,
  `searched_full_frame`, `left_foot` / `right_foot` (lists of `[x, y]`, heel then toe; may be
  empty), `head_pitch_deg`.
- `expected.events`: every event in order, with the `frame` (index into `observations`) whose
  `step` returned it, plus `kind`, `t_s`, `reported_t_s`, `foot`, `value`.
- `expected.final_touch_rate`: `touch_rate` at the last observation's time.

### `filters.json`
- `one_euro`: call `update(t_s, x)` on one filter per row; it returns `out`.
- `kalman`: one filter; each row is `update(t_s, z)`, `predict(t_s)` or `peek(t_s)`, returning `out`.

### `search_step.json`
`sequences.<name>` is a list of frames, each starting a fresh state at the first row. Per row,
call `needs_full_frame(state, t_s, crop, crop_det)`; then `search_step` with `full_det` only if it
was true (otherwise none). Check `needs_full_frame`, `detection`, `searched_full_frame` and the
new `state`.

### `pipeline/<clip>.json`
Cached model outputs, rounded to 3 decimals; the expected outputs were computed from the rounded
values. Per frame, `input` has `t_s`, `landmarks` (33 rows of `[x, y, visibility]`, all null
when no person), `crop` `[x1, y1, x2, y2]` or null, and `crop_candidates` / `full_candidates`
(`[x1, y1, x2, y2, confidence]`, full-frame pixels). Use `full_candidates` only on frames where
`needs_full_frame` is true: the phone would not have run the full-frame detector otherwise.
`expected` has `ball` (chosen detection), `ball_xy` (tracked position), `predicted`,
`searched_full_frame`, `ran_full_frame`, `events` and `touch_rate`.
