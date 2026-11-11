# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status
Early skeleton: build steps 1-6 have a first version, tested on synthetic data but not yet on a real tick-tock clip, so all thresholds are placeholders. Not built yet: Face Landmarker (head pitch uses a rough pose-only estimate). `evaluate.py` scores a report against hand labels. The full project brief (scope, build order, open items) is in `../CLAUDE.local.md`, which is not in this repo and may be absent on other machines.

Gaffer's Vision is a computer-vision soccer training app. A phone camera watches one drill, the tick-tock, and reports tick-tocks by foot, tick-tocks per minute (players work toward 150 in a minute), head-down time and lost-control moments. All current work is Python on a laptop, using recorded clips. Do not build a phone app yet; the mobile OS is undecided.

## The tick-tock drill
The tick-tock is the only drill in scope; do not design for toe taps or other drills. The player stays roughly in place and knocks the ball side to side between the insides of their feet in a quick, steady rhythm, ideally with their head up. Each touch reverses the ball's horizontal direction. This is a working definition, pending client confirmation.

The headline count is **tick-tocks**: a touch that sends the ball from one foot, across the midpoint between the feet, to the other foot, which makes the next touch. Setup taps, recovery taps after losing control, and two touches in a row with one foot are not tick-tocks; they are reported separately as **contacts** (every touch on the ball). Hand labels mark tick-tocks only, at push-off, once the ball is clearly sent back; that is about 0.15 s after the first contact the pipeline marks, so `eval_touch_tolerance_s` is 0.2 s. Ball speed in m/s is not a goal metric.

## Commands
Requires Python 3.12 and `ffprobe` (part of FFmpeg) on the PATH. Keep the virtual environment outside OneDrive; thousands of synced package files slow it down and can corrupt it.

```bash
# Setup, Windows (Git Bash)
py -3.12 -m venv ~/.venvs/gaffers-vision
source ~/.venvs/gaffers-vision/Scripts/activate
pip install -r requirements.txt

# Setup, macOS: same, but create with python3.12 and activate with
source ~/.venvs/gaffers-vision/bin/activate

# Tests
python -m pytest
python -m pytest tests/test_metrics.py::test_head_down_just_under_one_second_is_not_reported

# Run a clip: writes cache/<clip>.npz and output/<clip>_stride<N>/
python -m src.run_clip data/clips/clip.mp4
python -m src.run_clip data/clips/clip.mp4 --stride 2   # 60 fps clip replayed at 30 fps, without re-running models
python -m src.run_clip data/clips/clip.mp4 --refresh    # re-run models after changing detection or crop settings

# Hand-label a clip by key press (touches, lost control, head down): writes data/labels/<clip>.csv
python -m src.label_clip data/clips/<tune|test>/<session>/<clip>.mp4

# Score a run against hand labels: writes output/<clip>_stride<N>/evaluation.json
python -m src.evaluate output/<clip>_stride1/report.json
```

Clips live in `data/clips/tune/<session>/` (used to choose thresholds) or `data/clips/test/<session>/` (held out, scored once at the end), with one row per clip in `data/manifest.csv` and hand labels in `data/labels/<clip>.csv`; see `data/README.md`.

`requirements.txt` pins every package and was generated on Windows with `pip freeze`. After adding a dependency, regenerate it and exclude pip and setuptools.

## Architecture
Per-frame pipeline: frame + real timestamp -> pose (MediaPipe Pose) -> ball detection -> head pitch (MediaPipe Face Landmarker, ~5-10 Hz) -> causal filters (One Euro for keypoints, Kalman for the ball) -> rule-based metrics.

- **Ball search modes:** the detector normally runs on a foot crop derived from pose. After repeated misses it searches the full frame. A crop miss alone never counts as lost control. `ball.py` owns the switch; `metrics.py` receives a `searched_full_frame` flag.
- **Foot crop size:** the untuned detector misses balls that look too large or too small in its input (it works at about 40-80 px). The crop is therefore square, sized so the last seen ball fills `crop_ball_frac` of it, padded with gray past the frame edge, and run at `crop_imgsz`.
- **Ball tracking:** `BallTracker` in `ball.py` picks, among several candidate boxes, the one nearest the Kalman prediction, and predicts through gaps up to `track_max_predict_s` (never while searching). Metrics receive detections only; predictions are for the overlay and future frame skipping, since feeding them to the touch rule added false touches on real clips.
- **Contacts and tick-tocks** (`metrics.py`): a `contact` event is any direction change next to a foot, reported at once. Each contact waits as pending; it becomes a `touch` (tick-tock) event when the ball is detected past the midpoint between the feet (mean of each foot's heel and toe, on every frame) and the other foot makes the next contact within `ticktock_max_arrival_s`. A same-foot contact, lost control or the time limit drops it. Tick-tocks are therefore reported about one touch gap late (~0.4 s on real clips), above the 200-300 ms live-feedback target; contacts are not. `evaluate.py` scores `touch` events.
- **Scale:** meters per pixel comes from the detected ball diameter on each frame, using the configured ball size. Speed is in-plane only.

## Hard constraints
- **Real-time and causal:** code in `src/` metrics and filters must never read future frames. The budget is about 33 ms per frame at 30 fps, sustained for a full drill. Offline shortcuts are for labeling and notebooks only.
- **One code path:** recorded clips replay through the same per-frame `step(state, observation) -> (new_state, events)` logic the phone will run. No separate batch version.
- **`metrics.py` stays pure:** standard library only, with thresholds passed in as a parameter, because it will be ported to Swift or Kotlin.
- **Thresholds live in `config.py`,** never inline, with durations in seconds so they mean the same at 30 and 60 fps.
- **Fully offline:** no cloud services or network calls in the core flow.
- **Rules first:** no Random Forest, LSTM or ByteTrack until the rule-based baseline works and its failures are known.
- **Licensing:** Ultralytics YOLOv8 is AGPL; compare Apache-licensed options before committing to a detector.
- **Git:** keep clips and model weights out; pin dependency versions.

