# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status
Early skeleton: build steps 1-6 have a first version, tested on synthetic data but not yet on a real tick-tock clip, so all thresholds are placeholders. Not built yet: Face Landmarker (head pitch uses a rough pose-only estimate) and `evaluate.py`. The full project brief (scope, build order, open items) is in `../CLAUDE.local.md`, which is not in this repo and may be absent on other machines.

Gaffer's Vision is a computer-vision soccer training app. A phone camera watches one drill, the tick-tock, and reports touches by foot, ball speed in m/s, head-down time and lost-control moments. All current work is Python on a laptop, using recorded clips. Do not build a phone app yet; the mobile OS is undecided.

## The tick-tock drill
The tick-tock is the only drill in scope; do not design for toe taps or other drills. The player stays roughly in place and knocks the ball side to side between the insides of their feet in a quick, steady rhythm, ideally with their head up. Each touch reverses the ball's horizontal direction, and touches usually alternate feet, though repeats on one foot must still be counted correctly. This is a working definition, pending client confirmation.

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
```

`requirements.txt` pins every package and was generated on Windows with `pip freeze`. After adding a dependency, regenerate it and exclude pip and setuptools.

## Architecture
Per-frame pipeline: frame + real timestamp -> pose (MediaPipe Pose) -> ball detection -> head pitch (MediaPipe Face Landmarker, ~5-10 Hz) -> causal filters (One Euro for keypoints, Kalman for the ball) -> rule-based metrics.

- **Ball search modes:** the detector normally runs on a foot crop derived from pose. After repeated misses it searches the full frame. A crop miss alone never counts as lost control. `ball.py` owns the switch; `metrics.py` receives a `searched_full_frame` flag.
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
