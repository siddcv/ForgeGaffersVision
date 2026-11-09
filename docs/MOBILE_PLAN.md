# Mobile App Plan

**Status:** Draft. Platform decision pending team and client input.
**Owner:** Rayaa Williams

This page describes how the Python prototype in `src/` becomes a phone app, and how we will choose between iOS and Android. It does not change the current priority: get the rule-based pipeline working on real tick-tock clips first.

## Goal

A player props up a phone, records the tick-tock drill, and sees live feedback: touches by foot, ball speed, head-down time and lost-control moments. Everything runs on the phone, with no internet connection needed.

## What carries over from the prototype

The prototype was built so it can be ported without being redesigned:

- **One per-frame code path.** Recorded clips already replay through the same `step(state, observation) -> (new_state, events)` logic the phone will run.
- **`metrics.py` uses the standard library only.** It can be translated line by line into Swift or Kotlin.
- **Thresholds live in `config.py`**, in seconds, so they work the same at 30 and 60 fps.
- **Tests in `tests/`** define the expected behavior. The ported metrics should pass the same test cases.
- **`evaluate.py`** scores any run against the hand labels, so phone results can be checked against the Python results on the same clips.

## Pipeline on the phone

| Stage | Prototype (Python) | iOS | Android |
|---|---|---|---|
| Camera frames | OpenCV reads a clip | AVFoundation, 720p at 30 fps | CameraX |
| Body pose | MediaPipe Pose Landmarker (lite) | MediaPipe Tasks for iOS (GPU or CPU) | MediaPipe Tasks for Android |
| Ball detection | YOLO on a foot crop | Detector converted to Core ML (Neural Engine) | Detector exported to LiteRT (TFLite) |
| Head pitch | Rough pose-only estimate | Apple Vision face pitch, or MediaPipe Face Landmarker | MediaPipe Face Landmarker |
| Smoothing | Kalman + One Euro filters (`filters.py`) | Ported to Swift | Ported to Kotlin |
| Metrics | `metrics.py` | Ported to Swift | Ported to Kotlin |
| Feedback UI | `overlay.py` drawing on frames | SwiftUI overlay | Jetpack Compose overlay |

The timing budget is about 33 ms per frame at 30 fps, sustained for the full drill.

## Platform options

| Option | Pros | Cons |
|---|---|---|
| **Native iOS** (Swift) | Consistent, fast hardware across a few iPhone models; the Neural Engine runs Core ML models quickly on little battery | Requires a Mac and Xcode; testing on real devices needs an Apple developer setup |
| **Native Android** (Kotlin) | Development works on Windows or Mac; easy to install test builds | Many phone models with different cameras and speeds, so performance varies more |
| **Cross-platform** (Flutter or React Native) | One codebase for both | Live camera frames plus two on-device models need native code anyway, which adds complexity and slows each frame |

## Recommendation

Build **natively for one platform first**, then port to the second once the drill feedback is accurate. Cross-platform frameworks save little here, because the hard parts (camera, models, per-frame timing) are native on both platforms.

Choose the first platform using these questions:

1. **Test devices:** Which phones do the team and our test players own? We need real devices to measure speed and accuracy.
2. **Clients:** Does Marcelo Worsley or Herminio Bodon have a platform preference for the players they work with?
3. **Development setup:** Who on the team has a Mac for iOS work?
4. **Target players:** Which phones do the players we are building for mostly use?

**Decision:** _To be filled in once the questions above are answered._

## iOS design notes

Research for the iOS option. Each recommendation should be confirmed by benchmarking on real iPhones before we commit to it.

### How the models run on Apple's chips

iPhones have three processors that can run models: the CPU, the GPU and the Neural Engine (ANE). The ANE is the fastest and uses the least battery, but only for layers it supports.

| Decision | Recommendation |
|---|---|
| Ball detector runtime | Core ML is the most direct route to the ANE. LiteRT with its Core ML delegate keeps one model file for both platforms, but more layers may fall back to the CPU. Benchmark both, then choose. |
| Which processor runs what | A model can't be forced onto the ANE. Set compute units to "all" so Core ML chooses, and any unsupported layer falls back to the GPU or CPU. Pick a detector whose layers the ANE fully supports, and confirm with Xcode's Core ML performance report, which shows where each layer runs. |
| Pose | Keep MediaPipe Pose. Apple's own body-pose model has ankles but no heel or toe points, which the foot crop and touch detection need. MediaPipe runs on the GPU or CPU, not the ANE. |
| Head pitch | Apple Vision face detection reports head pitch natively and runs cheaply on Apple hardware. MediaPipe Face Landmarker is the same on both platforms. Compare them on the `head-angles` takes. |
| Workload split | Ball detection on the ANE, pose on the GPU, face at 5–10 times a second. Spreading work across processors is how the pipeline fits in 33 ms. |

### Exporting the ball detector

| Decision | Recommendation |
|---|---|
| Input shape | Fixed 320×320. The ANE handles fixed shapes much better than variable ones. |
| Precision | 16-bit to start, Core ML's default for the ANE. Try 8-bit weights to shrink the file, and full 8-bit only if the benchmark shows a gain on target iPhones. |
| Picking the final box (NMS) | Do it in Swift, not inside the model. With one ball only the top box is needed, and NMS inside the model can run on the CPU anyway. |
| Conversion | Convert from PyTorch with coremltools. It runs on macOS or Linux but not natively on Windows, so use WSL on Windows laptops. Running and profiling the converted model needs a Mac. |
| Detector choice | Check that each candidate (YOLOX-nano, MediaPipe Object Detector, YOLOv8n) converts cleanly and runs fully on the ANE. This is another reason to compare them now, alongside licensing. |

### Camera and preprocessing

| Decision | Recommendation |
|---|---|
| Capture resolution | 720p instead of 1080p. The models resize to 256–320 px anyway, so 720p cuts conversion work, and the ball stays about 100–150 px. Check this on a 720p clip. |
| Frame rate | 30 fps, set explicitly on the camera. |
| Late frames | Drop late frames (`alwaysDiscardsLateVideoFrames`) instead of queueing them, so feedback never falls behind. |
| Timestamps | Use each camera frame's presentation timestamp, not a frame counter, matching the Python pipeline. |
| Orientation and mirroring | Feed the models upright, unmirrored frames and mirror only the on-screen preview. Otherwise left and right swap. |
| Pixel format and copies | Keep frames in the camera's native format and crop and resize them in place, without converting to images in between. |
| Preprocessing spec | Write it down now in Python: input size, letterboxing, channel order, normalization. The Swift version must match exactly, or detections get quietly worse. |

### Per-frame pipeline

| Decision | Recommendation |
|---|---|
| Crop source | Running pose, then ball detection, is simplest and has no lag. Basing the crop on the previous frame's pose lets them run in parallel on different processors. Measure the serial version first and switch only if needed. |
| Threading | Camera, then models on a background queue, then metrics, then the screen on the main thread. Never run models on the main thread. |
| Metric logic | A Swift version of `metrics.step`, using plain value types and passing the Python tests. |

### Overheating and proof

| Decision | Recommendation |
|---|---|
| Overheating | Watch the phone's thermal state and step down when it gets hot: ball detection every other frame with the Kalman filter filling in, then a lower frame rate. |
| Proof | A 60-second sustained run on the oldest supported iPhone, measuring both frame time and how far feedback falls behind. |

## Milestones

1. **Pick the platform** using the questions above.
2. **Write the preprocessing spec in Python** (input size, letterboxing, channel order, normalization) so the phone version can match it.
3. **Compare detectors:** check that each candidate converts cleanly, runs on the target hardware, and has a usable license.
4. **Camera and pose on device:** live skeleton overlay on the phone, measuring frame rate.
5. **Ball detector on device:** run the converted detector on the foot crop and check where each layer runs.
6. **Port filters and metrics:** pass the same test cases as the Python version.
7. **Feedback screen:** live touch count, ball speed, and head-down and lost-control alerts.
8. **Session history:** save each session so players can see progress over time.
9. **Prove it:** a 60-second sustained run on the oldest supported phone, then replay the same labeled clips through phone and Python and compare with `evaluate.py`.

## Risks

| Risk | Mitigation |
|---|---|
| Running pose and ball models together is too slow | Split work across processors, use lightweight models, run detection on the foot crop only, and run the head check at a lower rate |
| Detector layers fall back from the Neural Engine to the CPU | Check Xcode's Core ML performance report and choose a detector that runs fully on the ANE |
| Detector license | Ultralytics YOLOv8 is AGPL. Compare Apache-licensed detectors before shipping |
| Phone preprocessing differs from Python | Write the preprocessing spec now and test the Swift version against it |
| Left and right swap on the selfie camera | Feed models unmirrored frames and mirror only the preview |
| Results differ between Python and the phone | Reuse the same test cases and replay the same labeled clips through both |
| Phone heats up or drains battery on long drills | Watch thermal state, step down detection rate when hot, and measure a sustained run early |

