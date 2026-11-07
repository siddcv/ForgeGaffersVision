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

## Pipeline on the phone

| Stage | Prototype (Python) | iOS | Android |
|---|---|---|---|
| Camera frames | OpenCV reads a clip | AVFoundation | CameraX |
| Body pose | MediaPipe Pose Landmarker (lite) | MediaPipe Tasks for iOS | MediaPipe Tasks for Android |
| Ball detection | YOLO on a foot crop | Detector exported to Core ML | Detector exported to LiteRT (TFLite) |
| Smoothing | Kalman + One Euro filters (`filters.py`) | Ported to Swift | Ported to Kotlin |
| Metrics | `metrics.py` | Ported to Swift | Ported to Kotlin |
| Feedback UI | `overlay.py` drawing on frames | SwiftUI overlay | Jetpack Compose overlay |

The timing budget is about 33 ms per frame at 30 fps, sustained for the full drill.

## Platform options

| Option | Pros | Cons |
|---|---|---|
| **Native iOS** (Swift) | Consistent, fast hardware across a few iPhone models; Core ML runs models efficiently | Requires a Mac and Xcode; testing on real devices needs an Apple developer setup |
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

## Milestones

1. **Pick the platform** using the questions above.
2. **Camera and pose on device:** live skeleton overlay on the phone, measuring frame rate.
3. **Ball detector on device:** export the chosen detector and run it on the foot crop.
4. **Port filters and metrics:** pass the same test cases as the Python version.
5. **Feedback screen:** live touch count, ball speed, and head-down and lost-control alerts.
6. **Session history:** save each session so players can see progress over time.
7. **Compare against the prototype:** run the same clips through both and check that results match.

## Risks

| Risk | Mitigation |
|---|---|
| Running pose and ball models together is too slow | Use lightweight models, run detection on the foot crop only, and run the head check at a lower rate |
| Detector license | Ultralytics YOLOv8 is AGPL. Compare Apache-licensed detectors before shipping |
| Results differ between Python and the phone | Reuse the same test cases and replay the same labeled clips through both |
| Phone heats up or drains battery on long drills | Measure on real devices early and reduce frame size or rate if needed |
