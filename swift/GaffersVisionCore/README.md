# GaffersVisionCore

The per-frame tick-tock logic for the phone app, ported from the Python reference in `src/`.
Plain Swift and Foundation only: no camera, models or UI, so it builds and tests on macOS, Linux
and Windows, and the iPhone app (Xcode) adds it as a local Swift package.

| Swift | Python reference | What it does |
|---|---|---|
| `Config.swift` | `src/config.py` | every threshold, in seconds |
| `Metrics.swift` | `src/metrics.py` | contacts, tick-tocks, speed, lost control, head-down, touch rate |
| `Filters.swift` | `src/filters.py` | One Euro (keypoints), Kalman (ball) |
| `Ball.swift` | `src/ball.py` (plain-code parts) | foot crop, crop-to-frame mapping, search switch, ball tracker |
| `Pose.swift` | `src/pose.py` (geometry) | landmark indices, visible points, pose-only head pitch |
| `Pipeline.swift` | `src/pipeline.py` | `FramePipeline.step`: everything after the models, once per frame |
| `Geometry.swift` | (Python tuples) | `Point`, `Detection`, `CropBox` |

## Run the tests
```bash
cd swift/GaffersVisionCore
swift test
```
The tests replay every portable test vector in `../../tests/vectors` (written by the Python code)
and pass when the Swift output matches: strings, booleans, integers and counts exactly, numbers
within `1e-6 + 1e-9 * |expected|`. See `tests/vectors/README.md` for the format.

## Changing a rule
Python stays the reference. Change the rule or threshold in `src/` first, then:
1. `python -m scripts.export_test_vectors` (from the repository root) and `python -m pytest`
2. update the matching Swift file until `swift test` passes again
3. commit the Python change, the regenerated vectors and the Swift change together

## Using it in the app
Per camera frame, after running the models:
```swift
let pipeline = FramePipeline(Config())
// landmarks: 33 MediaPipe Pose landmarks in full-frame pixels (Landmark.missing when no person)
let crop = pipeline.footCrop(landmarks, frameWidth: width, frameHeight: height)
// run the Core ML detector on the crop, map its boxes to full-frame pixels (toFullFrame)
let record = pipeline.step(
    tS: frameTimestamp, landmarks: landmarks, crop: crop,
    cropCandidates: cropBoxes,
    fullCandidates: { runFullFrameDetector() },  // called only when the search switch needs it
    headPitch: .face(faceLandmarkerPitch)
)
// record.events: contact, touch (tick-tock), speed, lost_control, reset, head_down, head_up
// record.touchRate: tick-tocks in the last minute and the live pace
```
Image preparation for the models must follow `docs/PREPROCESSING.md` exactly.
