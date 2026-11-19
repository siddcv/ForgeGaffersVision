# iPhone Port Plan

**Status (2026-11-19):** Phase 0 nearly done, Phase 1 finishing with the first real drills, and
the core of Phase 4 (the Swift logic, passing every test vector) done early. Phase 2, the first
iPhone app, is next and needs a Mac with Xcode.
**Decision:** iPhone first (native Swift, Core ML). Android comes later, if at all.

This plan turns the Python prototype in `src/` into a live, fully offline iPhone app. It builds on
`MOBILE_PLAN.md` (platform options and iOS design notes) and `FINE_TUNING.md` (ball detector data).
Where those documents and this one disagree, this one is newer.

## Progress at a glance

| Phase | Status | What exists |
|---|---|---|
| 0. Prep on Windows | Nearly done | Shared per-frame function, preprocessing spec, test vectors, ONNX export + benchmark, Core ML export script. Open: detector licence, rule gaps |
| 1. Laptop live demo | Finishing | `src/live.py` with a phone as webcam (Camo), START/STOP per drill, raw + overlay recording, camera report, manifest rows, `src/evaluate_live.py` scoring. Open: first real drills |
| 2. iOS spike | Not started | Needs a Mac with Xcode |
| 3. Ball detector on device | Not started | `scripts/export_coreml.py` ready to run on a Mac |
| 4. Port the logic | Core done | `swift/GaffersVisionCore`: Swift port of the per-frame logic, passing all 28 test vectors. Open: on-device parity test |
| 5. Product | Not started | START/STOP and countdown exist in the laptop demo, to copy |

The phone app is a native iOS app, built and installed from Xcode. For testing it is a one-screen
developer app (camera, models, overlay, count, START/STOP), installed over a cable with a free
Apple ID; no App Store. The counting logic is already in the Swift package; the app runs the
camera and the models and draws the results.

## Decisions

| Question | Decision | Why |
|---|---|---|
| Platform | Native iOS (Swift) | The advisor's rule: "if we can't do both, do iOS". A few fast, consistent iPhone models, and the Neural Engine for the ball detector. |
| Live on the laptop first, or iOS first? | A thin laptop live demo, then iOS as soon as a Mac is available. Do not optimise laptop speed. | The pipeline is already causal (one `step()` per frame, no future frames), so going live is wiring, not redesign. Laptop speed does not transfer to the phone, which runs other runtimes on other chips. On a phone, live is the first thing built, not the last. |
| Pose model | Keep MediaPipe Pose Landmarker (lite) | Officially supported on iOS (MediaPipe Tasks, `MediaPipeTasksVision`), same `.task` file. Runs on the GPU, not the Neural Engine, which leaves the Neural Engine free for the ball detector. Apache-2.0. Apple's own body pose has no heel or toe points. |
| When to replace the pose model | Only if an iPhone benchmark shows pose is the bottleneck, or the left/right foot swaps cannot be fixed | Fallback: RTMPose Halpe-26 (Apache, heel and toe points, PyTorch, so it converts to Core ML). YOLO-pose is ruled out (ankles only, AGPL). |
| Ball detector runtime | Core ML, compute units "all", fixed 320x320 input, 16-bit | The most direct route to the Neural Engine. |
| Ball detector model | Decide between YOLOX-nano and YOLOv8n before fine-tuning (see Licence) | The labels train either one; the licence and a clean Core ML conversion decide. |
| Head pitch | Compare Apple Vision face pitch with MediaPipe Face Landmarker, at 5-10 Hz | Head down is owned by a teammate; listed so the pipeline budget includes it. |
| Capture | 720p, 30 fps, late frames dropped, real presentation timestamps | The models work at 256-320 px, so 1080p only adds conversion work. Check the ball's pixel size on a 720p clip. |

## Licence (the code will be open source)
- Ultralytics YOLOv8 is AGPL-3.0. That works for an open-source app only if the whole app is released
  under AGPL with full source, and Ultralytics treats weights trained with its code as AGPL too.
- GPL-family licences conflict with App Store terms (VLC was pulled from the App Store in 2011).
  Check with Northwestern tech transfer or the advisor before training on YOLOv8.
- YOLOX-nano (Apache-2.0) and MediaPipe (Apache-2.0) avoid the question. **Default: YOLOX-nano**,
  unless it fails the Core ML conversion check or is clearly less accurate.

## Feasibility
Measured on the laptop CPU (PyTorch, 1080p frames, IMG_1710, noisy machine):

| Stage | Median per frame |
|---|---|
| MediaPipe pose lite | 34 ms |
| YOLOv8n foot crop @320 | 64-89 ms |
| YOLOv8n full frame @640 (only while searching, ~16% of frames) | 97-134 ms |
| All plain-code logic (filters, search switch, tracker, `metrics.step`) | 0.1-0.2 ms |

The same untuned YOLOv8n exported to a fixed-size ONNX model, run in ONNX Runtime on the laptop CPU
(`python -m scripts.bench_detector`, IMG_1710, 300 real foot crops). No retraining or other
optimization was needed; the export alone gives the same detections as PyTorch (242 of 242 shared
detections with IoU 1.0, 2 extra borderline ones at the 0.15 confidence threshold):

| Stage | 1 thread | 2 threads | 4 threads |
|---|---|---|---|
| Crop preprocessing (crop, resize, RGB, normalize) | 3.5 ms | | |
| Foot crop @320, model only | 40 ms | 25 ms | 21 ms |
| Full frame @640, model only (searching, ~7% of frames) | 162 ms | 101 ms | 81 ms |

Model file: 12.7 MB at 32-bit, about half that at 16-bit. On the Mac, `python -m scripts.export_coreml`
writes the `.mlpackage` files and checks them against ONNX.

Expected on an iPhone 12 or newer. These are estimates from published benchmarks for these model
families, **not yet measured on our models**:

| Stage | Estimate |
|---|---|
| Pose lite (GPU) | ~5-15 ms |
| Ball detector @320 (Neural Engine) | ~2-8 ms |
| Preprocessing (native pixel buffers) | ~1-3 ms |
| Logic | <1 ms |
| **Total** | **~10-25 ms, inside the 33 ms budget** |

Models total about 12 MB. A 60 s drill is too short for heavy thermal throttling. The plain-code
logic costs essentially nothing; the two neural networks are the whole budget.

### Frame rate matters for touch counting
The touch rules need frames close together, so on-device speed is not optional. Replaying all 9
labelled tune clips (516 tick-tocks) at lower frame rates, with `touch_max_gap_s` at 0.4 s:

| Frames processed | Tick-tock recall | False alarms |
|---|---|---|
| 60 fps | 90.3% | 23 |
| 30 fps | 89.5% | 21 |
| 15 fps | 86.4% | 19 |
| 8.6 fps | 70.7% | 17 |

Before `touch_max_gap_s` went from 0.25 to 0.4 s, 15 fps gave 74% and 8.6 fps 55%. The laptop
demo processed 8.4 fps when a 60 fps clip was played live with Face Landmarker on (live recall 57%
vs 99% for the full-rate replay of the same clip, `src/evaluate_live.py`), so laptop live counts
undercount; judge accuracy from replays. **The phone must hold close to 30 fps.**

## Known issues the port must handle

| Issue | Fix |
|---|---|
| Tick-tocks are reported ~0.34-0.46 s late (they wait for the other foot's answer), above the 200-300 ms live target | Show contacts instantly and confirm tick-tocks quietly, or count on arrival at the other foot instead of its return (about half the delay). Decide in Phase 0. |
| Pose runs before the ball detector (the crop needs the feet) | Start serial. If too slow, build the crop from the previous frame's feet and the Kalman-predicted ball so pose (GPU) and ball (Neural Engine) run in parallel. |
| Python crops may extend past the frame edge with gray padding; Vision's `regionOfInterest` cannot | Clamp the crop to the frame on iOS, and test the effect in Python first. |
| The selfie camera mirrors the image, which swaps left and right | Feed the models unmirrored frames; mirror only the on-screen preview. |
| MediaPipe swaps the feet on some clips (night, narrow stance) | A model problem, not a platform one; it follows the app to iOS. Track separately. |
| The cache step runs both detectors every frame | Dev only. Live code runs the full frame only while searching, as `search_step` already assumes. |
| INT8 quantization can drop low-confidence detections | Stay at 16-bit until the fine-tuned detector's INT8 recall is checked on cached crops. |
| Head-down thresholds (17 / 22 deg Face Landmarker pitch) come from one clip; on 20261102_134206 pitch never rose above 20.5 deg, so head-down showed for the whole clip | Calibrate per player or per setup (e.g. relative to the player's upright pitch at the start of a drill). Teammate-owned; needs pitch readings from more setups. |
| Foot attribution: the "direction" rule credits the foot the ball moved toward even when that foot is far away (false tick-tock at 29.13 s on 20261102_134206) | Use the direction foot only if it is within touch distance, else the nearest foot. Fix in Python, score on all tune clips, regenerate vectors, update Swift. |
| Phone-as-webcam apps send landscape video, which pushes the phone back and shrinks the ball | Laptop demo only: Camo portrait setting or `--rotate`. The iPhone app captures portrait directly. |

## Phases

### Phase 0: now, on Windows (no Mac)
1. [x] **One shared per-frame function.** `FramePipeline.step` in `src/pipeline.py`, used by replay
       and (next) live, and mirrored later in Swift. Full-frame candidates are requested only when
       `ball.needs_full_frame` says so (6.7% of frames across the cached clips).
2. [x] **Preprocessing spec** (`docs/PREPROCESSING.md`): input sizes, crop rules, padding, channel
       order, normalization. The Swift version must match it exactly.
3. [ ] **Detector choice:** ONNX export done and checked (same detections as PyTorch);
       `scripts/export_coreml.py` is ready but has not been run on a Mac yet (it does not need
       Xcode, only Python + `coremltools`). Still open: the licence answer (YOLOv8n vs YOLOX-nano).
       Fine-tuning waits until the drill results show the detector is a big share of errors.
4. [x] **Portable test vectors** (`tests/vectors/`, see its README): 25 metric scenarios, the
       filters, the search switch, and 10 s of a real clip through `FramePipeline.step`, with
       Face Landmarker pitch as an input. Regenerated after each intended logic change.
5. [ ] **Fix the rule gaps while they are cheap in Python:**
       - [x] Frame-rate robustness: `touch_max_gap_s` 0.25 -> 0.4 s (see Feasibility).
       - [x] Drill window: START/STOP per drill, and each recorded drill writes its own manifest row.
       - [ ] Foot attribution far from the ball (Known issues).
       - [ ] Starting from rest (TODO in `metrics.py`; the miss at 1.42 s on 20261102_134206).
       - [ ] The tick-tock delay on screen (~0.4 s): decide what the live screen shows.

### Phase 1: thin laptop live demo
- [x] `src/live.py`: camera -> the shared per-frame function -> live overlay window. On the laptop
      (1080p clip played live, 2 detector threads, before Face Landmarker): ~71 ms per frame.
- [x] Phone as a webcam: runs on a teammate's Mac with Camo.
- [x] Head tracking live: Face Landmarker pitch at 10 Hz, shared with replay (`HeadPitchSampler`).
- [x] START / STOP per drill (button or space, 3 s countdown); each drill saves its raw video,
      a timestamps sidecar, the overlay video as shown, a summary and a manifest row.
- [x] Camera report at start (size, orientation, delivered fps; landscape warning).
- [x] `src/evaluate_live.py`: scores a live drill next to a full-rate replay of its recording and
      sorts every error into both / live only / replay only.
- [ ] First real drills: hand count vs screen, camera placement, how the tick-tock delay feels,
      then label and score them.

### Phase 2: iOS spike (Mac + iPhone)
Before writing any app code, two numbers on a real iPhone:
- [ ] Ball detector speed: open the exported `.mlpackage` in Xcode, connect the iPhone, Performance tab.
- [ ] Pose speed: build Google's MediaPipe Pose Landmarker iOS sample app (mediapipe-samples on
      GitHub) onto the iPhone.

Then the developer test app:
- [ ] Xcode app: AVFoundation, portrait, 30 fps -> MediaPipe Pose -> skeleton overlay, with
      `swift/GaffersVisionCore` added as a local Swift package.
- [ ] Measure the frame rate on the oldest supported iPhone: the first real feasibility number.
- [ ] **Clip-input mode next to camera mode** (AVAssetReader feeding the same pipeline). It mirrors the
      Python replay, so labelled clips run on the phone and are scored with `evaluate.py`. This is the
      most important design decision in the iOS app.

### Phase 3: ball detector on device
- [ ] Add the Core ML model and run it through Vision on the foot crop
      (`VNCoreMLRequest`, `regionOfInterest`, `imageCropAndScaleOption = .scaleFill`).
- [ ] Pick the top "ball" score in Swift; no NMS is needed for one ball.
- [ ] Full-frame search mode after `ball_search_after_s` of crop misses.
- [ ] Xcode Performance report: confirm the detector runs on the Neural Engine, and fix any
      layers that fall back to the CPU.

### Phase 4: port the logic
- [x] Swift versions of `metrics.py`, `filters.py` (Kalman, One Euro), the search switch,
      `BallTracker`, the foot crop and `FramePipeline.step`, with plain value types and all
      thresholds in one `Config` struct: `swift/GaffersVisionCore` (Swift + Foundation only).
- [x] Pass the JSON test vectors: `swift test` passes all 28 (Windows, Swift 6.4.2). A deliberate
      bug (feet swapped) gave 281 failures, so the vectors catch mistakes. Python stays the
      reference: change Python, regenerate the vectors, update Swift until they pass.
- [ ] **Parity test:** the same labelled clips through Python and the iPhone, with matching events
      and matching `evaluate.py` scores.

### Phase 5: product
- [ ] Live screen: tick-tock count by foot, pace, head-down and lost-control alerts; contacts shown instantly.
- [ ] Explicit start (countdown) and stop, which also removes setup taps and after-drill taps.
      (Built in the laptop demo; copy its behaviour.)
- [ ] 60 s sustained run on the oldest supported iPhone: frame time and how far feedback falls behind.
- [ ] Thermal step-down: ball detection every other frame with the Kalman filter filling in, then a
      lower frame rate.
- [ ] Alignment screen, then session history.

## Blockers and owners

| Item | Needed for | Status |
|---|---|---|
| A Mac with Xcode | Phase 2 onward | **Critical path.** A teammate's Mac runs the laptop demo; install Xcode on it (about 10-15 GB from the Mac App Store, needs admin rights). |
| An Apple ID | Installing on our own iPhones | Free; builds expire after 7 days. |
| Apple Developer Program ($99/year) | TestFlight builds for test players | Needed by Phase 5. |
| Licence answer for YOLOv8 + App Store | Detector choice | Ask tech transfer or the advisor; default to YOLOX-nano meanwhile. |
| Test iPhones, including the oldest we will support | Phase 2 benchmarks | Decide the oldest supported model. |

## Using Core ML: reference
Convert (macOS, Linux or WSL):

```bash
# YOLOv8n through Ultralytics
yolo export model=yolov8n.pt format=coreml imgsz=320 half=True nms=False
```

```python
# Any PyTorch model, e.g. YOLOX-nano
import coremltools as ct, torch
traced = torch.jit.trace(model.eval(), torch.rand(1, 3, 320, 320))
mlmodel = ct.convert(
    traced,
    inputs=[ct.ImageType(name="image", shape=(1, 3, 320, 320),
                         scale=1/255, color_layout=ct.colorlayout.RGB)],  # preprocessing built in
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.iOS16,
)
mlmodel.save("BallDetector.mlpackage")
```

Run (Swift), after dragging the `.mlpackage` into Xcode:

```swift
let config = MLModelConfiguration()
config.computeUnits = .all                      // let Core ML use the Neural Engine
let model = try VNCoreMLModel(for: BallDetector(configuration: config).model)

let request = VNCoreMLRequest(model: model) { req, _ in
    // raw output: take the highest ball score as this frame's detection
}
request.imageCropAndScaleOption = .scaleFill   // the crop is already square
request.regionOfInterest = footCropNormalized  // Vision crops the frame, no copy; clamp to the frame

try VNImageRequestHandler(cvPixelBuffer: frame, orientation: .up).perform([request])
```

Profile: open the `.mlpackage` in Xcode, go to the **Performance** tab, and run it on a connected
iPhone. It shows latency and which chip runs each layer.
