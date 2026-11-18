# iPhone Port Plan

**Status:** Plan. Phase 0 can start now on Windows; Phase 2 onward needs a Mac.
**Decision:** iPhone first (native Swift, Core ML). Android comes later, if at all.

This plan turns the Python prototype in `src/` into a live, fully offline iPhone app. It builds on
`MOBILE_PLAN.md` (platform options and iOS design notes) and `FINE_TUNING.md` (ball detector data).
Where those documents and this one disagree, this one is newer.

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

## Phases

### Phase 0: now, on Windows (no Mac)
1. [ ] **One shared per-frame function.** Move the per-frame body of `replay()` in `run_clip.py`
       into one function used by replay and live, and mirrored later in Swift.
2. [ ] **Preprocessing spec** (`docs/PREPROCESSING.md`): input sizes, crop rules, padding, channel
       order, normalization. The Swift version must match it exactly.
3. [ ] **Detector choice:** check the licence, and convert each candidate to Core ML in WSL
       (`coremltools` does not run natively on Windows; conversion works in WSL, running and profiling
       need a Mac). Then run the fine-tuning pilot in `FINE_TUNING.md`.
4. [ ] **Portable test vectors:** export the `tests/test_metrics.py` cases as JSON (input frames,
       expected events) so the Swift port passes the same answers.
5. [ ] **Fix the rule gaps while they are cheap in Python:** starting from rest (TODO in
       `metrics.py`), crossings not seen between contacts, the drill window, and the tick-tock delay.

### Phase 1: thin laptop live demo (a few days)
- [ ] `src/live.py`: camera -> the shared per-frame function -> live overlay window; records the
      raw video so live sessions can be labelled and replayed.
- [ ] Use the phone as a webcam (Camo, DroidCam or Iriun) placed low and in front, like the clips.
- [ ] Goals: feel the tick-tock delay live and try camera placement for the alignment screen.
      Not performance.

### Phase 2: iOS spike (Mac + iPhone)
- [ ] Xcode app: AVFoundation at 720p / 30 fps -> MediaPipe Pose -> skeleton overlay.
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
- [ ] Swift versions of `filters.py` (Kalman, One Euro), `search_step`, `BallTracker` and
      `metrics.py`, using plain value types, all thresholds from one config struct.
- [ ] Pass the JSON test vectors.
- [ ] **Parity test:** the same labelled clips through Python and the iPhone, with matching events
      and matching `evaluate.py` scores.

### Phase 5: product
- [ ] Live screen: tick-tock count by foot, pace, head-down and lost-control alerts; contacts shown instantly.
- [ ] Explicit start (countdown) and stop, which also removes setup taps and after-drill taps.
- [ ] 60 s sustained run on the oldest supported iPhone: frame time and how far feedback falls behind.
- [ ] Thermal step-down: ball detection every other frame with the Kalman filter filling in, then a
      lower frame rate.
- [ ] Alignment screen, then session history.

## Blockers and owners

| Item | Needed for | Status |
|---|---|---|
| A Mac with Xcode | Phase 2 onward | **Critical path.** Find who has one, or ask whether Northwestern can lend one. |
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
