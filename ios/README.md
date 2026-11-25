# Gaffer's Vision iPhone app

Live tick-tock feedback running entirely on the phone: camera, MediaPipe Pose, the Core ML ball
detector and the tested per-frame logic in `swift/GaffersVisionCore`. The laptop's `src/live.py`,
on a phone.

| File | What it does | Python counterpart |
|---|---|---|
| `GaffersVision/CameraSource.swift` | front/back camera, 720p portrait, unmirrored, real timestamps, drops late frames | `FrameSource` in `src/live.py` |
| `GaffersVision/PoseDetector.swift` | MediaPipe Pose Landmarker (lite), video mode | `src/pose.py` |
| `GaffersVision/BallDetector.swift` | foot crop / letterbox, Core ML, decode + NMS, back to frame pixels | `src/onnx_detector.py`, `ball.crop_image` |
| `GaffersVision/LiveEngine.swift` | per-frame loop, READY / countdown / drill, counts, summary | `main()` in `src/live.py` |
| `GaffersVision/HeadPitchSampler.swift` | Face Landmarker on a head crop, 10 Hz, pitch for head-down | `HeadPitchSampler` in `src/face_pose.py` |
| `GaffersVision/HeadPitchMath.swift` | pitch from the face transform, head box and crop | `pitch_from_transformation_matrix`, `pose.head_box` |
| `GaffersVision/Framing.swift` | after START, the countdown waits for head-to-toe framing; drill length; RECENTER YOURSELF | `framing_problems`, `FramingGate`, `DrillFlow` |
| `GaffersVision/ContentView.swift` | screen: picture, overlay, counts, button, summary | `overlay.draw`, `DrillButton` |

Not in this first version: recording drills to video, saving a summary file. Head-down uses the pitch
cutoff in `src/config.py` (12 deg), calibrated on one clip only: note the pitch shown on
screen while looking ahead and while looking at the ball. Image preparation follows `docs/PREPROCESSING.md`.

## Build it (Mac with Xcode, one time)

**1. Tools** (in Terminal):
```bash
# Homebrew, if `brew` is not found: see https://brew.sh
brew install xcodegen cocoapods
```

**2. Model files** into the repo's `models/` folder (git-ignored):
```bash
cd ForgeGaffersVision
source ~/.venvs/gaffers-vision/bin/activate
python -m scripts.export_coreml          # writes models/yolov8n_320.mlpackage and _640
ls models/pose_landmarker_lite.task || curl -L -o models/pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
ls models/face_landmarker.task || curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

**3. Generate the project and add MediaPipe:**
```bash
cd ios
xcodegen
pod install
open GaffersVision.xcworkspace           # the .xcworkspace, not the .xcodeproj
```

**4. Signing:** in Xcode, click the blue GaffersVision project, target GaffersVision, tab
*Signing & Capabilities*: tick *Automatically manage signing*, and pick your Apple ID under
*Team* (a free Apple ID works). If it says the bundle identifier is taken, change it to something
unique, e.g. `edu.northwestern.<yourname>.gaffersvision`.

**5. Phone:** plug in the iPhone, unlock it, tap *Trust*. On the phone turn on
Settings > Privacy & Security > Developer Mode (it restarts). Pick the iPhone at the top of Xcode,
press Run (Cmd+R).

**6. First launch only:** the phone says "Untrusted Developer". On the phone: Settings > General >
VPN & Device Management > your Apple ID > Trust. Run again. Allow camera access.

With a free Apple ID the app stops opening after 7 days; press Run again to reinstall.

## Using it
- Prop the phone up in portrait, screen facing the player (front camera; the camera button
  switches to the back camera). Low, 2-3 m away, as in `docs/LIVE_RECORDING_SETUP.txt`.
- Pick the drill length (30 s by default, 45 s, 60 s or Open), press START, then walk into place. The app says
  STEP INTO FRAME (and what is missing) until head, both feet and the ball are in frame, then
  counts down 3-2-1 (stepping out goes back to waiting). A sound marks the drill's start and end
  (ring switch on). A timed drill stops itself; STOP ends any drill early.
- Bottom-left shows fps and per-frame time. Write these down: they are the phone numbers for
  `docs/IOS_PORT_PLAN.md` Phase 2.
- Left/right check: tap with only the left foot; the L count must go up.

## When a build fails
Copy the first red error from Xcode's Issue navigator (Cmd+5) and send it to Siddharth. Usual ones:
- *No such module 'MediaPipeTasksVision'*: opened the `.xcodeproj`; open the `.xcworkspace`.
- *Sandbox: rsync deny* / script errors: check `ENABLE_USER_SCRIPT_SANDBOXING` is NO
  (project.yml sets it; re-run `xcodegen` and `pod install`).
- *yolov8n_320.mlpackage not found* when running xcodegen: step 2 was skipped.
- After changing `project.yml`: run `xcodegen` and `pod install` again.

## Changing the logic
Never here: change Python first, regenerate the test vectors, update `swift/GaffersVisionCore`
until `swift test` passes (see its README). This app only feeds that package.
