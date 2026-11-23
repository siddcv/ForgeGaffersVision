# Putting the app on an iPhone: steps that worked

What we did on 2026-11-23 to go from the repo to Gaffer's Vision running on an iPhone. It follows
`README.md` in this folder, plus the problems we hit and how we fixed them. Use it to repeat the
setup on another Mac.

## Result (2026-11-23)
- The app builds, installs over a USB cable and opens on the iPhone.
- Not measured yet: fps and ms per frame, the left/right check, a hand count against the screen,
  and head pitch readings (see "After the first run" below).

| | Version used |
|---|---|
| macOS | 27.0.1 (Apple Silicon) |
| Xcode | 27.2 (27B52) |
| XcodeGen | 2.48.0 |
| CocoaPods | 1.17.0 |
| MediaPipeTasksVision | 0.10.21 (pinned in `Podfile`) |
| coremltools | 9.0 |

Core ML export check (`output/coreml_check.json`, Mac, compute units "all"):

| Model | Size | Top-box IoU vs ONNX | Median time |
|---|---|---|---|
| `yolov8n_320.mlpackage` | 6.4 MB | 0.997 | 1.7 ms (5.8 ms CPU only) |
| `yolov8n_640.mlpackage` | 6.5 MB | 0.995 | 3.8 ms (16.9 ms CPU only) |

The Mac timings are only a hint. The iPhone numbers come from the app and from Xcode's
Performance tab.

## What you need
- A Mac with Xcode installed (Mac App Store, about 10-15 GB) and Homebrew (https://brew.sh).
- An iPhone running iOS 17 or later, and a **data** USB cable for it (USB-C or Lightning). The
  first pairing needs the cable. Wireless pairing may work without one, but we did not test it.
- A free Apple ID. The Developer Program is not needed for your own phones.
- A personal iPhone. A school or work phone that is "supervised" or "managed" may refuse apps
  from other developers.

## 1. Model files
From the repo root, with the macOS virtual environment (see the docstring in
`scripts/export_coreml.py` for its packages):
```bash
source ~/.venvs/gaffers-mac/bin/activate
python -m scripts.export_coreml
ls models/pose_landmarker_lite.task || curl -L -o models/pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
ls models/face_landmarker.task || curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
ls models
```
`models/` must contain `yolov8n_320.mlpackage`, `yolov8n_640.mlpackage`,
`pose_landmarker_lite.task` and `face_landmarker.task` (the `.onnx` files are not used by the app).
These files are git-ignored, so every Mac has to make or copy them.

## 2. Generate the Xcode project
```bash
brew install xcodegen cocoapods
cd ios
xcodegen                          # writes GaffersVision.xcodeproj from project.yml
pod install                       # adds MediaPipe, writes GaffersVision.xcworkspace
open GaffersVision.xcworkspace    # always the .xcworkspace, never the .xcodeproj
```
After any change to `project.yml`, run `xcodegen` and then `pod install` again.

## 3. Connect the iPhone
1. Plug the iPhone into the Mac with the cable and unlock it. Tap **Trust This Computer** and
   enter the passcode.
2. In Xcode, open Window > Devices and Simulators (Shift+Cmd+2) and check that the iPhone is
   listed. Wait while it says "Preparing" or "Copying shared cache" (a few minutes the first time).
3. Choose the iPhone in the device menu at the top of the Xcode window (next to "GaffersVision"),
   not "Any iOS Device" and not a simulator. The Simulator has no camera, so the app cannot run there.

## 4. Developer Mode (on the iPhone)
Developer Mode is in **Settings > Privacy & Security**, at the very bottom under "Security". It
**stays hidden until Xcode has tried to run an app on the phone**:
1. With the phone chosen in Xcode, press **Cmd+R** once. The run may stop with an error such as
   "Developer Mode disabled"; that is expected.
2. On the phone, close Settings completely, open Settings > Privacy & Security again and scroll to
   the bottom.
3. Turn on **Developer Mode**, tap **Restart**, and after the restart tap **Turn On**.

## 5. Signing (in Xcode)
1. Click the blue **GaffersVision** project in the left sidebar, then target **GaffersVision**,
   then the **Signing & Capabilities** tab.
2. Tick **Automatically manage signing** and choose your Apple ID as the **Team** (add it under
   Xcode > Settings > Accounts if it is not listed).
3. If Xcode says the bundle identifier is taken, change it to something unique, e.g.
   `edu.northwestern.<yourname>.gaffersvision`.

**Error we hit:** "Communication with Apple failed. Your team has no devices from which to generate
a provisioning profile." It only means Xcode has not seen the iPhone yet. Connect the phone
(step 3) and choose it in the device menu; the error then clears by itself. If it does not, untick
*Automatically manage signing*, tick it again and choose the Team again. There is nothing to do on
developer.apple.com; adding device IDs by hand is only for the paid program.

## 6. Build and install
Press **Cmd+R**. The first build is slow because it compiles MediaPipe and the Core ML models. If the
Mac asks for keychain access, choose **Always Allow**.

**Error we hit:** `'Landmark' is ambiguous for type lookup in this context` in `PoseDetector.swift`
and `HeadPitchSampler.swift`. Both `GaffersVisionCore` and MediaPipe define a `Landmark` type. Fixed
by writing `GaffersVisionCore.Landmark` in those two files (3 lines). If you see this error again, the
fix is not on your branch yet.

Checking the build from Terminal without signing (handy before opening Xcode):
```bash
cd ios
xcodebuild -workspace GaffersVision.xcworkspace -scheme GaffersVision \
  -destination generic/platform=iOS CODE_SIGNING_ALLOWED=NO build | tail -1   # ** BUILD SUCCEEDED **
```

## 7. Trust the developer (on the iPhone, first launch only)
On first launch the phone says *"Your device management settings do not allow using apps from
developer 'Apple Development: <your Apple ID>' on this iPhone"* (or "Untrusted Developer").
1. Open **Settings > General > VPN & Device Management**.
2. Under **Developer App**, tap **Apple Development: <your Apple ID>**, then **Trust**, then
   **Trust** again.
3. The phone needs internet for this step, because it checks the certificate with Apple.
4. Open the app again and allow camera access.

The app is now on the home screen as **Gaffer's Vision**, with iOS's blank placeholder icon (no
icon is set yet). With a free Apple ID it stops opening after **7 days**; plug in and press Cmd+R to
reinstall. To skip the cable next time, tick **Connect via network** for the phone in the Devices
window.

## What the app shows
- The front camera, mirrored for display only. A green skeleton, with cyan dots on the left foot
  and pink dots on the right, a yellow foot-crop box, and an orange box on the ball (or an orange
  circle where the tracker predicts it).
- **STEP INTO FRAME** and a grey START until the head, both feet and the ball are in view; then
  **READY** and a green START. START gives a 3-2-1 countdown, then the drill timer, then STOP.
- Top panel: tick-tocks L/R (flash green, about 0.4 s after the touch, because a tick-tock waits
  for the other foot's answer), contacts L/R (flash yellow at once), last minute, pace, head
  pitch, **HEAD UP!** and **LOST CONTROL**. **RECENTER YOURSELF** appears if you leave the frame
  mid-drill.
- Bottom left: fps, ms per frame, and TRACK or SEARCH (searching the full frame for the ball).
- After STOP: a summary card with counts, processed fps, median ms per frame and full-frame
  searches. Nothing is saved yet, so photograph the card.

## After the first run: numbers to record here
Phone setup: portrait, low, 2-3 m away, screen facing the player, head to toe and the ball in frame.

| Check | Target | Result |
|---|---|---|
| iPhone model and iOS version | | |
| Processed fps (summary card) | about 30 | |
| Median ms per frame | under 33 | |
| Left foot only: only L goes up | yes | |
| Hand count vs screen tick-tocks, one drill | close | |
| Pitch looking ahead / looking at the ball | cutoff is 12 deg | |
| `yolov8n_320` in Xcode Performance tab: runs on the Neural Engine | yes | |

These feed Phase 2 of `docs/IOS_PORT_PLAN.md`.
