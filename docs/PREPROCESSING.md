# Preprocessing spec

How each camera frame becomes model inputs, and how model outputs become the pixel coordinates the
per-frame logic uses. The phone app must match this exactly: a small mismatch (channel order,
padding, rounding) does not crash anything, it just makes detections quietly worse.

The Python reference is `src/onnx_detector.py` (detector input and output), `src/ball.py`
(`foot_crop`, `crop_image`, `to_full_frame`) and `src/pose.py`. Their tests are
`tests/test_onnx_detector.py` and `tests/test_ball.py`.

## Coordinates and time
- **One coordinate system everywhere:** pixels in the **upright, unmirrored** full frame, origin
  top-left, x right, y down. Every box and keypoint is converted to it before the logic sees it.
- **Upright:** apply the rotation the camera or file reports before anything else.
- **Unmirrored:** the front camera preview is mirrored; mirror only what is drawn on screen, never
  the frames given to the models. Otherwise the player's left and right feet swap.
- **Time:** each frame's presentation timestamp in seconds, from the camera (or the file's
  timestamps when replaying a clip), never `frame_index / fps`. Phones record variable frame rate.
- **Capture:** 30 fps target, 720p on the phone (the models resize to 256-640 px anyway). Drop late
  frames rather than queueing them.

## 1. Pose (MediaPipe Pose Landmarker, lite)
| Item | Value |
|---|---|
| Model | `pose_landmarker_lite.task` (float16), the same file on every platform |
| Mode | video / live stream, timestamps in whole milliseconds and strictly increasing (add 1 ms on a tie) |
| Input | the full upright frame, RGB. MediaPipe resizes it internally; do not crop or letterbox first |
| Output used | the first person's 33 landmarks: `x * frame_width`, `y * frame_height`, `visibility` |
| No person | all 33 landmarks are "none" (NaN in Python, null in the test vectors) |

Landmarks used by the logic: heels 29 (left) and 30 (right), toes 31 and 32, nose 0, ears 7 and 8.
A landmark counts only if `visibility >= pose_min_visibility` (0.5).

## 2. Foot crop (`ball.foot_crop`)
Built from the **raw** (unsmoothed) visible heel and toe points of this frame:
1. `span = max(max x - min x, max y - min y)` over those points; no visible point means no crop.
2. Side length:
   - once a ball has been seen: `max(ball_diameter_px / crop_ball_frac, span * crop_span_min_scale)`
     (`0.18` and `1.2`), where `ball_diameter_px` is the shorter side of the last detected box;
   - before that: `max(span * crop_span_scale, crop_min_frac * frame_height)` (`2.0` and `0.25`).
3. `side = int(side)`; centre = midpoint of the points' bounding box;
   `x1 = int(cx - side / 2)`, `y1 = int(cy - side / 2)`, `x2 = x1 + side`, `y2 = y1 + side`.
   `int()` truncates toward zero, also for negative values (Swift `Int(_:)` does the same).
4. No crop if it lies entirely outside the frame.

The crop is **square** and **may extend past the frame edge**. The part outside the frame is filled
with gray `114` in all three channels (`crop_image`). Keeping it square means no letterbox is
needed, so the ball keeps the size step 2 chose (the untuned detector works best at 40-80 px).

## 3. Ball detector input
| Item | Foot crop | Full frame (search mode only) |
|---|---|---|
| Model input | 320 x 320 | 640 x 640 |
| Resize | square crop scaled to 320 x 320, bilinear | longer side scaled to 640, bilinear, aspect kept |
| Padding | none (already square) | letterbox, see below |
| Channels | RGB | RGB |
| Values | `pixel / 255` as float, in [0, 1] | same |
| Layout | 1 x 3 x H x W (Core ML `ImageType` takes the image directly and does the scaling) | same |

Letterbox (`onnx_detector.letterbox`), for a frame of width `w` and height `h`:
```
scale = min(640 / h, 640 / w)
new_w, new_h = round(w * scale), round(h * scale)        # round half to even
pad_w, pad_h = (640 - new_w) / 2, (640 - new_h) / 2
left, right = round(pad_w - 0.1), round(pad_w + 0.1)     # odd padding: the extra pixel goes right / bottom
top, bottom = round(pad_h - 0.1), round(pad_h + 0.1)
fill = gray 114
```
A 1920 x 1080 frame gives `scale = 1/3`, 640 x 360 picture, 140 gray rows above and below.

## 4. Ball detector output
Raw YOLOv8 head, no NMS inside the model: shape `1 x 84 x N` (N = 2100 at 320, 8400 at 640).
- Rows 0-3: box centre x, centre y, width, height, in **model input pixels**.
- Rows 4-83: one score per COCO class, already passed through a sigmoid. The ball is class
  `32` ("sports ball"), so its score is row `4 + 32`. A fine-tuned one-class model will have a single
  score row instead; update `ball_class_id` and this line together.

Decoding (`onnx_detector.decode`):
1. Keep anchors with ball score `>= ball_confidence` (0.15), highest first.
2. Greedy NMS: drop a box whose IoU with an already kept box is above `ball_nms_iou` (0.7).
3. Keep at most `ball_candidates` (3). The tracker, not the detector, picks among them.

Back to full-frame pixels:
- Full frame: `x = (x_model - left) / scale`, `y = (y_model - top) / scale`.
- Foot crop: `x = x_model * side / 320 + x1`, `y = y_model * side / 320 + y1` (`to_full_frame`
  adds the crop origin).

Ball diameter for scale and crop sizing = the **shorter** side of the box, since motion blur
stretches the longer one.

## 5. What the logic receives
Per frame: `t_s`, the 33 landmarks, the crop (or none), the crop candidates and, only when
`ball.needs_full_frame` is true, the full-frame candidates. From there `FramePipeline.step` in
`src/pipeline.py` does the rest (smoothing, tracker, search switch, metrics). The test vectors in
`tests/vectors/` check that part.

## Known differences to resolve on iOS
| Difference | What to do |
|---|---|
| Vision's `regionOfInterest` cannot extend past the frame, so it cannot add the gray padding | Prefer building the 320 x 320 crop ourselves (Core Image or vImage, gray fill) so the input matches. If using `regionOfInterest`, clamp the crop to the frame and first measure the effect in Python on the tune clips |
| Bilinear resize differs slightly between OpenCV, Core Image and Vision | Check with the Core ML parity script (`scripts/export_coreml.py`) and a replayed clip; small box shifts are expected, missing detections are not |
| The cached full-frame detections came from PyTorch's rectangular letterbox (e.g. 640 x 384), not the fixed 640 x 640 the exported model needs | Only the full-frame mode is affected; the crop path matches exactly (242 of 242 boxes in `scripts/bench_detector.py`). Re-check full-frame recall when the detector is fine-tuned |
| Camera pixel format is usually YUV (`420f`), not RGB | Convert once, or let Vision / Core ML convert; confirm the model sees RGB, not BGR |
