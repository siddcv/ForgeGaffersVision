# Ball detector fine-tuning (TODO)

Status: not started. The detector is the untuned YOLOv8n COCO "sports ball" class at confidence 0.15.
It finds the ball in 57-72% of frames on the classroom clips and fails worst when the ball blends into
the background (20261101_185024, night park). Ball-detection causes were about 7 of 65 touch errors on
the tune clips, plus the misses where the ball was not seen crossing the midpoint, so a better detector
has a known ceiling on how much it improves touch counting.

## TODO: decide the licence first
- [ ] The code will be open-sourced. Ultralytics YOLOv8 is AGPL-3.0: fine if the whole app is released
      under AGPL with full source, including any changes to Ultralytics. Ultralytics treats weights
      trained with its code as AGPL too, so publish the weights under AGPL as well.
- [ ] The catch is app-store distribution: GPL-family licences conflict with App Store terms (VLC was
      pulled from the App Store over this in 2011). Google Play is generally fine. iOS is the leading
      platform, so check with Northwestern tech transfer or the advisor before training.
- [ ] YOLOX-nano (Apache-2.0, ~1 GFLOP at 416) avoids the question. The labels are the expensive
      part and train either detector, so the choice can follow the labelling.

## TODO: how much data
Target: **30-50 clips covering 8-10 distinct setups, about 1,500-3,000 labelled frames.** Distinct
conditions matter more than the clip count. Neighbouring frames are nearly identical, so a 30 s
clip at 60 fps has only about 30-60 genuinely different frames. One class in a narrow domain is an
easy fine-tune: the pretrained model already knows "ball" and only has to learn our balls, floors
and lighting. Failures come from conditions it has not seen.

Conditions to vary:

| Axis | Aim for |
|---|---|
| Ball | 4-5 colours and patterns (white/blue, all-white, dark, bright orange, worn); sizes 4 and 5 |
| Place and floor | wood indoor, grass, turf, concrete, dirt (relevant to Ethiopia and Cameroon users) |
| Lighting | indoor overhead, daylight, low sun and shadow, night under floodlights |
| Player | 5-8 people, different shoes and trousers (the ball sits right against them) |
| Camera | 2-3 phones, a few distances and heights |

About 8-10 combinations x 3-5 clips each = 30-50 clips.

## TODO: which frames to label
- [ ] Label 40-80 frames per clip, sampled to over-represent the hard cases:
  - foot contact with occlusion (the most common failure)
  - motion blur in mid-flight
  - ball against a similar-coloured background
  - ball away from the feet (training examples for full-frame search mode)
  - some frames with no ball, so the model does not fire on shoes or cones
- [ ] Export both a **foot crop** and a **full frame** for each labelled frame, since the detector runs
      in both modes. Evaluate each mode separately.
- [ ] Labelling shortcut: pre-label with the current pipeline's boxes. Offline tricks are allowed
      for labelling, so fill gaps by tracking forward and backward between detections, and use SAM to
      draw the hard boxes. Correct rather than draw from scratch (about 1-2 hours per 500 frames).
      Tool: see the next section.

## TODO: labelling tool (Roboflow or CVAT)
Roboflow is a web platform for computer-vision datasets. Use it for **labelling only**, during
development; that does not break the "fully offline" rule, which is about the app.

| Roboflow part | Use it? |
|---|---|
| Annotate: draw boxes in the browser, pre-label with a model ("Label Assist") or click-to-box with SAM | **Yes, the main use** |
| Import / export: import our pipeline's pre-labels in YOLO format, export YOLO or COCO | **Yes** |
| Dataset versions and augmentation (blur, brightness, flips) | Optional |
| Universe: public datasets, including many soccer-ball sets | Maybe; see watch-outs |
| Hosted training and inference | **No.** Train locally or on Colab; the app runs offline |

Workflow:
1. [ ] Frame sampler (to build) writes hard frames plus pipeline pre-labels in YOLO format.
2. [ ] Upload, then correct the boxes with Label Assist or SAM.
3. [ ] Export YOLO format and train locally or on Colab (YOLOX-nano or YOLOv8n).
4. [ ] Score with `evaluate.py` (touch precision and recall), not only Roboflow's box metrics.

Watch-outs:
- [ ] **Privacy.** The free plan makes datasets **public** on Universe, and our frames show
      identifiable players, possibly minors. Get consent for public release, or pay for a private
      workspace (ask about education or research discounts), or use **CVAT** instead: open source,
      SAM-assisted labelling, runs locally. Both export the same YOLO format, so switching later is cheap.
- [ ] **Universe datasets are a different setting.** Most are broadcast or stadium footage (a tiny,
      distant ball seen from high up), not our low, close phone view with feet covering the ball. They
      may help a little as extra pre-training data but do not replace our own labels. Check each
      dataset's licence (often CC BY 4.0, which needs credit).
- [ ] **Model choice stays ours.** RF-DETR (Roboflow, Apache-2.0) is a candidate worth testing, but
      transformer detectors are a riskier Core ML / Neural Engine conversion than YOLOX-nano, so
      YOLOX-nano stays the default.

## TODO: pilot before recording more
1. [ ] Label about 400 frames from the current tune clips.
2. [ ] Fine-tune YOLOv8n and/or YOLOX-nano.
3. [ ] Plot a learning curve: train on 25%, 50% and 100% of the frames and check recall on a held-out clip.
       - Still climbing at 100%: more data helps; record more conditions.
       - Flat: the remaining errors come from somewhere else (pose foot swaps, touch rules, blur
         that cannot be recovered).
4. [ ] Measure the payoff as touch precision and recall from `evaluate.py`, not only box recall.
5. [ ] Check INT8 vs FP32 recall on the cached crops before trusting phone results. Quantization
       mostly hurts low-confidence detections.

Tooling to build for the pilot: a frame sampler that pulls hard frames from the tune clips,
pre-labels them with the current pipeline, and exports crops and full frames in YOLO format.

## Hold-out rule
Clips in `data/clips/test/` stay out of detector training as well as threshold tuning. Ideally hold
out a whole session, a different ball or place, or a different player; otherwise the test set
flatters the detector.
