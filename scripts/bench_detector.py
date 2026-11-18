"""Export the ball detector to ONNX and time it on real foot crops, on a few CPU threads.

A closer stand-in for a phone than PyTorch timings: the exported fixed-size model in its real
runtime, with limited threads. It also checks that the ONNX detections match the PyTorch ones in
the cache, so the export and the hand-written preprocessing are known to be right.
A laptop still cannot predict phone numbers (different chips, no Neural Engine, no thermals).

Usage: python -m scripts.bench_detector data/clips/tune/<session>/<clip>.mp4 [--frames 300]
Needs the clip's cache (run src.run_clip on it first).
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from src import ball as ball_module
from src.config import Config
from src.onnx_detector import OnnxBallDetector, export_onnx, iou, letterbox, to_tensor
from src.video_io import iter_frames

SIZES = (320, 640)


def load_inputs(clip: str, cache, cfg: Config, frames: int) -> list:
    """(frame index, full frame, square crop image, crop box) for frames that have a crop."""
    inputs = []
    for frame in iter_frames(clip):
        if frame.index >= len(cache["t_s"]) or len(inputs) >= frames:
            break
        row = cache["crops"][frame.index]
        if np.isnan(row[0]):
            continue
        crop = tuple(int(v) for v in row)
        image = ball_module.crop_image(frame.bgr, crop, cfg.crop_pad_value)
        inputs.append((frame.index, frame.bgr, image, crop))
    return inputs


def timed(fn, items, warmup: int = 10) -> list:
    for item in items[:warmup]:
        fn(item)
    times = []
    for item in items:
        start = time.perf_counter()
        fn(item)
        times.append((time.perf_counter() - start) * 1000)
    return times


def summary(times: list) -> dict:
    ordered = sorted(times)
    return {
        "median_ms": round(statistics.median(ordered), 2),
        "p90_ms": round(ordered[int(0.9 * (len(ordered) - 1))], 2),
    }


def parity(detector: OnnxBallDetector, inputs, cache, cfg: Config) -> dict:
    """Top crop detection from ONNX vs the PyTorch top detection stored in the cache."""
    counts = {"both_found_same_box": 0, "both_found_different_box": 0, "both_missed": 0,
              "only_pytorch": 0, "only_onnx": 0}
    ious = []
    for index, _, image, crop in inputs:
        onnx = detector.detect(image, cfg.crop_imgsz)
        onnx = ball_module.to_full_frame(onnx[0], crop) if onnx else None
        row = cache["crop_balls"][index]
        row = row[0] if row.ndim == 2 else row
        torch = None if np.isnan(row[0]) else tuple(row)
        if onnx and torch:
            overlap = iou(onnx, torch)
            ious.append(overlap)
            counts["both_found_same_box" if overlap >= 0.9 else "both_found_different_box"] += 1
        elif torch:
            counts["only_pytorch"] += 1
        elif onnx:
            counts["only_onnx"] += 1
        else:
            counts["both_missed"] += 1
    counts["median_iou"] = round(float(np.median(ious)), 4) if ious else None
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clip")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--pytorch", action="store_true", help="also time PyTorch, for comparison")
    args = parser.parse_args()

    cfg = Config()
    cache = dict(np.load(Path("cache") / f"{Path(args.clip).stem}.npz"))
    paths = {size: export_onnx(cfg.detector_weights, size) for size in SIZES}
    inputs = load_inputs(args.clip, cache, cfg, args.frames)
    full_inputs = inputs[::5]
    print(f"{len(inputs)} crops, {len(full_inputs)} full frames, from {args.clip}")

    report = {
        "clip": Path(args.clip).name,
        "frame_size": list(inputs[0][1].shape[1::-1]),
        "model_mb": {size: round(path.stat().st_size / 1e6, 2) for size, path in paths.items()},
        "preprocess_crop": summary(timed(lambda x: to_tensor(letterbox(x[2], cfg.crop_imgsz)[0]), inputs)),
        "onnx": {},
    }
    for threads in args.threads:
        detector = OnnxBallDetector(cfg, paths, threads)
        crop_tensors = [to_tensor(letterbox(x[2], cfg.crop_imgsz)[0]) for x in inputs]
        full_tensors = [to_tensor(letterbox(x[1], cfg.full_imgsz)[0]) for x in full_inputs]
        report["onnx"][f"{threads}_threads"] = {
            "crop_320_model_only": summary(timed(lambda t: detector.run(t, cfg.crop_imgsz), crop_tensors)),
            "full_640_model_only": summary(timed(lambda t: detector.run(t, cfg.full_imgsz), full_tensors)),
        }
        print(f"  {threads} threads done")
    report["parity_with_pytorch_cache"] = parity(OnnxBallDetector(cfg, paths), inputs, cache, cfg)

    if args.pytorch:
        import torch

        detector = ball_module.YoloBallDetector(cfg)
        report["pytorch"] = {}
        for threads in args.threads:
            torch.set_num_threads(threads)
            report["pytorch"][f"{threads}_threads"] = {
                "crop_320_end_to_end": summary(timed(lambda x: detector.detect(x[2], cfg.crop_imgsz), inputs[:100])),
            }

    out = Path("output") / "bench_detector.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
