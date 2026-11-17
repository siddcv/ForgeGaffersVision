"""Export the ball detector to Core ML, check it matches ONNX, and time it. macOS only.

Run on a Mac (Apple Silicon), from the repo root:
    python3.12 -m venv ~/.venvs/gaffers-mac && source ~/.venvs/gaffers-mac/bin/activate
    pip install ultralytics==8.4.194 onnx==1.23.2 onnxruntime==1.32.0 onnxslim==0.1.99 coremltools
    python -m scripts.export_coreml                       # uses Ultralytics' sample image
    python -m scripts.export_coreml --image some_frame.jpg

Writes models/yolov8n_<size>.mlpackage (copy these back; they go into the Xcode project) and
output/coreml_check.json. The Mac's Neural Engine timing is only a hint for an iPhone's; the
real number comes from Xcode's Performance report on a connected iPhone.
"""
import argparse
import json
import platform
import shutil
import statistics
import time
from pathlib import Path

import numpy as np

from scripts.bench_detector import export_onnx
from src.config import Config
from src.onnx_detector import OnnxBallDetector, decode, iou, letterbox, to_tensor

SIZES = (320, 640)


def export_coreml(weights: str, imgsz: int) -> Path:
    target = Path("models") / f"{Path(weights).stem}_{imgsz}.mlpackage"
    if not target.exists():
        from ultralytics import YOLO

        # 16-bit, fixed size, no NMS inside the model: picking the box happens in Swift.
        exported = YOLO(weights).export(format="coreml", imgsz=imgsz, half=True, nms=False, verbose=False)
        target.parent.mkdir(exist_ok=True)
        shutil.move(exported, target)
    return target


def coreml_output(model, rgb_image) -> np.ndarray:
    from PIL import Image

    outputs = model.predict({model.get_spec().description.input[0].name: Image.fromarray(rgb_image)})
    return np.asarray(next(iter(outputs.values())), dtype=np.float32)


def time_ms(fn, runs: int = 50, warmup: int = 10) -> dict:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    return {"median_ms": round(statistics.median(times), 2), "p90_ms": round(times[int(0.9 * (runs - 1))], 2)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", help="test image (default: Ultralytics' bus.jpg)")
    args = parser.parse_args()
    if platform.system() != "Darwin":
        raise SystemExit("Core ML runs only on macOS. On Windows use scripts.bench_detector.")

    import cv2
    import coremltools as ct
    from ultralytics.utils import ASSETS

    cfg = Config()
    bgr = cv2.imread(args.image or str(ASSETS / "bus.jpg"))
    onnx = OnnxBallDetector(cfg, {size: export_onnx(cfg.detector_weights, size) for size in SIZES})
    report = {"coremltools": ct.__version__, "machine": platform.machine(), "mac": platform.mac_ver()[0]}

    for size in SIZES:
        path = export_coreml(cfg.detector_weights, size)
        image = letterbox(bgr, size)[0]
        rgb = np.ascontiguousarray(image[:, :, ::-1])
        reference = onnx.run(to_tensor(image), size)
        entry = {"model_mb": round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6, 2)}
        for label, units in (("all", ct.ComputeUnit.ALL), ("cpu_only", ct.ComputeUnit.CPU_ONLY)):
            model = ct.models.MLModel(str(path), compute_units=units)
            output = coreml_output(model, rgb)
            # Compare the highest-scoring boxes for every class, so the check works on any image.
            ref_boxes = decode(reference, int(np.argmax(reference[0, 4:].max(axis=1))), 0.25, 0.7, 5)
            got_boxes = decode(output, int(np.argmax(reference[0, 4:].max(axis=1))), 0.25, 0.7, 5)
            entry[label] = {
                "max_abs_diff_scores": round(float(np.abs(output[0, 4:] - reference[0, 4:]).max()), 4),
                "top_box_iou": round(iou(ref_boxes[0], got_boxes[0]), 4) if ref_boxes and got_boxes else None,
                **time_ms(lambda: coreml_output(model, rgb)),
            }
        report[f"{size}"] = entry
        print(f"{size}: {json.dumps(entry)}")

    out = Path("output") / "coreml_check.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"Saved {out}. Copy models/*.mlpackage back for the Xcode project.")


if __name__ == "__main__":
    main()
