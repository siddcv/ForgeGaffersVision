import numpy as np
import pytest

from src.onnx_detector import PAD_VALUE, decode, letterbox, to_source, to_tensor


def test_square_crop_is_resized_without_padding():
    image, scale, pad_x, pad_y = letterbox(np.zeros((640, 640, 3), np.uint8), 320)
    assert image.shape == (320, 320, 3) and scale == 0.5 and (pad_x, pad_y) == (0, 0)


def test_wide_frame_is_padded_top_and_bottom_with_gray():
    image, scale, pad_x, pad_y = letterbox(np.zeros((1080, 1920, 3), np.uint8), 640)
    assert image.shape == (640, 640, 3)
    assert pad_x == 0 and pad_y == 140  # 1080 * (640 / 1920) = 360 rows of picture
    assert (image[0] == PAD_VALUE).all() and (image[320] == 0).all()


def test_box_maps_back_to_source_pixels():
    _, scale, pad_x, pad_y = letterbox(np.zeros((1080, 1920, 3), np.uint8), 640)
    source = (960.0, 540.0, 1000.0, 580.0, 0.9)
    model_box = (source[0] * scale + pad_x, source[1] * scale + pad_y,
                 source[2] * scale + pad_x, source[3] * scale + pad_y, 0.9)
    assert to_source(model_box, scale, pad_x, pad_y) == pytest.approx(source)


def test_to_tensor_swaps_to_rgb_and_scales():
    bgr = np.zeros((2, 2, 3), np.uint8)
    bgr[..., 0] = 255  # blue
    tensor = to_tensor(bgr)
    assert tensor.shape == (1, 3, 2, 2) and tensor.dtype == np.float32
    assert tensor[0, 2].max() == 1.0 and tensor[0, 0].max() == 0.0


def _output(boxes_and_scores, classes=3, class_id=1):
    out = np.zeros((1, 4 + classes, len(boxes_and_scores)), np.float32)
    for i, (cx, cy, w, h, score) in enumerate(boxes_and_scores):
        out[0, :4, i] = cx, cy, w, h
        out[0, 4 + class_id, i] = score
    return out


def test_decode_keeps_best_box_and_drops_overlapping_and_weak_ones():
    out = _output([
        (100, 100, 20, 20, 0.5),
        (101, 100, 20, 20, 0.9),  # same ball, higher score
        (200, 200, 20, 20, 0.4),  # another candidate
        (300, 300, 20, 20, 0.1),  # below threshold
    ])
    boxes = decode(out, class_id=1, confidence=0.15, iou_threshold=0.7, keep=3)
    assert [round(b[4], 2) for b in boxes] == [0.9, 0.4]
    assert boxes[0][:4] == pytest.approx((91, 90, 111, 110))


def test_decode_ignores_other_classes():
    assert decode(_output([(100, 100, 20, 20, 0.9)], class_id=0), 1, 0.15, 0.7, 3) == []
