from __future__ import annotations

import numpy as np

from onnxocr import api_utils, inference_engine, onnx_paddleocr
from onnxocr.operators import DetResizeForTest
from onnxocr.predict_det import TextDetector


def test_model_registry_passes_ocr_model_size(monkeypatch):
    captured = {}

    class FakeOcr:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(api_utils, "ONNXPaddleOcr", FakeOcr)

    api_utils.ModelRegistry(use_gpu=True, ocr_model_size="small").get_ocr_model()

    assert captured["use_gpu"] is True
    assert captured["use_angle_cls"] is True
    assert captured["ocr_model_size"] == "small"


def test_model_registry_passes_use_angle_cls_false(monkeypatch):
    captured = {}

    class FakeOcr:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(api_utils, "ONNXPaddleOcr", FakeOcr)

    api_utils.ModelRegistry(use_gpu=False, ocr_model_size="small", use_angle_cls=False).get_ocr_model()

    assert captured["use_gpu"] is False
    assert captured["use_angle_cls"] is False
    assert captured["ocr_model_size"] == "small"


def test_default_session_options_suppress_warnings(monkeypatch):
    monkeypatch.delenv("ONNXOCR_ORT_LOG_SEVERITY", raising=False)

    options = inference_engine._default_session_options()

    assert options.log_severity_level == 3


def test_default_session_options_allow_log_severity_override(monkeypatch):
    monkeypatch.setenv("ONNXOCR_ORT_LOG_SEVERITY", "2")

    options = inference_engine._default_session_options()

    assert options.log_severity_level == 2


def test_default_session_options_sets_global_onnxruntime_logger(monkeypatch):
    captured = []
    monkeypatch.setenv("ONNXOCR_ORT_LOG_SEVERITY", "3")
    monkeypatch.setattr(inference_engine.onnxruntime, "set_default_logger_severity", captured.append)

    inference_engine._default_session_options()

    assert captured == [3]


def test_default_session_options_configures_threads_and_spinning(monkeypatch):
    monkeypatch.setenv("ONNXOCR_ORT_INTRA_OP_THREADS", "2")
    monkeypatch.setenv("ONNXOCR_ORT_INTER_OP_THREADS", "1")
    monkeypatch.setenv("ONNXOCR_ORT_ALLOW_SPINNING", "0")

    entries = []
    orig_session_options = inference_engine.SessionOptions

    class SpySessionOptions(orig_session_options):
        def add_session_config_entry(self, key, value):
            entries.append((key, value))
            return super().add_session_config_entry(key, value)

    monkeypatch.setattr(inference_engine, "SessionOptions", SpySessionOptions)

    options = inference_engine._default_session_options()

    assert options.intra_op_num_threads == 2
    assert options.inter_op_num_threads == 1
    assert options.execution_mode == inference_engine.onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    assert ("session.intra_op.allow_spinning", "0") in entries
    assert ("session.inter_op.allow_spinning", "0") in entries


def test_det_resize_for_test_caps_max_side_len():
    op = DetResizeForTest(limit_side_len=736, limit_type="min", max_side_len=1600)

    small = np.zeros((400, 500, 3), dtype=np.uint8)
    small_out = op({"image": small})["image"]
    assert small_out.shape[:2] == (736, 928)

    large = np.zeros((3000, 4000, 3), dtype=np.uint8)
    large_out = op({"image": large})["image"]
    assert max(large_out.shape[:2]) == 1600
    assert large_out.shape[0] % 32 == 0 and large_out.shape[1] % 32 == 0


def test_build_ppocrv6_defaults_reads_det_max_side_len(monkeypatch):
    monkeypatch.setenv("ONNXOCR_DET_MAX_SIDE_LEN", "1600")
    defaults = onnx_paddleocr._build_ppocrv6_defaults({"ocr_model_size": "small"})
    assert defaults["det_max_side_len"] == 1600.0

    monkeypatch.setenv("ONNXOCR_DET_MAX_SIDE_LEN", "0")
    defaults_disabled = onnx_paddleocr._build_ppocrv6_defaults({"ocr_model_size": "small"})
    assert defaults_disabled["det_max_side_len"] is None


def test_det_resize_for_test_caps_max_pixels_preserving_long_screenshot_legibility():
    op = DetResizeForTest(limit_side_len=736, limit_type="min", max_pixels=4500000)

    long_screenshot = np.zeros((6664, 1550, 3), dtype=np.uint8)
    out = op({"image": long_screenshot})["image"]
    assert out.shape[0] % 32 == 0 and out.shape[1] % 32 == 0
    assert out.shape[1] >= 960
    assert out.shape[0] * out.shape[1] <= 4700000


def test_build_ppocrv6_defaults_reads_det_max_pixels(monkeypatch):
    monkeypatch.setenv("ONNXOCR_DET_MAX_PIXELS", "4500000")
    defaults = onnx_paddleocr._build_ppocrv6_defaults({"ocr_model_size": "small"})
    assert defaults["det_max_pixels"] == 4500000.0

    monkeypatch.setenv("ONNXOCR_DET_MAX_PIXELS", "0")
    defaults_disabled = onnx_paddleocr._build_ppocrv6_defaults({"ocr_model_size": "small"})
    assert defaults_disabled["det_max_pixels"] is None


def test_text_detector_should_tile_and_compute_ranges():
    det = TextDetector.__new__(TextDetector)
    det.det_tile_enable = True
    det.det_tile_height = 1600
    det.det_tile_overlap = 160
    det.det_tile_aspect_ratio = 2.5

    assert det._should_tile(np.zeros((877, 621, 3), dtype=np.uint8)) is False
    assert det._should_tile(np.zeros((2560, 1920, 3), dtype=np.uint8)) is False
    assert det._should_tile(np.zeros((6664, 1550, 3), dtype=np.uint8)) is True

    ranges = det._compute_tile_ranges(6664)
    assert ranges[0][0] == 0
    assert ranges[-1][1] == 6664
    for i, (y0, y1) in enumerate(ranges):
        assert 0 <= y0 < y1 <= 6664
        assert (y1 - y0) <= 1600
        if i > 0:
            assert ranges[i - 1][1] - y0 >= 160


def test_text_detector_detect_tiled_filters_seams_and_deduplicates():
    det = TextDetector.__new__(TextDetector)
    det.det_tile_enable = True
    det.det_tile_height = 1600
    det.det_tile_overlap = 160
    det.det_tile_aspect_ratio = 2.5

    def box(x1, y1, x2, y2):
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

    # Tile 0 (0..1600), Tile 1 (1440..3040):
    # - Shared line at global y=1480..1530 appears in both tiles (tile 0 margin=70, tile 1 margin=40)
    # - Clipped seam fragment at tile 0 bottom y=1596..1600 should be dropped
    # - Clipped seam fragment at tile 1 top y=0..4 (global 1440..1444) should be dropped
    # - Two distinct lines inside tile 0 at y=100..140 and y=150..190 must both be kept
    tile_outputs = [
        np.array([box(10, 100, 200, 140), box(10, 150, 200, 190), box(10, 1480, 300, 1530), box(10, 1596, 200, 1600)]),
        np.array([box(10, 0, 200, 4), box(12, 40, 298, 90), box(10, 800, 250, 850)]),
    ]
    call_idx = {"i": 0}

    def fake_detect_single(tile_img):
        idx = call_idx["i"]
        call_idx["i"] += 1
        return tile_outputs[idx]

    det._detect_single = fake_detect_single
    merged = det(np.zeros((2500, 800, 3), dtype=np.uint8))

    assert len(merged) == 4
    y_mins = sorted(float(np.min(b[:, 1])) for b in merged)
    assert y_mins == [100.0, 150.0, 1480.0, 2240.0]


def test_build_ppocrv6_defaults_reads_det_tile_env(monkeypatch):
    monkeypatch.setenv("ONNXOCR_DET_LIMIT_SIDE_LEN", "736")
    monkeypatch.setenv("ONNXOCR_DET_TILE_ENABLE", "1")
    monkeypatch.setenv("ONNXOCR_DET_TILE_HEIGHT", "1600")
    monkeypatch.setenv("ONNXOCR_DET_TILE_OVERLAP", "160")
    monkeypatch.setenv("ONNXOCR_DET_TILE_ASPECT_RATIO", "2.5")

    defaults = onnx_paddleocr._build_ppocrv6_defaults({"ocr_model_size": "small"})
    assert defaults["det_limit_side_len"] == 736
    assert isinstance(defaults["det_limit_side_len"], int)
    assert defaults["det_tile_enable"] is True
    assert defaults["det_tile_height"] == 1600
    assert defaults["det_tile_overlap"] == 160
    assert defaults["det_tile_aspect_ratio"] == 2.5


def test_text_detector_init_handles_float_det_limit_side_len(monkeypatch):
    import argparse
    from onnxocr import utils

    parser = utils.infer_args()
    args = parser.parse_args([])
    args.det_limit_side_len = 736.0
    args.use_gpu = False

    class FakeSession:
        def get_inputs(self):
            return [argparse.Namespace(name="x")]

        def get_outputs(self):
            return [argparse.Namespace(name="y")]

        def run(self, output_names, input_feed):
            x = input_feed["x"]
            assert x.shape == (1, 3, 736, 736)
            return [np.zeros((1, 1, 736, 736), dtype=np.float32)]

    monkeypatch.setattr(TextDetector, "get_onnx_session", lambda self, *a, **kw: FakeSession())
    det = TextDetector(args)
    assert det.det_onnx_session is not None
