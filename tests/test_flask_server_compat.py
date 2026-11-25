from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path

import cv2
import fitz
import numpy as np
import pytest


def load_flask_server():
    module_path = Path(__file__).resolve().parents[1] / "flask_server.py"
    spec = importlib.util.spec_from_file_location("flask_server", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def image_base64() -> str:
    image = np.full((8, 8, 3), 255, dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return base64.b64encode(buffer).decode("ascii")


class FakeOcrModel:
    def __call__(self, image):
        assert image.shape == (8, 8, 3)
        return (
            [np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=np.float32)],
            [["你好。", np.float32(0.99)]],
        )

    def ocr(self, image):
        return [[[[[0, 0], [4, 0], [4, 4], [0, 4]], ("你好。", 0.99)]]]


class FakeRegistry:
    def get_ocr_model(self):
        return FakeOcrModel()


def test_legacy_predict_keeps_wire_shape():
    module = load_flask_server()
    app = module.create_app(FakeRegistry())

    response = app.test_client().post("/predict/ch_pp-ocrv3", json={"images": [image_base64()]})
    payload = json.loads(response.get_data(as_text=True))

    assert response.status_code == 200
    assert response.content_type == "text/html; charset=utf-8"
    assert payload["status"] == "000"
    assert payload["msg"] == ""
    assert payload["results"][0][0][0] == [0.0, 0.0]
    assert payload["results"][1][0][0] == "你好。"
    assert payload["results"][1][0][1] == pytest.approx(0.99)


def test_legacy_predict_rejects_missing_image():
    module = load_flask_server()
    app = module.create_app(FakeRegistry())

    response = app.test_client().post("/predict/ch_pp-ocrv3", json={})
    payload = json.loads(response.get_data(as_text=True))

    assert response.status_code == 400
    assert payload == {"status": "112", "msg": "Image is required.", "results": ""}


def test_pdf_ocr_keeps_legacy_page_result_shape():
    module = load_flask_server()
    app = module.create_app(FakeRegistry())

    document = fitz.open()
    page = document.new_page(width=8, height=8)
    pdf_base64 = base64.b64encode(document.tobytes()).decode("ascii")
    document.close()

    response = app.test_client().post("/pdf/ocr", json={"pdfBase64": pdf_base64})
    payload = json.loads(response.get_data(as_text=True))

    assert response.status_code == 200
    assert payload["status"] == "200"
    assert len(payload["results"]) == 1
    assert payload["results"][0][0][0] == "你好。"
    assert payload["results"][0][0][1] == pytest.approx(0.99)


def test_current_ocr_and_health_endpoints():
    module = load_flask_server()
    app = module.create_app(FakeRegistry())
    client = app.test_client()

    health = client.get("/health")
    ocr = client.post("/ocr", json={"image": image_base64()})

    assert health.status_code == 200
    assert health.get_json() == {"status": "ok"}
    assert ocr.status_code == 200
    assert ocr.get_json()["results"][0]["text"] == "你好。"


def test_create_app_configures_cv2_threads_and_angle_cls(monkeypatch):
    module = load_flask_server()
    captured = {}
    cv2_calls = []

    class CapturingRegistry:
        def __init__(self, use_gpu=False, ocr_model_size="small", use_angle_cls=True):
            captured["use_gpu"] = use_gpu
            captured["ocr_model_size"] = ocr_model_size
            captured["use_angle_cls"] = use_angle_cls

    monkeypatch.setattr(module, "ModelRegistry", CapturingRegistry)
    monkeypatch.setattr(module.cv2, "setNumThreads", cv2_calls.append)
    monkeypatch.delenv("ONNXOCR_USE_ANGLE_CLS", raising=False)
    monkeypatch.setenv("ONNXOCR_CV2_THREADS", "1")

    module.create_app()

    assert cv2_calls == [1]
    assert captured == {"use_gpu": False, "ocr_model_size": "small", "use_angle_cls": False}

    monkeypatch.setenv("ONNXOCR_USE_ANGLE_CLS", "1")
    module.create_app()
    assert captured["use_angle_cls"] is True
