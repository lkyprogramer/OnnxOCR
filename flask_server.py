"""CPU compatibility API for the legacy OnnxOCR service.

The legacy endpoints intentionally keep their original wire shape while the
new ``/ocr`` endpoint exposes the current service format.
"""

from __future__ import annotations

import base64
import json
import os
import time
from io import BytesIO
from typing import Any

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
from PIL import Image

try:
    import pymupdf as fitz
except ImportError:  # Older PyMuPDF releases expose the legacy module name.
    import fitz

from onnxocr.api_utils import ModelRegistry, decode_base64_image, format_ocr_results


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _configure_cv2_threads() -> None:
    raw = os.getenv("ONNXOCR_CV2_THREADS", "").strip()
    if not raw:
        return
    threads = int(raw)
    if threads >= 0:
        cv2.setNumThreads(threads)


def create_app(model_registry: ModelRegistry | None = None) -> Flask:
    _configure_cv2_threads()
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("ONNXOCR_MAX_UPLOAD_MB", "200")) * 1024 * 1024
    models = model_registry or ModelRegistry(
        use_gpu=_env_bool("ONNXOCR_USE_GPU", default=False),
        ocr_model_size=os.getenv("ONNXOCR_OCR_MODEL_SIZE", "small"),
        use_angle_cls=_env_bool("ONNXOCR_USE_ANGLE_CLS", default=False),
    )

    @app.get("/")
    def index():
        return jsonify(
            {
                "name": "OnnxOCR CPU Compatibility API",
                "status": "ok",
                "endpoints": ["/health", "/ocr", "/predict/ch_pp-ocrv3", "/pdf/ocr"],
            }
        )

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/predict/ch_pp-ocrv3")
    def predict_legacy():
        try:
            data = request.get_json(silent=True) or {}
            images = data.get("images")
            if not isinstance(images, list) or not images or images[0] is None:
                return _legacy_response("112", "Image is required.", "", status_code=400)

            image = decode_base64_image(str(images[0]))
            result = models.get_ocr_model()(image)
            return _legacy_response("000", "", result)
        except ValueError as exc:
            return _legacy_response("112", str(exc), "", status_code=400)
        except Exception as exc:  # pragma: no cover - production boundary
            return _legacy_response("500", str(exc), "", status_code=500)

    @app.post("/pdf/ocr")
    def pdf_ocr():
        try:
            data = request.get_json(silent=True) or {}
            pdf_base64 = data.get("pdfBase64")
            if not pdf_base64:
                return _legacy_response("112", "pdfBase64 is required.", "", status_code=400)

            results = []
            for image in _pdf_to_images(str(pdf_base64)):
                result = models.get_ocr_model()(image)
                results.append(result[1])
            return _legacy_response("200", "", results)
        except ValueError as exc:
            return _legacy_response("112", str(exc), "", status_code=400)
        except Exception as exc:  # pragma: no cover - production boundary
            return _legacy_response("500", str(exc), "", status_code=500)

    @app.post("/ocr")
    def ocr():
        try:
            image = _read_current_image()
            start_time = time.time()
            result = models.get_ocr_model().ocr(image)
            return jsonify(
                {
                    "success": True,
                    "processing_time": time.time() - start_time,
                    "results": format_ocr_results(result),
                }
            )
        except ValueError as exc:
            return jsonify({"success": False, "error": str(exc)}), 400
        except Exception as exc:  # pragma: no cover - production boundary
            return jsonify({"success": False, "error": f"OCR failed: {exc}"}), 500

    return app


def _legacy_response(status: str, message: str, results: Any, status_code: int = 200) -> Response:
    payload = {"status": status, "msg": message, "results": results}
    return Response(json.dumps(payload, ensure_ascii=False, cls=NumpyEncoder), status=status_code)


def _pdf_to_images(pdf_base64: str) -> list[np.ndarray]:
    try:
        pdf_bytes = base64.b64decode(pdf_base64)
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Invalid PDF data: {exc}") from exc

    images: list[np.ndarray] = []
    try:
        for page in pdf_document:
            pixmap = page.get_pixmap()
            pil_image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            buffer = BytesIO()
            pil_image.save(buffer, format="PNG")
            image_np = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
            image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            if image is not None:
                images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    finally:
        pdf_document.close()
    return images


def _read_current_image() -> np.ndarray:
    data = request.get_json(silent=True) or {}
    if "image" in data:
        return decode_base64_image(str(data["image"]))

    file = request.files.get("image") or request.files.get("file")
    if file:
        image_bytes = base64.b64encode(file.read()).decode("ascii")
        return decode_base64_image(image_bytes)
    raise ValueError("Image is required. Send JSON {'image': '<base64>'} or multipart file field 'image'.")


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("ONNXOCR_PORT", "6013")))
