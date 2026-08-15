"""YOLOv8 detector (optional, higher accuracy).

Requires the ``ultralytics`` package and PyTorch. The first run downloads
the model weights, so an internet connection is needed. Falls back gracefully
by raising a clear error that the caller can catch.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import BaseDetector, Detection

_VEHICLE_CLASSES = (2, 7)  # COCO: car, truck


class YOLODetector(BaseDetector):
    """Ultralytics YOLO wrapper restricted to vehicle classes."""

    name = "yolo"

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.35, device: Optional[str] = None) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "ultralytics is not installed. Install it with: pip install -r requirements-optional.txt"
            ) from exc
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = device

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            frame,
            conf=self.confidence,
            classes=list(_VEHICLE_CLASSES),
            verbose=False,
            device=self.device,
        )
        detections: list[Detection] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                detections.append(
                    Detection((x1, y1, x2, y2), result.names[int(box.cls[0])], float(box.conf[0]))
                )
        return detections
