"""Global colour-based vehicle detector.

Vehicles are located by segmenting saturated colour regions (typical car paint)
from neutral surfaces such as asphalt and white lane markings. Morphological
operations clean the mask before contour extraction. This works without deep
learning and is a good middle ground between the per-slot heuristic and YOLO.
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import BaseDetector, Detection


class SaturationDetector(BaseDetector):
    """Finds vehicle boxes via HSV saturation segmentation + contours."""

    name = "saturation"

    def __init__(
        self,
        saturation_threshold: int = 70,
        min_area: int = 2200,
        max_area: int = 120000,
        close_kernel: int = 11,
        open_kernel: int = 3,
    ) -> None:
        self.saturation_threshold = saturation_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.close_kernel = close_kernel
        self.open_kernel = open_kernel

    def detect(self, frame: np.ndarray) -> list[Detection]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        mask = (saturation > self.saturation_threshold).astype(np.uint8) * 255

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((self.open_kernel, self.open_kernel), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((self.close_kernel, self.close_kernel), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[Detection] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area or area > self.max_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            detections.append(Detection((x, y, x + w, y + h), "car", min(1.0, area / 40000.0)))

        return detections
