"""Per-slot occupancy detector using interior edge density.

This is the classic approach for camera-based parking-lot occupancy: instead of
localising cars globally, each slot's interior region is analysed in isolation.
An occupied slot shows a high density of strong edges (vehicle body, roof,
windows), while an empty slot shows only low-contrast asphalt. The slot borders
are excluded by shrinking the region of interest, so painted markings do not
trigger false positives. No deep-learning dependencies are required.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..config import ParkingMap
from .base import BaseDetector, Detection


class RegionDetector(BaseDetector):
    """Labels slots as occupied based on interior edge density."""

    name = "region"

    def __init__(
        self,
        parking_map: ParkingMap,
        edge_threshold: float = 0.045,
        shrink: int = 26,
        canny_low: int = 60,
        canny_high: int = 160,
    ) -> None:
        self.parking_map = parking_map
        self.edge_threshold = edge_threshold
        self.shrink = shrink
        self.canny_low = canny_low
        self.canny_high = canny_high

    def detect(self, frame: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        detections: list[Detection] = []
        for slot in self.parking_map.slots:
            x1, y1, x2, y2 = slot.bbox
            x1, y1 = x1 + self.shrink, y1 + self.shrink
            x2, y2 = x2 - self.shrink, y2 - self.shrink
            if x2 <= x1 or y2 <= y1:
                continue

            roi = gray[y1:y2, x1:x2]
            edges = cv2.Canny(roi, self.canny_low, self.canny_high)
            edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
            ratio = float(cv2.countNonZero(edges)) / float(edges.size)

            if ratio >= self.edge_threshold:
                confidence = min(1.0, ratio / (self.edge_threshold * 1.6))
                detections.append(Detection(slot.bbox, "car", round(confidence, 3)))

        return detections
