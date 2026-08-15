"""Object detection abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Detection:
    """A detected vehicle expressed as an axis-aligned bounding box."""

    bbox: tuple
    label: str = "car"
    confidence: float = 1.0

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def y1(self) -> float:
        return self.bbox[1]

    @property
    def x2(self) -> float:
        return self.bbox[2]

    @property
    def y2(self) -> float:
        return self.bbox[3]

    @property
    def center(self) -> tuple:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


class BaseDetector(ABC):
    """Common interface for computer-vision vehicle detectors."""

    name: str = "base"

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return vehicle detections for a single image frame."""
