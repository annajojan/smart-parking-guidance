"""Structured occupancy model and detection-to-slot matching.

Converts raw bounding-box detections into a structured, JSON-serialisable
representation of the parking lot state: every slot is labelled as
``available`` or ``occupied`` with a confidence score.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import List, Optional

import cv2
import numpy as np

from .config import ParkingMap
from .detectors.base import Detection


def _intersection_area(box_a: tuple, box_b: tuple) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    h = max(0.0, min(ay2, by2) - max(ay1, by1))
    return w * h


def _center_inside_polygon(center: tuple, polygon: list) -> bool:
    pts = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(pts, center, False) >= 0


@dataclass
class SlotState:
    """Occupancy state of a single parking slot."""

    slot_id: str
    zone: str
    level: str
    status: str
    confidence: float
    bbox: tuple
    centroid: tuple


@dataclass
class OccupancyFrame:
    """Structured snapshot of the whole parking lot."""

    timestamp: str
    source: str
    total_slots: int
    available_slots: int
    occupied_slots: int
    slots: List[SlotState]

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "source": self.source,
            "total_slots": self.total_slots,
            "available_slots": self.available_slots,
            "occupied_slots": self.occupied_slots,
            "slots": [asdict(s) for s in self.slots],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def status_of(self, slot_id: str) -> Optional[str]:
        for slot in self.slots:
            if slot.slot_id == slot_id:
                return slot.status
        return None


def match_detections_to_slots(
    parking_map: ParkingMap,
    detections: List[Detection],
    coverage_threshold: float = 0.22,
    source: str = "frame",
    timestamp: Optional[str] = None,
) -> OccupancyFrame:
    """Label every slot as occupied when a vehicle covers enough of it."""
    slot_states: List[SlotState] = []
    for slot in parking_map.slots:
        best_coverage = 0.0
        for det in detections:
            covered = _intersection_area(slot.bbox, det.bbox) / slot.area
            best_coverage = max(best_coverage, covered)
            if det.center is not None and _center_inside_polygon(det.center, slot.polygon):
                best_coverage = max(best_coverage, coverage_threshold)
        if best_coverage >= coverage_threshold:
            status = "occupied"
            confidence = min(1.0, best_coverage)
        else:
            status = "available"
            confidence = 1.0 - min(1.0, best_coverage)

        slot_states.append(
            SlotState(
                slot_id=slot.slot_id,
                zone=slot.zone,
                level=slot.level,
                status=status,
                confidence=round(confidence, 3),
                bbox=slot.bbox,
                centroid=[round(v, 1) for v in slot.centroid],
            )
        )

    available = sum(1 for s in slot_states if s.status == "available")
    return OccupancyFrame(
        timestamp=timestamp or time.strftime("%Y-%m-%dT%H:%M:%S"),
        source=source,
        total_slots=len(slot_states),
        available_slots=available,
        occupied_slots=len(slot_states) - available,
        slots=slot_states,
    )
