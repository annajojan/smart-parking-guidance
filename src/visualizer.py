"""Overlay parking-slot occupancy onto a frame for human inspection."""

from __future__ import annotations

from typing import List, Optional

import cv2

from .config import ParkingMap
from .detectors.base import Detection
from .occupancy import OccupancyFrame

_STATUS_COLORS = {
    "available": (80, 200, 90),
    "occupied": (60, 60, 220),
}

_TEXT_BG = (20, 20, 22, 150)


def draw_overlay(
    frame,
    occupancy: OccupancyFrame,
    parking_map: ParkingMap,
    detections: Optional[List[Detection]] = None,
    guidance_text: Optional[str] = None,
) -> None:
    """Draw slot states, vehicle boxes and guidance onto ``frame`` in place."""
    h, w = frame.shape[:2]

    for state in occupancy.slots:
        color = _STATUS_COLORS[state.status]
        x1, y1, x2, y2 = state.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        label = f"{state.slot_id} {'FREE' if state.status == 'available' else 'FULL'}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1 - 2), color, -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 250, 250), 1, cv2.LINE_AA)

    if detections:
        for det in detections:
            x1, y1, x2, y2 = (int(v) for v in det.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (240, 220, 60), 2)

    cv2.rectangle(frame, (0, 0), (w, 34), (30, 30, 34), -1)
    status_line = (
        f"Free: {occupancy.available_slots}/{occupancy.total_slots}   "
        f"Occupied: {occupancy.occupied_slots}   |   {parking_map.name}   |   {occupancy.source}"
    )
    cv2.putText(frame, status_line, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA)

    if guidance_text:
        _draw_text_block(frame, guidance_text, 0, h - 96)


def _draw_text_block(frame, text: str, x: int, y: int, max_width: int = 60) -> None:
    h, w = frame.shape[:2]
    lines: List[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)

    box_h = len(lines) * 20 + 14
    box_x = max(0, min(x, w - 400))
    box_y = max(0, min(y, h - box_h))
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x, box_y), (box_x + w - 8, box_y + box_h), (18, 18, 22), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, dst=frame)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (box_x + 12, box_y + 24 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
