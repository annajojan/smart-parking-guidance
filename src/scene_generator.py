"""Synthetic bird's-eye parking lot renderer used to generate demo footage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import ParkingMap, Slot

CAR_COLORS = [
    (66, 72, 190),
    (196, 102, 66),
    (96, 158, 64),
    (48, 148, 214),
    (190, 86, 150),
]

ASPHALT = (104, 103, 99)
MARKING = (238, 240, 243)


@dataclass
class SyntheticScene:
    image: np.ndarray
    occupied_ids: List[str]
    cars: Dict[str, dict]


def _draw_car(image: np.ndarray, slot: Slot, rng: np.random.Generator) -> dict:
    x1, y1, x2, y2 = slot.bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    width = int((x2 - x1) * 0.72)
    height = int((y2 - y1) * 0.62)
    left = int(cx - width / 2)
    top = int(cy - height / 2)
    right = left + width
    bottom = top + height

    color = CAR_COLORS[int(rng.integers(0, len(CAR_COLORS)))]
    outline = tuple(int(c * 0.22) for c in color)

    cv2.rectangle(image, (left + 4, top + 6), (right + 4, bottom + 6), (38, 40, 44), -1)
    cv2.rectangle(image, (left, top), (right, bottom), color, -1)
    cv2.rectangle(image, (left, top), (right, bottom), outline, 2)

    roof = tuple(int(c * 0.62) for c in color)
    cv2.rectangle(image, (left + 8, top + 26), (right - 8, bottom - 14), roof, -1)

    windshield = (36, 40, 46)
    cv2.rectangle(image, (left + 10, top + 8), (right - 10, top + 20), windshield, -1)
    cv2.rectangle(image, (left + 10, bottom - 12), (right - 10, bottom - 2), windshield, -1)

    for wx, wy in ((left + 6, top + 12), (right - 26, top + 12), (left + 6, bottom - 16), (right - 26, bottom - 16)):
        cv2.rectangle(image, (wx, wy), (wx + 20, wy + 8), (24, 26, 30), -1)

    return {"bbox": (left, top, right, bottom), "color": color}


def generate_scene(
    parking_map: ParkingMap,
    occupied_ids: Optional[Sequence[str]] = None,
    seed: int = 42,
) -> SyntheticScene:
    """Render a synthetic frame of the parking lot.

    If ``occupied_ids`` is None, roughly half of the slots are occupied at random
    using the given seed.
    """
    width, height = parking_map.image_size
    rng = np.random.default_rng(seed)

    image = np.full((height, width, 3), ASPHALT, dtype=np.uint8)
    noise = rng.integers(-14, 15, size=(height, width, 3), dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    x1, y1, x2, y2 = parking_map.lot
    cv2.rectangle(image, (x1, y1), (x2, y2), MARKING, 6)

    for slot in parking_map.slots:
        pts = np.array(slot.polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [pts], True, MARKING, 6)

    lane_top, lane_bottom = parking_map.drive_lane
    cv2.line(image, (x1, lane_top), (x2, lane_top), (190, 192, 196), 3)
    cv2.line(image, (x1, lane_bottom), (x2, lane_bottom), (190, 192, 196), 3)

    dash = 48
    for cx in range(x1 + 20, x2, dash * 2):
        cv2.line(image, (cx, (lane_top + lane_bottom) // 2), (min(cx + dash, x2), (lane_top + lane_bottom) // 2), (190, 192, 196), 4)

    for name, (ex, ey) in parking_map.entrances.items():
        cv2.circle(image, (ex, ey), 10, (60, 120, 220), -1)
        cv2.putText(
            image,
            f"{name.upper()} ENTRY",
            (ex - 70, ey - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 32, 36),
            2,
        )

    all_ids = [slot.slot_id for slot in parking_map.slots]
    if occupied_ids is None:
        count = max(1, int(len(all_ids) * 0.6))
        occupied_ids = [str(x) for x in rng.choice(all_ids, size=count, replace=False)]

    occupied_set = set(occupied_ids)
    cars: Dict[str, dict] = {}
    for slot in parking_map.slots:
        if slot.slot_id in occupied_set:
            cars[slot.slot_id] = _draw_car(image, slot, rng)

    return SyntheticScene(image=image, occupied_ids=list(occupied_ids), cars=cars)


def save_scene(scene: SyntheticScene, out_dir: str | Path, tag: str = "scene") -> Tuple[Path, Path]:
    """Write the rendered image and its ground-truth occupancy to disk."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import json

    image_path = out_dir / f"{tag}.png"
    cv2.imwrite(str(image_path), scene.image)

    truth_path = out_dir / f"{tag}_ground_truth.json"
    truth_path.write_text(json.dumps({"occupied_ids": scene.occupied_ids}, indent=2), encoding="utf-8")

    return image_path, truth_path
