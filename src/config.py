"""Configuration and parking map loading."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple


def load_env_file(path: str | os.PathLike = ".env") -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ (never overrides)."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass
class Slot:
    """A single parking bay defined by a polygon in image coordinates."""

    slot_id: str
    zone: str
    level: str
    polygon: List[List[int]]

    def __post_init__(self) -> None:
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        self._bbox = (min(xs), min(ys), max(xs), max(ys))

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return self._bbox

    @property
    def centroid(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self._bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self._bbox
        return (x2 - x1) * (y2 - y1)


@dataclass
class Zone:
    """A group of slots sharing a name and level."""

    name: str
    level: str
    slots: List[Slot] = field(default_factory=list)


@dataclass
class ParkingMap:
    """Geometry of the monitored parking area."""

    name: str
    floors: int
    image_size: Tuple[int, int]
    lot: Tuple[int, int, int, int]
    drive_lane: Tuple[int, int]
    entrances: dict
    zones: List[Zone] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | os.PathLike) -> "ParkingMap":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        lot = data["lot"]
        lane = data.get("drive_lane", {})
        if isinstance(lane, dict):
            drive_lane = (lane.get("top", 0), lane.get("bottom", 0))
        else:
            drive_lane = tuple(lane)
        zones: List[Zone] = []
        for z in data.get("zones", []):
            slots = [
                Slot(
                    slot_id=s["id"],
                    zone=z["name"],
                    level=z.get("level", "Level 1"),
                    polygon=s["polygon"],
                )
                for s in z.get("slots", [])
            ]
            zones.append(Zone(name=z["name"], level=z.get("level", "Level 1"), slots=slots))
        return cls(
            name=data.get("name", "Parking"),
            floors=data.get("floors", 1),
            image_size=tuple(data.get("image_size", [1280, 720])),
            lot=(lot["top_left"][0], lot["top_left"][1], lot["bottom_right"][0], lot["bottom_right"][1]),
            drive_lane=drive_lane,
            entrances=data.get("entrances", {}),
            zones=zones,
            metadata=data.get("metadata", {}),
        )

    @property
    def slots(self) -> List[Slot]:
        return [slot for zone in self.zones for slot in zone.slots]

    def get_slot(self, slot_id: str) -> Optional[Slot]:
        for slot in self.slots:
            if slot.slot_id == slot_id:
                return slot
        return None

    def zone_by_name(self, name: str) -> Optional[Zone]:
        for zone in self.zones:
            if zone.name == name:
                return zone
        return None
