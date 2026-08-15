import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ParkingMap
from src.occupancy import OccupancyFrame, SlotState

MAP_PATH = ROOT / "config" / "parking_map.json"


@pytest.fixture(scope="session")
def parking_map() -> ParkingMap:
    return ParkingMap.from_file(MAP_PATH)


@pytest.fixture(scope="session")
def map_path() -> Path:
    return MAP_PATH


def build_occupancy(parking_map: ParkingMap, statuses: dict) -> OccupancyFrame:
    """Build an OccupancyFrame from a {slot_id: status} mapping."""
    states = []
    for slot in parking_map.slots:
        states.append(
            SlotState(
                slot_id=slot.slot_id,
                zone=slot.zone,
                level=slot.level,
                status=statuses.get(slot.slot_id, "available"),
                confidence=1.0,
                bbox=slot.bbox,
                centroid=[round(v, 1) for v in slot.centroid],
            )
        )
    return OccupancyFrame(
        timestamp="2026-01-01T12:00:00",
        source="test",
        total_slots=len(states),
        available_slots=sum(1 for s in states if s.status == "available"),
        occupied_slots=sum(1 for s in states if s.status == "occupied"),
        slots=states,
    )
