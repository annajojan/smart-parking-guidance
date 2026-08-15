import json

from src.detectors.base import Detection
from src.occupancy import match_detections_to_slots


def test_fully_covered_slot_is_occupied(parking_map):
    a1 = parking_map.get_slot("A1")
    detection = Detection(a1.bbox, "car", 0.95)
    occ = match_detections_to_slots(parking_map, [detection])
    assert occ.status_of("A1") == "occupied"
    assert occ.status_of("B1") == "available"
    assert occ.occupied_slots == 1
    assert occ.available_slots == occ.total_slots - 1


def test_no_detections_means_all_available(parking_map):
    occ = match_detections_to_slots(parking_map, [])
    assert occ.occupied_slots == 0
    assert occ.available_slots == occ.total_slots


def test_partial_overlap_crosses_threshold(parking_map):
    a1 = parking_map.get_slot("A1")
    x1, y1, x2, y2 = a1.bbox
    partial = Detection((x1 + 10, y1 + 20, x2 - 10, y2 - 20), "car")
    occ = match_detections_to_slots(parking_map, [partial])
    assert occ.status_of("A1") == "occupied"


def test_small_detection_centered_in_slot_is_occupied(parking_map):
    center = parking_map.get_slot("B2").centroid
    cx, cy = int(center[0]), int(center[1])
    small = Detection((cx - 4, cy - 4, cx + 4, cy + 4), "car", 0.7)
    occ = match_detections_to_slots(parking_map, [small])
    assert occ.status_of("B2") == "occupied"


def test_detection_outside_lot_is_ignored(parking_map):
    detection = Detection((40, 40, 100, 100), "car")
    occ = match_detections_to_slots(parking_map, [detection])
    assert occ.occupied_slots == 0


def test_to_json_round_trip(parking_map):
    detection = Detection(parking_map.get_slot("A3").bbox, "car")
    occ = match_detections_to_slots(parking_map, [detection])
    data = json.loads(occ.to_json())
    assert data["occupied_slots"] == 1
    assert data["slots"][0]["status"] in ("available", "occupied")
    assert "timestamp" in data
