from tests.conftest import build_occupancy


def test_full_lot_message(parking_map):
    all_taken = {s.slot_id: "occupied" for s in parking_map.slots}
    from src.guidance.rule_based import RuleBasedGuidance

    result = RuleBasedGuidance().generate(build_occupancy(parking_map, all_taken), parking_map)
    assert result.source == "rule"
    assert "No parking slots are currently available." in result.text


def test_single_free_slot_mentioned(parking_map):
    statuses = {s.slot_id: "occupied" for s in parking_map.slots}
    statuses["A2"] = "available"
    from src.guidance.rule_based import RuleBasedGuidance

    result = RuleBasedGuidance().generate(build_occupancy(parking_map, statuses), parking_map)
    assert "A2" in result.text
    assert "Zone A" in result.text


def test_partial_occupancy_counts(parking_map):
    statuses = {"A1": "occupied", "B3": "occupied"}
    from src.guidance.rule_based import RuleBasedGuidance

    result = RuleBasedGuidance().generate(build_occupancy(parking_map, statuses), parking_map)
    assert "Total Slots: 8" in result.text
    assert "Occupied: 2" in result.text
    assert "Available: 6" in result.text


def test_deterministic(parking_map):
    statuses = {"A1": "occupied", "B3": "occupied", "B4": "occupied"}
    from src.guidance.rule_based import RuleBasedGuidance

    first = RuleBasedGuidance().generate(build_occupancy(parking_map, statuses), parking_map)
    second = RuleBasedGuidance().generate(build_occupancy(parking_map, statuses), parking_map)
    assert first.text == second.text
