"""Deterministic, dependency-free guidance generator.

Uses the parking geometry to select the best available slot and produce a
standardised, professional parking guidance output. Serves as the default
and as the offline fallback for the LLM generator.
"""

from __future__ import annotations

import math

from .base import BaseGuidance, GuidanceResult


class RuleBasedGuidance(BaseGuidance):
    """Rule-based parking guidance with standardised template output."""

    def __init__(self, preferred_entrance: str = "west") -> None:
        self.preferred_entrance = preferred_entrance

    def generate(self, occupancy, parking_map) -> GuidanceResult:
        available = [s for s in occupancy.slots if s.status == "available"]
        total = occupancy.total_slots

        if not available:
            text = (
                f"SMARTPARK PARKING GUIDANCE\n"
                f"\n"
                f"Location: {parking_map.name}\n"
                f"Level: Level 1\n"
                f"\n"
                f"PARKING STATUS\n"
                f"Total Slots: {total}\n"
                f"Occupied: {occupancy.occupied_slots}\n"
                f"Available: {occupancy.available_slots}\n"
                f"\n"
                f"No parking slots are currently available."
            )
            return GuidanceResult(text=text, source="rule")

        entrance = parking_map.entrances.get(
            self.preferred_entrance,
            next(iter(parking_map.entrances.values())),
        )

        def distance_to_entry(slot) -> float:
            cx, cy = slot.centroid
            return math.hypot(cx - entrance[0], cy - entrance[1])

        ranked = sorted(available, key=distance_to_entry)
        best = ranked[0]

        entrance_name = self.preferred_entrance.capitalize()

        text = (
            f"SMARTPARK PARKING GUIDANCE\n"
            f"\n"
            f"Location: {parking_map.name}\n"
            f"Level: {best.level}\n"
            f"\n"
            f"PARKING STATUS\n"
            f"Total Slots: {total}\n"
            f"Occupied: {occupancy.occupied_slots}\n"
            f"Available: {occupancy.available_slots}\n"
            f"\n"
            f"RECOMMENDED SLOT\n"
            f"Slot: {best.slot_id}\n"
            f"Zone: {best.zone}\n"
            f"Status: AVAILABLE\n"
            f"\n"
            f"DIRECTIONS\n"
            f"1. Enter through {entrance_name} entrance.\n"
            f"2. Proceed to Zone {best.zone}.\n"
            f"3. Follow the signs toward {best.slot_id}.\n"
            f"4. Park in the designated {best.slot_id} slot.\n"
            f"\n"
            f"Destination: {best.slot_id}"
        )

        return GuidanceResult(text=text, source="rule")
