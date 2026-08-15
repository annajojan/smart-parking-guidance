"""Deterministic, dependency-free guidance generator.

Uses the parking geometry to produce clear, human-friendly directions without
requiring any external API. It selects the free slot closest to the entry
point and describes how to reach it. Serves as the default and as the offline
fallback for the LLM generator.
"""

from __future__ import annotations

import math

from .base import BaseGuidance, GuidanceResult

_ORDINALS = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
}


class RuleBasedGuidance(BaseGuidance):
    """Rule-based natural-language parking directions."""

    def __init__(self, preferred_entrance: str = "west") -> None:
        self.preferred_entrance = preferred_entrance

    @staticmethod
    def _bay_index(slot) -> int:
        suffix = "".join(ch for ch in slot.slot_id if ch.isdigit())
        return int(suffix) if suffix else 1

    @staticmethod
    def _side(slot, direction: int) -> str:
        is_top_row = slot.zone in ("A", "Top")
        if direction > 0:
            return "left" if is_top_row else "right"
        return "right" if is_top_row else "left"

    def generate(self, occupancy, parking_map) -> GuidanceResult:
        available = [s for s in occupancy.slots if s.status == "available"]

        total = occupancy.total_slots
        free = len(available)

        if free == 0:
            text = (
                f"Sorry, all {total} spots in {parking_map.name} are currently taken. "
                "Please wait near the entrance for a vehicle to leave, or follow the "
                "overflow signage to the adjacent lot."
            )
            return GuidanceResult(text=text, source="rule")

        entrance = parking_map.entrances.get(
            self.preferred_entrance,
            next(iter(parking_map.entrances.values())),
        )
        direction = 1 if self.preferred_entrance == "west" else -1
        scale = float(parking_map.metadata.get("pixels_to_metres", 0.02))

        def distance_to_entry(slot) -> float:
            cx, cy = slot.centroid
            return math.hypot(cx - entrance[0], cy - entrance[1])

        ranked = sorted(available, key=distance_to_entry)
        best = ranked[0]
        best_dist = distance_to_entry(best) * scale

        ordinal = _ORDINALS.get(self._bay_index(best), f"{self._bay_index(best)}th")
        side = self._side(best, direction)

        lines = [
            f"Good news - {free} of {total} spots are currently free.",
            (
                f"The nearest free spot is {best.slot_id} in Zone {best.zone}, "
                f"on your {side}, about {best_dist:.0f} metres ahead."
            ),
            (
                f"Drive down the aisle and take the {ordinal} bay on your {side}. "
                "Pull in carefully - the spot is waiting for you."
            ),
        ]
        if len(ranked) > 1:
            backups = ", ".join(s.slot_id for s in ranked[1:4])
            lines.append(f"Prefer a backup? {backups} are also free nearby.")

        return GuidanceResult(text=" ".join(lines), source="rule")
