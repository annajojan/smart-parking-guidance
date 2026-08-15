"""Guidance generation abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class GuidanceResult:
    """Natural-language parking directions plus where they came from."""

    text: str
    source: str


class BaseGuidance(ABC):
    """Common interface for guidance generators."""

    @abstractmethod
    def generate(self, occupancy, parking_map) -> GuidanceResult:
        """Return human-friendly directions for a structured occupancy frame."""
