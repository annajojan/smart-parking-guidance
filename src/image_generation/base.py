"""Abstract base for image generators."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseImageGenerator(ABC):
    """Common interface for local image generators."""

    name: str = "base"

    @abstractmethod
    def generate(self, occupancy, parking_map):
        """Generate a parking visualisation image.

        Args:
            occupancy: OccupancyFrame with per-slot status.
            parking_map: ParkingMap with geometry.

        Returns:
            A PIL Image showing the parking situation.
        """
