"""Fast "how close is this point to the route?" lookups.

The old code compared every point with every route sample using the exact
Earth-distance formula -- millions of calculations per request. Here the
route samples are put into a grid whose cells are as wide as the search
radius, so for any point only the samples in the 3 x 3 neighbouring cells
can possibly be within range. The exact formula is then used only on those
few candidates, so results are identical to before, just much faster.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

Point = tuple[float, float]  # (longitude, latitude)


class RouteIndex:
    def __init__(self, samples: list[Point], radius_meters: float,
                 distance: Callable[[Point, Point], float]) -> None:
        self.distance = distance
        self.radius = radius_meters
        mean_lat = sum(p[1] for p in samples) / len(samples) if samples else 0.0
        # Cells at least `radius` wide in both directions (slightly larger
        # for safety, since longitude degrees shrink towards the poles).
        self.lat_cell = (radius_meters * 1.05) / 111_000
        self.lon_cell = (radius_meters * 1.05) / (
            111_000 * max(math.cos(math.radians(mean_lat)) - 0.01, 0.05)
        )
        self.cells: dict[tuple[int, int], list[Point]] = {}
        for point in samples:
            self.cells.setdefault(self._cell(point), []).append(point)

    def _cell(self, point: Point) -> tuple[int, int]:
        return (math.floor(point[1] / self.lat_cell), math.floor(point[0] / self.lon_cell))

    def nearest_within(self, point: Point) -> Optional[float]:
        """Exact distance to the nearest route sample, if within the radius."""
        row, col = self._cell(point)
        best: Optional[float] = None
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                for sample in self.cells.get((row + dr, col + dc), ()):
                    d = self.distance(point, sample)
                    if best is None or d < best:
                        best = d
        if best is None or best > self.radius:
            return None
        return best

    def nearest_within_any(self, points: list[Point]) -> Optional[float]:
        """Smallest distance between any of `points` and the route, if within radius."""
        best: Optional[float] = None
        for point in points:
            d = self.nearest_within(point)
            if d is not None and (best is None or d < best):
                best = d
        return best


class PointIndex:
    """Grid over target points (lamps, street samples) for nearest lookups.

    `candidates(point)` returns every target that could be within `radius`
    of `point` (it may also return a few slightly further away). If a target
    is within the radius it is guaranteed to be among the candidates.
    """

    def __init__(self, items: list[tuple[Point, object]], radius_meters: float) -> None:
        mean_lat = sum(p[1] for p, _ in items) / len(items) if items else 0.0
        self.lat_cell = (radius_meters * 1.05) / 111_000
        self.lon_cell = (radius_meters * 1.05) / (
            111_000 * max(math.cos(math.radians(mean_lat)) - 0.01, 0.05)
        )
        self.cells: dict[tuple[int, int], list[tuple[Point, object]]] = {}
        for point, payload in items:
            self.cells.setdefault(self._cell(point), []).append((point, payload))

    def _cell(self, point: Point) -> tuple[int, int]:
        return (math.floor(point[1] / self.lat_cell), math.floor(point[0] / self.lon_cell))

    def candidates(self, point: Point) -> list[tuple[Point, object]]:
        row, col = self._cell(point)
        found: list[tuple[Point, object]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                found.extend(self.cells.get((row + dr, col + dc), ()))
        return found
