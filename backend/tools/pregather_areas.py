"""Gather local context in advance for the most-walked parts of the main cities,
so even the first user there sees it instantly.

Walks a grid of points (about every 700 m) over each city centre, finds the
neighbourhood at each point, and gathers news and opinions for every
neighbourhood that has no fresh local context yet -- with exactly the same
code the app uses (local_context_gatherer.gather_area), into the same
Supabase tables.

Free-tier friendly: stops after MAX_AREAS new neighbourhoods per run
(each costs about 2 Tavily searches and a few AI calls) and pauses between
areas. Already-gathered areas are skipped, so later runs only fill gaps.

Run (GitHub Actions or locally):
  SUPABASE_URL=... SUPABASE_SECRET_KEY=... TAVILY_API_KEY=... GEMINI_API_KEY=... \
  python backend/tools/pregather_areas.py
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import local_context_gatherer as gatherer  # noqa: E402

# name, centre (lat, lon), radius in km -- most visited first
CITIES = [
    ("Madrid", 40.4168, -3.7038, 2.5),
    ("Barcelona", 41.3874, 2.1686, 2.5),
    ("Valencia", 39.4699, -0.3763, 2.0),
    ("Paris", 48.8566, 2.3522, 3.0),
    ("London", 51.5072, -0.1276, 3.0),
    ("Rome", 41.9028, 12.4964, 2.0),
    ("Milan", 45.4642, 9.1900, 2.0),
    ("Lisbon", 38.7223, -9.1393, 2.0),
    ("Amsterdam", 52.3676, 4.9041, 2.0),
    ("Berlin", 52.5200, 13.4050, 2.5),
    ("Seville", 37.3891, -5.9845, 1.5),
    ("Porto", 41.1579, -8.6291, 1.5),
    ("Castellón de la Plana", 39.9864, -0.0513, 1.5),
    ("Ljubljana", 46.0569, 14.5058, 1.2),
]

GRID_STEP_KM = 0.7
MAX_AREAS = int(os.getenv("MEILI_PREGATHER_MAX_AREAS", "120"))
PAUSE_BETWEEN_AREAS = 3.0


def grid_points(lat: float, lon: float, radius_km: float) -> list[tuple[float, float]]:
    """(lon, lat) points about GRID_STEP_KM apart inside a circle, centre first."""
    steps = int(radius_km / GRID_STEP_KM)
    dlat = GRID_STEP_KM / 110.54
    dlon = GRID_STEP_KM / (111.32 * math.cos(math.radians(lat)))
    points = []
    for i in range(-steps, steps + 1):
        for j in range(-steps, steps + 1):
            if math.hypot(i, j) * GRID_STEP_KM <= radius_km:
                points.append((math.hypot(i, j), lon + j * dlon, lat + i * dlat))
    return [(x, y) for _, x, y in sorted(points)]   # centre outwards


def main() -> int:
    if gatherer._db() is None:
        print("Database not configured (SUPABASE_URL / SUPABASE_SECRET_KEY).")
        return 1
    if not os.getenv("TAVILY_API_KEY"):
        print("::warning::TAVILY_API_KEY is not set: only news will be searched, no opinions.")

    gathered = skipped = found_notes = 0
    for city, lat, lon, radius in CITIES:
        if gathered >= MAX_AREAS:
            break
        seen: set[str] = set()
        print(f"\n== {city}")
        for point_lon, point_lat in grid_points(lat, lon, radius):
            if gathered >= MAX_AREAS:
                print(f"Reached {MAX_AREAS} new areas for this run; the next run continues.")
                break
            area = gatherer.reverse_neighbourhood(point_lon, point_lat)
            if not area:
                continue
            key = gatherer.area_key(area["city"], area["name"])
            if key in seen:
                continue
            seen.add(key)
            if gatherer._area_known(key):
                skipped += 1
                continue
            try:
                result = gatherer.gather_area(area, budget_seconds=60)
            except Exception as error:  # noqa: BLE001
                print(f"  ! {area['name']}: {str(error)[:150]}")
                continue
            gathered += 1
            found_notes += result["records"]
            print(f"  + {area['name']}: {result['records']} notes from {result['articles_found']} pages")
            time.sleep(PAUSE_BETWEEN_AREAS)

    print(f"\nDone: {gathered} neighbourhoods gathered ({found_notes} notes), "
          f"{skipped} already had fresh local context.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
