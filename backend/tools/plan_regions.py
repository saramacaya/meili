"""Decide which regions the monthly map build covers.

Prints a JSON list of Geofabrik region paths (e.g. "europe/spain") for the
workflow's build matrix:

  1. Every region in regions.txt (always refreshed, in that order).
  2. Then up to MAX_NEW_REGIONS regions people asked for: the backend notes
     areas it had no map data for (coverage_requests.py). Each note becomes
     the whole country it is in, or, when the country's download is bigger
     than MAX_REGION_MB (for example the US, China or India), the largest
     state/province inside it that fits. Most-requested first.

Regions under EXCLUDED_PREFIXES are never added automatically.
Needs the R2_* environment variables; without them only regions.txt is used.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

INDEX_URL = "https://download.geofabrik.de/index-v1.json"
BASE_URL = "https://download.geofabrik.de/"
MAX_NEW_REGIONS = 5
MAX_REGION_MB = 1500
EXCLUDED_PREFIXES = ("russia",)   # left out on purpose (decision 3 Nov 2026)
REGIONS_FILE = Path(__file__).with_name("regions.txt")


def fixed_regions() -> list[str]:
    lines = REGIONS_FILE.read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def _in_ring(lon: float, lat: float, ring: list) -> bool:
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def contains(geometry: dict, lon: float, lat: float) -> bool:
    if not geometry:
        return False
    polygons = geometry["coordinates"] if geometry.get("type") == "MultiPolygon" else [geometry.get("coordinates") or []]
    for rings in polygons:
        if rings and _in_ring(lon, lat, rings[0]) and not any(_in_ring(lon, lat, hole) for hole in rings[1:]):
            return True
    return False


def region_path(feature: dict) -> str | None:
    """'europe/spain' from the feature's .osm.pbf download link."""
    url = ((feature.get("properties") or {}).get("urls") or {}).get("pbf") or ""
    if not url.startswith(BASE_URL) or not url.endswith("-latest.osm.pbf"):
        return None
    return url[len(BASE_URL):-len("-latest.osm.pbf")]


def choose_region(features: list[dict], lon: float, lat: float, size_mb) -> str | None:
    """The biggest region containing the point whose download fits, starting
    at country level (one level below a continent)."""
    by_id = {f["properties"]["id"]: f for f in features}
    hits = [f for f in features if contains(f.get("geometry"), lon, lat)]
    if not hits:
        return None

    def depth(feature: dict) -> int:
        level, parent = 0, feature["properties"].get("parent")
        while parent and parent in by_id and level < 10:
            level += 1
            parent = by_id[parent]["properties"].get("parent")
        return level

    # depth 0 = continent, 1 = country, 2+ = states, provinces, districts.
    for feature in sorted((f for f in hits if depth(f) >= 1), key=depth):
        path = region_path(feature)
        if not path or path.startswith(EXCLUDED_PREFIXES):
            continue
        size = size_mb(path)
        if size is not None and size <= MAX_REGION_MB:
            return path
    return None


def download_size_mb(path: str) -> float | None:
    try:
        response = requests.head(f"{BASE_URL}{path}-latest.osm.pbf", allow_redirects=True, timeout=30)
        length = response.headers.get("Content-Length")
        return int(length) / 1_000_000 if response.ok and length else None
    except requests.RequestException:
        return None


def requested_regions(notes: list[dict], features: list[dict], known: set[str], size_mb=download_size_mb) -> list[str]:
    counts: dict[str, int] = {}
    for note in sorted(notes, key=lambda n: -int(n.get("count", 0))):
        try:
            lat, lon = float(note["lat"]), float(note["lng"])
        except (KeyError, TypeError, ValueError):
            continue
        path = choose_region(features, lon, lat, size_mb)
        if path and path not in known:
            counts[path] = counts.get(path, 0) + int(note.get("count", 1))
    ranked = sorted(counts, key=lambda p: -counts[p])
    return ranked[:MAX_NEW_REGIONS]


def main() -> None:
    regions = fixed_regions()
    extra: list[str] = []
    try:
        import coverage_requests
        notes = coverage_requests.list_requests()
        if notes:
            index = requests.get(INDEX_URL, timeout=120).json()
            extra = requested_regions(notes, index.get("features", []), set(regions))
        print(f"{len(notes)} requested areas -> new regions: {extra or 'none'}", file=sys.stderr)
    except Exception as error:  # noqa: BLE001 — never block the normal build
        print(f"Requested areas skipped: {error}", file=sys.stderr)
    print(json.dumps(regions + extra))


if __name__ == "__main__":
    main()
