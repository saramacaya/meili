"""Neighbourhood search for "places you avoid" (OpenStreetMap Nominatim).

Works worldwide: a search returns named areas (neighbourhoods, districts,
quarters...) with their outline when OpenStreetMap has one. About a third of
European neighbourhoods are mapped only as a point; those come back as a
circle (POINT_RADIUS_M) so the app can still mark them.

Results are cached, and calls share the 1-request-per-second limit with the
other Nominatim lookups (local_context_gatherer), as Nominatim's usage
policy requires.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Optional

import outbound
from local_context_gatherer import USER_AGENT, _last_nominatim_call, _nominatim_lock

SEARCH_URL = "https://nominatim.openstreetmap.org/search"
POINT_RADIUS_M = 500
# Bigger than this is a whole town or city, not a neighbourhood to route around.
MAX_AREA_KM2 = 30.0
CACHE_SECONDS = 7 * 24 * 3600
MAX_CACHE = 2000

# Nominatim "addresstype" values that name an area people live in or walk through.
AREA_TYPES = {
    "neighbourhood", "suburb", "quarter", "city_district", "borough", "district",
    "residential", "village", "hamlet", "croft", "isolated_dwelling", "allotments",
}

_cache: dict[tuple, tuple[float, list[dict]]] = {}
_cache_lock = threading.Lock()


def _area_km2(geometry: dict) -> float:
    """Approximate area of a GeoJSON Polygon / MultiPolygon (outer rings minus holes)."""
    def ring_area(ring: list) -> float:
        if len(ring) < 3:
            return 0.0
        lat0 = math.radians(sum(p[1] for p in ring) / len(ring))
        kx, ky = 111.32 * math.cos(lat0), 110.57
        total = 0.0
        for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
            total += (x1 * kx) * (y2 * ky) - (x2 * kx) * (y1 * ky)
        return abs(total) / 2

    def polygon_area(rings: list) -> float:
        return max(0.0, ring_area(rings[0]) - sum(ring_area(r) for r in rings[1:])) if rings else 0.0

    if geometry.get("type") == "Polygon":
        return polygon_area(geometry.get("coordinates") or [])
    if geometry.get("type") == "MultiPolygon":
        return sum(polygon_area(p) for p in geometry.get("coordinates") or [])
    return 0.0


def _context(address: dict) -> str:
    """'València, Spain' style second line for a result."""
    city = (address.get("city") or address.get("town") or address.get("village")
            or address.get("municipality") or address.get("county"))
    parts = [p for p in (city, address.get("country")) if p]
    return ", ".join(dict.fromkeys(parts))


def _to_result(item: dict) -> Optional[dict]:
    address = item.get("address") or {}
    kind = item.get("addresstype") or item.get("type")
    if item.get("category") not in ("boundary", "place") and kind not in AREA_TYPES:
        return None  # stations, shops, streets...
    if kind not in AREA_TYPES and item.get("category") == "place" and item.get("type") in ("city", "town", "state", "country"):
        return None
    try:
        lat, lng = float(item["lat"]), float(item["lon"])
    except (KeyError, TypeError, ValueError):
        return None
    name = item.get("name") or (item.get("display_name") or "").split(",")[0].strip()
    if not name:
        return None
    result: dict[str, Any] = {
        "id": f"{(item.get('osm_type') or 'x')[:1]}{item.get('osm_id')}",
        "name": name,
        "context": _context(address),
        "center": {"lat": lat, "lng": lng},
    }
    geometry = item.get("geojson") or {}
    if geometry.get("type") in ("Polygon", "MultiPolygon"):
        area = _area_km2(geometry)
        if area > MAX_AREA_KM2 or area <= 0:
            return None
        result.update(shape="outline", geometry=geometry, area_km2=round(area, 2))
    else:
        result.update(shape="circle", radius_m=POINT_RADIUS_M,
                      area_km2=round(math.pi * (POINT_RADIUS_M / 1000) ** 2, 2))
    return result


def search_areas(query: str, lat: Optional[float] = None, lng: Optional[float] = None,
                 language: Optional[str] = None, limit: int = 6) -> list[dict]:
    """Named areas matching the query, nearest to (lat, lng) first when given."""
    query = " ".join(query.split())[:120]
    near = (round(lat, 1), round(lng, 1)) if lat is not None and lng is not None else None
    key = (query.lower(), near, (language or "")[:5])
    now = time.time()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < CACHE_SECONDS:
            return hit[1]

    params: dict[str, Any] = {
        "q": query, "format": "jsonv2", "limit": 15, "addressdetails": 1,
        "polygon_geojson": 1, "polygon_threshold": 0.0001, "dedupe": 1,
    }
    if near:
        # Prefer results around the user, without excluding the rest of the world.
        params["viewbox"] = f"{lng - 0.4},{lat + 0.4},{lng + 0.4},{lat - 0.4}"
    # As a query parameter (not a header) so it also survives the proxy.
    params["accept-language"] = language or "en"
    with _nominatim_lock:  # Nominatim policy: at most 1 request per second
        wait = 1.1 - (time.monotonic() - _last_nominatim_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_nominatim_call[0] = time.monotonic()
        response = outbound.nominatim_get(SEARCH_URL, params, USER_AGENT)

    results: list[dict] = []
    seen: set[str] = set()
    for item in response.json() or []:
        result = _to_result(item)
        if not result or result["id"] in seen:
            continue
        seen.add(result["id"])
        results.append(result)
    if near:
        results.sort(key=lambda r: math.hypot(r["center"]["lat"] - lat,
                                              (r["center"]["lng"] - lng) * math.cos(math.radians(lat))))
    # The same place can be mapped as both a point and an outline: keep the outline.
    best: dict[tuple, dict] = {}
    for result in results:
        key = (result["name"].lower(), result["context"])
        if key not in best or (best[key]["shape"] == "circle" and result["shape"] == "outline"):
            best[key] = result
    results = [r for r in results if best.get((r["name"].lower(), r["context"])) is r][:limit]

    with _cache_lock:
        if len(_cache) >= MAX_CACHE:
            for stale in sorted(_cache, key=lambda k: _cache[k][0])[: MAX_CACHE // 4]:
                del _cache[stale]
        _cache[key] = (now, results)
    return results
