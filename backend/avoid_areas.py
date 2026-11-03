"""Personal "places you avoid" layer on top of a shared route comparison.

The app sends the signed-in person's avoided neighbourhoods with each
request: each one is a list of 100 m grid cells (avoid_areas.blocks, same
grid as the app's src/lib/avoid-blocks.ts) plus when it applies.

Two places use them:
* /routes/preview-real: when every walking route Google suggests passes
  through an avoided place, extra routes are requested through points just
  outside it (detour_points), so there is a route that goes around it.
* /safety/routes/compare (apply): when at least one route avoids every
  avoided place, only those routes get labels and are shown. Routes through
  an avoided place are shown only when there is no way around (for example
  when the start or the destination is inside it). Displayed safety scores
  are not changed: they describe the streets, while avoiding is personal.

It runs after the shared analysis and its cache, so one person's places never
leak into another person's results.
"""

from __future__ import annotations

import copy
import math
from datetime import datetime
from typing import Any, Callable, Iterable

BLOCK_METERS = 100
METERS_PER_DEGREE = 111320
SAMPLE_METERS = 20
# Night for "at night" places, in the walk's local time (departure carries its offset).
NIGHT_STARTS_HOUR = 20
NIGHT_ENDS_HOUR = 7
# Ranking penalty: passing through any avoided place costs at least this much,
# up to the full amount when a third or more of the walk is inside.
MIN_PENALTY = 12.0
FULL_PENALTY = 30.0


def cell_id(lat: float, lng: float) -> str:
    """Same 100 m latitude-band grid id as the app (avoidCellId)."""
    row = math.floor((lat + 90) * METERS_PER_DEGREE / BLOCK_METERS)
    centre_lat = (row + 0.5) * BLOCK_METERS / METERS_PER_DEGREE - 90
    scale = METERS_PER_DEGREE * max(0.01, math.cos(math.radians(centre_lat)))
    return f"{row}:{math.floor((lng + 180) * scale / BLOCK_METERS)}"


def cell_center(cell: str) -> tuple[float, float] | None:
    """(lat, lng) of a grid cell's centre, the inverse of cell_id."""
    try:
        row_text, col_text = cell.split(":")
        row, col = int(row_text), int(col_text)
    except (ValueError, AttributeError):
        return None
    lat = (row + 0.5) * BLOCK_METERS / METERS_PER_DEGREE - 90
    scale = METERS_PER_DEGREE * max(0.01, math.cos(math.radians(lat)))
    return lat, (col + 0.5) * BLOCK_METERS / scale - 180


def is_active(time_of_day: str, departure: datetime | None) -> bool:
    if time_of_day != "night":
        return True  # "always" and legacy "sometimes"
    if departure is None:
        return True  # time unknown: avoid it to be safe
    hour = departure.hour
    return hour >= NIGHT_STARTS_HOUR or hour < NIGHT_ENDS_HOUR


def _samples(geometry: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """Points every ~20 m along a [lon, lat] line, as (lat, lng)."""
    points = [(lat, lon) for lon, lat in geometry]
    out: list[tuple[float, float]] = points[:1]
    for (lat1, lng1), (lat2, lng2) in zip(points, points[1:]):
        dx = (lng2 - lng1) * METERS_PER_DEGREE * math.cos(math.radians((lat1 + lat2) / 2))
        dy = (lat2 - lat1) * METERS_PER_DEGREE
        steps = max(1, math.ceil(math.hypot(dx, dy) / SAMPLE_METERS))
        out.extend((lat1 + (lat2 - lat1) * i / steps, lng1 + (lng2 - lng1) * i / steps) for i in range(1, steps + 1))
    return out


def route_hits(geometry: Iterable[tuple[float, float]], areas: list[dict], departure: datetime) -> list[dict]:
    """Avoided places this route passes, with the share of the walk inside each."""
    samples = _samples(geometry)
    if not samples:
        return []
    cells = [cell_id(lat, lng) for lat, lng in samples]
    hits = []
    for area in areas:
        if not is_active(area.get("time_of_day", "always"), departure):
            continue
        blocks = set(area.get("blocks") or [])
        inside = sum(1 for c in cells if c in blocks)
        if inside:
            hits.append({"name": area.get("name") or "a place you avoid", "share": round(inside / len(cells), 3)})
    return sorted(hits, key=lambda h: -h["share"])


def active_areas(areas: list[dict], departure: datetime | None,
                 endpoints: Iterable[tuple[float, float]] = ()) -> list[dict]:
    """Areas that apply at this time and can be walked around.

    An area containing the start or the destination ([lon, lat]) can't be
    avoided, so it is left out (routes are still marked by route_hits).
    """
    end_cells = {cell_id(lat, lon) for lon, lat in endpoints}
    out = []
    for area in areas:
        blocks = set(area.get("blocks") or [])
        if not blocks or not is_active(area.get("time_of_day", "always"), departure):
            continue
        if blocks & end_cells:
            continue
        out.append({**area, "blocks": blocks})
    return out


def passes_through(geometry: Iterable[tuple[float, float]], areas: list[dict]) -> bool:
    """True when a [lon, lat] line enters any of the (active) areas."""
    if not areas:
        return False
    cells = {cell_id(lat, lng) for lat, lng in _samples(geometry)}
    return any(cells & area["blocks"] for area in areas)


DETOUR_MARGINS_METERS = (150.0, 450.0)
MAX_DETOUR_AREAS = 2


def detour_points(origin: tuple[float, float], destination: tuple[float, float],
                  areas: list[dict], routes: list[list[tuple[float, float]]]) -> list[list[tuple[float, float]]]:
    """Ways around the avoided places the routes cross.

    Each way is one or two points ([lon, lat]) for Google to route through
    ("via" points): beside the place on one side, just outside it, before
    and after it along the walk, so the path goes around it instead of
    cutting a corner through it.
    """
    (lon1, lat1), (lon2, lat2) = origin, destination
    mid_lat = (lat1 + lat2) / 2
    m_lat = METERS_PER_DEGREE
    m_lon = METERS_PER_DEGREE * max(0.01, math.cos(math.radians(mid_lat)))
    dx, dy = (lon2 - lon1) * m_lon, (lat2 - lat1) * m_lat
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length    # along the direct line
    px, py = -uy, ux                     # perpendicular to it

    def to_lonlat(x: float, y: float) -> tuple[float, float]:
        return lon1 + x / m_lon, lat1 + y / m_lat

    route_cells = [{cell_id(lat, lng) for lat, lng in _samples(g)} for g in routes]
    crossed = [a for a in areas if any(c & a["blocks"] for c in route_cells)]
    crossed.sort(key=lambda a: -len(a["blocks"]))
    ways: list[list[tuple[float, float]]] = []
    half = BLOCK_METERS / 2
    for area in crossed[:MAX_DETOUR_AREAS]:
        centres = [p for p in (cell_center(c) for c in area["blocks"]) if p]
        if not centres:
            continue
        # Area cells in metres from the origin: along (a) and sideways (s).
        pts = [((lng - lon1) * m_lon, (lat - lat1) * m_lat) for lat, lng in centres]
        along = [x * ux + y * uy for x, y in pts]
        side = [x * px + y * py for x, y in pts]
        a_min, a_max = min(along) - half, max(along) + half
        for margin in DETOUR_MARGINS_METERS:
            for sign in (1, -1):
                s = (max(side) + half + margin) if sign > 0 else (min(side) - half - margin)
                # Before and after the place, kept between start and destination.
                a_in = min(max(a_min - margin, 0.05 * length), 0.95 * length)
                a_out = min(max(a_max + margin, 0.05 * length), 0.95 * length)
                way = [to_lonlat(a_in * ux + s * px, a_in * uy + s * py)]
                if a_out - a_in > 50:
                    way.append(to_lonlat(a_out * ux + s * px, a_out * uy + s * py))
                ways.append(way)
    return ways


def penalty(hits: list[dict]) -> float:
    if not hits:
        return 0.0
    share = min(1.0, sum(h["share"] for h in hits))
    return MIN_PENALTY + (FULL_PENALTY - MIN_PENALTY) * min(1.0, share * 3)


def _add_detour_cost(result: dict) -> None:
    """How much longer each route that goes around is than the quickest route
    through an avoided place, so the app can say "about 8 min longer to avoid
    X". Only set when a quicker route through one exists."""
    through = [r for r in result["routes"]
               if r.get("avoided_areas") and isinstance(r.get("estimated_time_minutes"), (int, float))]
    if not through:
        return
    quickest = min(through, key=lambda r: r["estimated_time_minutes"])
    names: list[str] = []
    for hit in quickest.get("avoided_areas") or []:
        if hit["name"] not in names:
            names.append(hit["name"])
    for route in result["routes"]:
        minutes = route.get("estimated_time_minutes")
        if route.get("avoided_areas") or not isinstance(minutes, (int, float)):
            continue
        extra = minutes - quickest["estimated_time_minutes"]
        if extra > 0:
            route["avoid_detour"] = {
                "extra_minutes": round(extra),
                "extra_percent": round(100 * extra / max(1, quickest["estimated_time_minutes"])),
                "through_minutes": quickest["estimated_time_minutes"],
                "area_names": names,
            }


def apply(response: dict, routes: list[Any], areas: list[dict], departure: datetime,
          assign_labels: Callable[[list[dict]], dict], pick_display: Callable[[list[dict], dict], list[str]]) -> dict:
    """Copy of a comparison response with avoided places marked and labels re-picked."""
    if not areas or not response.get("routes"):
        return response
    result = copy.deepcopy(response)
    geometry_by_id = {route.route_id: route.geometry for route in routes}
    label_inputs = []
    clean_ids = set()
    for route in result["routes"]:
        hits = route_hits(geometry_by_id.get(route["route_id"], []), areas, departure)
        route["avoided_areas"] = hits
        route["passes_avoided_area"] = bool(hits)
        score = route.get("route_value_score")
        if route.get("status") == "analysis_failed" or not isinstance(score, (int, float)):
            continue
        if not hits:
            clean_ids.add(route["route_id"])
        label_inputs.append({
            "route_id": route["route_id"],
            # Ranking only; the shown score stays the street score.
            "route_value_score": score - penalty(hits),
            "estimated_time_minutes": route.get("estimated_time_minutes"),
        })
    # When some route goes around every avoided place, only those routes are
    # labelled and shown; the others stay in the response, marked.
    if clean_ids:
        label_inputs = [r for r in label_inputs if r["route_id"] in clean_ids]
        _add_detour_cost(result)
    result["avoid_areas_respected"] = bool(clean_ids) or not any(
        r.get("avoided_areas") for r in result["routes"])
    if label_inputs:
        assignment = assign_labels(label_inputs)
        result["labels"] = assignment
        for route in result["routes"]:
            route["labels"] = assignment.get("labels_by_route_id", {}).get(route["route_id"], [])
        result["display_route_ids"] = pick_display(label_inputs, assignment)
    return result
