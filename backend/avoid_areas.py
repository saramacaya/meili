"""Personal "places you avoid" layer on top of a shared route comparison.

The app sends the signed-in person's avoided neighbourhoods with each
comparison: each one is a list of 100 m grid cells (avoid_areas.blocks, same
grid as the app's src/lib/avoid-blocks.ts) plus when it applies. This module
marks which routes pass through them and moves the labels (best rated,
balanced) away from those routes. Displayed safety scores are not changed:
they describe the streets, while avoiding is a personal preference.

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


def is_active(time_of_day: str, departure: datetime) -> bool:
    if time_of_day != "night":
        return True  # "always" and legacy "sometimes"
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


def penalty(hits: list[dict]) -> float:
    if not hits:
        return 0.0
    share = min(1.0, sum(h["share"] for h in hits))
    return MIN_PENALTY + (FULL_PENALTY - MIN_PENALTY) * min(1.0, share * 3)


def apply(response: dict, routes: list[Any], areas: list[dict], departure: datetime,
          assign_labels: Callable[[list[dict]], dict], pick_display: Callable[[list[dict], dict], list[str]]) -> dict:
    """Copy of a comparison response with avoided places marked and labels re-picked."""
    if not areas or not response.get("routes"):
        return response
    result = copy.deepcopy(response)
    geometry_by_id = {route.route_id: route.geometry for route in routes}
    label_inputs = []
    for route in result["routes"]:
        hits = route_hits(geometry_by_id.get(route["route_id"], []), areas, departure)
        route["avoided_areas"] = hits
        score = route.get("route_value_score")
        if route.get("status") == "analysis_failed" or not isinstance(score, (int, float)):
            continue
        label_inputs.append({
            "route_id": route["route_id"],
            # Ranking only; the shown score stays the street score.
            "route_value_score": score - penalty(hits),
            "estimated_time_minutes": route.get("estimated_time_minutes"),
        })
    if label_inputs:
        assignment = assign_labels(label_inputs)
        result["labels"] = assignment
        for route in result["routes"]:
            route["labels"] = assignment["labels_by_route_id"].get(route["route_id"], [])
        result["display_route_ids"] = pick_display(label_inputs, assignment)
    return result
