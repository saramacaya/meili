"""Neighbourhood baseline: compare a walk with the streets around it.

Why: Meili's activity score counts shops, cafes and other places open along a
walk. In a quiet suburb every street has almost none, so every route scored
"low activity" and looked worse than a European city centre, although it is
simply typical for the area. Lighting has the same problem in reverse: where
OpenStreetMap has little lighting data, everything is "unknown" (neutral).

How: the server scores a few "probe" walks next to the route (the same path
shifted sideways by 400 m and 800 m on both sides) with the same data and the
same scoring. From those it learns two things:

* expected_activity: how much activity a typical walk here has. A route's
  activity is then judged against its neighbourhood, but only ever lifted:
  being as quiet as everything around it reads as typical (50), not "low".
  Busy areas are unchanged, so a route can never score worse than before
  because of its neighbourhood.
* area_center: the raw score a typical walk here gets. Until a city cell has
  enough real walks (score_calibration), this is the score shown as 50.

Everything here is pure (no I/O); main._area_baseline runs the probes.
"""

from __future__ import annotations

import copy
import math
from statistics import median
from typing import Any, Optional

PROBE_OFFSETS_METERS = (-800.0, -400.0, 400.0, 800.0)
MIN_PROBES = 3                 # fewer usable probes: no baseline
MIN_TILE_COVERAGE = 0.8        # share of probe map tiles that must be stored
SHORTFALL_SCALE = 40.0         # see lift_activity


def offset_geometry(geometry: list[tuple[float, float]], offset_meters: float) -> list[tuple[float, float]]:
    """The route shifted sideways (perpendicular to start->end) by offset_meters.

    Coordinates are (longitude, latitude). Positive offsets go to the left of
    the direction of travel.
    """
    if len(geometry) < 2:
        return list(geometry)
    (lon1, lat1), (lon2, lat2) = geometry[0], geometry[-1]
    mid_lat = (lat1 + lat2) / 2
    metres_per_deg_lat = 111_000.0
    metres_per_deg_lon = 111_000.0 * max(0.2, math.cos(math.radians(mid_lat)))
    dx = (lon2 - lon1) * metres_per_deg_lon
    dy = (lat2 - lat1) * metres_per_deg_lat
    length = math.hypot(dx, dy)
    if length < 1.0:            # round trip: shift north-south instead
        nx, ny = 0.0, 1.0
    else:
        nx, ny = -dy / length, dx / length   # left-hand normal
    d_lon = nx * offset_meters / metres_per_deg_lon
    d_lat = ny * offset_meters / metres_per_deg_lat
    return [(lon + d_lon, lat + d_lat) for lon, lat in geometry]


def lift_activity(score: float, expected: Optional[float]) -> float:
    """A section's activity judged against its neighbourhood, only ever raised.

    expected is the typical activity of walks around here (0-100). The
    relative score puts `expected` at 50; the result is the higher of that
    and the original, so quiet areas stop looking alarming and busy areas
    (expected 50 or more) are unchanged.
    """
    score = max(0.0, min(100.0, float(score)))
    if expected is None:
        return score
    expected = max(0.0, min(100.0, float(expected)))
    if expected >= 50.0:
        return score            # relative score would never be higher
    if score >= expected:
        relative = 50.0 + 50.0 * (score - expected) / (100.0 - expected)
    else:
        # A shortfall only counts in proportion to how much activity there
        # usually is (at least SHORTFALL_SCALE), so in an almost empty suburb
        # a section with nothing is still close to typical.
        relative = 50.0 - 50.0 * (expected - score) / max(expected, SHORTFALL_SCALE)
    return round(max(score, relative), 1)


def apply_activity_lift(active_places_analysis: dict[str, Any], expected: Optional[float]) -> dict[str, Any]:
    """A copy of the activity analysis with every section lifted (see above).

    The original section scores are kept as raw_activity_score, so the
    explanation screen can still show what was actually found nearby.
    """
    result = copy.deepcopy(active_places_analysis)
    if expected is None or result.get("data_unavailable"):
        return result
    segments = result.get("activity_segments") or []
    for segment in segments:
        raw = float(segment.get("activity_score", 0.0))
        segment["raw_activity_score"] = raw
        segment["activity_score"] = lift_activity(raw, expected)
    if segments:
        result["raw_route_activity_score"] = result.get("route_activity_score")
        result["route_activity_score"] = round(
            sum(float(s["activity_score"]) for s in segments) / len(segments), 1)
    result["neighbourhood_expected_activity"] = round(float(expected), 1)
    return result


def summarise(probe_activity: list[float], probe_raw_scores: list[float]) -> Optional[dict[str, Any]]:
    """Baseline from the probe walks, or None when there are too few."""
    if len(probe_activity) < MIN_PROBES or len(probe_raw_scores) < MIN_PROBES:
        return None
    return {
        "expected_activity": round(median(probe_activity), 1),
        "area_center": round(median(probe_raw_scores), 1),
        "probe_count": len(probe_raw_scores),
    }


def compare_to_area(raw_route_score: float, area_center: Optional[float]) -> Optional[str]:
    """'typical', 'better' or 'worse' than the streets around it (±5 raw points)."""
    if area_center is None:
        return None
    difference = raw_route_score - area_center
    if difference > 5.0:
        return "better"
    if difference < -5.0:
        return "worse"
    return "typical"
