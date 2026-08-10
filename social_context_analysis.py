"""Social-context scoring and map overlays for Meili.

Route coordinates use GeoJSON order: [longitude, latitude].

Map overlays are now generated directly from the same opinion records used
for scoring (see ``generate_overlays_for_city``) instead of a hand-maintained
static GeoJSON file. This removes the old failure mode where the overlay
file and the scoring data could silently drift apart -- there is now exactly
one source of truth per city: ``public_opinions.json``.

Regional evidence selection is coordinate-driven: every registered city's
opinion/overlay dataset is loaded and matched purely by route geometry. The
caller never needs to know or declare which municipality a route falls in.
Coverage grows automatically as new city folders are added to
``data/social_context`` and registered in ``city_registry.json`` (for example
Alicante) -- no code change is required here to pick them up.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any


DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "data" / "social_context"
DATA_ROOT = Path(os.getenv("MEILI_SOCIAL_CONTEXT_DATA", DEFAULT_DATA_ROOT))

CITY_REGISTRY_FILENAME = "city_registry.json"

TIME_WEIGHTS = {
    "unspecified": 1.0,
    "daytime_and_unspecified": 1.0,
    "daytime": 1.0,
    "evening": 1.0,
    "night": 1.0,
    "late_night": 1.0,
    "weekend": 1.0,
}

# ---------------------------------------------------------------------------
# Geographic-precision classification
#
# ``locations[].precision`` on each opinion record is free-text set by
# whoever authored the record. This maps every value actually seen in the
# data (plus a safe fallback for anything new/unrecognised) onto exactly
# four buckets:
#
#   point         a single, specific place: a venue, station, exact spot.
#   street        a specific street, promenade, or a named point along one.
#   neighbourhood a broad named zone -- not one street, square or venue.
#   unknown       missing, unrecognised, or too ambiguous to place at all
#                 (e.g. a record naming several distinct plazas at once).
#
# Only "point" and "street" are ever drawn on the map or given a direct
# numeric score effect. "neighbourhood" and "unknown" are non-mappable --
# they can only ever appear as background text context.
# ---------------------------------------------------------------------------

POINT_PRECISIONS = {"exact_place", "named_place"}
STREET_PRECISIONS = {"street_section", "linear_place", "linear_feature_anchor"}
NEIGHBOURHOOD_PRECISIONS = {
    "neighbourhood",
    "neighbourhood_section",
    "district",
    "city_sector",
    "citywide",
    # "named_area" (e.g. "Casitas Rosas area", "Marina docks") names a zone
    # around a place rather than the place itself, and in the data seen so
    # far always carries a 500m+ radius -- treated as neighbourhood-broad
    # rather than a precise point. Judgment call -- see the delivery report.
    "named_area",
}

MAPPABLE_PRECISION_CLASSES = {"point", "street"}
NON_MAPPABLE_PRECISION_CLASSES = {"neighbourhood", "unknown"}

# Fixed display radius for a mappable overlay circle -- deliberately not
# derived from the record's own (often much larger) evidence radius, which
# is still used for scoring/matching purposes further below.
OVERLAY_RADIUS_METERS = {"point": 200.0, "street": 250.0}

# Evidence tiers grounded enough that, combined with negative sentiment and
# decent confidence, they justify red rather than orange on the map.
STRONG_EVIDENCE_TIERS = {"confirmed_incident", "municipal_data", "resident_complaint"}

OVERLAY_COLOURS = {
    "red": "#DC2626",
    "orange": "#F97316",
}


def location_precision_class(precision: str | None) -> str:
    """Buckets a raw ``locations[].precision`` tag into one of four classes.

    Anything not explicitly recognised (missing, or a tag such as
    "multiple_named_places" that names more than one distinct place) falls
    through to "unknown" -- the same non-mappable, text-only treatment as
    "neighbourhood", never treated as if it were precise.
    """
    key = (precision or "").strip().lower()
    if key in POINT_PRECISIONS:
        return "point"
    if key in STREET_PRECISIONS:
        return "street"
    if key in NEIGHBOURHOOD_PRECISIONS:
        return "neighbourhood"
    return "unknown"


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as error:
        raise RuntimeError(f"Missing social-context file: {path}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid JSON in social-context file: {path}") from error


@lru_cache(maxsize=1)
def load_city_registry() -> dict[str, Any]:
    """Loads the registry describing every municipality with prepared data.

    Adding regional coverage (e.g. Alicante) is a data change only: drop a new
    folder under ``data/social_context`` and add one entry here.
    """
    registry_path = DATA_ROOT / CITY_REGISTRY_FILENAME
    registry = _read_json(registry_path)
    if not isinstance(registry.get("cities"), dict) or not registry["cities"]:
        raise RuntimeError(f"City registry at {registry_path} has no cities configured")
    return registry


def registered_city_ids() -> list[str]:
    return sorted(load_city_registry()["cities"].keys())


@lru_cache(maxsize=8)
def load_opinions(city_id: str) -> list[dict[str, Any]]:
    cities = load_city_registry()["cities"]
    if city_id not in cities:
        raise RuntimeError(f"Unknown city_id: {city_id}")
    relative_path = cities[city_id].get("opinions_file", f"{city_id}/public_opinions.json")
    records = _read_json(DATA_ROOT / relative_path)
    if not isinstance(records, list):
        raise RuntimeError(f"Opinions for {city_id} must be a JSON list")
    return records


def _circle_polygon(
    latitude: float,
    longitude: float,
    radius_meters: float,
    points: int = 32,
) -> list[list[list[float]]]:
    """A closed N-point ring approximating a circle, GeoJSON [lon, lat] order."""
    coordinates: list[list[float]] = []
    for index in range(points):
        angle = 2 * math.pi * index / points
        delta_lat = (radius_meters * math.cos(angle)) / 111_320
        delta_lon = (radius_meters * math.sin(angle)) / (
            111_320 * math.cos(math.radians(latitude))
        )
        coordinates.append([round(longitude + delta_lon, 6), round(latitude + delta_lat, 6)])
    coordinates.append(coordinates[0])
    return [coordinates]


def _overlay_severity(record: dict[str, Any], sentiment: str) -> str:
    """"red" for strong, specific negative evidence -- "orange" for
    everything else that still qualifies for a visible area (limited,
    mixed, or lower-confidence evidence)."""
    confidence = record.get("confidence") or 0
    strong_tier = record.get("evidence_tier") in STRONG_EVIDENCE_TIERS
    if sentiment == "negative" and strong_tier and confidence >= 3:
        return "red"
    return "orange"


def generate_overlays_for_city(city_id: str) -> dict[str, Any]:
    """Builds the map-overlay FeatureCollection directly from opinion
    records -- the single source of truth also used for scoring.

    Only records whose location precision classifies as "point" or "street"
    ever produce a visible shape (see ``location_precision_class``).
    Neighbourhood-level and unrecognised evidence never appears on the map,
    no matter how many records mention that neighbourhood.

    Only negative or mixed-sentiment records are drawn at all -- this has
    always been a "flagged safety concern" layer, not a general opinion map.
    """
    features: list[dict[str, Any]] = []

    for record in load_opinions(city_id):
        if not record.get("rating_eligible", False):
            continue
        sentiment = record.get("overall_sentiment")
        if sentiment not in {"negative", "mixed"}:
            continue

        for location_index, location in enumerate(record.get("locations") or []):
            if location.get("type") != "circle":
                continue
            precision_class = location_precision_class(location.get("precision"))
            if precision_class not in MAPPABLE_PRECISION_CLASSES:
                continue
            try:
                latitude = float(location["latitude"])
                longitude = float(location["longitude"])
            except (KeyError, TypeError, ValueError):
                continue

            radius = OVERLAY_RADIUS_METERS[precision_class]
            severity = _overlay_severity(record, sentiment)
            colour = OVERLAY_COLOURS[severity]
            feature_id = f"{record.get('record_id')}-{location_index}"
            label = location.get("label") or record.get("street_or_place") or record.get("area")

            features.append({
                "type": "Feature",
                "id": feature_id,
                "properties": {
                    "overlay_id": feature_id,
                    "city_id": city_id,
                    "record_id": record.get("record_id"),
                    "label": label,
                    "scope": "place",
                    "classification": "community_safety_concern",
                    "severity": severity,
                    "precision_class": precision_class,
                    "confidence": record.get("confidence"),
                    "evidence_tier": record.get("evidence_tier"),
                    "post_date": record.get("post_date"),
                    "display": {
                        "fill_color": colour,
                        "fill_opacity": 0.12,
                        "stroke_color": colour,
                        "stroke_opacity": 0.45 if severity == "red" else 0.32,
                        "stroke_width": 1,
                    },
                    "interaction": {
                        "tap_title": (
                            "Community safety concern"
                            if severity == "red"
                            else "Limited or mixed local evidence"
                        ),
                        "tap_subtitle": label,
                        "show_evidence_count": False,
                        "wording_guardrail": (
                            "Describe this as reported or perceived concern for this "
                            "specific spot, not proof that every nearby street is unsafe."
                        ),
                    },
                    "boundary_status": "fixed_radius_not_an_administrative_boundary",
                    "geometry_source": (
                        f"circle_around_evidence_point_fixed_{int(radius)}m_"
                        f"for_{precision_class}_precision"
                    ),
                    "geometry_role": "visual_evidence_overlay",
                    "rating_double_count": False,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": _circle_polygon(latitude, longitude, radius),
                },
            })

    return {
        "type": "FeatureCollection",
        "name": f"{city_id}_social_safety_concern_overlays",
        "features": features,
    }


@lru_cache(maxsize=8)
def load_overlays(city_id: str) -> dict[str, Any]:
    cities = load_city_registry()["cities"]
    if city_id not in cities:
        raise RuntimeError(f"Unknown city_id: {city_id}")
    return generate_overlays_for_city(city_id)


def load_all_opinions() -> dict[str, list[dict[str, Any]]]:
    """Loads every registered city's opinion records, keyed by city_id.

    A city whose data file is missing or malformed is skipped rather than
    failing the whole regional lookup -- one municipality's data problem
    should not take down evidence matching everywhere else.
    """
    all_opinions: dict[str, list[dict[str, Any]]] = {}
    for city_id in registered_city_ids():
        try:
            all_opinions[city_id] = load_opinions(city_id)
        except RuntimeError:
            continue
    return all_opinions


def load_all_overlays(
    bbox: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    """Merges every registered city's overlay FeatureCollection into one.

    When ``bbox`` (west, south, east, north) is supplied, only features whose
    representative point falls inside it are kept -- this lets the frontend
    ask for "whatever is visible in the current map view" without needing to
    know which municipality that view covers (Priority 7: overlays should
    stay visible for the current geographic view, not just the selected
    route).
    """
    merged_features: list[dict[str, Any]] = []
    cities_included: list[str] = []

    for city_id in registered_city_ids():
        try:
            geojson = load_overlays(city_id)
        except RuntimeError:
            continue

        cities_included.append(city_id)

        for feature in geojson.get("features", []):
            if bbox is not None:
                point = _feature_representative_point(feature)
                if point is None:
                    continue
                west, south, east, north = bbox
                longitude, latitude = point
                if not (west <= longitude <= east and south <= latitude <= north):
                    continue

            feature_with_city = dict(feature)
            properties = dict(feature.get("properties") or {})
            properties.setdefault("city_id", city_id)
            feature_with_city["properties"] = properties
            merged_features.append(feature_with_city)

    return {
        "type": "FeatureCollection",
        "cities_included": cities_included,
        "features": merged_features,
    }


def _flatten_coordinates(value: Any) -> list[tuple[float, float]]:
    if (
        isinstance(value, list)
        and len(value) >= 2
        and isinstance(value[0], (int, float))
        and isinstance(value[1], (int, float))
    ):
        return [(float(value[0]), float(value[1]))]

    points: list[tuple[float, float]] = []
    if isinstance(value, list):
        for child in value:
            points.extend(_flatten_coordinates(child))
    return points


def _feature_representative_point(feature: dict[str, Any]) -> tuple[float, float] | None:
    points = _flatten_coordinates((feature.get("geometry") or {}).get("coordinates", []))
    if not points:
        return None
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def _distance_meters(
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    longitude_a, latitude_a = first
    longitude_b, latitude_b = second
    radius = 6_371_000.0
    latitude_delta = math.radians(latitude_b - latitude_a)
    longitude_delta = math.radians(longitude_b - longitude_a)
    latitude_a_radians = math.radians(latitude_a)
    latitude_b_radians = math.radians(latitude_b)
    haversine = (
        math.sin(latitude_delta / 2) ** 2
        + math.cos(latitude_a_radians)
        * math.cos(latitude_b_radians)
        * math.sin(longitude_delta / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(haversine))


def _arrival_time(
    route_index: int,
    route_point_count: int,
    departure_datetime: datetime,
    route_duration_seconds: int,
) -> datetime:
    progress = route_index / max(1, route_point_count - 1)
    return datetime.fromtimestamp(
        departure_datetime.timestamp() + route_duration_seconds * progress,
        tz=departure_datetime.tzinfo,
    )


def _period(moment: datetime) -> str:
    hour = moment.hour
    if 7 <= hour < 18:
        return "daytime"
    if 18 <= hour < 22:
        return "evening"
    if 22 <= hour or hour < 1:
        return "night"
    return "late_night"


def _time_weight(time_context: str | None, moment: datetime) -> float:
    context = (time_context or "unspecified").lower()
    current = _period(moment)

    if context in {"unspecified", "daytime_and_unspecified"}:
        return 1.0
    if context == "weekend":
        return 1.0 if moment.weekday() >= 5 else 0.35
    if context == current:
        return 1.0
    if context == "night" and current == "late_night":
        return 1.0
    if context == "late_night" and current == "night":
        return 0.65
    if {context, current} == {"evening", "night"}:
        return 0.55
    return 0.25


def _best_location_match(
    record: dict[str, Any],
    route_geometry: list[tuple[float, float]],
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None

    for location in record.get("locations") or []:
        if location.get("type") != "circle":
            continue
        try:
            centre = (float(location["longitude"]), float(location["latitude"]))
            radius = float(location["radius_meters"])
        except (KeyError, TypeError, ValueError):
            continue
        if radius <= 0:
            continue

        nearest_index, nearest_point = min(
            enumerate(route_geometry),
            key=lambda item: _distance_meters(item[1], centre),
        )
        nearest_distance = _distance_meters(nearest_point, centre)
        if nearest_distance > radius:
            continue

        # Keep a non-zero contribution anywhere inside the stated evidence area.
        distance_weight = 0.15 + 0.85 * (1.0 - nearest_distance / radius)
        candidate = {
            "location": location,
            "route_point_index": nearest_index,
            "distance_meters": nearest_distance,
            "distance_weight": distance_weight,
        }
        if best is None or candidate["distance_weight"] > best["distance_weight"]:
            best = candidate

    return best


def _route_bbox(
    route_geometry: list[tuple[float, float]],
    padding_meters: float = 600.0,
) -> tuple[float, float, float, float]:
    """Returns (west, south, east, north) with a small buffer.

    The buffer must comfortably exceed the largest opinion-evidence radius in
    the dataset (currently a few hundred metres) so that a coarse pre-filter
    never discards a record a precise per-record match would have kept.
    """
    longitudes = [point[0] for point in route_geometry]
    latitudes = [point[1] for point in route_geometry]
    average_latitude = sum(latitudes) / len(latitudes)

    latitude_padding = padding_meters / 111_000
    longitude_padding = padding_meters / (
        111_000 * max(math.cos(math.radians(average_latitude)), 0.01)
    )

    return (
        min(longitudes) - longitude_padding,
        min(latitudes) - latitude_padding,
        max(longitudes) + longitude_padding,
        max(latitudes) + latitude_padding,
    )


def relevant_city_ids_for_route(
    route_geometry: list[tuple[float, float]],
) -> list[str]:
    """Returns which registered cities have any evidence near this route.

    This is informational (useful for debugging/telemetry) -- the actual
    matching in :func:`analyse_social_context` always checks every
    registered city's records directly by geometry, so a route is never
    silently excluded just because it wasn't assigned to the "right" city.
    """
    west, south, east, north = _route_bbox(route_geometry)
    relevant: list[str] = []

    for city_id, opinions in load_all_opinions().items():
        for record in opinions:
            for location in record.get("locations") or []:
                if location.get("type") != "circle":
                    continue
                try:
                    longitude = float(location["longitude"])
                    latitude = float(location["latitude"])
                except (KeyError, TypeError, ValueError):
                    continue
                if west <= longitude <= east and south <= latitude <= north:
                    relevant.append(city_id)
                    break
            else:
                continue
            break

    return sorted(set(relevant))


def analyse_social_context(
    *,
    route_geometry: list[tuple[float, float]],
    departure_datetime: datetime,
    route_duration_seconds: int,
    city: str | None = None,
) -> dict[str, Any]:
    """Return the opinion component for one route.

    Evidence is selected purely from the route's own coordinates: every
    registered municipality's opinion records are checked geometrically, and
    whichever ones actually intersect the route contribute. ``city`` is
    accepted for backward compatibility with older callers but is ignored --
    it is exactly the frontend-supplied value that caused Parque Ribalta
    evidence to be missed when a route was mis-tagged as the wrong city.

    Neighbourhood-level and unrecognised ("unknown") precision evidence is
    matched the same way (still gated by its own authored radius, still
    time-weighted) but its numeric ``effect_points`` is forced to zero: it
    can never move a route's score or turn a section red, and it never
    appears in ``matched_records_display_eligible``. It is still returned,
    both in ``matched_records`` (audit trail) and grouped by area in
    ``neighbourhood_context_records``, so the frontend can show it as plain
    background text in "Why this route?" -- never as if it were evidence
    about one specific street.

    The supplied route must be sampled closely enough for area matching; Meili's
    existing route geometries meet that requirement.
    """
    if not route_geometry:
        raise ValueError("route_geometry cannot be empty")
    if route_duration_seconds <= 0:
        raise ValueError("route_duration_seconds must be positive")

    matched: list[dict[str, Any]] = []
    eligible_count = 0
    checked_city_ids: list[str] = []

    for city_id, opinions in load_all_opinions().items():
        checked_city_ids.append(city_id)

        for record in opinions:
            if not record.get("rating_eligible", False):
                continue
            eligible_count += 1
            location_match = _best_location_match(record, route_geometry)
            if location_match is None:
                continue

            arrival = _arrival_time(
                location_match["route_point_index"],
                len(route_geometry),
                departure_datetime,
                route_duration_seconds,
            )
            time_weight = _time_weight(record.get("time_context"), arrival)
            base_effect = float(record.get("maximum_exact_time_effect_points") or 0.0)
            raw_effect = base_effect * location_match["distance_weight"] * time_weight

            precision = str(location_match["location"].get("precision") or "").lower()
            precision_class = location_precision_class(precision)
            is_mappable = precision_class in MAPPABLE_PRECISION_CLASSES

            # Neighbourhood-level and unrecognised evidence never receives a
            # direct numerical effect -- it can still be audited/shown as
            # text context below, but it must never move a score or a
            # section colour on its own.
            effect = raw_effect if is_mappable else 0.0

            # A neutral or malformed record can remain auditable without
            # changing score -- but only skip it entirely (don't even keep
            # it for text context) if it had no effect to begin with *and*
            # isn't mappable either, i.e. there's nothing useful to show.
            if effect == 0 and raw_effect == 0:
                continue

            match_strength = "specific" if is_mappable else "broad_context"

            matched.append({
                "city_id": city_id,
                "record_id": record.get("record_id"),
                "area": record.get("area"),
                "neighbourhood": record.get("neighbourhood"),
                "place": record.get("street_or_place"),
                "sentiment": record.get("overall_sentiment"),
                "evidence_tier": record.get("evidence_tier"),
                "summary": record.get("opinion_summary"),
                "source_platform": record.get("source_platform"),
                "source_url": record.get("source_url"),
                "post_date": record.get("post_date"),
                "confidence": record.get("confidence"),
                "arrival_time": arrival.isoformat(),
                "nearest_distance_meters": round(location_match["distance_meters"], 1),
                "distance_weight": round(location_match["distance_weight"], 3),
                "time_weight": round(time_weight, 3),
                "effect_points": round(effect, 3),
                "uncapped_raw_effect_points": round(raw_effect, 3),
                "precision": precision or None,
                "precision_class": precision_class,
                "match_strength": match_strength,
            })

    uncapped = sum(item["effect_points"] for item in matched)
    capped = max(-3.0, min(2.0, uncapped))
    display_eligible = sorted(
        (item for item in matched if item["match_strength"] == "specific"),
        key=lambda item: abs(item["effect_points"]),
        reverse=True,
    )
    negative = sorted(
        (item for item in matched if item["effect_points"] < 0),
        key=lambda item: item["effect_points"],
    )
    positive = sorted(
        (item for item in matched if item["effect_points"] > 0),
        key=lambda item: item["effect_points"],
        reverse=True,
    )

    # Neighbourhood-level context, grouped by area, for the "Why this
    # route?" panel. "unknown"-precision records are intentionally excluded
    # here (not just from the map): we can't honestly label context by
    # neighbourhood name for evidence whose own geography is ambiguous.
    neighbourhood_groups: dict[str, dict[str, Any]] = {}
    for item in matched:
        if item["precision_class"] != "neighbourhood":
            continue
        key = item.get("neighbourhood") or item.get("area") or "this area"
        group = neighbourhood_groups.setdefault(key, {
            "area": key,
            "records": [],
        })
        group["records"].append(item)

    neighbourhood_context_records = sorted(
        neighbourhood_groups.values(),
        key=lambda group: len(group["records"]),
        reverse=True,
    )

    return {
        "city_ids_checked": sorted(checked_city_ids),
        "requested_city": city,
        "score_adjustment_points": round(capped, 3),
        "uncapped_adjustment_points": round(uncapped, 3),
        "component_limits": {"minimum": -3.0, "maximum": 2.0},
        "rating_eligible_records_checked": eligible_count,
        "matched_record_count": len(matched),
        "matched_records_display_eligible": display_eligible,
        "strongest_negative_factors": negative[:3],
        "strongest_positive_factors": positive[:3],
        "matched_records": matched,
        "neighbourhood_context_records": neighbourhood_context_records,
        "method_note": (
            "Every geographically matching, rating-eligible opinion from every "
            "registered municipality contributes -- evidence selection is driven "
            "entirely by the route's own coordinates, never by a declared city. "
            "Its pre-audited maximum effect is reduced by distance and time "
            "mismatch. Neighbourhood-level and unclassified evidence is matched "
            "and audited the same way but never receives a direct numerical "
            "effect and is never drawn on the map -- it can only appear as "
            "plain-text neighbourhood context."
        ),
    }
