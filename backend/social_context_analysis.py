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


# ---------------------------------------------------------------------------
# Area influence model (v2)
#
# Each record describes an area as a circle (centre + radius). Influence is
# full in the core of the circle and fades smoothly to zero at its edge, and
# it is measured per route section, so a route that only grazes an area is
# barely affected and a route through its centre is affected most.
# ---------------------------------------------------------------------------

SECTION_COUNT = 10
CORE_FRACTION = 0.5            # inner half of the radius = full influence
MIN_ROUTE_INFLUENCE = 0.25     # below this the area is not shown at all
CITY_WIDE_RADIUS_METERS = 3000 # larger circles are city background, not areas
SECTION_CAP_NEGATIVE = -8.0
SECTION_CAP_POSITIVE = 4.0
MAX_POINTS_SPECIFIC = 10.0     # point / street evidence
MAX_POINTS_AREA = 5.0          # neighbourhood-level evidence
RECENCY_HALF_LIFE_MONTHS = 18.0
MAX_AREAS_SHOWN = 3

EVIDENCE_STRENGTH = {
    "confirmed_incident": 1.0,
    "municipal_data": 1.0,
    "resident_complaint": 0.7,
    "perception_only": 0.4,
}
SENTIMENT_SIGN = {"negative": -1.0, "positive": 1.0, "mixed": -0.3, "neutral": 0.0}


def area_weight(distance: float, radius: float) -> float:
    """1 in the core, smooth fade to 0 at the edge, 0 outside."""
    core = radius * CORE_FRACTION
    if distance <= core:
        return 1.0
    if distance >= radius:
        return 0.0
    return 0.5 * (1.0 + math.cos(math.pi * (distance - core) / (radius - core)))


def _densify(route: list[tuple[float, float]], step_m: float = 25.0) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for a, b in zip(route, route[1:]):
        d = _distance_meters(a, b)
        n = max(1, int(d / step_m))
        for i in range(n):
            out.append((a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n))
    out.append(route[-1])
    return out


def _recency_weight(post_date: str | None, now: datetime) -> float:
    if not post_date:
        return 0.6
    try:
        posted = datetime.fromisoformat(str(post_date)[:10])
    except ValueError:
        return 0.6
    months = max(0.0, (now.replace(tzinfo=None) - posted).days / 30.4)
    return 0.5 ** (months / RECENCY_HALF_LIFE_MONTHS)


def _record_strength(record: dict[str, Any], now: datetime) -> float:
    tier = EVIDENCE_STRENGTH.get(str(record.get("evidence_tier") or ""), 0.3)
    try:
        confidence = max(1.0, min(3.0, float(record.get("confidence") or 2))) / 3.0
    except (TypeError, ValueError):
        confidence = 2 / 3
    # Automatically gathered records count less until reviewed or confirmed
    # by user ratings (verification_weight 0.7 -> up to 1.0).
    try:
        verification = max(0.0, min(1.0, float(record.get("verification_weight", 1.0))))
    except (TypeError, ValueError):
        verification = 1.0
    return tier * confidence * _recency_weight(record.get("post_date"), now) * verification


def _section_influence(record, samples) -> tuple[list[float], dict[str, Any] | None, int]:
    """Per-section influence of the record's best circle, plus that circle and
    the index of the route sample closest to its centre."""
    best = None
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
        weights = [area_weight(_distance_meters(p, centre), radius) for p in samples]
        n = len(samples)
        sections = []
        for k in range(SECTION_COUNT):
            chunk = weights[k * n // SECTION_COUNT:max(k * n // SECTION_COUNT + 1, (k + 1) * n // SECTION_COUNT)]
            sections.append(sum(chunk) / len(chunk) if chunk else 0.0)
        if best is None or max(sections) > max(best[0]):
            closest = max(range(n), key=lambda i: weights[i])
            best = (sections, location, closest)
    if best is None:
        return [0.0] * SECTION_COUNT, None, 0
    return best


def _area_key(record: dict[str, Any]) -> str:
    name = str(record.get("neighbourhood") or record.get("area") or "this area")
    import re
    first = re.split(r"\s*(?:/|\(|,|\s-\s|–)\s*", name)[0].strip()
    import unicodedata
    plain = "".join(c for c in unicodedata.normalize("NFD", first) if unicodedata.category(c) != "Mn")
    return plain.lower()


def _display_name(records: list[dict[str, Any]]) -> str:
    names = [str(r.get("area") or r.get("neighbourhood") or "this area").split("/")[0].strip() for r in records]
    return min(names, key=len) if names else "this area"


def load_area_summaries(city_id: str) -> dict[str, Any]:
    """AI-written one-line summaries per area (generated offline, optional)."""
    path = DATA_ROOT / city_id / "area_summaries.json"
    try:
        return _read_json(path) if path.exists() else {}
    except Exception:  # noqa: BLE001
        return {}


def analyse_social_context(
    *,
    route_geometry: list[tuple[float, float]],
    departure_datetime: datetime,
    route_duration_seconds: int,
    city: str | None = None,
    extra_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Opinion / incident component for one route (area influence model v2).

    * Influence is centre-weighted and per section (see ``area_weight``).
    * Records are grouped per area; at most ``MAX_AREAS_SHOWN`` areas are
      returned in ``area_context`` (most influential first), plus at most one
      city-wide background line in ``city_context``.
    * Every record can move the score of the sections it actually covers, in
      proportion to evidence strength, confidence, recency, time of day and
      influence. Per-section totals are capped at -8 / +4 points.
    """
    if not route_geometry:
        raise ValueError("route_geometry cannot be empty")
    if route_duration_seconds <= 0:
        raise ValueError("route_duration_seconds must be positive")

    samples = _densify(route_geometry) if len(route_geometry) > 1 else list(route_geometry)
    now = departure_datetime
    matched: list[dict[str, Any]] = []
    city_background: list[dict[str, Any]] = []
    eligible_count = 0
    checked_city_ids: list[str] = []
    section_totals = [0.0] * SECTION_COUNT
    section_ids: list[list[str]] = [[] for _ in range(SECTION_COUNT)]
    summaries_by_city: dict[str, dict[str, Any]] = {}

    sources = dict(load_all_opinions())
    if extra_records:
        sources["auto"] = extra_records   # automatically gathered (Supabase)
    for city_id, opinions in sources.items():
        checked_city_ids.append(city_id)
        for record in opinions:
            if not record.get("rating_eligible", False):
                continue
            eligible_count += 1
            sections, location, closest = _section_influence(record, samples)
            influence = max(sections)
            if location is None or influence < MIN_ROUTE_INFLUENCE:
                continue

            arrival = _arrival_time(closest, len(samples), departure_datetime, route_duration_seconds)
            time_weight = _time_weight(record.get("time_context"), arrival)
            sentiment = str(record.get("overall_sentiment") or "neutral").lower()
            precision = str(location.get("precision") or "").lower()
            precision_class = location_precision_class(precision)
            radius = float(location.get("radius_meters") or 0)

            entry = {
                "city_id": city_id,
                "record_id": record.get("record_id"),
                "area": record.get("area"),
                "neighbourhood": record.get("neighbourhood"),
                "place": record.get("street_or_place"),
                "sentiment": sentiment,
                "evidence_tier": record.get("evidence_tier"),
                "summary": record.get("opinion_summary"),
                "source_platform": record.get("source_platform"),
                "source_url": record.get("source_url"),
                "post_date": record.get("post_date"),
                "confidence": record.get("confidence"),
                "arrival_time": arrival.isoformat(),
                "influence": round(influence, 3),
                "time_weight": round(time_weight, 3),
                "precision": precision or None,
                "precision_class": precision_class,
                "origin": record.get("origin") or "reviewed",
            }

            # Whole-city records (e.g. a city-wide survey) are background only.
            # Decided by what the record covers, not by circle size alone: a big
            # district like Marjaleria is still an area.
            if record.get("coverage") == "city" or radius >= 5000:
                entry["effect_points"] = 0.0
                entry["match_strength"] = "city_background"
                city_background.append(entry)
                continue

            specific = precision_class in MAPPABLE_PRECISION_CLASSES
            max_points = MAX_POINTS_SPECIFIC if specific else MAX_POINTS_AREA
            strength = _record_strength(record, now) * time_weight
            sign = SENTIMENT_SIGN.get(sentiment, 0.0)
            per_section = [sign * max_points * strength * w for w in sections]
            for k, effect in enumerate(per_section):
                if abs(effect) >= 0.05:
                    section_totals[k] += effect
                    if entry["record_id"]:
                        section_ids[k].append(str(entry["record_id"]))

            entry["section_effects"] = [round(e, 2) for e in per_section]
            entry["effect_points"] = round(sum(per_section) / SECTION_COUNT, 3)
            entry["match_strength"] = "specific" if specific else "area"
            matched.append(entry)

    section_adjustments = [
        round(max(SECTION_CAP_NEGATIVE, min(SECTION_CAP_POSITIVE, t)), 2) for t in section_totals
    ]

    # ---- one entry per area -------------------------------------------------
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in matched:
        groups.setdefault(_area_key(item), []).append(item)

    area_context = []
    for key, records in groups.items():
        records.sort(key=lambda r: (r["influence"], abs(r["effect_points"])), reverse=True)
        concerns = sum(1 for r in records if r["sentiment"] in ("negative", "mixed"))
        positives = sum(1 for r in records if r["sentiment"] == "positive")
        net = sum(SENTIMENT_SIGN.get(r["sentiment"], 0.0) * r["influence"] for r in records)
        verdict = (
            "concerns" if net < -0.15 and positives == 0
            else "positive" if net > 0.15 and concerns == 0
            else "mixed"
        )
        influence = max(r["influence"] for r in records)
        city_id = records[0]["city_id"]
        if city_id not in summaries_by_city:
            summaries_by_city[city_id] = load_area_summaries(city_id)
        ai = summaries_by_city[city_id].get(key) or {}
        area_context.append({
            "area": _display_name(records),
            "area_key": key,
            "verdict": verdict,
            "concern_count": concerns,
            "positive_count": positives,
            "influence": round(influence, 2),
            "influence_label": "strong" if influence >= 0.7 else "some" if influence >= 0.45 else "slight",
            "score_effect_points": round(sum(r["effect_points"] for r in records), 2),
            "summary": ai.get("summary") or records[0]["summary"],
            "summary_source": "ai_from_records" if ai.get("summary") else "top_record",
            "auto_gathered": all(r.get("origin") == "auto" for r in records),
            "sources": [
                {k: r.get(k) for k in ("record_id", "summary", "source_platform", "source_url", "post_date", "sentiment", "evidence_tier", "origin")}
                for r in records
            ],
        })
    area_context.sort(key=lambda g: (g["influence"], abs(g["score_effect_points"])), reverse=True)

    city_context = None
    if city_background:
        top = max(city_background, key=lambda r: (r.get("confidence") or 0))
        city_context = {k: top.get(k) for k in ("area", "summary", "sentiment", "source_platform", "source_url", "post_date")}

    total = sum(section_adjustments) / SECTION_COUNT
    return {
        "model_version": "area_influence_v2",
        "city_ids_checked": sorted(checked_city_ids),
        "requested_city": city,
        "score_adjustment_points": round(total, 3),
        "section_adjustments": section_adjustments,
        "section_record_ids": section_ids,
        "component_limits": {"section_minimum": SECTION_CAP_NEGATIVE, "section_maximum": SECTION_CAP_POSITIVE},
        "rating_eligible_records_checked": eligible_count,
        "matched_record_count": len(matched),
        "matched_records": matched,
        "area_context": area_context[:MAX_AREAS_SHOWN],
        "more_areas_count": max(0, len(area_context) - MAX_AREAS_SHOWN),
        "city_context": city_context,
        # Backward compatibility for older app versions:
        "matched_records_display_eligible": [m for m in matched if m["match_strength"] == "specific"],
        "neighbourhood_context_records": [
            {"area": g["area"], "records": g["sources"]} for g in area_context[:MAX_AREAS_SHOWN]
        ],
        "method_note": (
            "Areas count fully in their core and fade to nothing at their edge, "
            "measured per route section; routes that only graze an area are not "
            "affected. Each record's effect depends on evidence type, confidence, "
            "recency and time of day, and is capped at -8/+4 points per section."
        ),
    }
