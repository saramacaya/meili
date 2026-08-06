"""Open-establishment analysis for Meili walking routes."""

from datetime import datetime, timedelta
import math
import time
import threading
import unicodedata
from typing import Optional
from zoneinfo import ZoneInfo

import requests
from fastapi import APIRouter, HTTPException
from opening_hours import OpeningHours
from pydantic import BaseModel, Field


router = APIRouter()
VALENCIA_TIMEZONE = ZoneInfo("Europe/Madrid")
OVERPASS_API_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

class ActivePlacesAnalysisRequest(BaseModel):
    route_geometry: list[tuple[float, float]]
    evaluation_datetime: datetime

    route_duration_seconds: int = Field(
        ...,
        ge=60,
        le=21_600
    )

    arrival_uncertainty_ratio: float = Field(
        default=0.10,
        ge=0,
        le=0.50
    )

    search_radius_meters: float = Field(
        default=75,
        ge=25,
        le=200
    )

def distance_meters(
    point_a: tuple[float, float],
    point_b: tuple[float, float]
) -> float:
    longitude_a, latitude_a = point_a
    longitude_b, latitude_b = point_b
    earth_radius_meters = 6_371_000

    latitude_a_radians = math.radians(latitude_a)
    latitude_b_radians = math.radians(latitude_b)
    latitude_delta = math.radians(latitude_b - latitude_a)
    longitude_delta = math.radians(longitude_b - longitude_a)

    haversine = (
        math.sin(latitude_delta / 2) ** 2
        + math.cos(latitude_a_radians)
        * math.cos(latitude_b_radians)
        * math.sin(longitude_delta / 2) ** 2
    )

    return (
        2
        * earth_radius_meters
        * math.asin(math.sqrt(haversine))
    )


def densify_route(
    coordinates: list[tuple[float, float]],
    interval_meters: float = 20
) -> list[tuple[float, float]]:
    if len(coordinates) < 2:
        return coordinates

    samples = [coordinates[0]]

    for start, end in zip(coordinates, coordinates[1:]):
        segment_length = distance_meters(start, end)
        steps = max(
            1,
            math.ceil(segment_length / interval_meters)
        )

        for step in range(1, steps + 1):
            fraction = step / steps

            samples.append((
                start[0] + (end[0] - start[0]) * fraction,
                start[1] + (end[1] - start[1]) * fraction
            ))

    return samples

def calculate_place_route_progress(
    place_coordinate: tuple[float, float],
    route_coordinates: list[tuple[float, float]]
) -> float:
    """
    Returns the approximate position of a place along the route.

    Examples:
    0.0 = route beginning
    0.5 = halfway
    1.0 = route end
    """
    route_samples = densify_route(
        route_coordinates,
        interval_meters=10
    )

    if len(route_samples) < 2:
        return 0.0

    cumulative_distances = [0.0]

    for previous, current in zip(
        route_samples,
        route_samples[1:]
    ):
        cumulative_distances.append(
            cumulative_distances[-1]
            + distance_meters(previous, current)
        )

    total_distance = cumulative_distances[-1]

    if total_distance <= 0:
        return 0.0

    nearest_sample_index = min(
        range(len(route_samples)),
        key=lambda index: distance_meters(
            place_coordinate,
            route_samples[index]
        )
    )

    progress = (
        cumulative_distances[nearest_sample_index]
        / total_distance
    )

    return max(0.0, min(1.0, progress))

def calculate_place_arrival_time(
    route_progress: float,
    departure_datetime: datetime,
    route_duration_seconds: int
) -> datetime:
    elapsed_seconds = (
        route_duration_seconds
        * route_progress
    )

    return (
        departure_datetime
        + timedelta(seconds=elapsed_seconds)
    )


def flatten_geojson_coordinates(
    value
) -> list[tuple[float, float]]:
    if (
        isinstance(value, list)
        and len(value) >= 2
        and isinstance(value[0], (int, float))
        and isinstance(value[1], (int, float))
    ):
        return [(float(value[0]), float(value[1]))]

    points = []

    if isinstance(value, list):
        for child in value:
            points.extend(
                flatten_geojson_coordinates(child)
            )

    return points
def normalise_text(value: str) -> str:
    """
    Makes text easier to match.

    Example:
    'Plaça de la Reina' becomes 'placa de la reina'
    """
    value = value.lower().strip()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = " ".join(value.split())
    return value

ACTIVE_PLACE_TAGS = {
    "cafe",
    "restaurant",
    "fast_food",
    "convenience",
    "supermarket",
    "hotel",
    "hostel",
    "guest_house"
}

HELP_POINT_TAGS = {
    "pharmacy",
    "hospital",
    "clinic",
    "doctors",
    "police",
    "fire_station"
}

NIGHTLIFE_TAGS = {
    "bar",
    "pub",
    "nightclub"
}


def classify_place(tags: dict) -> Optional[str]:
    """
    Classifies an OpenStreetMap place for the safety analysis.

    Returns:
    - 'active_place'
    - 'help_point'
    - 'nightlife'
    - None when the place is not relevant
    """
    possible_values = {
        normalise_text(str(tags.get("amenity", ""))),
        normalise_text(str(tags.get("shop", ""))),
        normalise_text(str(tags.get("tourism", "")))
    }

    if possible_values & HELP_POINT_TAGS:
        return "help_point"

    if possible_values & ACTIVE_PLACE_TAGS:
        return "active_place"

    if possible_values & NIGHTLIFE_TAGS:
        return "nightlife"

    return None

def get_establishment_type(tags: dict) -> Optional[str]:
    """
    Returns the specific OpenStreetMap establishment type.

    Examples:
    - restaurant
    - cafe
    - fast_food
    - pharmacy
    - hotel
    - bar
    """
    recognised_types = (
        ACTIVE_PLACE_TAGS
        | HELP_POINT_TAGS
        | NIGHTLIFE_TAGS
    )

    for key in ("amenity", "shop", "tourism"):
        value = normalise_text(
            str(tags.get(key, ""))
        )

        if value in recognised_types:
            return value

    return None

def estimate_opening_status(
    establishment_type: Optional[str],
    evaluation_datetime: datetime
) -> str:
    """
    Estimates whether a place is likely to be open when
    OpenStreetMap does not provide opening hours.

    Estimated statuses must not be treated as confirmed.
    """
    if not establishment_type:
        return "unknown"

    hour = evaluation_datetime.hour
    weekday = evaluation_datetime.weekday()
    is_weekend = weekday >= 5

    if establishment_type in {
        "hotel",
        "hostel",
        "guest_house",
        "hospital",
        "police",
        "fire_station"
    }:
        return "estimated_open"

    if establishment_type == "cafe":
        if 7 <= hour < 21:
            return "estimated_open"

        return "estimated_closed"

    if establishment_type == "restaurant":
        if (
            12 <= hour < 16
            or 19 <= hour < 24
        ):
            return "estimated_open"

        return "estimated_closed"

    if establishment_type == "fast_food":
        if hour >= 10 or hour < 1:
            return "estimated_open"

        return "estimated_closed"

    if establishment_type in {
        "convenience",
        "supermarket"
    }:
        if 9 <= hour < 22:
            return "estimated_open"

        return "estimated_closed"

    if establishment_type == "pharmacy":
        if 9 <= hour < 21:
            return "estimated_open"

        return "unknown"

    if establishment_type in {
        "clinic",
        "doctors"
    }:
        if weekday < 5 and 9 <= hour < 20:
            return "estimated_open"

        return "estimated_closed"

    if establishment_type in {
        "bar",
        "pub"
    }:
        estimated_closing_hour = (
            2 if is_weekend else 1
        )

        if (
            hour >= 18
            or hour < estimated_closing_hour
        ):
            return "estimated_open"

        return "estimated_closed"

    if establishment_type == "nightclub":
        if hour >= 23 or hour < 6:
            return "estimated_open"

        return "estimated_closed"

    return "unknown"

def determine_place_opening_status(
    opening_hours_value: Optional[str],
    evaluation_datetime: datetime,
    latitude: float,
    longitude: float,
    establishment_type: Optional[str]
) -> str:
    """
    Returns:
    - confirmed_open
    - confirmed_closed
    - estimated_open
    - estimated_closed
    - unknown
    """

    # Only estimate when OSM genuinely has no opening-hours data.
    if not opening_hours_value:
        return estimate_opening_status(
            establishment_type=establishment_type,
            evaluation_datetime=evaluation_datetime
        )

    try:
        local_datetime = evaluation_datetime.astimezone(
            VALENCIA_TIMEZONE
        )

        opening_hours = OpeningHours(
            opening_hours_value,
            timezone=VALENCIA_TIMEZONE
        )

        if opening_hours.is_open(local_datetime):
            return "confirmed_open"

        if opening_hours.is_closed(local_datetime):
            return "confirmed_closed"

        return "unknown"

    except Exception as error:
        print(
            "Opening-hours parser failed:",
            repr(opening_hours_value),
            type(error).__name__,
            repr(str(error))
        )

        # Do not estimate here.
        # An existing but unparseable value is different
        # from having no opening-hours data.
        return "unknown"

def determine_place_activity_status(
    opening_status: str,
    opening_hours_value: Optional[str],
    evaluation_datetime: datetime,
    transition_window_minutes: int = 30
) -> str:
    """
    Converts factual opening status into expected street activity.

    Returns:
    - open_activity
    - opening_activity
    - closing_activity
    - estimated_activity
    - closed_activity
    - unknown_activity
    """

    if opening_status == "confirmed_open":
        return "open_activity"

    if opening_status == "estimated_open":
        return "estimated_activity"

    if opening_status == "estimated_closed":
        return "closed_activity"

    if opening_status == "unknown":
        return "unknown_activity"

    if (
        opening_status != "confirmed_closed"
        or not opening_hours_value
    ):
        return "closed_activity"

    try:
        local_datetime = evaluation_datetime.astimezone(
            VALENCIA_TIMEZONE
        )

        opening_hours = OpeningHours(
            opening_hours_value,
            timezone=VALENCIA_TIMEZONE
        )

        # Check whether the establishment was open recently.
        # This captures people leaving, cleaning and closing.
        for minutes_before in range(
            1,
            transition_window_minutes + 1
        ):
            earlier_datetime = (
                local_datetime
                - timedelta(minutes=minutes_before)
            )

            if opening_hours.is_open(earlier_datetime):
                return "closing_activity"

        # Check whether the establishment will open soon.
        # This captures staff arrival, preparation and deliveries.
        for minutes_after in range(
            1,
            transition_window_minutes + 1
        ):
            later_datetime = (
                local_datetime
                + timedelta(minutes=minutes_after)
            )

            if opening_hours.is_open(later_datetime):
                return "opening_activity"

        return "closed_activity"

    except Exception as error:
        print(
            "Transition-activity evaluation failed:",
            repr(opening_hours_value),
            type(error).__name__,
            repr(str(error))
        )

        return "closed_activity"

ACTIVITY_STATUS_WEIGHTS = {
    "open_activity": 1.00,
    "closing_activity": 0.60,
    "opening_activity": 0.40,
    "estimated_activity": 0.30,
    "unknown_activity": 0.05,
    "closed_activity": 0.00
}


ACTIVITY_CATEGORY_WEIGHTS = {
    "active_place": 1.00,
    "help_point": 1.20,
    "nightlife": 0.40
}


# How much weighted activity a segment needs to reach a 100 score.
#
# Recalibrated (Priority 5): the previous value of 3.0 meant a segment
# needed roughly three simultaneously-open, immediately-adjacent places to
# reach 100. In the first real run, routes with 32-33 confirmed/estimated
# active places along the whole walk (spread across 10 segments, so ~3-4
# per segment, each rarely both fully "open_activity" AND within a few
# metres of the route) still scored a flat 30/100 -- the target was too
# strict for how sparse and imperfect real OSM opening-hours data actually
# is. 1.6 means two moderately-confident, reasonably-close active places
# (or one help point) are enough to call a segment fully active, which
# matches what the feedback described as "roughly 32-33 confirmed or
# estimated active" places no longer being undervalued.
SEGMENT_ACTIVITY_TARGET_POINTS = 1.6


def calculate_place_activity_contribution(
    place: dict,
    search_radius_meters: float
) -> float:
    """
    Calculates how much one place contributes to route activity.

    Contribution depends on:
    - activity confidence/status
    - establishment category
    - distance from the route
    """
    if search_radius_meters <= 0:
        return 0.0

    status_weight = ACTIVITY_STATUS_WEIGHTS.get(
        place.get("activity_status"),
        0.0
    )

    category_weight = ACTIVITY_CATEGORY_WEIGHTS.get(
        place.get("category"),
        0.0
    )

    distance_to_route = float(
        place.get(
            "distance_to_route_meters",
            search_radius_meters
        )
    )

    distance_weight = max(
        0.0,
        1.0 - (
            distance_to_route
            / search_radius_meters
        )
    )

    contribution = (
        status_weight
        * category_weight
        * distance_weight
    )

    return round(contribution, 3)


def calculate_route_activity_analysis(
    places: list[dict],
    segment_count: int = 10
) -> dict:
    """
    Divides the route into equal-distance segments and
    calculates an activity score for every segment.

    The final route score is the average segment score.
    """
    if segment_count <= 0:
        segment_count = 10

    segments = [
        {
            "segment_number": index + 1,
            "start_percentage": round(
                index * 100 / segment_count,
                1
            ),
            "end_percentage": round(
                (index + 1) * 100 / segment_count,
                1
            ),
            "raw_activity_points": 0.0,
            "contributing_place_count": 0
        }
        for index in range(segment_count)
    ]

    for place in places:
        route_progress_percentage = float(
            place.get(
                "route_progress_percentage",
                0.0
            )
        )

        route_progress_percentage = max(
            0.0,
            min(100.0, route_progress_percentage)
        )

        segment_index = min(
            segment_count - 1,
            int(
                route_progress_percentage
                / 100
                * segment_count
            )
        )

        contribution = float(
            place.get(
                "activity_contribution",
                0.0
            )
        )

        segments[segment_index][
            "raw_activity_points"
        ] += contribution

        if contribution > 0:
            segments[segment_index][
                "contributing_place_count"
            ] += 1

    for segment in segments:
        raw_points = segment[
            "raw_activity_points"
        ]

        segment["raw_activity_points"] = round(
            raw_points,
            3
        )

        segment["activity_score"] = round(
            100
            * min(
                1.0,
                raw_points
                / SEGMENT_ACTIVITY_TARGET_POINTS
            )
        )

    segment_scores = [
        segment["activity_score"]
        for segment in segments
    ]

    route_activity_score = round(
        sum(segment_scores)
        / len(segment_scores)
    )

    return {
        "route_activity_score": (
            route_activity_score
        ),
        "minimum_segment_activity_score": min(
            segment_scores
        ),
        "low_activity_segment_count": sum(
            score < 30
            for score in segment_scores
        ),
        "segment_count": segment_count,
        "segments": segments
    }


def _overpass_places_query(bbox: str) -> str:
    return f"""
    [out:json][timeout:25];
    (
      nwr["amenity"~"^(cafe|restaurant|fast_food|pharmacy|hospital|clinic|doctors|police|fire_station|bar|pub|nightclub)$"]({bbox});
      nwr["shop"~"^(convenience|supermarket)$"]({bbox});
      nwr["tourism"~"^(hotel|hostel|guest_house)$"]({bbox});
    );
    out tags center;
    """


def _route_padded_bbox(
    route_coordinates: list[tuple[float, float]],
    search_radius_meters: float
) -> str:
    longitudes = [coordinate[0] for coordinate in route_coordinates]
    latitudes = [coordinate[1] for coordinate in route_coordinates]

    latitude_padding = search_radius_meters / 111_000
    average_latitude = sum(latitudes) / len(latitudes)
    longitude_padding = search_radius_meters / (
        111_000 * max(math.cos(math.radians(average_latitude)), 0.01)
    )

    south = min(latitudes) - latitude_padding
    west = min(longitudes) - longitude_padding
    north = max(latitudes) + latitude_padding
    east = max(longitudes) + longitude_padding

    return f"{south},{west},{north},{east}"


def _run_overpass_query(query: str, timeout_seconds: int = 35) -> dict:
    data = None
    retrieval_errors = []

    for overpass_url in OVERPASS_API_URLS:
        try:
            response = requests.post(
                overpass_url,
                data={"data": query},
                headers={
                    "Accept": "application/json",
                    "User-Agent": "Meili safety-routing prototype"
                },
                timeout=timeout_seconds
            )
            response.raise_for_status()
            data = response.json()
            break
        except (requests.RequestException, ValueError) as error:
            retrieval_errors.append(f"{overpass_url}: {error}")

    if data is None:
        raise HTTPException(
            status_code=502,
            detail=(
                "OpenStreetMap place data could not be retrieved from the "
                "available Overpass servers: " + " | ".join(retrieval_errors)
            )
        )

    return data


def _place_from_element(element: dict) -> Optional[dict]:
    tags = element.get("tags", {})
    category = classify_place(tags)

    if category is None:
        return None

    latitude = element.get("lat")
    longitude = element.get("lon")

    if latitude is None or longitude is None:
        center = element.get("center", {})
        latitude = center.get("lat")
        longitude = center.get("lon")

    if latitude is None or longitude is None:
        return None

    return {
        "osm_type": element.get("type"),
        "osm_id": element.get("id"),
        "establishment_type": get_establishment_type(tags),
        "name": tags.get("name") or tags.get("brand") or "Unnamed place",
        "category": category,
        "latitude": float(latitude),
        "longitude": float(longitude),
        "opening_hours": tags.get("opening_hours")
    }


SHARED_PLACES_CACHE: dict[str, dict] = {}
SHARED_PLACES_CACHE_LOCK = threading.Lock()
SHARED_PLACES_CACHE_TTL_SECONDS = 15 * 60


def fetch_shared_osm_places(
    route_geometries: list[list[tuple[float, float]]],
    search_radius_meters: float
) -> tuple[list[dict], dict]:
    """
    Downloads relevant OpenStreetMap places once for a bounding box that
    covers every route being compared, instead of one Overpass request per
    route (Priority 3/4: routes being compared together should not pay for
    a separate slow external download each).

    The raw elements are cached in memory for 15 minutes, keyed by the
    rounded bounding box, so repeat comparisons over the same area of the
    Comunitat Valenciana reuse the same download.
    """
    all_coordinates = [
        coordinate
        for geometry in route_geometries
        for coordinate in geometry
    ]

    if not all_coordinates:
        return [], {
            "cache_hit": False,
            "raw_places_found": 0
        }

    bbox = _route_padded_bbox(all_coordinates, search_radius_meters)
    cache_key = bbox

    now = time.time()

    with SHARED_PLACES_CACHE_LOCK:
        cached_entry = SHARED_PLACES_CACHE.get(cache_key)
        if cached_entry is not None and (now - cached_entry["created_at"]) < SHARED_PLACES_CACHE_TTL_SECONDS:
            elements = cached_entry["elements"]
            return elements, {"cache_hit": True, "raw_places_found": len(elements)}

    data = _run_overpass_query(_overpass_places_query(bbox), timeout_seconds=35)
    elements = data.get("elements", [])

    with SHARED_PLACES_CACHE_LOCK:
        SHARED_PLACES_CACHE[cache_key] = {
            "created_at": now,
            "elements": elements
        }

    return elements, {"cache_hit": False, "raw_places_found": len(elements)}


def filter_shared_places_for_route(
    route_coordinates: list[tuple[float, float]],
    shared_elements: list[dict],
    search_radius_meters: float
) -> tuple[list[dict], dict]:
    """
    Keeps only shared OSM places close to one particular route.

    This performs no network request -- it is the per-route counterpart to
    :func:`fetch_shared_osm_places`, mirroring the existing
    fetch_shared_osm_lighting_data / filter_shared_osm_*_for_route pattern
    already used for lighting comparisons.
    """
    if not route_coordinates:
        return [], {
            "raw_places_found": len(shared_elements),
            "relevant_places_near_route": 0
        }

    route_samples = densify_route(route_coordinates, interval_meters=30)
    relevant_places = []

    for element in shared_elements:
        place = _place_from_element(element)
        if place is None:
            continue

        place_coordinate = (place["longitude"], place["latitude"])
        nearest_route_distance = min(
            distance_meters(place_coordinate, route_sample)
            for route_sample in route_samples
        )

        if nearest_route_distance > search_radius_meters:
            continue

        place["distance_to_route_meters"] = round(nearest_route_distance, 1)
        relevant_places.append(place)

    retrieval_debug = {
        "raw_places_found": len(shared_elements),
        "relevant_places_near_route": len(relevant_places),
        "places_with_opening_hours": sum(
            1 for place in relevant_places if place["opening_hours"]
        )
    }

    return relevant_places, retrieval_debug


def fetch_osm_places_near_route(
    route_coordinates: list[tuple[float, float]],
    search_radius_meters: float
) -> tuple[list[dict], dict]:
    """
    Retrieves relevant places from OpenStreetMap and keeps
    only places within the requested distance of the route.

    Kept for the single-route endpoint; batch comparisons should prefer
    fetch_shared_osm_places + filter_shared_places_for_route so multiple
    routes share one download.
    """
    if not route_coordinates:
        return [], {
            "raw_places_found": 0,
            "relevant_places_near_route": 0
        }

    bbox = _route_padded_bbox(route_coordinates, search_radius_meters)
    data = _run_overpass_query(_overpass_places_query(bbox), timeout_seconds=35)

    return filter_shared_places_for_route(
        route_coordinates=route_coordinates,
        shared_elements=data.get("elements", []),
        search_radius_meters=search_radius_meters
    )


def score_places_for_route(
    places: list[dict],
    route_geometry: list[tuple[float, float]],
    evaluation_datetime: datetime,
    route_duration_seconds: int,
    search_radius_meters: float
) -> list[dict]:
    """
    Adds arrival time, opening status and activity contribution to each
    place. Shared between the single-route and batch-comparison code paths
    so the underlying scoring logic never diverges between them.
    """
    scored_places = []

    for place in places:
        place = dict(place)
        place_coordinate = (place["longitude"], place["latitude"])

        route_progress = calculate_place_route_progress(
            place_coordinate=place_coordinate,
            route_coordinates=route_geometry
        )

        estimated_arrival_datetime = calculate_place_arrival_time(
            route_progress=route_progress,
            departure_datetime=evaluation_datetime,
            route_duration_seconds=route_duration_seconds
        )

        place["route_progress_percentage"] = round(route_progress * 100, 1)
        place["estimated_arrival_datetime"] = estimated_arrival_datetime.isoformat()

        place["opening_status"] = determine_place_opening_status(
            opening_hours_value=place["opening_hours"],
            evaluation_datetime=estimated_arrival_datetime,
            latitude=place["latitude"],
            longitude=place["longitude"],
            establishment_type=place["establishment_type"]
        )

        place["activity_status"] = determine_place_activity_status(
            opening_status=place["opening_status"],
            opening_hours_value=place["opening_hours"],
            evaluation_datetime=estimated_arrival_datetime,
            transition_window_minutes=30
        )

        place["activity_contribution"] = calculate_place_activity_contribution(
            place=place,
            search_radius_meters=search_radius_meters
        )

        scored_places.append(place)

    return scored_places


def build_active_places_response(places: list[dict], evaluation_datetime: datetime) -> dict:
    """Shared response-shaping logic for both the single-route endpoint and
    the batch comparison endpoint, so their output stays identical in shape."""
    activity_analysis = calculate_route_activity_analysis(
        places=places,
        segment_count=10
    )

    def _of_category(category):
        return [place for place in places if place["category"] == category]

    def _of_opening_status(status):
        return [place for place in places if place["opening_status"] == status]

    def _of_activity_status(status):
        return [place for place in places if place["activity_status"] == status]

    active_places = _of_category("active_place")
    help_points = _of_category("help_point")
    nightlife_places = _of_category("nightlife")

    return {
        "status": "active_places_analysed",
        "opening_hours_logic_version": "v3_explicit_timezone",
        "source": "OpenStreetMap contributors",
        "source_license": "ODbL",
        "evaluation_datetime": evaluation_datetime.isoformat(),
        "route_activity_score": activity_analysis["route_activity_score"],
        "minimum_segment_activity_score": activity_analysis["minimum_segment_activity_score"],
        "low_activity_segment_count": activity_analysis["low_activity_segment_count"],
        "activity_segments": activity_analysis["segments"],
        "total_relevant_places": len(places),
        "active_place_count": len(active_places),
        "help_point_count": len(help_points),
        "nightlife_count": len(nightlife_places),
        "places_with_opening_hours_count": sum(1 for place in places if place["opening_hours"]),
        "places_with_unknown_hours_count": sum(1 for place in places if not place["opening_hours"]),
        "confirmed_open_count": len(_of_opening_status("confirmed_open")),
        "confirmed_closed_count": len(_of_opening_status("confirmed_closed")),
        "estimated_open_count": len(_of_opening_status("estimated_open")),
        "estimated_closed_count": len(_of_opening_status("estimated_closed")),
        "unknown_opening_status_count": len(_of_opening_status("unknown")),
        "transition_activity_window_minutes": 30,
        "open_activity_count": len(_of_activity_status("open_activity")),
        "closing_activity_count": len(_of_activity_status("closing_activity")),
        "opening_activity_count": len(_of_activity_status("opening_activity")),
        "estimated_activity_count": len(_of_activity_status("estimated_activity")),
        "closed_activity_count": len(_of_activity_status("closed_activity")),
        "unknown_activity_count": len(_of_activity_status("unknown_activity")),
        "active_places": active_places,
        "help_points": help_points,
        "nightlife_places": nightlife_places,
        "data_confidence": "testing",
        "interpretation": (
            "This checks the availability of relevant OpenStreetMap places and "
            "their opening-hours data near the route. Missing opening hours "
            "mean unknown, not closed."
        )
    }


@router.post("/safety/active-places/analyse")
def analyse_active_places(
    request: ActivePlacesAnalysisRequest
):
    if len(request.route_geometry) < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                "At least two route coordinates "
                "are required."
            )
        )

    places, retrieval_debug = (
        fetch_osm_places_near_route(
            route_coordinates=request.route_geometry,
            search_radius_meters=(
                request.search_radius_meters
            )
        )
    )

    scored_places = score_places_for_route(
        places=places,
        route_geometry=request.route_geometry,
        evaluation_datetime=request.evaluation_datetime,
        route_duration_seconds=request.route_duration_seconds,
        search_radius_meters=request.search_radius_meters
    )

    response = build_active_places_response(
        places=scored_places,
        evaluation_datetime=request.evaluation_datetime
    )
    response["search_radius_meters"] = request.search_radius_meters
    response["retrieval_debug"] = retrieval_debug
    return response
