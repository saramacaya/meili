from typing import Literal, Optional
import os
import requests
import uuid
import httpx
import math

import unicodedata

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from supabase import create_client
from datetime import datetime, timedelta
from opening_hours import OpeningHours
from zoneinfo import ZoneInfo

VALENCIA_TIMEZONE = ZoneInfo("Europe/Madrid")

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENROUTESERVICE_API_KEY = os.getenv("OPENROUTESERVICE_API_KEY")
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")
GOOGLE_ROUTES_API_KEY = os.getenv("GOOGLE_ROUTES_API_KEY")

GOOGLE_ROUTES_URL = (
    "https://routes.googleapis.com/"
    "directions/v2:computeRoutes"
)

supabase = None

if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(
    title="Meili Backend",
    description="Backend for the Meili behavioural route-choice prototype.",
    version="0.1.0"
)

from fastapi.middleware.cors import CORSMiddleware


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RouteRequest(BaseModel):
    origin: str
    destination: str

    origin_latitude: float = Field(..., ge=-90, le=90)
    origin_longitude: float = Field(..., ge=-180, le=180)

    destination_latitude: float = Field(..., ge=-90, le=90)
    destination_longitude: float = Field(..., ge=-180, le=180)

    initial_preference: Literal["fastest", "balanced", "safest"]
    user_type: str

class RouteSelectionRequest(BaseModel):
    user_id: str
    origin: str
    destination: str
    initial_preference: Literal["fastest", "balanced", "safest"]
    final_choice_type: Literal["fastest", "balanced", "safest"]
    user_type: str
    fastest_route_id: str
    balanced_route_id: str
    safest_route_id: str

class FeedbackRequest(BaseModel):
    user_id: str
    chosen_route_id: str
    origin: str
    destination: str
    final_choice_type: Literal["fastest", "balanced", "safest"]
    user_type: str
    perceived_safety_rating: int = Field(..., ge=1, le=5)
    would_choose_again: bool
    comment: Optional[str] = None

class UserCreateRequest(BaseModel):
    user_type: str

class PlaceAutocompleteRequest(BaseModel):
    input: str
    session_token: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class PlaceDetailsRequest(BaseModel):
    place_id: str
    session_token: Optional[str] = None

class RouteCoordinate(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    label: Optional[str] = None
    place_id: Optional[str] = None


class RealRoutePreviewRequest(BaseModel):
    origin: RouteCoordinate
    destination: RouteCoordinate

class StreetlightAnalysisRequest(BaseModel):
    route_geometry: list[tuple[float, float]]
    coverage_radius_meters: float = Field(
        default=25,
        ge=5,
        le=100
    )
    
VALENCIA_STREET_FURNITURE_URL = (
    "https://geoportal.valencia.es/server/rest/services/OPENDATA/"
    "UrbanismoEInfraestructuras/MapServer/323/query"
)

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


def feature_representative_point(
    feature: dict
) -> Optional[tuple[float, float]]:
    points = flatten_geojson_coordinates(
        feature.get("geometry", {}).get(
            "coordinates",
            []
        )
    )

    if not points:
        return None

    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points)
    )


def is_streetlight(feature: dict) -> bool:
    values = {
        str(value).strip().upper()
        for value in feature.get(
            "properties",
            {}
        ).values()
        if value is not None
    }

    return "FAROLA" in values or "FANAL" in values


def fetch_valencia_streetlights(
    route_coordinates: list[tuple[float, float]]
) -> tuple[list[tuple[float, float]], dict]:
    longitudes = [
        coordinate[0]
        for coordinate in route_coordinates
    ]

    latitudes = [
        coordinate[1]
        for coordinate in route_coordinates
    ]

    padding_degrees = 0.001

    envelope = ",".join(
        str(value)
        for value in (
            min(longitudes) - padding_degrees,
            min(latitudes) - padding_degrees,
            max(longitudes) + padding_degrees,
            max(latitudes) + padding_degrees
        )
    )

    shared_params = {
        "where": "1=1",
        "geometry": envelope,
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "outSR": "4326",
        "spatialRel": "esriSpatialRelIntersects"
    }

    try:
        # First, obtain every object ID in the search area.
        id_response = requests.get(
            VALENCIA_STREET_FURNITURE_URL,
            params={
                **shared_params,
                "f": "json",
                "returnIdsOnly": "true",
                "returnGeometry": "false"
            },
            timeout=20
        )

        id_response.raise_for_status()
        id_data = id_response.json()

        if "error" in id_data:
            raise ValueError(id_data["error"])

        object_ids = id_data.get("objectIds") or []
        object_ids = sorted(set(object_ids))

        all_features = []
        batch_size = 200

        # Download the objects in manageable batches.
        for batch_start in range(
            0,
            len(object_ids),
            batch_size
        ):
            object_id_batch = object_ids[
                batch_start:batch_start + batch_size
            ]

            feature_response = requests.get(
                VALENCIA_STREET_FURNITURE_URL,
                params={
                    "f": "geojson",
                    "objectIds": ",".join(
                        str(object_id)
                        for object_id in object_id_batch
                    ),
                    "outFields": "*",
                    "outSR": "4326",
                    "returnGeometry": "true"
                },
                timeout=20
            )

            feature_response.raise_for_status()
            feature_data = feature_response.json()

            if "error" in feature_data:
                raise ValueError(feature_data["error"])

            all_features.extend(
                feature_data.get("features", [])
            )

    except (requests.RequestException, ValueError) as error:
        raise HTTPException(
            status_code=502,
            detail=(
                "Valencia streetlight data could "
                f"not be retrieved: {error}"
            )
        )

    streetlights = []
    rejected_objects = []

    for feature in all_features:
        if is_streetlight(feature):
            point = feature_representative_point(
                feature
            )

            if point is not None:
                streetlights.append(point)
        else:
            rejected_objects.append(
                feature.get("properties", {})
            )

    retrieval_debug = {
        "object_ids_found": len(object_ids),
        "raw_features_downloaded": len(all_features),
        "download_batches": (
            (len(object_ids) + batch_size - 1)
            // batch_size
        ),
        "recognised_streetlights": len(streetlights),
        "rejected_objects_count": len(rejected_objects),
        "rejected_element_categories": sorted(
            {
                str(properties.get("elemento", "")).strip()
                for properties in rejected_objects
            }
    )
    }

    return streetlights, retrieval_debug

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

OVERPASS_API_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter"
]


def fetch_osm_places_near_route(
    route_coordinates: list[tuple[float, float]],
    search_radius_meters: float
) -> tuple[list[dict], dict]:
    """
    Retrieves relevant places from OpenStreetMap and keeps
    only places within the requested distance of the route.
    """
    if not route_coordinates:
        return [], {
            "raw_places_found": 0,
            "relevant_places_near_route": 0
        }

    route_samples = densify_route(
        route_coordinates,
        interval_meters=30
    )

    longitudes = [
        coordinate[0]
        for coordinate in route_coordinates
    ]

    latitudes = [
        coordinate[1]
        for coordinate in route_coordinates
    ]

    latitude_padding = (
        search_radius_meters / 111_000
    )

    average_latitude = sum(latitudes) / len(latitudes)

    longitude_padding = (
        search_radius_meters
        / (
            111_000
            * max(
                math.cos(
                    math.radians(average_latitude)
                ),
                0.01
            )
        )
    )

    south = min(latitudes) - latitude_padding
    west = min(longitudes) - longitude_padding
    north = max(latitudes) + latitude_padding
    east = max(longitudes) + longitude_padding

    bbox = f"{south},{west},{north},{east}"

    query = f"""
    [out:json][timeout:25];
    (
      nwr["amenity"~"^(cafe|restaurant|fast_food|pharmacy|hospital|clinic|doctors|police|fire_station|bar|pub|nightclub)$"]({bbox});
      nwr["shop"~"^(convenience|supermarket)$"]({bbox});
      nwr["tourism"~"^(hotel|hostel|guest_house)$"]({bbox});
    );
    out tags center;
    """

    data = None
    retrieval_errors = []

    for overpass_url in OVERPASS_API_URLS:
        try:
            response = requests.post(
                overpass_url,
                data={"data": query},
                headers={
                    "Accept": "application/json",
                    "User-Agent": (
                        "Meili safety-routing prototype"
                    )
                },
                timeout=35
            )
            response.raise_for_status()
            data = response.json()
            break

        except (
            requests.RequestException,
            ValueError
        ) as error:
            retrieval_errors.append(
                f"{overpass_url}: {error}"
            )

    if data is None:
        raise HTTPException(
            status_code=502,
            detail=(
                "OpenStreetMap place data could not "
                "be retrieved from the available "
                "Overpass servers: "
                + " | ".join(retrieval_errors)
            )
        )

    raw_elements = data.get("elements", [])
    relevant_places = []

    for element in raw_elements:
        tags = element.get("tags", {})
        category = classify_place(tags)

        if category is None:
            continue

        latitude = element.get("lat")
        longitude = element.get("lon")

        if latitude is None or longitude is None:
            center = element.get("center", {})
            latitude = center.get("lat")
            longitude = center.get("lon")

        if latitude is None or longitude is None:
            continue

        place_coordinate = (
            float(longitude),
            float(latitude)
        )

        nearest_route_distance = min(
            distance_meters(
                place_coordinate,
                route_sample
            )
            for route_sample in route_samples
        )

        if nearest_route_distance > search_radius_meters:
            continue

        relevant_places.append({
            "osm_type": element.get("type"),
            "osm_id": element.get("id"),
            "establishment_type": (
                get_establishment_type(tags)
            ),
            "name": (
                tags.get("name")
                or tags.get("brand")
                or "Unnamed place"
            ),
            "category": category,
            "latitude": float(latitude),
            "longitude": float(longitude),
            "distance_to_route_meters": round(
                nearest_route_distance,
                1
            ),
            "opening_hours": tags.get(
                "opening_hours"
            )
        })

    retrieval_debug = {
        "raw_places_found": len(raw_elements),
        "relevant_places_near_route": len(
            relevant_places
        ),
        "places_with_opening_hours": sum(
            1
            for place in relevant_places
            if place["opening_hours"]
        )
    }

    return relevant_places, retrieval_debug

def get_real_walking_route(
    origin_longitude: float,
    origin_latitude: float,
    destination_longitude: float,
    destination_latitude: float
):
    if not OPENROUTESERVICE_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OPENROUTESERVICE_API_KEY is missing from the .env file."
        )

    url = (
        "https://api.heigit.org/openrouteservice/"
        "v2/directions/foot-walking/geojson"
    )

    headers = {
        "Authorization": OPENROUTESERVICE_API_KEY.strip(),
        "Content-Type": "application/json",
        "Accept": "application/geo+json"
    }

    body = {
        "coordinates": [
            [origin_longitude, origin_latitude],
            [destination_longitude, destination_latitude]
        ],
        "alternative_routes": {
            "target_count": 3,
            "share_factor": 0.6,
            "weight_factor": 1.4
        }
    }

    try:
        response = requests.post(
            url,
            json=body,
            headers=headers,
            timeout=30
        )

        response.raise_for_status()
        route_data = response.json()

    except requests.RequestException as error:
        error_message = str(error)

        if error.response is not None:
            error_message = error.response.text

        raise HTTPException(
            status_code=502,
            detail=f"Routing service error: {error_message}"
        )

    features = route_data.get("features", [])

    if not features:
        raise HTTPException(
            status_code=404,
            detail="No walking route was found between these coordinates."
        )

    routes = []

    for index, feature in enumerate(features, start=1):
        summary = feature["properties"]["summary"]

        routes.append({
            "route_id": f"alternative_{index}",
            "estimated_time_minutes": round(summary["duration"] / 60),
            "distance_meters": round(summary["distance"]),
            "geometry": feature["geometry"]["coordinates"]
        })

    return {
        "route_type": "real_walking_alternatives",
        "number_of_routes": len(routes),
        "routes": routes
    }

MOCK_JOURNEYS = {
    ("ruzafa", "placa de la reina"): {
        "display_origin": "Ruzafa",
        "display_destination": "Plaça de la Reina",
        "routes": [
            {
                "id": "ruzafa_reina_fastest",
                "route_type": "fastest",
                "route_name": "Fastest Route",
                "estimated_time_minutes": 13,
                "distance_meters": 1300,
                "safety_score": 60,
                "explanation": "Shortest walking path through side streets, but with a lower safety score.",
                "tradeoff_summary": "Saves time, but has the lowest safety score."
            },
            {
                "id": "ruzafa_reina_balanced",
                "route_type": "balanced",
                "route_name": "Balanced Route",
                "estimated_time_minutes": 16,
                "distance_meters": 1400,
                "safety_score": 74,
                "explanation": "Mixes lit avenues with shorter connectors to balance time and perceived safety.",
                "tradeoff_summary": "Takes 3 extra minutes for a 14-point safety improvement."
            },
            {
                "id": "ruzafa_reina_safest",
                "route_type": "safest",
                "route_name": "Safest Route",
                "estimated_time_minutes": 19,
                "distance_meters": 1600,
                "safety_score": 85,
                "explanation": "Routes along better-lit main streets and busier public areas.",
                "tradeoff_summary": "Takes 6 extra minutes for a 25-point safety improvement."
            }
        ]
    },

    ("valencia nord", "mercado central"): {
        "display_origin": "Valencia Nord",
        "display_destination": "Mercado Central",
        "routes": [
            {
                "id": "nord_mercado_fastest",
                "route_type": "fastest",
                "route_name": "Fastest Route",
                "estimated_time_minutes": 11,
                "distance_meters": 950,
                "safety_score": 65,
                "explanation": "A direct route through central streets with moderate safety characteristics.",
                "tradeoff_summary": "Fastest option, but includes a few busier crossings."
            },
            {
                "id": "nord_mercado_balanced",
                "route_type": "balanced",
                "route_name": "Balanced Route",
                "estimated_time_minutes": 14,
                "distance_meters": 1100,
                "safety_score": 78,
                "explanation": "Uses wider streets and avoids some lower-rated segments.",
                "tradeoff_summary": "Adds 3 minutes for a 13-point safety improvement."
            },
            {
                "id": "nord_mercado_safest",
                "route_type": "safest",
                "route_name": "Safest Route",
                "estimated_time_minutes": 17,
                "distance_meters": 1300,
                "safety_score": 87,
                "explanation": "Prioritises better-lit, more active streets near central public areas.",
                "tradeoff_summary": "Adds 6 minutes for a 22-point safety improvement."
            }
        ]
    },

    ("turia gardens", "placa de la reina"): {
        "display_origin": "Turia Gardens",
        "display_destination": "Plaça de la Reina",
        "routes": [
            {
                "id": "turia_reina_fastest",
                "route_type": "fastest",
                "route_name": "Fastest Route",
                "estimated_time_minutes": 15,
                "distance_meters": 1250,
                "safety_score": 67,
                "explanation": "A shorter route using smaller connecting streets.",
                "tradeoff_summary": "Saves time, but has more uneven segment scores."
            },
            {
                "id": "turia_reina_balanced",
                "route_type": "balanced",
                "route_name": "Balanced Route",
                "estimated_time_minutes": 18,
                "distance_meters": 1450,
                "safety_score": 80,
                "explanation": "Balances directness with more visible and active streets.",
                "tradeoff_summary": "Adds 3 minutes for a 13-point safety improvement."
            },
            {
                "id": "turia_reina_safest",
                "route_type": "safest",
                "route_name": "Safest Route",
                "estimated_time_minutes": 22,
                "distance_meters": 1700,
                "safety_score": 90,
                "explanation": "Uses the most active and better-lit route segments.",
                "tradeoff_summary": "Adds 7 minutes for a 23-point safety improvement."
            }
        ]
    }
}

def google_duration_to_seconds(duration: str) -> int:
    if not duration or not duration.endswith("s"):
        return 0

    try:
        return int(round(float(duration[:-1])))
    except ValueError:
        return 0

@app.get("/health")
def health_check():
    return {
        "status": "Meili backend is running"
    }

@app.post("/safety/streetlights/analyse")
def analyse_streetlight_coverage(
    request: StreetlightAnalysisRequest
):
    if len(request.route_geometry) < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                "At least two route coordinates "
                "are required."
            )
        )

    route_coordinates = request.route_geometry

    samples = densify_route(route_coordinates)

    streetlights, retrieval_debug = (
    fetch_valencia_streetlights(
        route_coordinates
    )
)

    nearest_distances = []

    for sample in samples:
        if streetlights:
            nearest_distances.append(
                min(
                    distance_meters(
                        sample,
                        streetlight
                    )
                    for streetlight in streetlights
                )
            )
        else:
            nearest_distances.append(None)

    covered_samples = sum(
        distance is not None
        and distance <= request.coverage_radius_meters
        for distance in nearest_distances
    )

    coverage_percentage = round(
        100 * covered_samples / len(samples)
    )

    known_distances = [
        distance
        for distance in nearest_distances
        if distance is not None
    ]

    worst_sample = None
    nearest_streetlight_to_worst_sample = None

    if known_distances:
        worst_sample_index = max(
            range(len(nearest_distances)),
            key=lambda index: (
                nearest_distances[index]
                if nearest_distances[index] is not None
                else -1
            )
        )

        worst_sample = samples[worst_sample_index]

        nearest_streetlight_to_worst_sample = min(
            streetlights,
            key=lambda streetlight: distance_meters(
                worst_sample,
                streetlight
            )
        )

    return {
        "status": "streetlight_coverage_analysed",
        "source": (
            "Ajuntament de Valencia - "
            "Mobiliario urbano"
        ),
        "source_license": "CC BY 4.0",
        "coverage_radius_meters": (
            request.coverage_radius_meters
        ),
        "route_sample_count": len(samples),
        "streetlights_found_near_route": (
            len(streetlights)
        ),
        "covered_sample_percentage": (
            coverage_percentage
        ),
        "median_distance_to_nearest_streetlight_meters": (
            round(
                sorted(known_distances)[
                    len(known_distances) // 2
                ],
                1
            )
            if known_distances
            else None
        ),
        "maximum_distance_to_nearest_streetlight_meters": (
            round(max(known_distances), 1)
            if known_distances
            else None
        ),
        "data_confidence": "limited",
        "interpretation": (
            "This measures mapped streetlight "
            "structures near the route. It does "
            "not confirm that a light works, its "
            "brightness, or visibility at street level."
        )
    }

@app.post("/safety/active-places/analyse")
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

    for place in places:
        place_coordinate = (
            place["longitude"],
            place["latitude"]
        )

        route_progress = (
            calculate_place_route_progress(
                place_coordinate=place_coordinate,
                route_coordinates=(
                    request.route_geometry
                )
            )
        )

        estimated_arrival_datetime = (
            calculate_place_arrival_time(
                route_progress=route_progress,
                departure_datetime=(
                    request.evaluation_datetime
                ),
                route_duration_seconds=(
                    request.route_duration_seconds
                )
            )
        )

        place["route_progress_percentage"] = round(
            route_progress * 100,
            1
        )

        place["estimated_arrival_datetime"] = (
            estimated_arrival_datetime.isoformat()
        )

        place["opening_status"] = (
            determine_place_opening_status(
                opening_hours_value=(
                    place["opening_hours"]
                ),
                evaluation_datetime=(
                    estimated_arrival_datetime
                ),
                latitude=place["latitude"],
                longitude=place["longitude"],
                establishment_type=place[
                    "establishment_type"
                ]
            )
        )

        place["activity_status"] = (
            determine_place_activity_status(
                opening_status=(
                    place["opening_status"]
                ),
                opening_hours_value=(
                    place["opening_hours"]
                ),
                evaluation_datetime=(
                    estimated_arrival_datetime
                ),
                transition_window_minutes=30
            )
        )

    active_places = [
        place
        for place in places
        if place["category"] == "active_place"
    ]

    help_points = [
        place
        for place in places
        if place["category"] == "help_point"
    ]

    nightlife_places = [
        place
        for place in places
        if place["category"] == "nightlife"
    ]

    places_with_opening_hours = [
        place
        for place in places
        if place["opening_hours"]
    ]

    places_with_unknown_hours = [
        place
        for place in places
        if not place["opening_hours"]
    ]

    confirmed_open_places = [
        place
        for place in places
        if place["opening_status"] == "confirmed_open"
    ]

    confirmed_closed_places = [
        place
        for place in places
        if place["opening_status"] == "confirmed_closed"
    ]

    estimated_open_places = [
        place
        for place in places
        if place["opening_status"] == "estimated_open"
    ]

    estimated_closed_places = [
        place
        for place in places
        if place["opening_status"] == "estimated_closed"
    ]

    unknown_status_places = [
        place
        for place in places
        if place["opening_status"] == "unknown"
    ]

    open_activity_places = [
    place
    for place in places
    if place["activity_status"] == "open_activity"
    ]   

    closing_activity_places = [
        place
        for place in places
        if place["activity_status"] == "closing_activity"
    ]

    opening_activity_places = [
        place
        for place in places
        if place["activity_status"] == "opening_activity"
    ]

    estimated_activity_places = [
        place
        for place in places
        if place["activity_status"] == "estimated_activity"
    ]

    closed_activity_places = [
        place
        for place in places
        if place["activity_status"] == "closed_activity"
    ]

    unknown_activity_places = [
        place
        for place in places
        if place["activity_status"] == "unknown_activity"
    ]

    return {
        "status": "active_places_analysed",
        "opening_hours_logic_version": "v3_explicit_timezone",
        "source": "OpenStreetMap contributors",
        "source_license": "ODbL",
        "evaluation_datetime": (
            request.evaluation_datetime.isoformat()
        ),
        "search_radius_meters": (
            request.search_radius_meters
        ),
        "retrieval_debug": retrieval_debug,
        "total_relevant_places": len(places),
        "active_place_count": len(active_places),
        "help_point_count": len(help_points),
        "nightlife_count": len(nightlife_places),
        "places_with_opening_hours_count": len(
            places_with_opening_hours
        ),
        "places_with_unknown_hours_count": len(
            places_with_unknown_hours
        ),
        "confirmed_open_count": len(
            confirmed_open_places
        ),
        "confirmed_closed_count": len(
            confirmed_closed_places
        ),
        "estimated_open_count": len(
            estimated_open_places
        ),
        "estimated_closed_count": len(
            estimated_closed_places
        ),
        "unknown_opening_status_count": len(
            unknown_status_places
        ),
        "transition_activity_window_minutes": 30,
        "open_activity_count": len(
            open_activity_places
        ),
        "closing_activity_count": len(
            closing_activity_places
        ),
        "opening_activity_count": len(
            opening_activity_places
        ),
        "estimated_activity_count": len(
            estimated_activity_places
        ),
        "closed_activity_count": len(
            closed_activity_places
        ),
        "unknown_activity_count": len(
            unknown_activity_places
        ),
        "active_places": active_places,
        "help_points": help_points,
        "nightlife_places": nightlife_places,
        "data_confidence": "testing",
        "interpretation": (
            "This test checks the availability of "
            "relevant OpenStreetMap places and their "
            "opening-hours data near the route. Missing "
            "opening hours mean unknown, not closed."
        )
    }

@app.post("/routes/test-real")
def test_real_route(request: RouteRequest):
    route = get_real_walking_route(
        origin_longitude=request.origin_longitude,
        origin_latitude=request.origin_latitude,
        destination_longitude=request.destination_longitude,
        destination_latitude=request.destination_latitude
    )

    return {
        "status": "real_route_generated",
        "requested_origin": request.origin,
        "requested_destination": request.destination,
        "route": route
    }
@app.post("/places/autocomplete")
async def autocomplete_places(request: PlaceAutocompleteRequest):
    search_text = request.input.strip()

    if len(search_text) < 2:
        return {
            "status": "success",
            "suggestions": [],
            "session_token": request.session_token
        }

    if not GOOGLE_PLACES_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Google Places API is not configured."
        )

    session_token = request.session_token or str(uuid.uuid4())

    payload = {
        "input": search_text,
        "languageCode": "en",
        "regionCode": "es",
        "sessionToken": session_token
    }

    if request.latitude is not None and request.longitude is not None:
        payload["locationBias"] = {
            "circle": {
                "center": {
                    "latitude": request.latitude,
                    "longitude": request.longitude
                },
                "radius": 50000.0
            }
        }

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
        "X-Goog-FieldMask": (
            "suggestions.placePrediction.placeId,"
            "suggestions.placePrediction.text.text,"
            "suggestions.placePrediction.structuredFormat.mainText.text,"
            "suggestions.placePrediction.structuredFormat.secondaryText.text,"
            "suggestions.placePrediction.types"
        )
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            google_response = await client.post(
                "https://places.googleapis.com/v1/places:autocomplete",
                json=payload,
                headers=headers
            )

        if google_response.status_code != 200:
            raise HTTPException(
                status_code=google_response.status_code,
                detail=google_response.text
            )

        google_data = google_response.json()
        suggestions = []

        for suggestion in google_data.get("suggestions", []):
            prediction = suggestion.get("placePrediction")

            if not prediction:
                continue

            structured = prediction.get("structuredFormat", {})
            main_text = structured.get("mainText", {}).get("text")
            secondary_text = structured.get("secondaryText", {}).get("text")
            full_text = prediction.get("text", {}).get("text")

            suggestions.append({
                "place_id": prediction.get("placeId"),
                "name": main_text or full_text,
                "address": secondary_text,
                "label": full_text,
                "types": prediction.get("types", [])
            })

        return {
            "status": "success",
            "suggestions": suggestions[:5],
            "session_token": session_token
        }

    except httpx.RequestError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Could not contact Google Places: {str(error)}"
        )
@app.post("/places/details")
async def get_place_details(request: PlaceDetailsRequest):
    if not request.place_id.strip():
        raise HTTPException(
            status_code=400,
            detail="A place_id is required."
        )

    if not GOOGLE_PLACES_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Google Places API is not configured."
        )

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
        "X-Goog-FieldMask": (
            "id,"
            "formattedAddress,"
            "location,"
            "types"
        )
    }

    params = {
        "languageCode": "en",
        "regionCode": "es"
    }

    if request.session_token:
        params["sessionToken"] = request.session_token

    place_url = (
        "https://places.googleapis.com/v1/places/"
        f"{request.place_id}"
    )

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            google_response = await client.get(
                place_url,
                headers=headers,
                params=params
            )

        if google_response.status_code != 200:
            raise HTTPException(
                status_code=google_response.status_code,
                detail=google_response.text
            )

        place = google_response.json()
        location = place.get("location", {})

        return {
            "status": "success",
            "place_id": place.get("id"),
            "formatted_address": place.get("formattedAddress"),
            "latitude": location.get("latitude"),
            "longitude": location.get("longitude"),
            "types": place.get("types", [])
        }

    except httpx.RequestError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Could not retrieve place details: {str(error)}"
        )

@app.post("/routes/preview-real")
async def preview_real_routes(request: RealRoutePreviewRequest):
    if not GOOGLE_ROUTES_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Google Routes API is not configured."
        )

    payload = {
        "origin": {
            "location": {
                "latLng": {
                    "latitude": request.origin.latitude,
                    "longitude": request.origin.longitude
                }
            }
        },
        "destination": {
            "location": {
                "latLng": {
                    "latitude": request.destination.latitude,
                    "longitude": request.destination.longitude
                }
            }
        },
        "travelMode": "WALK",
        "computeAlternativeRoutes": True,
        "languageCode": "en",
        "units": "METRIC"
    }

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_ROUTES_API_KEY,
        "X-Goog-FieldMask": (
            "routes.duration,"
            "routes.distanceMeters,"
            "routes.routeLabels,"
            "routes.polyline.encodedPolyline"
        )
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            google_response = await client.post(
                GOOGLE_ROUTES_URL,
                json=payload,
                headers=headers
            )

    except httpx.RequestError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Could not contact Google Routes: {str(error)}"
        )

    if google_response.status_code != 200:
        raise HTTPException(
            status_code=google_response.status_code,
            detail=google_response.text
        )

    google_data = google_response.json()
    google_routes = google_data.get("routes", [])

    if not google_routes:
        raise HTTPException(
            status_code=404,
            detail="No walking route was found between these locations."
        )

    candidates = []

    for route in google_routes:
        duration_seconds = google_duration_to_seconds(
            route.get("duration", "")
        )

        distance_meters = route.get("distanceMeters", 0)

        candidates.append({
            "duration_seconds": duration_seconds,
            "estimated_time_minutes": max(
                1,
                round(duration_seconds / 60)
            ),
            "distance_meters": distance_meters,
            "distance_km": round(distance_meters / 1000, 2),
            "encoded_polyline": route.get("polyline", {}).get("encodedPolyline"),
            "provider_labels": route.get("routeLabels", []),
            "safety_score": None,
            "safety_status": "not_scored"
        })

    candidates.sort(key=lambda route: route["duration_seconds"])

    for index, candidate in enumerate(candidates):
        candidate["candidate_id"] = f"candidate_{index + 1}"

        if index == 0:
            candidate["candidate_type"] = "fastest"
            candidate["display_name"] = "Fastest"
        else:
            candidate["candidate_type"] = f"alternative_{index}"
            candidate["display_name"] = f"Alternative {index}"

    return {
        "status": "real_routes_found",
        "provider": "google_routes",
        "travel_mode": "walking",
        "origin": request.origin.dict(),
        "destination": request.destination.dict(),
        "route_count": len(candidates),
        "routes": candidates,
        "safety_scoring_applied": False,
        "note": (
            "These are genuine walking-route candidates. "
            "Safety scoring has not yet been applied."
        )
    }

@app.post("/routes/generate")
def generate_routes(request: RouteRequest):
    if None in (
        request.origin_latitude,
        request.origin_longitude,
        request.destination_latitude,
        request.destination_longitude
    ):
        raise HTTPException(
            status_code=400,
            detail="Origin and destination coordinates are required."
        )

    real_route_result = get_real_walking_route(
        origin_longitude=request.origin_longitude,
        origin_latitude=request.origin_latitude,
        destination_longitude=request.destination_longitude,
        destination_latitude=request.destination_latitude
    )

    real_routes = real_route_result["routes"]

    requested_coordinates = {
        "origin": {
            "latitude": request.origin_latitude,
            "longitude": request.origin_longitude
        },
        "destination": {
            "latitude": request.destination_latitude,
            "longitude": request.destination_longitude
        }
    }

    if supabase is None:
        return {
            "status": "real_routes_generated_not_saved",
            "message": (
                "Real walking alternatives were generated, but Supabase "
                "is not configured, so they were not saved."
            ),
            "requested_origin": request.origin,
            "requested_destination": request.destination,
            "requested_coordinates": requested_coordinates,
            "user_type": request.user_type,
            "initial_preference": request.initial_preference,
            "recommended_route_type": None,
            "number_of_routes": len(real_routes),
            "routes": real_routes,
            "note": (
                "These are real walking alternatives. Safety scoring has "
                "not yet been applied."
            )
        }

    try:
        user_response = (
            supabase
            .table("users")
            .insert({
                "user_type": request.user_type
            })
            .execute()
        )

        if not user_response.data:
            return {
                "status": "error",
                "message": "The user could not be created."
            }

        created_user = user_response.data[0]
        user_id = created_user["id"]

        route_rows = []

        for route in real_routes:
            route_rows.append({
                "user_id": user_id,
                "origin_text": request.origin,
                "destination_text": request.destination,
                "initial_preference": request.initial_preference,
                "route_type": route["route_id"],
                "estimated_time_minutes": route[
                    "estimated_time_minutes"
                ],
                "distance_meters": route["distance_meters"],
                "safety_score": None,
                "route_geometry_json": route["geometry"],
                "explanation": (
                    "Real walking alternative. Safety scoring has not "
                    "yet been applied."
                )
            })

        routes_response = (
            supabase
            .table("routes")
            .insert(route_rows)
            .execute()
        )

        saved_routes = routes_response.data or []

        database_id_by_route_type = {
            saved_route["route_type"]: saved_route["id"]
            for saved_route in saved_routes
        }

        routes_with_database_ids = []

        for route in real_routes:
            route_copy = route.copy()
            route_copy["database_id"] = (
                database_id_by_route_type.get(route["route_id"])
            )
            routes_with_database_ids.append(route_copy)

        return {
            "status": "real_routes_generated_and_saved",
            "user_id": user_id,
            "requested_origin": request.origin,
            "requested_destination": request.destination,
            "requested_coordinates": requested_coordinates,
            "user_type": request.user_type,
            "initial_preference": request.initial_preference,
            "recommended_route_type": None,
            "number_of_routes": len(routes_with_database_ids),
            "routes": routes_with_database_ids,
            "note": (
                "These are real walking alternatives saved to Supabase. "
                "Safety scoring has not yet been applied."
            )
        }

    except Exception as error:
        return {
            "status": "error",
            "message": str(error)
        }
@app.post("/routes/select")
def select_route(request: RouteSelectionRequest):
    origin_key = normalise_text(request.origin)
    destination_key = normalise_text(request.destination)

    journey_key = (origin_key, destination_key)

    if journey_key in MOCK_JOURNEYS:
        journey = MOCK_JOURNEYS[journey_key]
        matched_demo_route = True
    else:
        journey = MOCK_JOURNEYS[("ruzafa", "placa de la reina")]
        matched_demo_route = False

    routes = journey["routes"]

    fastest_route = next(route for route in routes if route["route_type"] == "fastest")
    chosen_route = next(
        route for route in routes
        if route["route_type"] == request.final_choice_type
    )

    extra_time_minutes = (
        chosen_route["estimated_time_minutes"]
        - fastest_route["estimated_time_minutes"]
    )

    safety_gain = (
        chosen_route["safety_score"]
        - fastest_route["safety_score"]
    )

    changed_preference = request.initial_preference != request.final_choice_type

    route_id_by_type = {
        "fastest": request.fastest_route_id,
        "balanced": request.balanced_route_id,
        "safest": request.safest_route_id
    }

    chosen_route_id = route_id_by_type[request.final_choice_type]

    if supabase is None:
        return {
            "status": "selection_calculated_not_saved",
            "message": "Supabase is not configured, so the route choice was calculated but not saved.",
            "requested_origin": request.origin,
            "requested_destination": request.destination,
            "matched_origin": journey["display_origin"],
            "matched_destination": journey["display_destination"],
            "matched_demo_route": matched_demo_route,
            "user_id": request.user_id,
            "user_type": request.user_type,
            "initial_preference": request.initial_preference,
            "final_choice_type": request.final_choice_type,
            "changed_preference": changed_preference,
            "chosen_route": chosen_route,
            "chosen_route_id": chosen_route_id,
            "extra_time_minutes": extra_time_minutes,
            "safety_gain": safety_gain
        }

    try:
        choice_response = supabase.table("route_choices").insert({
            "user_id": request.user_id,
            "chosen_route_id": chosen_route_id,
            "fastest_route_id": request.fastest_route_id,
            "balanced_route_id": request.balanced_route_id,
            "safest_route_id": request.safest_route_id,
            "initial_preference": request.initial_preference,
            "final_choice_type": request.final_choice_type,
            "extra_time_minutes": extra_time_minutes,
            "safety_gain": safety_gain,
            "framing_group": None
        }).execute()

        saved_choice = choice_response.data[0] if choice_response.data else None

        return {
            "status": "route_choice_saved",
            "route_choice": saved_choice,
            "requested_origin": request.origin,
            "requested_destination": request.destination,
            "matched_origin": journey["display_origin"],
            "matched_destination": journey["display_destination"],
            "matched_demo_route": matched_demo_route,
            "user_id": request.user_id,
            "user_type": request.user_type,
            "initial_preference": request.initial_preference,
            "final_choice_type": request.final_choice_type,
            "changed_preference": changed_preference,
            "chosen_route": chosen_route,
            "chosen_route_id": chosen_route_id,
            "extra_time_minutes": extra_time_minutes,
            "safety_gain": safety_gain,
            "note": "The route choice has been saved to Supabase."
        }

    except Exception as error:
        return {
            "status": "error",
            "message": str(error)
        }

@app.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    origin_key = normalise_text(request.origin)
    destination_key = normalise_text(request.destination)

    journey_key = (origin_key, destination_key)

    if journey_key in MOCK_JOURNEYS:
        journey = MOCK_JOURNEYS[journey_key]
        matched_demo_route = True
    else:
        journey = MOCK_JOURNEYS[("ruzafa", "placa de la reina")]
        matched_demo_route = False

    routes = journey["routes"]

    chosen_route = next(
        route for route in routes
        if route["route_type"] == request.final_choice_type
    )

    if supabase is None:
        return {
            "status": "feedback_validated_not_saved",
            "message": "Supabase is not configured, so feedback was validated but not saved.",
            "requested_origin": request.origin,
            "requested_destination": request.destination,
            "matched_origin": journey["display_origin"],
            "matched_destination": journey["display_destination"],
            "matched_demo_route": matched_demo_route,
            "user_id": request.user_id,
            "chosen_route_id": request.chosen_route_id,
            "user_type": request.user_type,
            "final_choice_type": request.final_choice_type,
            "chosen_route_name": chosen_route["route_name"],
            "perceived_safety_rating": request.perceived_safety_rating,
            "would_choose_again": request.would_choose_again,
            "comment": request.comment
        }

    try:
        feedback_response = supabase.table("feedback").insert({
            "user_id": request.user_id,
            "route_id": request.chosen_route_id,
            "perceived_safety_rating": request.perceived_safety_rating,
            "would_choose_again": request.would_choose_again,
            "comment": request.comment
        }).execute()

        saved_feedback = feedback_response.data[0] if feedback_response.data else None

        return {
            "status": "feedback_saved",
            "feedback": saved_feedback,
            "requested_origin": request.origin,
            "requested_destination": request.destination,
            "matched_origin": journey["display_origin"],
            "matched_destination": journey["display_destination"],
            "matched_demo_route": matched_demo_route,
            "user_id": request.user_id,
            "chosen_route_id": request.chosen_route_id,
            "user_type": request.user_type,
            "final_choice_type": request.final_choice_type,
            "chosen_route_name": chosen_route["route_name"],
            "perceived_safety_rating": request.perceived_safety_rating,
            "would_choose_again": request.would_choose_again,
            "comment": request.comment,
            "note": "Feedback has been saved to Supabase."
        }

    except Exception as error:
        return {
            "status": "error",
            "message": str(error)
        }

@app.get("/database/health")
def database_health():
    if supabase is None:
        return {
            "status": "not_configured",
            "message": "Supabase URL or key is missing from the backend .env file."
        }

    try:
        response = supabase.table("users").select("id").limit(1).execute()

        return {
            "status": "connected",
            "table_checked": "users",
            "rows_returned": len(response.data)
        }

    except Exception as error:
        return {
            "status": "error",
            "message": str(error)
        }

@app.post("/users/create")
def create_user(request: UserCreateRequest):
    if supabase is None:
        return {
            "status": "not_configured",
            "message": "Supabase URL or key is missing from the backend .env file."
        }

    try:
        response = supabase.table("users").insert({
            "user_type": request.user_type
        }).execute()

        created_user = response.data[0] if response.data else None

        return {
            "status": "user_created",
            "user": created_user
        }

    except Exception as error:
        return {
            "status": "error",
            "message": str(error)
        }
