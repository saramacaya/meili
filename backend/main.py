from typing import Literal, Optional
import os
import requests
import uuid
import httpx
import math
import threading
import time

import unicodedata

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from supabase import create_client
from datetime import datetime, timedelta
from opening_hours import OpeningHours
from zoneinfo import ZoneInfo
from nasa_route_analysis import analyse_nasa_route_samples

from combined_lighting_analysis import combine_lighting_sources
from open_establishments import (
    densify_route,
    distance_meters,
    flatten_geojson_coordinates,
    normalise_text,
    router as open_establishments_router,
)

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

app.include_router(open_establishments_router)

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

class OsmLightingAnalysisRequest(BaseModel):
    route_geometry: list[tuple[float, float]]

    match_radius_meters: float = Field(
        default=15,
        ge=5,
        le=50
    )

    sample_interval_meters: float = Field(
        default=15,
        ge=5,
        le=50
    )


class OsmStreetLampAnalysisRequest(BaseModel):
    route_geometry: list[tuple[float, float]]

    coverage_radius_meters: float = Field(
        default=25,
        ge=5,
        le=100
    )

    sample_interval_meters: float = Field(
        default=15,
        ge=5,
        le=50
    )


class LightingRouteCandidate(BaseModel):
    route_id: str
    geometry: list[tuple[float, float]]
    estimated_time_minutes: Optional[int] = None
    distance_meters: Optional[float] = None


class LightingComparisonRequest(BaseModel):
    routes: list[LightingRouteCandidate]

    sample_interval_meters: float = Field(
        default=15,
        ge=5,
        le=50
    )

    official_lamp_radius_meters: float = Field(
        default=25,
        ge=5,
        le=100
    )

    osm_lit_match_radius_meters: float = Field(
        default=15,
        ge=5,
        le=50
    )

    osm_lamp_radius_meters: float = Field(
        default=25,
        ge=5,
        le=100
    )

    month: Optional[str] = None

class OsmStreetLampAreaScanRequest(BaseModel):
    south: float = Field(..., ge=-90, le=90)
    west: float = Field(..., ge=-180, le=180)
    north: float = Field(..., ge=-90, le=90)
    east: float = Field(..., ge=-180, le=180)

    sample_limit: int = Field(
        default=20,
        ge=0,
        le=100
    )

class NasaNightLightsAnalysisRequest(BaseModel):
    route_geometry: list[tuple[float, float]]

    sample_interval_meters: float = Field(
        default=15,
        ge=5,
        le=100
    )

    # Leave empty to use the latest month in Supabase.
    # Example historical value: "2026-05".
    month: Optional[str] = None

class CombinedLightingAnalysisRequest(BaseModel):
    route_geometry: list[
        tuple[float, float]
    ]

    sample_interval_meters: float = Field(
        default=15,
        ge=5,
        le=50
    )

    official_lamp_radius_meters: float = Field(
        default=25,
        ge=5,
        le=100
    )

    osm_lit_match_radius_meters: float = Field(
        default=15,
        ge=5,
        le=50
    )

    osm_lamp_radius_meters: float = Field(
        default=25,
        ge=5,
        le=100
    )

    # Leave empty to use the latest NASA month.
    month: Optional[str] = None



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


OVERPASS_API_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter"
]


OSM_LIT_POSITIVE_VALUES = {
    "yes",
    "24/7",
    "automatic"
}

OSM_LIT_NEGATIVE_VALUES = {
    "no",
    "disused"
}


def classify_osm_lit_value(value: Optional[str]) -> str:
    """
    Converts an OSM lit=* value into a simple evidence class.

    Returns:
    - lit
    - unlit
    - conditional_or_other
    - unknown
    """
    if value is None:
        return "unknown"

    normalised_value = normalise_text(str(value))

    if normalised_value in OSM_LIT_POSITIVE_VALUES:
        return "lit"

    if normalised_value in OSM_LIT_NEGATIVE_VALUES:
        return "unlit"

    return "conditional_or_other"

OSM_LIGHTING_CACHE: dict[str, dict] = {}
OSM_LIGHTING_CACHE_LOCK = threading.Lock()
OSM_LIGHTING_CACHE_TTL_SECONDS = 15 * 60


def fetch_shared_osm_lighting_data(
    route_geometries: list[
        list[tuple[float, float]]
    ],
    padding_meters: float
) -> dict:
    """
    Downloads OSM lighting data once for one bounding box
    covering every route in the comparison.

    The result is cached in memory for 15 minutes.
    """
    all_coordinates = [
        coordinate
        for geometry in route_geometries
        for coordinate in geometry
    ]

    if not all_coordinates:
        return {
            "lit_way_elements": [],
            "street_lamp_elements": [],
            "debug": {
                "cache_hit": False,
                "raw_lit_ways_found": 0,
                "raw_street_lamps_found": 0
            }
        }

    longitudes = [
        coordinate[0]
        for coordinate in all_coordinates
    ]

    latitudes = [
        coordinate[1]
        for coordinate in all_coordinates
    ]

    average_latitude = (
        sum(latitudes) / len(latitudes)
    )

    latitude_padding = (
        padding_meters / 111_000
    )

    longitude_padding = (
        padding_meters
        / (
            111_000
            * max(
                math.cos(
                    math.radians(
                        average_latitude
                    )
                ),
                0.01
            )
        )
    )

    south = min(latitudes) - latitude_padding
    west = min(longitudes) - longitude_padding
    north = max(latitudes) + latitude_padding
    east = max(longitudes) + longitude_padding

    bbox = (
        f"{south},{west},{north},{east}"
    )

    # Rounded coordinates make repeat requests for the
    # same route group use the same cache entry.
    cache_key = (
        f"{south:.5f},"
        f"{west:.5f},"
        f"{north:.5f},"
        f"{east:.5f}"
    )

    current_time = time.time()

    with OSM_LIGHTING_CACHE_LOCK:
        cached_entry = OSM_LIGHTING_CACHE.get(
            cache_key
        )

        if cached_entry is not None:
            cache_age = (
                current_time
                - cached_entry["created_at"]
            )

            if (
                cache_age
                < OSM_LIGHTING_CACHE_TTL_SECONDS
            ):
                cached_result = (
                    cached_entry["result"].copy()
                )

                cached_result["debug"] = (
                    cached_result["debug"].copy()
                )

                cached_result["debug"][
                    "cache_hit"
                ] = True

                return cached_result

    query = f"""
    [out:json][timeout:40];
    (
      way["highway"]["lit"]({bbox});
      node["highway"="street_lamp"]({bbox});
    );
    out body geom;
    """

    data = None
    retrieval_errors = []
    successful_server = None

    for overpass_url in OVERPASS_API_URLS:
        try:
            response = requests.post(
                overpass_url,
                data={
                    "data": query
                },
                headers={
                    "Accept": (
                        "application/json"
                    ),
                    "User-Agent": (
                        "Meili safety-routing "
                        "prototype"
                    )
                },
                timeout=60
            )

            response.raise_for_status()
            data = response.json()
            successful_server = overpass_url
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
                "Shared OpenStreetMap lighting "
                "data could not be retrieved from "
                "the available Overpass servers: "
                + " | ".join(retrieval_errors)
            )
        )

    elements = data.get(
        "elements",
        []
    )

    lit_way_elements = [
        element
        for element in elements
        if element.get("type") == "way"
        and element.get(
            "tags",
            {}
        ).get("lit") is not None
    ]

    street_lamp_elements = [
        element
        for element in elements
        if element.get("type") == "node"
        and element.get(
            "tags",
            {}
        ).get("highway") == "street_lamp"
    ]

    result = {
        "lit_way_elements": (
            lit_way_elements
        ),
        "street_lamp_elements": (
            street_lamp_elements
        ),
        "debug": {
            "cache_hit": False,
            "overpass_server": (
                successful_server
            ),
            "bbox": {
                "south": south,
                "west": west,
                "north": north,
                "east": east
            },
            "raw_lit_ways_found": len(
                lit_way_elements
            ),
            "raw_street_lamps_found": len(
                street_lamp_elements
            )
        }
    }

    with OSM_LIGHTING_CACHE_LOCK:
        OSM_LIGHTING_CACHE[cache_key] = {
            "created_at": current_time,
            "result": result
        }

    return result

def filter_shared_osm_lit_ways_for_route(
    route_coordinates: list[
        tuple[float, float]
    ],
    shared_lit_way_elements: list[dict],
    match_radius_meters: float
) -> tuple[list[dict], dict]:
    """
    Keeps only shared OSM lit=* ways that are close
    to one particular route.

    This function performs no network request.
    """
    if not route_coordinates:
        return [], {
            "raw_lit_ways_found": len(
                shared_lit_way_elements
            ),
            "lit_ways_near_route": 0,
            "lit_value_counts": {}
        }

    route_samples = densify_route(
        route_coordinates,
        interval_meters=15
    )

    nearby_lit_ways = []

    for element in shared_lit_way_elements:
        geometry = element.get(
            "geometry",
            []
        )

        way_coordinates = [
            (
                float(point["lon"]),
                float(point["lat"])
            )
            for point in geometry
            if point.get("lon") is not None
            and point.get("lat") is not None
        ]

        if len(way_coordinates) < 2:
            continue

        way_samples = densify_route(
            way_coordinates,
            interval_meters=10
        )

        nearest_route_distance = min(
            distance_meters(
                route_sample,
                way_sample
            )
            for route_sample in route_samples
            for way_sample in way_samples
        )

        if (
            nearest_route_distance
            > match_radius_meters
        ):
            continue

        tags = element.get(
            "tags",
            {}
        )

        lit_value = tags.get("lit")

        nearby_lit_ways.append({
            "osm_type": element.get("type"),
            "osm_id": element.get("id"),
            "name": tags.get("name"),
            "highway": tags.get("highway"),
            "lit_value": lit_value,
            "lit_classification": (
                classify_osm_lit_value(
                    lit_value
                )
            ),
            "distance_to_route_meters": round(
                nearest_route_distance,
                1
            ),
            "sampled_geometry": way_samples
        })

    lit_values = {
        way["lit_value"]
        for way in nearby_lit_ways
        if way["lit_value"] is not None
    }

    retrieval_debug = {
        "raw_lit_ways_found": len(
            shared_lit_way_elements
        ),
        "lit_ways_near_route": len(
            nearby_lit_ways
        ),
        "lit_value_counts": {
            value: sum(
                1
                for way in nearby_lit_ways
                if way["lit_value"] == value
            )
            for value in sorted(lit_values)
        }
    }

    return nearby_lit_ways, retrieval_debug


def filter_shared_osm_street_lamps_for_route(
    route_coordinates: list[
        tuple[float, float]
    ],
    shared_street_lamp_elements: list[dict],
    coverage_radius_meters: float
) -> tuple[list[dict], dict]:
    """
    Keeps only shared highway=street_lamp nodes that
    are close to one particular route.

    This function performs no network request.
    """
    if not route_coordinates:
        return [], {
            "raw_street_lamps_found": len(
                shared_street_lamp_elements
            ),
            "street_lamps_near_route": 0,
            "lamps_with_reference_number": 0,
            "lamps_with_model_or_type_data": 0
        }

    route_samples = densify_route(
        route_coordinates,
        interval_meters=5
    )

    nearby_street_lamps = []

    for element in shared_street_lamp_elements:
        latitude = element.get("lat")
        longitude = element.get("lon")

        if (
            latitude is None
            or longitude is None
        ):
            continue

        lamp_coordinate = (
            float(longitude),
            float(latitude)
        )

        nearest_route_distance = min(
            distance_meters(
                lamp_coordinate,
                route_sample
            )
            for route_sample in route_samples
        )

        if (
            nearest_route_distance
            > coverage_radius_meters
        ):
            continue

        tags = element.get(
            "tags",
            {}
        )

        nearby_street_lamps.append({
            "osm_type": element.get("type"),
            "osm_id": element.get("id"),
            "latitude": float(latitude),
            "longitude": float(longitude),
            "distance_to_route_meters": round(
                nearest_route_distance,
                1
            ),
            "ref": tags.get("ref"),
            "operator": tags.get("operator"),
            "lamp_type": tags.get(
                "lamp_type"
            ),
            "lamp_mount": tags.get(
                "lamp_mount"
            ),
            "lamp_model": tags.get(
                "lamp_model"
            ),
            "light_source": tags.get(
                "light_source"
            ),
            "height": tags.get("height"),
            "direction": tags.get(
                "direction"
            )
        })

    retrieval_debug = {
        "raw_street_lamps_found": len(
            shared_street_lamp_elements
        ),
        "street_lamps_near_route": len(
            nearby_street_lamps
        ),
        "lamps_with_reference_number": sum(
            1
            for lamp in nearby_street_lamps
            if lamp["ref"]
        ),
        "lamps_with_model_or_type_data": sum(
            1
            for lamp in nearby_street_lamps
            if (
                lamp["lamp_model"]
                or lamp["lamp_type"]
                or lamp["light_source"]
            )
        )
    }

    return (
        nearby_street_lamps,
        retrieval_debug
    )

def fetch_osm_lit_ways_near_route(
    route_coordinates: list[tuple[float, float]],
    match_radius_meters: float
) -> tuple[list[dict], dict]:
    """
    Retrieves OSM highway ways carrying lit=* and keeps only
    those spatially close to the walking route.
    """
    if not route_coordinates:
        return [], {
            "raw_lit_ways_found": 0,
            "lit_ways_near_route": 0
        }

    route_samples = densify_route(
        route_coordinates,
        interval_meters=15
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
        match_radius_meters / 111_000
    )

    average_latitude = sum(latitudes) / len(latitudes)

    longitude_padding = (
        match_radius_meters
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
    way["highway"]["lit"]({bbox});
    out tags geom;
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
                "OpenStreetMap lighting data could not "
                "be retrieved from the available "
                "Overpass servers: "
                + " | ".join(retrieval_errors)
            )
        )

    raw_elements = data.get("elements", [])
    nearby_lit_ways = []

    for element in raw_elements:
        geometry = element.get("geometry", [])

        way_coordinates = [
            (
                float(point["lon"]),
                float(point["lat"])
            )
            for point in geometry
            if point.get("lon") is not None
            and point.get("lat") is not None
        ]

        if len(way_coordinates) < 2:
            continue

        way_samples = densify_route(
            way_coordinates,
            interval_meters=10
        )

        nearest_route_distance = min(
            distance_meters(
                route_sample,
                way_sample
            )
            for route_sample in route_samples
            for way_sample in way_samples
        )

        if nearest_route_distance > match_radius_meters:
            continue

        tags = element.get("tags", {})
        lit_value = tags.get("lit")

        nearby_lit_ways.append({
            "osm_type": element.get("type"),
            "osm_id": element.get("id"),
            "name": tags.get("name"),
            "highway": tags.get("highway"),
            "lit_value": lit_value,
            "lit_classification": (
                classify_osm_lit_value(lit_value)
            ),
            "distance_to_route_meters": round(
                nearest_route_distance,
                1
            ),
            "sampled_geometry": way_samples
        })

    retrieval_debug = {
        "raw_lit_ways_found": len(raw_elements),
        "lit_ways_near_route": len(
            nearby_lit_ways
        ),
        "lit_value_counts": {
            value: sum(
                1
                for way in nearby_lit_ways
                if way["lit_value"] == value
            )
            for value in sorted({
                way["lit_value"]
                for way in nearby_lit_ways
            })
        }
    }

    return nearby_lit_ways, retrieval_debug


def fetch_osm_street_lamps_near_route(
    route_coordinates: list[tuple[float, float]],
    coverage_radius_meters: float
) -> tuple[list[dict], dict]:
    """
    Retrieves individual OSM street-lamp nodes near the route.

    OSM maps these as nodes tagged:
    highway=street_lamp
    """
    if not route_coordinates:
        return [], {
            "raw_street_lamps_found": 0,
            "street_lamps_near_route": 0
        }

    # Use closely spaced samples so each lamp can be projected
    # onto an approximate position along the walking route.
    route_samples = densify_route(
        route_coordinates,
        interval_meters=5
    )

    cumulative_route_distances = [0.0]

    for previous, current in zip(
        route_samples,
        route_samples[1:]
    ):
        cumulative_route_distances.append(
            cumulative_route_distances[-1]
            + distance_meters(previous, current)
        )

    total_route_distance = (
        cumulative_route_distances[-1]
        if cumulative_route_distances
        else 0.0
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
        coverage_radius_meters / 111_000
    )

    average_latitude = sum(latitudes) / len(latitudes)

    longitude_padding = (
        coverage_radius_meters
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
    node["highway"="street_lamp"]({bbox});
    out body;
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
                "OpenStreetMap street-lamp data could "
                "not be retrieved from the available "
                "Overpass servers: "
                + " | ".join(retrieval_errors)
            )
        )

    raw_elements = data.get("elements", [])
    nearby_street_lamps = []

    for element in raw_elements:
        latitude = element.get("lat")
        longitude = element.get("lon")

        if latitude is None or longitude is None:
            continue

        lamp_coordinate = (
            float(longitude),
            float(latitude)
        )

        nearest_sample_index = min(
            range(len(route_samples)),
            key=lambda index: distance_meters(
                lamp_coordinate,
                route_samples[index]
            )
        )

        distance_to_route = distance_meters(
            lamp_coordinate,
            route_samples[nearest_sample_index]
        )

        if distance_to_route > coverage_radius_meters:
            continue

        approximate_distance_along_route = (
            cumulative_route_distances[
                nearest_sample_index
            ]
        )

        route_progress = (
            approximate_distance_along_route
            / total_route_distance
            if total_route_distance > 0
            else 0.0
        )

        tags = element.get("tags", {})

        nearby_street_lamps.append({
            "osm_type": element.get("type"),
            "osm_id": element.get("id"),
            "latitude": float(latitude),
            "longitude": float(longitude),
            "distance_to_route_meters": round(
                distance_to_route,
                1
            ),
            "route_progress_percentage": round(
                route_progress * 100,
                1
            ),
            "approximate_distance_along_route_meters": round(
                approximate_distance_along_route,
                1
            ),
            "ref": tags.get("ref"),
            "operator": tags.get("operator"),
            "lamp_type": tags.get("lamp_type"),
            "lamp_mount": tags.get("lamp_mount"),
            "lamp_model": tags.get("lamp_model"),
            "light_source": tags.get("light_source"),
            "height": tags.get("height"),
            "direction": tags.get("direction")
        })

    nearby_street_lamps.sort(
        key=lambda lamp: (
            lamp[
                "approximate_distance_along_route_meters"
            ],
            lamp["osm_id"] or 0
        )
    )

    previous_distance_along_route = None

    for lamp in nearby_street_lamps:
        current_distance_along_route = lamp[
            "approximate_distance_along_route_meters"
        ]

        if previous_distance_along_route is None:
            lamp[
                "approximate_spacing_from_previous_lamp_meters"
            ] = None
        else:
            lamp[
                "approximate_spacing_from_previous_lamp_meters"
            ] = round(
                max(
                    0.0,
                    current_distance_along_route
                    - previous_distance_along_route
                ),
                1
            )

        previous_distance_along_route = (
            current_distance_along_route
        )

    retrieval_debug = {
        "raw_street_lamps_found": len(raw_elements),
        "street_lamps_near_route": len(
            nearby_street_lamps
        ),
        "lamps_with_reference_number": sum(
            1
            for lamp in nearby_street_lamps
            if lamp["ref"]
        ),
        "lamps_with_model_or_type_data": sum(
            1
            for lamp in nearby_street_lamps
            if lamp["lamp_model"]
            or lamp["lamp_type"]
            or lamp["light_source"]
        )
    }

    return nearby_street_lamps, retrieval_debug



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

@app.post("/safety/nasa-night-lights/analyse")
def analyse_nasa_night_lights(
    request: NasaNightLightsAnalysisRequest
):
    if len(request.route_geometry) < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                "At least two route coordinates "
                "are required."
            )
        )

    route_samples = densify_route(
        request.route_geometry,
        interval_meters=(
            request.sample_interval_meters
        )
    )

    analysis = analyse_nasa_route_samples(
        route_samples=route_samples,
        requested_month=request.month
    )

    return {
        "status": "nasa_night_lights_analysed",
        "sample_interval_meters": (
            request.sample_interval_meters
        ),
        **analysis
    }

@app.post("/safety/lighting/combined")
def analyse_combined_lighting(
    request: CombinedLightingAnalysisRequest
):
    if len(request.route_geometry) < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                "At least two route coordinates "
                "are required."
            )
        )

    route_samples = densify_route(
        request.route_geometry,
        interval_meters=(
            request.sample_interval_meters
        )
    )

    # Each source is attempted separately.
    # One unavailable source should not make the
    # complete combined endpoint fail.
    source_errors = {}

    official_streetlights = []
    official_debug = {}

    try:
        (
            official_streetlights,
            official_debug
        ) = fetch_valencia_streetlights(
            request.route_geometry
        )

    except HTTPException as error:
        source_errors[
            "official_valencia_lamps"
        ] = error.detail

    osm_lit_ways = []
    osm_lit_debug = {}

    try:
        (
            osm_lit_ways,
            osm_lit_debug
        ) = fetch_osm_lit_ways_near_route(
            route_coordinates=(
                request.route_geometry
            ),
            match_radius_meters=(
                request.osm_lit_match_radius_meters
            )
        )

    except HTTPException as error:
        source_errors[
            "osm_lit"
        ] = error.detail

    osm_street_lamps = []
    osm_lamp_debug = {}

    try:
        (
            osm_street_lamps,
            osm_lamp_debug
        ) = fetch_osm_street_lamps_near_route(
            route_coordinates=(
                request.route_geometry
            ),
            coverage_radius_meters=(
                request.osm_lamp_radius_meters
            )
        )

    except HTTPException as error:
        source_errors[
            "osm_individual_lamps"
        ] = error.detail

    nasa_analysis = None

    try:
        nasa_analysis = (
            analyse_nasa_route_samples(
                route_samples=route_samples,
                requested_month=request.month
            )
        )

    except HTTPException as error:
        source_errors[
            "nasa_background"
        ] = error.detail

    combined_analysis = (
        combine_lighting_sources(
            route_samples=route_samples,
            official_streetlights=(
                official_streetlights
            ),
            osm_lit_ways=osm_lit_ways,
            osm_street_lamps=(
                osm_street_lamps
            ),
            nasa_analysis=nasa_analysis,
            official_coverage_radius_meters=(
                request.official_lamp_radius_meters
            ),
            osm_lit_match_radius_meters=(
                request.osm_lit_match_radius_meters
            ),
            osm_lamp_coverage_radius_meters=(
                request.osm_lamp_radius_meters
            )
        )
    )

    return {
        "status": (
            "combined_lighting_analysed"
        ),
        "sample_interval_meters": (
            request.sample_interval_meters
        ),
        "month_requested": request.month,
        "month_used": (
            nasa_analysis.get("month")
            if nasa_analysis
            else None
        ),
        "source_availability": {
            "official_valencia_lamps": (
                "available"
                if official_streetlights
                else "missing_or_unavailable"
            ),
            "osm_lit": (
                "available"
                if osm_lit_ways
                else "missing_or_unavailable"
            ),
            "osm_individual_lamps": (
                "available"
                if osm_street_lamps
                else "missing_or_unavailable"
            ),
            "nasa_background": (
                "available"
                if nasa_analysis
                else "unavailable"
            )
        },
        "source_counts": {
            "official_streetlights_near_route": (
                len(official_streetlights)
            ),
            "osm_lit_ways_near_route": len(
                osm_lit_ways
            ),
            "osm_individual_lamps_near_route": (
                len(osm_street_lamps)
            ),
            "nasa_unique_cells": (
                nasa_analysis.get(
                    "unique_nasa_cell_count"
                )
                if nasa_analysis
                else 0
            )
        },
        "source_debug": {
            "official_valencia_lamps": (
                official_debug
            ),
            "osm_lit": osm_lit_debug,
            "osm_individual_lamps": (
                osm_lamp_debug
            ),
            "nasa_memory_cache_hit": (
                nasa_analysis.get(
                    "memory_cache_hit"
                )
                if nasa_analysis
                else None
            )
        },
        "source_errors": source_errors,
        **combined_analysis
    }

@app.post("/safety/lighting/compare")
def compare_lighting_routes(
    request: LightingComparisonRequest
):
    if len(request.routes) < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                "At least two route candidates "
                "are required."
            )
        )

    if len(request.routes) > 5:
        raise HTTPException(
            status_code=400,
            detail=(
                "A maximum of five route candidates "
                "can be compared at once."
            )
        )

    for route in request.routes:
        if len(route.geometry) < 2:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Route {route.route_id} must contain "
                    "at least two coordinates."
                )
            )

    route_geometries = [
        route.geometry
        for route in request.routes
    ]

    shared_osm_padding = max(
        request.osm_lit_match_radius_meters,
        request.osm_lamp_radius_meters
    )

    shared_source_errors = {}

    shared_lit_way_elements = []
    shared_street_lamp_elements = []

    shared_osm_debug = {
        "cache_hit": False,
        "overpass_server": None,
        "bbox": None,
        "raw_lit_ways_found": 0,
        "raw_street_lamps_found": 0
    }

    try:
        shared_osm_data = (
            fetch_shared_osm_lighting_data(
                route_geometries=route_geometries,
                padding_meters=shared_osm_padding
            )
        )

        shared_lit_way_elements = (
            shared_osm_data[
                "lit_way_elements"
            ]
        )

        shared_street_lamp_elements = (
            shared_osm_data[
                "street_lamp_elements"
            ]
        )

        shared_osm_debug = (
            shared_osm_data["debug"]
        )

    except HTTPException as error:
        shared_source_errors[
            "osm_lit"
        ] = error.detail

        shared_source_errors[
            "osm_individual_lamps"
        ] = error.detail

    route_results = []

    for route in request.routes:
        route_samples = densify_route(
            route.geometry,
            interval_meters=(
                request.sample_interval_meters
            )
        )

        source_errors = dict(
            shared_source_errors
        )

        official_streetlights = []
        official_debug = {}

        try:
            (
                official_streetlights,
                official_debug
            ) = fetch_valencia_streetlights(
                route.geometry
            )

        except HTTPException as error:
            source_errors[
                "official_valencia_lamps"
            ] = error.detail

        (
            osm_lit_ways,
            osm_lit_debug
        ) = filter_shared_osm_lit_ways_for_route(
            route_coordinates=route.geometry,
            shared_lit_way_elements=(
                shared_lit_way_elements
            ),
            match_radius_meters=(
                request.osm_lit_match_radius_meters
            )
        )

        (
            osm_street_lamps,
            osm_lamp_debug
        ) = (
            filter_shared_osm_street_lamps_for_route(
                route_coordinates=route.geometry,
                shared_street_lamp_elements=(
                    shared_street_lamp_elements
                ),
                coverage_radius_meters=(
                    request.osm_lamp_radius_meters
                )
            )
        )

        nasa_analysis = None

        try:
            nasa_analysis = (
                analyse_nasa_route_samples(
                    route_samples=route_samples,
                    requested_month=request.month
                )
            )

        except HTTPException as error:
            source_errors[
                "nasa_background"
            ] = error.detail

        combined_analysis = (
            combine_lighting_sources(
                route_samples=route_samples,
                official_streetlights=(
                    official_streetlights
                ),
                osm_lit_ways=osm_lit_ways,
                osm_street_lamps=(
                    osm_street_lamps
                ),
                nasa_analysis=nasa_analysis,
                official_coverage_radius_meters=(
                    request.official_lamp_radius_meters
                ),
                osm_lit_match_radius_meters=(
                    request.osm_lit_match_radius_meters
                ),
                osm_lamp_coverage_radius_meters=(
                    request.osm_lamp_radius_meters
                )
            )
        )

        route_results.append({
            "route_id": route.route_id,
            "estimated_time_minutes": (
                route.estimated_time_minutes
            ),
            "distance_meters": (
                route.distance_meters
            ),
            "sample_interval_meters": (
                request.sample_interval_meters
            ),
            "month_requested": request.month,
            "month_used": (
                nasa_analysis.get("month")
                if nasa_analysis
                else None
            ),
            "source_availability": {
                "official_valencia_lamps": (
                    "available"
                    if official_streetlights
                    else "missing_or_unavailable"
                ),
                "osm_lit": (
                    "unavailable"
                    if "osm_lit" in source_errors
                    else (
                        "available"
                        if osm_lit_ways
                        else "no_mapped_data_found"
                    )
                ),
                "osm_individual_lamps": (
                    "unavailable"
                    if (
                        "osm_individual_lamps"
                        in source_errors
                    )
                    else (
                        "available"
                        if osm_street_lamps
                        else "no_mapped_data_found"
                    )
                ),
                "nasa_background": (
                    "available"
                    if nasa_analysis
                    else "unavailable"
                )
            },
            "source_counts": {
                "official_streetlights_near_route": (
                    len(official_streetlights)
                ),
                "osm_lit_ways_near_route": (
                    len(osm_lit_ways)
                ),
                "osm_individual_lamps_near_route": (
                    len(osm_street_lamps)
                ),
                "nasa_unique_cells": (
                    nasa_analysis.get(
                        "unique_nasa_cell_count"
                    )
                    if nasa_analysis
                    else 0
                )
            },
            "source_debug": {
                "official_valencia_lamps": (
                    official_debug
                ),
                "osm_lit": osm_lit_debug,
                "osm_individual_lamps": (
                    osm_lamp_debug
                ),
                "nasa_memory_cache_hit": (
                    nasa_analysis.get(
                        "memory_cache_hit"
                    )
                    if nasa_analysis
                    else None
                )
            },
            "source_errors": source_errors,
            **combined_analysis
        })

    return {
        "status": "lighting_routes_compared",
        "route_count": len(route_results),
        "shared_osm_download": {
            "one_download_used_for_all_routes": True,
            **shared_osm_debug
        },
        "routes": route_results,
        "interpretation": (
            "All route candidates were analysed using "
            "the same shared OpenStreetMap lighting "
            "download. This avoids making separate "
            "Overpass requests for every route."
        )
    }

@app.get("/health")
def health_check():
    return {
        "status": "Meili backend is running"
    }

@app.post("/safety/osm-street-lamps/area-scan")
def scan_osm_street_lamps_in_area(
    request: OsmStreetLampAreaScanRequest
):
    if request.south >= request.north:
        raise HTTPException(
            status_code=400,
            detail="south must be lower than north."
        )

    if request.west >= request.east:
        raise HTTPException(
            status_code=400,
            detail="west must be lower than east."
        )

    latitude_span = request.north - request.south
    longitude_span = request.east - request.west

    if latitude_span > 1 or longitude_span > 1:
        raise HTTPException(
            status_code=400,
            detail=(
                "The scan area is too large. "
                "Use a box no larger than one degree."
            )
        )

    bbox = (
        f"{request.south},"
        f"{request.west},"
        f"{request.north},"
        f"{request.east}"
    )

    count_query = f"""
    [out:json][timeout:60];
    node["highway"="street_lamp"]({bbox});
    out count;
    """

    count_data = None
    retrieval_errors = []

    for overpass_url in OVERPASS_API_URLS:
        try:
            response = requests.post(
                overpass_url,
                data={"data": count_query},
                headers={
                    "Accept": "application/json",
                    "User-Agent": (
                        "Meili safety-routing prototype"
                    )
                },
                timeout=75
            )

            response.raise_for_status()
            count_data = response.json()
            break

        except (
            requests.RequestException,
            ValueError
        ) as error:
            retrieval_errors.append(
                f"{overpass_url}: {error}"
            )

    if count_data is None:
        raise HTTPException(
            status_code=502,
            detail=(
                "OpenStreetMap street-lamp data "
                "could not be retrieved: "
                + " | ".join(retrieval_errors)
            )
        )

    total_lamps = 0

    for element in count_data.get("elements", []):
        if element.get("type") != "count":
            continue

        tags = element.get("tags", {})

        try:
            total_lamps = int(
                tags.get("nodes")
                or tags.get("total")
                or 0
            )
        except (TypeError, ValueError):
            total_lamps = 0

    sample_lamps = []

    if total_lamps > 0 and request.sample_limit > 0:
        sample_query = f"""
        [out:json][timeout:60];
        node["highway"="street_lamp"]({bbox});
        out body {request.sample_limit};
        """

        sample_data = None

        for overpass_url in OVERPASS_API_URLS:
            try:
                response = requests.post(
                    overpass_url,
                    data={"data": sample_query},
                    headers={
                        "Accept": "application/json",
                        "User-Agent": (
                            "Meili safety-routing prototype"
                        )
                    },
                    timeout=75
                )

                response.raise_for_status()
                sample_data = response.json()
                break

            except (
                requests.RequestException,
                ValueError
            ):
                continue

        if sample_data is not None:
            for element in sample_data.get(
                "elements",
                []
            ):
                latitude = element.get("lat")
                longitude = element.get("lon")

                if (
                    latitude is None
                    or longitude is None
                ):
                    continue

                tags = element.get("tags", {})

                sample_lamps.append({
                    "osm_id": element.get("id"),
                    "latitude": float(latitude),
                    "longitude": float(longitude),
                    "ref": tags.get("ref"),
                    "operator": tags.get("operator"),
                    "lamp_type": tags.get(
                        "lamp_type"
                    ),
                    "lamp_model": tags.get(
                        "lamp_model"
                    ),
                    "light_source": tags.get(
                        "light_source"
                    )
                })

    return {
        "status": "osm_street_lamp_area_scanned",
        "source": "OpenStreetMap contributors",
        "source_license": "ODbL",
        "osm_tag_used": "highway=street_lamp",
        "bounding_box": {
            "south": request.south,
            "west": request.west,
            "north": request.north,
            "east": request.east
        },
        "total_mapped_street_lamps_in_area": (
            total_lamps
        ),
        "has_any_mapped_street_lamps": (
            total_lamps > 0
        ),
        "sample_lamps": sample_lamps,
        "interpretation": (
            "This checks the entire bounding box. "
            "Zero means no individual street-lamp "
            "points are mapped in OSM inside the box, "
            "not that no physical lamps exist."
        )
    }

@app.post("/safety/osm-street-lamps/analyse")
def analyse_osm_street_lamps(
    request: OsmStreetLampAnalysisRequest
):
    if len(request.route_geometry) < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                "At least two route coordinates "
                "are required."
            )
        )

    route_samples = densify_route(
        request.route_geometry,
        interval_meters=(
            request.sample_interval_meters
        )
    )

    street_lamps, retrieval_debug = (
        fetch_osm_street_lamps_near_route(
            route_coordinates=(
                request.route_geometry
            ),
            coverage_radius_meters=(
                request.coverage_radius_meters
            )
        )
    )

    sample_results = []
    known_distances = []

    for sample in route_samples:
        if street_lamps:
            nearest_lamp = min(
                street_lamps,
                key=lambda lamp: distance_meters(
                    sample,
                    (
                        lamp["longitude"],
                        lamp["latitude"]
                    )
                )
            )

            nearest_distance = distance_meters(
                sample,
                (
                    nearest_lamp["longitude"],
                    nearest_lamp["latitude"]
                )
            )

            known_distances.append(nearest_distance)

            sample_results.append({
                "longitude": sample[0],
                "latitude": sample[1],
                "covered_by_mapped_osm_lamp": (
                    nearest_distance
                    <= request.coverage_radius_meters
                ),
                "nearest_osm_lamp_id": nearest_lamp[
                    "osm_id"
                ],
                "distance_to_nearest_osm_lamp_meters": round(
                    nearest_distance,
                    1
                )
            })
        else:
            sample_results.append({
                "longitude": sample[0],
                "latitude": sample[1],
                "covered_by_mapped_osm_lamp": False,
                "nearest_osm_lamp_id": None,
                "distance_to_nearest_osm_lamp_meters": None
            })

    covered_sample_count = sum(
        result["covered_by_mapped_osm_lamp"]
        for result in sample_results
    )

    sorted_known_distances = sorted(known_distances)

    median_distance = None

    if sorted_known_distances:
        middle = len(sorted_known_distances) // 2

        if len(sorted_known_distances) % 2 == 1:
            median_distance = sorted_known_distances[
                middle
            ]
        else:
            median_distance = (
                sorted_known_distances[middle - 1]
                + sorted_known_distances[middle]
            ) / 2

    spacing_values = [
        lamp[
            "approximate_spacing_from_previous_lamp_meters"
        ]
        for lamp in street_lamps
        if lamp[
            "approximate_spacing_from_previous_lamp_meters"
        ] is not None
    ]

    sorted_spacing_values = sorted(spacing_values)
    median_spacing = None

    if sorted_spacing_values:
        middle = len(sorted_spacing_values) // 2

        if len(sorted_spacing_values) % 2 == 1:
            median_spacing = sorted_spacing_values[middle]
        else:
            median_spacing = (
                sorted_spacing_values[middle - 1]
                + sorted_spacing_values[middle]
            ) / 2

    total_samples = len(sample_results)

    return {
        "status": "osm_street_lamps_analysed",
        "source": "OpenStreetMap contributors",
        "source_license": "ODbL",
        "osm_tag_used": "highway=street_lamp",
        "coverage_radius_meters": (
            request.coverage_radius_meters
        ),
        "sample_interval_meters": (
            request.sample_interval_meters
        ),
        "route_sample_count": total_samples,
        "osm_street_lamps_found_near_route": len(
            street_lamps
        ),
        "covered_sample_count": covered_sample_count,
        "covered_sample_percentage": round(
            100
            * covered_sample_count
            / total_samples
        ),
        "median_distance_to_nearest_osm_lamp_meters": (
            round(median_distance, 1)
            if median_distance is not None
            else None
        ),
        "maximum_distance_to_nearest_osm_lamp_meters": (
            round(max(known_distances), 1)
            if known_distances
            else None
        ),
        "approximate_median_spacing_between_mapped_lamps_meters": (
            round(median_spacing, 1)
            if median_spacing is not None
            else None
        ),
        "approximate_maximum_spacing_between_mapped_lamps_meters": (
            round(max(spacing_values), 1)
            if spacing_values
            else None
        ),
        "street_lamps": street_lamps,
        "sample_results": sample_results,
        "retrieval_debug": retrieval_debug,
        "data_confidence": "supplementary_incomplete",
        "interpretation": (
            "These are individual street lamps mapped "
            "by OpenStreetMap contributors. Missing lamp "
            "points do not prove that no lamp exists. "
            "Coverage and spacing are calculated from "
            "the mapped points near the route."
        )
    }


@app.post("/safety/osm-lighting/analyse")
def analyse_osm_lighting(
    request: OsmLightingAnalysisRequest
):
    if len(request.route_geometry) < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                "At least two route coordinates "
                "are required."
            )
        )

    route_samples = densify_route(
        request.route_geometry,
        interval_meters=(
            request.sample_interval_meters
        )
    )

    lit_ways, retrieval_debug = (
        fetch_osm_lit_ways_near_route(
            route_coordinates=(
                request.route_geometry
            ),
            match_radius_meters=(
                request.match_radius_meters
            )
        )
    )

    sample_results = []

    for sample in route_samples:
        nearest_way = None
        nearest_distance = None

        for way in lit_ways:
            distance_to_way = min(
                distance_meters(
                    sample,
                    way_sample
                )
                for way_sample in way[
                    "sampled_geometry"
                ]
            )

            if (
                nearest_distance is None
                or distance_to_way < nearest_distance
            ):
                nearest_distance = distance_to_way
                nearest_way = way

        if (
            nearest_way is not None
            and nearest_distance is not None
            and nearest_distance
            <= request.match_radius_meters
        ):
            sample_results.append({
                "longitude": sample[0],
                "latitude": sample[1],
                "lighting_evidence": nearest_way[
                    "lit_classification"
                ],
                "lit_value": nearest_way[
                    "lit_value"
                ],
                "matched_osm_way_id": nearest_way[
                    "osm_id"
                ],
                "distance_to_matched_way_meters": round(
                    nearest_distance,
                    1
                )
            })
        else:
            sample_results.append({
                "longitude": sample[0],
                "latitude": sample[1],
                "lighting_evidence": "unknown",
                "lit_value": None,
                "matched_osm_way_id": None,
                "distance_to_matched_way_meters": None
            })

    evidence_counts = {
        evidence_class: sum(
            1
            for result in sample_results
            if result["lighting_evidence"]
            == evidence_class
        )
        for evidence_class in (
            "lit",
            "unlit",
            "conditional_or_other",
            "unknown"
        )
    }

    total_samples = len(sample_results)

    public_lit_ways = [
        {
            key: value
            for key, value in way.items()
            if key != "sampled_geometry"
        }
        for way in lit_ways
    ]

    return {
        "status": "osm_lighting_analysed",
        "source": "OpenStreetMap contributors",
        "source_license": "ODbL",
        "match_radius_meters": (
            request.match_radius_meters
        ),
        "sample_interval_meters": (
            request.sample_interval_meters
        ),
        "route_sample_count": total_samples,
        "mapped_lighting_sample_count": (
            total_samples
            - evidence_counts["unknown"]
        ),
        "mapped_lighting_percentage": round(
            100
            * (
                total_samples
                - evidence_counts["unknown"]
            )
            / total_samples
        ),
        "lit_sample_count": evidence_counts["lit"],
        "unlit_sample_count": evidence_counts[
            "unlit"
        ],
        "conditional_or_other_sample_count": (
            evidence_counts[
                "conditional_or_other"
            ]
        ),
        "unknown_sample_count": evidence_counts[
            "unknown"
        ],
        "matched_lit_ways": public_lit_ways,
        "sample_results": sample_results,
        "retrieval_debug": retrieval_debug,
        "data_confidence": "supplementary",
        "interpretation": (
            "OSM lit tags provide mapped lighting "
            "evidence near the route. A missing tag "
            "is treated as unknown, not unlit. The "
            "tag does not measure brightness or "
            "whether a lamp currently works."
        )
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
