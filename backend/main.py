import copy
import asyncio
from typing import Literal, Optional
import os
import requests
import uuid
import httpx
import math
import threading
import time


from dotenv import load_dotenv

# Render's network cannot always reach IPv6 addresses ("Network is
# unreachable", Errno 101). Force IPv4 for every outbound request.
import socket as _socket
import urllib3.util.connection as _urllib3_connection
_urllib3_connection.allowed_gai_family = lambda: _socket.AF_INET
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from supabase import create_client
from datetime import datetime
from nasa_route_analysis import analyse_nasa_route_samples, load_nasa_month_data

from combined_lighting_analysis import combine_lighting_sources
from open_establishments import (
    ActivePlacesAnalysisRequest,
    analyse_active_places,
    build_active_places_response,
    densify_route,
    distance_meters,
    filter_shared_places_for_route,
    flatten_geojson_coordinates,
    normalise_text,
    router as open_establishments_router,
    score_places_for_route,
)

from street_scoring import assign_route_labels, value_route_sections
import hashlib
import hmac
import json as _json
from collections import OrderedDict
import tile_cache
import local_context_gatherer
import score_calibration
from street_scoring import _classification as _score_class, _colour as _score_colour
from fast_geo import RouteIndex

import area_search
from social_context_analysis import (
    analyse_social_context,
    load_all_overlays,
)

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
from meili_auth import SupabaseAuthMiddleware

# Added before CORS so CORS wraps it: a 401 then still carries CORS headers
# and the app can read it (and refresh the session) instead of a network error.
app.add_middleware(SupabaseAuthMiddleware)

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

class UserCreateRequest(BaseModel):
    user_type: str

class PlaceAutocompleteRequest(BaseModel):
    input: str
    session_token: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    # The user's device language (e.g. "es", "it-IT"). Optional: without it
    # Google infers the language from what was typed.
    language: Optional[str] = None


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

class SocialContextAnalysisRequest(BaseModel):
    # city is accepted only for backward compatibility with older
    # frontend builds. It is never used: evidence selection is always
    # driven by route_geometry so a wrong/missing city can no longer
    # hide evidence (this was the Parque Ribalta bug).
    city: Optional[str] = None
    route_geometry: list[tuple[float, float]]
    departure_datetime: datetime
    route_duration_seconds: int = Field(..., gt=0)

class StreetScoringRequest(BaseModel):
    # See SocialContextAnalysisRequest.city -- accepted for backward
    # compatibility only, never used for evidence selection.
    city: Optional[str] = None
    route_geometry: list[tuple[float, float]]
    departure_datetime: datetime
    route_duration_seconds: int = Field(..., ge=60, le=21_600)
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


# Priority 1/4: a coarse box around Valencia city, used only to skip a
# pointless network call to Valencia's own official streetlight dataset
# when a route is clearly nowhere near Valencia (e.g. Castellon,
# Alicante). It intentionally errs generous -- it must never exclude a
# real Valencia route, it only needs to exclude routes that obviously
# cannot be Valencia.
VALENCIA_OFFICIAL_COVERAGE_BOUNDS = {
    "south": 39.35,
    "west": -0.48,
    "north": 39.58,
    "east": -0.28,
}


def _route_intersects_valencia_official_coverage(
    route_coordinates: list[tuple[float, float]]
) -> bool:
    bounds = VALENCIA_OFFICIAL_COVERAGE_BOUNDS
    return any(
        bounds["west"] <= longitude <= bounds["east"]
        and bounds["south"] <= latitude <= bounds["north"]
        for longitude, latitude in route_coordinates
    )


def fetch_valencia_streetlights(
    route_coordinates: list[tuple[float, float]]
) -> tuple[list[tuple[float, float]], dict]:
    if not _route_intersects_valencia_official_coverage(route_coordinates):
        return [], {
            "skipped_reason": "route_outside_valencia_official_coverage_area",
            "object_ids_found": 0,
            "raw_features_downloaded": 0,
            "download_batches": 0,
            "recognised_streetlights": 0,
            "rejected_objects_count": 0,
            "rejected_element_categories": []
        }

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
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
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
    route_index = RouteIndex(route_samples, match_radius_meters, distance_meters)

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

        nearest_route_distance = route_index.nearest_within_any(way_samples)

        if nearest_route_distance is None:
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
    lamp_index = RouteIndex(route_samples, coverage_radius_meters, distance_meters)

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

        nearest_route_distance = lamp_index.nearest_within(lamp_coordinate)

        if nearest_route_distance is None:
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

@app.get("/system/keepalive")
def keepalive_check():
    """
    Touches Supabase with a trivial read so the project's own activity
    tracker sees real usage and doesn't auto-pause it after a period of
    inactivity. Meant to be hit periodically by a scheduled job (see
    .github/workflows/keepalive.yml), not by the app itself.
    """
    if supabase is None:
        return {
            "status": "keepalive_skipped",
            "message": "Supabase is not configured on this backend."
        }

    try:
        response = (
            supabase
            .table("users")
            .select("id")
            .limit(1)
            .execute()
        )

        return {
            "status": "keepalive_ok",
            "supabase_reachable": True,
            "rows_seen": len(response.data or [])
        }

    except Exception as error:
        return {
            "status": "keepalive_failed",
            "supabase_reachable": False,
            "message": str(error)
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


async def _google_autocomplete_request(payload: dict) -> dict:
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
        "X-Goog-FieldMask": (
            "suggestions.placePrediction.placeId,"
            "suggestions.placePrediction.text.text,"
            "suggestions.placePrediction.structuredFormat.mainText.text,"
            "suggestions.placePrediction.structuredFormat.secondaryText.text,"
            "suggestions.placePrediction.types,"
            "suggestions.placePrediction.distanceMeters"
        )
    }
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
    return google_response.json()


def _parse_place_suggestions(google_data: dict) -> list:
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
            "types": prediction.get("types", []),
            # Present when the request included an origin point.
            "distance_meters": prediction.get("distanceMeters"),
        })
    return suggestions


async def _google_text_search_request(payload: dict) -> dict:
    """Google Places Text Search: tolerant of typos, other languages and
    descriptive queries ("eiffel tower", "torre eiffel", "sagrada familla")."""
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
        "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.types,places.location",
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        google_response = await client.post(
            "https://places.googleapis.com/v1/places:searchText",
            json=payload,
            headers=headers,
        )
    if google_response.status_code != 200:
        raise HTTPException(status_code=google_response.status_code, detail=google_response.text)
    return google_response.json()


def _parse_text_search_places(google_data: dict) -> list:
    suggestions = []
    for place in google_data.get("places", []):
        name = (place.get("displayName") or {}).get("text")
        address = place.get("formattedAddress")
        location = place.get("location") or {}
        suggestions.append({
            "place_id": place.get("id"),
            "name": name or address,
            "address": address,
            "label": ", ".join(part for part in (name, address) if part),
            "types": place.get("types", []),
            "latitude": location.get("latitude"),
            "longitude": location.get("longitude"),
        })
    return suggestions


def _merge_suggestions(*groups: list, limit: int = 6) -> list:
    """Keeps order of the groups, drops duplicates by place id."""
    merged, seen = [], set()
    for group in groups:
        for suggestion in group:
            place_id = suggestion.get("place_id")
            if not place_id or place_id in seen:
                continue
            seen.add(place_id)
            merged.append(suggestion)
    return merged[:limit]


@app.get("/areas/search")
def search_avoid_areas(q: str, lat: Optional[float] = None, lng: Optional[float] = None, language: Optional[str] = None):
    """Neighbourhoods matching q (worldwide), with their outline or a circle."""
    if len(q.strip()) < 2:
        raise HTTPException(status_code=400, detail="Type at least 2 characters.")
    if (lat is None) != (lng is None) or (lat is not None and not (-90 <= lat <= 90 and -180 <= lng <= 180)):
        raise HTTPException(status_code=400, detail="lat and lng must be given together and be valid.")
    try:
        return {"results": area_search.search_areas(q, lat, lng, language)}
    except requests.RequestException as error:
        raise HTTPException(status_code=502, detail=f"Neighbourhood search is unavailable: {type(error).__name__}")


@app.post("/places/autocomplete")
async def autocomplete_places(request: PlaceAutocompleteRequest):
    # Collapse stray spaces so "  sagrada   familia " behaves like the clean text.
    search_text = " ".join(request.input.split())

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

    # No fixed language or country: Google matches names in whatever language
    # was typed, and results come back in the user's own language if known.
    base_payload = {"input": search_text, "sessionToken": session_token}
    if request.latitude is not None and request.longitude is not None:
        # Google then returns each suggestion's distance from this point.
        base_payload["origin"] = {"latitude": request.latitude, "longitude": request.longitude}
    language = (request.language or "").strip()
    if language:
        base_payload["languageCode"] = language

    has_bias_point = request.latitude is not None and request.longitude is not None
    nearby_circle = (
        {"circle": {"center": {"latitude": request.latitude, "longitude": request.longitude},
                    "radius": 50000.0}}
        if has_bias_point else None
    )

    async def nearby_matches() -> list:
        if not has_bias_point:
            return []
        payload = dict(base_payload, locationRestriction=nearby_circle)
        return _parse_place_suggestions(await _google_autocomplete_request(payload))

    async def anywhere_matches() -> list:
        payload = dict(base_payload)
        if has_bias_point:
            payload["locationBias"] = nearby_circle
        return _parse_place_suggestions(await _google_autocomplete_request(payload))

    async def forgiving_matches() -> list:
        payload = {"textQuery": search_text, "pageSize": 5}
        if language:
            payload["languageCode"] = language
        if has_bias_point:
            payload["locationBias"] = nearby_circle
        return _parse_text_search_places(await _google_text_search_request(payload))

    try:
        # Nearby and worldwide matches at the same time: close places come
        # first, but a famous place far away is no longer hidden by a weak
        # nearby match.
        results = await asyncio.gather(nearby_matches(), anywhere_matches(), return_exceptions=True)
        nearby, anywhere = [r if isinstance(r, list) else [] for r in results]
        if all(isinstance(r, Exception) for r in results):
            raise results[1]

        suggestions = _merge_suggestions(nearby[:3], anywhere, nearby[3:])

        # Few or no prefix matches usually means a typo, another language or a
        # description: ask Google's more forgiving full-text search too.
        if len(suggestions) < 3 and len(search_text) >= 3:
            try:
                suggestions = _merge_suggestions(suggestions, await forgiving_matches())
            except HTTPException:
                pass  # keep whatever autocomplete found

        if has_bias_point:
            for suggestion in suggestions:
                if suggestion.get("distance_meters") is None and suggestion.get("latitude") is not None:
                    suggestion["distance_meters"] = round(distance_meters(
                        (request.longitude, request.latitude),
                        (suggestion["longitude"], suggestion["latitude"]),
                    ))

        return {
            "status": "success",
            "suggestions": suggestions,
            "session_token": session_token,
            "distance_from": "origin" if has_bias_point else None,
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

def _decode_polyline(encoded: str) -> list[tuple[float, float]]:
    """Google encoded polyline -> [(lon, lat), ...]"""
    points, index, lat, lng = [], 0, 0, 0
    while encoded and index < len(encoded):
        for is_lng in (False, True):
            shift, result = 0, 0
            while True:
                b = ord(encoded[index]) - 63
                index += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            delta = ~(result >> 1) if result & 1 else result >> 1
            if is_lng:
                lng += delta
            else:
                lat += delta
        points.append((lng / 1e5, lat / 1e5))
    return points


def _route_overlap(a: list[tuple[float, float]], b: list[tuple[float, float]], near_m: float = 30.0) -> float:
    """Share of route a's points that lie within near_m of route b."""
    if not a or not b:
        return 0.0
    a_s = a[:: max(1, len(a) // 120)]
    b_s = b[:: max(1, len(b) // 300)]
    close = sum(1 for p in a_s if min(distance_meters(p, q) for q in b_s) <= near_m)
    return close / len(a_s)


def _via_points(origin: tuple[float, float], destination: tuple[float, float]) -> list[tuple[float, float]]:
    """Two points beside the middle of the direct line (left and right), to make
    Google suggest genuinely different streets."""
    (lon1, lat1), (lon2, lat2) = origin, destination
    straight = distance_meters(origin, destination)
    if straight < 300:
        return []
    offset_m = max(150.0, min(1500.0, 0.3 * straight))
    mid_lat, mid_lon = (lat1 + lat2) / 2, (lon1 + lon2) / 2
    metres_per_deg_lat = 110_540.0
    metres_per_deg_lon = 111_320.0 * math.cos(math.radians(mid_lat))
    dx, dy = (lon2 - lon1) * metres_per_deg_lon, (lat2 - lat1) * metres_per_deg_lat
    length = math.hypot(dx, dy) or 1.0
    px, py = -dy / length, dx / length          # unit vector perpendicular to the line
    return [
        (mid_lon + side * px * offset_m / metres_per_deg_lon, mid_lat + side * py * offset_m / metres_per_deg_lat)
        for side in (1, -1)
    ]


MAX_ROUTE_CANDIDATES = 5
MAX_DETOUR_RATIO = 1.5


def _metres_between(a: tuple[float, float], b: tuple[float, float]) -> float:
    return distance_meters(a, b)


@app.post("/routes/preview-real")
async def preview_real_routes(request: RealRoutePreviewRequest):
    if not GOOGLE_ROUTES_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Google Routes API is not configured."
        )

    def point(lat: float, lon: float) -> dict:
        return {"location": {"latLng": {"latitude": lat, "longitude": lon}}}

    base_payload = {
        "origin": point(request.origin.latitude, request.origin.longitude),
        "destination": point(request.destination.latitude, request.destination.longitude),
        "travelMode": "WALK",
        "languageCode": "en",
        "units": "METRIC",
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

    # Google's normal alternatives (up to 3), plus two routes forced through a
    # point left and right of the direct line, all requested at the same time.
    origin_xy = (request.origin.longitude, request.origin.latitude)
    destination_xy = (request.destination.longitude, request.destination.latitude)
    payloads = [dict(base_payload, computeAlternativeRoutes=True)]
    if _metres_between(origin_xy, destination_xy) <= MAX_WALK_DISTANCE_METERS:
        for lon, lat in _via_points(origin_xy, destination_xy):
            payloads.append(dict(base_payload, computeAlternativeRoutes=False,
                                 intermediates=[{**point(lat, lon), "via": True}]))

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            responses = await asyncio.gather(
                *(client.post(GOOGLE_ROUTES_URL, json=p, headers=headers) for p in payloads),
                return_exceptions=True,
            )
    except httpx.RequestError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Could not contact Google Routes: {str(error)}"
        )

    main_response = responses[0]
    if isinstance(main_response, Exception):
        raise HTTPException(status_code=502, detail=f"Could not contact Google Routes: {main_response}")
    if main_response.status_code != 200:
        raise HTTPException(
            status_code=main_response.status_code,
            detail=main_response.text
        )

    google_routes = [dict(r, _meili_source="google") for r in main_response.json().get("routes", [])]
    for extra in responses[1:]:
        if not isinstance(extra, Exception) and extra.status_code == 200:
            google_routes.extend(dict(r, _meili_source="extra") for r in extra.json().get("routes", []))

    if not google_routes:
        raise HTTPException(
            status_code=404,
            detail="No walking route was found between these locations."
        )

    # Remove near-duplicates and big detours; keep at most 5, fastest first.
    google_routes.sort(key=lambda r: google_duration_to_seconds(r.get("duration", "")))
    fastest_seconds = max(1, google_duration_to_seconds(google_routes[0].get("duration", "")))
    kept: list[tuple[dict, list]] = []
    for route in google_routes:
        seconds = google_duration_to_seconds(route.get("duration", ""))
        if kept and seconds > MAX_DETOUR_RATIO * fastest_seconds:
            continue
        geometry = _decode_polyline(route.get("polyline", {}).get("encodedPolyline") or "")
        def same_walk(other_route: dict, other_geometry: list) -> bool:
            a, b = _route_overlap(geometry, other_geometry), _route_overlap(other_geometry, geometry)
            other_seconds = max(1, google_duration_to_seconds(other_route.get("duration", "")))
            similar_time = abs(seconds - other_seconds) / other_seconds <= 0.05
            # Mostly the same streets, or largely the same streets and the same time.
            return (a > 0.8 and b > 0.8) or (a > 0.65 and b > 0.65 and similar_time)

        if any(same_walk(other_route, other) for other_route, other in kept):
            continue
        kept.append((route, geometry))
        if len(kept) >= MAX_ROUTE_CANDIDATES:
            break
    google_routes = [route for route, _ in kept]

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
            # "google" = Google's own alternatives (score these first);
            # "extra" = routes forced left/right of the direct line (score after).
            "source": route.get("_meili_source", "google"),
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

@app.post("/safety/social-context/analyse")
def analyse_route_social_context(
    request: SocialContextAnalysisRequest
):
    try:
        # request.city is accepted for backward compatibility only and
        # is passed through purely for auditability in the response;
        # it never controls which evidence is loaded (Priority 1/2).
        return analyse_social_context(
            route_geometry=request.route_geometry,
            departure_datetime=request.departure_datetime,
            route_duration_seconds=request.route_duration_seconds,
            city=request.city,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )
    except RuntimeError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


@app.get("/social-context/overlays")
def get_social_context_overlays(
    south: Optional[float] = None,
    west: Optional[float] = None,
    north: Optional[float] = None,
    east: Optional[float] = None,
):
    """Returns flagged public-opinion areas for the map.

    No city parameter: overlays for every registered municipality are
    returned by default (Priority 7 -- overlays should stay visible for
    the current geographic view). Pass south/west/north/east to limit
    the result to the current map viewport instead of the whole region.
    """
    try:
        bbox = None
        if None not in (south, west, north, east):
            bbox = (west, south, east, north)
        return load_all_overlays(bbox=bbox)
    except RuntimeError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )

@app.post("/routes/generate")
def generate_routes():
    # Retired: the app no longer uses this endpoint, and it used demo data for
    # unknown trips. Kept as a clear error instead of silently wrong answers.
    raise HTTPException(status_code=410, detail="This endpoint is retired: route generation.")

@app.post("/safety/street-score/analyse")
def analyse_street_score(request: StreetScoringRequest):
    if len(request.route_geometry) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least two route coordinates are required.",
        )

    lighting = analyse_combined_lighting(
        CombinedLightingAnalysisRequest(
            route_geometry=request.route_geometry,
            month=request.month,
        )
    )

    active_places = analyse_active_places(
        ActivePlacesAnalysisRequest(
            route_geometry=request.route_geometry,
            evaluation_datetime=request.departure_datetime,
            route_duration_seconds=request.route_duration_seconds,
        )
    )

    social_context = analyse_social_context(
        route_geometry=request.route_geometry,
        departure_datetime=request.departure_datetime,
        route_duration_seconds=request.route_duration_seconds,
        city=request.city,
    )

    valuation = value_route_sections(
        lighting_analysis=lighting,
        active_places_analysis=active_places,
        social_context_analysis=social_context,
        departure_datetime=request.departure_datetime,
        route_duration_seconds=request.route_duration_seconds,
    )

    return {
        "status": "street_score_analysed",
        "city_ids_checked": social_context.get("city_ids_checked", []),
        **valuation,
        "component_summaries": {
            "lighting_mean_score": lighting[
                "combined_score_statistics"
            ]["mean_score"],
            "route_activity_score": active_places[
                "route_activity_score"
            ],
            "social_context_adjustment_points": social_context[
                "score_adjustment_points"
            ],
        },
        "component_details": {
            "lighting": lighting,
            "active_places": active_places,
            "social_context": social_context,
        },
    }

@app.post("/routes/select")
def select_route():
    # Retired: the app no longer uses this endpoint, and it used demo data for
    # unknown trips. Kept as a clear error instead of silently wrong answers.
    raise HTTPException(status_code=410, detail="This endpoint is retired: route selection.")

@app.post("/feedback")
def submit_feedback():
    # Retired: the app no longer uses this endpoint, and it used demo data for
    # unknown trips. Kept as a clear error instead of silently wrong answers.
    raise HTTPException(status_code=410, detail="This endpoint is retired: feedback (now saved by the app in its own database).")

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


# ---------------------------------------------------------------------------
# Priority 3: one combined three-route comparison endpoint.
#
# Each route candidate used to require separate calls to lighting, active-
# places and social-context analysis, each of which (for lighting and
# active-places) made its own live OpenStreetMap/ArcGIS/NASA requests with
# no sharing between routes. For three routes that meant repeating the same
# OSM downloads three times over, which is exactly what produced the
# five-to-ten-minute waits described in the first real run. This endpoint
# downloads OSM lighting data once and OSM places data once for a bounding
# box covering every route being compared, then reuses those shared
# downloads (and the already-cached NASA regional grid) for every route's
# lighting/activity/social-context scoring. Safest/Fastest/Balanced labels
# are assigned only once every route has a complete score (Priority 6).
# ---------------------------------------------------------------------------

class RouteComparisonCandidate(BaseModel):
    route_id: str
    geometry: list[tuple[float, float]]
    estimated_time_minutes: int = Field(..., gt=0)
    distance_meters: float = Field(..., ge=0)


class RouteComparisonRequest(BaseModel):
    routes: list[RouteComparisonCandidate]
    departure_datetime: datetime

    sample_interval_meters: float = Field(default=15, ge=5, le=50)
    official_lamp_radius_meters: float = Field(default=25, ge=5, le=100)
    osm_lit_match_radius_meters: float = Field(default=15, ge=5, le=50)
    osm_lamp_radius_meters: float = Field(default=25, ge=5, le=100)
    activity_search_radius_meters: float = Field(default=75, ge=25, le=200)

    # Leave empty to use the latest NASA month in Supabase.
    month: Optional[str] = None

    # Accepted for backward compatibility only; ignored. See
    # SocialContextAnalysisRequest.city for why.
    city: Optional[str] = None



# ---- Fast-path caches -------------------------------------------------------

ROUTE_RESULT_CACHE: "OrderedDict[str, tuple[float, dict, dict]]" = OrderedDict()
ROUTE_RESULT_TTL_SECONDS = 1800
HEAVY_ROUTE_CACHE: "OrderedDict[str, tuple[float, tuple]]" = OrderedDict()


def _route_cache_key(route, request) -> str:
    """Same walk, same 15-minute departure window -> same analysis."""
    slot = request.departure_datetime.replace(minute=(request.departure_datetime.minute // 15) * 15,
                                              second=0, microsecond=0)
    geometry = ";".join(f"{lon:.5f},{lat:.5f}" for lon, lat in route.geometry)
    raw = f"{geometry}|{route.estimated_time_minutes}|{slot.isoformat()}|{request.sample_interval_meters}"
    return hashlib.sha256(raw.encode()).hexdigest()

MAX_WALK_DISTANCE_METERS = 25_000   # about 5 hours on foot

tile_cache.VALENCIA_LAMP_FETCHER = lambda corners: fetch_valencia_streetlights(corners)

COMPARISON_CACHE: "OrderedDict[str, tuple[float, dict]]" = OrderedDict()
COMPARISON_CACHE_LOCK = threading.Lock()
COMPARISON_CACHE_TTL_SECONDS = 30 * 60
COMPARISON_CACHE_LIMIT = 200


def _comparison_cache_key(request: "RouteComparisonRequest") -> str:
    """Same routes + same 15-minute departure window = same answer."""
    departure = request.departure_datetime
    window = departure.replace(minute=(departure.minute // 15) * 15, second=0, microsecond=0)
    payload = {
        "routes": [
            [route.route_id, route.estimated_time_minutes,
             [[round(lon, 5), round(lat, 5)] for lon, lat in route.geometry]]
            for route in request.routes
        ],
        "window": window.isoformat(),
        "month": request.month,
        "params": [
            request.sample_interval_meters, request.official_lamp_radius_meters,
            request.osm_lit_match_radius_meters, request.osm_lamp_radius_meters,
            request.activity_search_radius_meters,
        ],
    }
    return hashlib.sha256(_json.dumps(payload).encode()).hexdigest()


def _comparison_cache_get(key: str):
    with COMPARISON_CACHE_LOCK:
        entry = COMPARISON_CACHE.get(key)
        if entry is None or time.time() - entry[0] > COMPARISON_CACHE_TTL_SECONDS:
            return None
        COMPARISON_CACHE.move_to_end(key)
        return entry[1]


def _comparison_cache_put(key: str, response: dict) -> None:
    # Do not cache answers where map data failed to load: try again next time.
    # (NASA is optional and regional, so its absence does not block caching.)
    blocking = {"osm_lit", "osm_individual_lamps", "active_places", "official_valencia_lamps"}
    routes = response.get("routes", [])
    if any(blocking & set(route.get("source_errors") or {}) for route in routes):
        return
    # Never cache incomplete answers: failed routes, partial map data, or local
    # context still being gathered (otherwise every refresh would get the same
    # unfinished answer back for 30 minutes).
    if not response.get("data_complete", True) or response.get("result") != "complete":
        return
    if any(route.get("status") == "analysis_failed" for route in routes):
        return
    if any(((route.get("component_details") or {}).get("social_context") or {}).get("local_context_status") == "gathering"
           for route in routes):
        return
    with COMPARISON_CACHE_LOCK:
        COMPARISON_CACHE[key] = (time.time(), response)
        while len(COMPARISON_CACHE) > COMPARISON_CACHE_LIMIT:
            COMPARISON_CACHE.popitem(last=False)


class TilePrefillRequest(BaseModel):
    south: float
    west: float
    north: float
    east: float


@app.post("/tiles/prefill")
def prefill_tiles(request: TilePrefillRequest, x_prefill_token: Optional[str] = Header(default=None)):
    """Fills the tile cache for an area in the background (e.g. a whole city)."""
    expected = os.getenv("PREFILL_TOKEN")
    # No token configured means prefill is switched off, not open to everyone.
    if not expected:
        raise HTTPException(status_code=503, detail="Prefill is not configured (PREFILL_TOKEN missing).")
    if not x_prefill_token or not hmac.compare_digest(x_prefill_token, expected):
        raise HTTPException(status_code=401, detail="Invalid prefill token.")
    if request.north - request.south > 0.3 or request.east - request.west > 0.3:
        raise HTTPException(status_code=400, detail="Area too large: keep each side under 0.3 degrees.")
    return tile_cache.start_prefill((request.south, request.west, request.north, request.east))


class TileWarmRequest(BaseModel):
    origin_latitude: float = Field(..., ge=-90, le=90)
    origin_longitude: float = Field(..., ge=-180, le=180)
    destination_latitude: float = Field(..., ge=-90, le=90)
    destination_longitude: float = Field(..., ge=-180, le=180)


@app.post("/tiles/warm")
def warm_tiles(request: TileWarmRequest):
    """Called by the app as soon as origin and destination are chosen, so map
    tiles download while Google is still computing the walking routes."""
    straight = distance_meters(
        (request.origin_longitude, request.origin_latitude),
        (request.destination_longitude, request.destination_latitude),
    )
    if straight > MAX_WALK_DISTANCE_METERS:
        return {"status": "too_far_to_walk", "straight_line_meters": round(straight)}
    return tile_cache.warm_tiles_for_trip(
        (request.origin_longitude, request.origin_latitude),
        (request.destination_longitude, request.destination_latitude),
    )


@app.get("/tiles/diagnose")
def diagnose_tiles():
    """Open in a browser: tests whether this server can reach each
    OpenStreetMap (Overpass) server right now."""
    return tile_cache.diagnose()


@app.get("/diagnose")
def diagnose_everything():
    """Open in a browser: one page that checks every outside service Meili
    depends on (map data, place names, news, AI model, database)."""
    return {
        "map_data": tile_cache.diagnose(),
        "local_context": local_context_gatherer.diagnose(),
    }


@app.get("/tiles/prefill/status")
def prefill_tiles_status():
    return tile_cache.prefill_status()


@app.on_event("startup")
def warm_up_on_startup():
    """Loads the latest NASA month in the background so the first route is fast."""
    def warm():
        try:
            load_nasa_month_data()
        except Exception:
            pass
    threading.Thread(target=warm, daemon=True, name="warm-up").start()



def _sections_without_map_data(route_samples, area, radius_meters) -> list[int]:
    """Indices (0-9) of route sections that cross a map tile that did not load."""
    loaded = area["osm_tiles"]
    missing_sections = []
    count = len(route_samples)
    for section in range(10):
        start = section * count // 10
        end = max(start + 1, (section + 1) * count // 10)
        points = route_samples[start:end]
        if not points:
            continue
        needed = tile_cache.tiles_for_route_corridor(points, radius_meters)
        if any(tile not in loaded for tile in needed):
            missing_sections.append(section)
    return missing_sections


def _route_data_confidence(lighting_analysis: dict, active_places_analysis: dict) -> dict:
    """How much real evidence the score rests on, in plain terms."""
    counts = lighting_analysis.get("classification_counts") or {}
    total = sum(counts.values()) or 1
    lighting_coverage = round(100 * (total - counts.get("unknown", 0)) / total)
    unavailable = len(active_places_analysis.get("unavailable_sections") or [])
    if active_places_analysis.get("data_unavailable"):
        unavailable = 10
    activity_coverage = 100 - unavailable * 10
    combined = 0.6 * lighting_coverage + 0.4 * activity_coverage
    if combined >= 70:
        level = "high"
    elif combined >= 45:
        level = "medium"
    elif combined >= 20:
        level = "low"
    else:
        level = "very_low"
    return {
        "level": level,
        "lighting_evidence_coverage_percentage": lighting_coverage,
        "activity_data_coverage_percentage": activity_coverage,
        "explanation": (
            f"Lighting evidence covers {lighting_coverage}% of the route; "
            f"business data loaded for {activity_coverage}% of it. "
            "Where there is no evidence the score stays neutral (50)."
        ),
    }



def _pick_display_routes(label_inputs: list[dict], label_assignment: dict, count: int = 3) -> list[str]:
    """Fastest, safest and best balance first; then the best remaining routes."""
    chosen: list[str] = []
    for key in ("fastest_route_id", "safest_route_id", "balanced_route_id"):
        route_id = label_assignment.get(key)
        if route_id and route_id not in chosen:
            chosen.append(route_id)
    if not label_inputs:
        return chosen
    fastest = min(r["estimated_time_minutes"] for r in label_inputs) or 1
    remaining = sorted(
        (r for r in label_inputs if r["route_id"] not in chosen),
        key=lambda r: r["route_value_score"] - 2.0 * (r["estimated_time_minutes"] / fastest - 1) * 100 / 10,
        reverse=True,
    )
    for route in remaining:
        if len(chosen) >= count:
            break
        chosen.append(route["route_id"])
    return chosen[:count]


@app.post("/safety/routes/compare")
def compare_routes(request: RouteComparisonRequest):
    # A single route (Google often returns only one for long walks) is scored
    # normally and simply holds every label.
    if len(request.routes) < 1:
        raise HTTPException(
            status_code=400,
            detail="At least one route is required."
        )

    # Meili is for walking: trips longer than about 5 hours on foot are not
    # analysed (they would need hundreds of map tiles and are not walks).
    walkable = [r for r in request.routes if (r.distance_meters or 0) <= MAX_WALK_DISTANCE_METERS]
    if not walkable:
        shortest = min(request.routes, key=lambda r: r.distance_meters or 0)
        return {
            "status": "too_far_to_walk",
            "route_count": len(request.routes),
            "max_walk_distance_meters": MAX_WALK_DISTANCE_METERS,
            "shortest_route_meters": shortest.distance_meters,
            "shortest_route_minutes": shortest.estimated_time_minutes,
            "data_complete": True,
            "refresh_recommended": False,
            "labels": {"labels_by_route_id": {}, "safest_route_id": None, "fastest_route_id": None,
                       "balanced_route_id": None, "default_route_id": None},
            "routes": [
                {"route_id": r.route_id, "estimated_time_minutes": r.estimated_time_minutes,
                 "distance_meters": r.distance_meters, "status": "too_far_to_walk",
                 "error": "This trip is too long to walk.", "labels": []}
                for r in request.routes
            ],
        }
    if len(walkable) < len(request.routes):
        request = request.model_copy(update={"routes": walkable})

    if len(request.routes) > 5:
        raise HTTPException(
            status_code=400,
            detail="A maximum of five route candidates can be compared at once."
        )

    for route in request.routes:
        if len(route.geometry) < 2:
            raise HTTPException(
                status_code=400,
                detail=f"Route {route.route_id} must contain at least two coordinates."
            )

    cache_key = _comparison_cache_key(request)
    cached_response = _comparison_cache_get(cache_key)
    if cached_response is not None:
        return cached_response

    route_geometries = [route.geometry for route in request.routes]
    # Load only the map tiles these routes touch (memory -> Supabase -> source).
    data_padding_meters = max(
        request.official_lamp_radius_meters * 2,
        request.osm_lit_match_radius_meters * 2,
        request.osm_lamp_radius_meters * 2,
        request.activity_search_radius_meters,
    ) + 25
    area = tile_cache.load_area(route_geometries, data_padding_meters)

    # Local context: stored records near these routes (reviewed + automatic),
    # and background gathering for areas Meili knows nothing about yet.
    all_points = [p for g in route_geometries for p in g]
    context_bbox = (
        min(p[1] for p in all_points), min(p[0] for p in all_points),
        max(p[1] for p in all_points), max(p[0] for p in all_points),
    )
    auto_records = local_context_gatherer.load_records_near(context_bbox)
    first_route = request.routes[0].geometry
    middle_point = first_route[len(first_route) // 2]
    calibration_cell = score_calibration.cell_for(middle_point[0], middle_point[1])
    calibration_samples = score_calibration.load_samples(calibration_cell)
    raw_scores_to_record: list[tuple[str, float]] = []
    local_context_job = local_context_gatherer.ensure_context(route_geometries)

    shared_source_errors: dict[str, str] = {}
    if area["osm_error"]:
        shared_source_errors["osm_lit"] = area["osm_error"]
        shared_source_errors["osm_individual_lamps"] = area["osm_error"]
        shared_source_errors["active_places"] = area["osm_error"]
    if area["valencia_error"]:
        shared_source_errors["official_valencia_lamps"] = area["valencia_error"]

    route_results = []
    label_inputs = []

    for route in request.routes:
        # Already scored in the last 30 minutes (e.g. the app's first, faster
        # request with fewer routes): reuse it instead of scoring again.
        route_key = _route_cache_key(route, request)
        cached_route = ROUTE_RESULT_CACHE.get(route_key)
        if cached_route and time.time() - cached_route[0] < ROUTE_RESULT_TTL_SECONDS:
            result_copy = copy.deepcopy(cached_route[1])
            result_copy["route_id"] = route.route_id
            route_results.append(result_copy)
            label_inputs.append(dict(cached_route[2], route_id=route.route_id))
            continue
        try:
            route_duration_seconds = route.estimated_time_minutes * 60
            # Heavy part (lighting, lamps, places, satellite data) cached per walk;
            # refresh checks while local context is gathered only redo the light part.
            heavy = HEAVY_ROUTE_CACHE.get(route_key)
            if heavy and time.time() - heavy[0] < ROUTE_RESULT_TTL_SECONDS:
                (active_places_analysis, lighting_analysis, official_debug, osm_lamp_debug, osm_lit_debug, source_errors) = copy.deepcopy(heavy[1])
            else:
                route_samples = densify_route(
                    route.geometry,
                    interval_meters=request.sample_interval_meters
                )
                source_errors = dict(shared_source_errors)

                route_data = tile_cache.route_slice(area, route.geometry, data_padding_meters)

                lamps_near_route = tile_cache.corridor_filter(
                    route_data["official_lamps"],
                    route_samples,
                    request.official_lamp_radius_meters * 2 + 10,
                )
                # With no lamp near the route, the scorer still gets the area's
                # lamps so it reads "data available, no lamp close" rather than
                # "no data". Counts shown to users use lamps_near_route only.
                official_streetlights = lamps_near_route or route_data["official_lamps"]
                official_debug = (
                    {"source": "tile_cache", "streetlights_near_route": len(lamps_near_route)}
                    if area["valencia_covered"]
                    else {"skipped_reason": "route_outside_valencia_official_coverage_area"}
                )

                osm_lit_ways, osm_lit_debug = filter_shared_osm_lit_ways_for_route(
                    route_coordinates=route.geometry,
                    shared_lit_way_elements=route_data["lit_ways"],
                    match_radius_meters=request.osm_lit_match_radius_meters
                )

                osm_street_lamps, osm_lamp_debug = filter_shared_osm_street_lamps_for_route(
                    route_coordinates=route.geometry,
                    shared_street_lamp_elements=route_data["street_lamps"],
                    coverage_radius_meters=request.osm_lamp_radius_meters
                )

                nasa_analysis = None

                try:
                    nasa_analysis = analyse_nasa_route_samples(
                        route_samples=route_samples,
                        requested_month=request.month
                    )
                except HTTPException as error:
                    source_errors["nasa_background"] = error.detail

                lighting_analysis = combine_lighting_sources(
                    route_samples=route_samples,
                    official_streetlights=official_streetlights,
                    osm_lit_ways=osm_lit_ways,
                    osm_street_lamps=osm_street_lamps,
                    nasa_analysis=nasa_analysis,
                    official_coverage_radius_meters=request.official_lamp_radius_meters,
                    osm_lit_match_radius_meters=request.osm_lit_match_radius_meters,
                    osm_lamp_coverage_radius_meters=request.osm_lamp_radius_meters
                )

                raw_places, places_debug = filter_shared_places_for_route(
                    route_coordinates=route.geometry,
                    shared_elements=route_data["places"],
                    search_radius_meters=request.activity_search_radius_meters
                )

                scored_places = score_places_for_route(
                    places=raw_places,
                    route_geometry=route.geometry,
                    evaluation_datetime=request.departure_datetime,
                    route_duration_seconds=route_duration_seconds,
                    search_radius_meters=request.activity_search_radius_meters
                )

                active_places_analysis = build_active_places_response(
                    places=scored_places,
                    evaluation_datetime=request.departure_datetime
                )
                active_places_analysis["search_radius_meters"] = request.activity_search_radius_meters
                active_places_analysis["retrieval_debug"] = places_debug
                if "active_places" in source_errors:
                    # Some map tiles are missing. Only the sections that run through
                    # missing tiles are treated as unknown (neutral); the rest keep
                    # their real activity score.
                    unavailable_sections = _sections_without_map_data(
                        route_samples, area, request.activity_search_radius_meters
                    )
                    if len(unavailable_sections) == 10:
                        active_places_analysis["data_unavailable"] = True
                    else:
                        active_places_analysis["unavailable_sections"] = unavailable_sections

                no_lighting = (
                    lighting_analysis.get("combined_score_statistics", {}).get("mean_score") is None
                )
                if no_lighting and active_places_analysis.get("data_unavailable"):
                    route_results.append({
                        "route_id": route.route_id,
                        "estimated_time_minutes": route.estimated_time_minutes,
                        "distance_meters": route.distance_meters,
                        "status": "analysis_failed",
                        "error": "No safety data sources were reachable for this route.",
                        "source_errors": source_errors,
                    })
                    continue

                # What the app's explanation screen needs: which lighting sources
                # had data, and how much.
                some_map_data = not active_places_analysis.get("data_unavailable")
                osm_state = (
                    "available" if "osm_lit" not in source_errors
                    else "partial" if some_map_data
                    else "unavailable"
                )
                lighting_analysis["source_availability"] = {
                    "osm_lit": osm_state,
                    "osm_individual_lamps": osm_state,
                    "official_valencia_lamps": (
                        "not_covered" if not area["valencia_covered"]
                        else "unavailable" if "official_valencia_lamps" in source_errors
                        else "available"
                    ),
                    "nasa_background": (
                        "available"
                        if nasa_analysis and nasa_analysis.get("valid_brightness_sample_count")
                        else "not_covered"
                    ),
                }
                lighting_analysis["source_counts"] = {
                    "osm_lit_ways_near_route": osm_lit_debug.get("lit_ways_near_route", 0),
                    "osm_individual_lamps_near_route": osm_lamp_debug.get("street_lamps_near_route", 0),
                    "official_streetlights_near_route": len(lamps_near_route) if area["valencia_covered"] else 0,
                    "nasa_unique_cells": (nasa_analysis or {}).get("unique_nasa_cell_count", 0),
                }

                if not ({"osm_lit", "osm_individual_lamps", "active_places", "official_valencia_lamps"}
                        & set(source_errors)):
                    HEAVY_ROUTE_CACHE[route_key] = (time.time(), copy.deepcopy((active_places_analysis, lighting_analysis, official_debug, osm_lamp_debug, osm_lit_debug, source_errors)))
                    while len(HEAVY_ROUTE_CACHE) > 300:
                        HEAVY_ROUTE_CACHE.popitem(last=False)
            social_context_analysis = analyse_social_context(
                route_geometry=route.geometry,
                departure_datetime=request.departure_datetime,
                route_duration_seconds=route_duration_seconds,
                city=request.city,
                extra_records=auto_records,
            )
            social_context_analysis["local_context_status"] = local_context_job["status"]
            social_context_analysis["local_context_areas_checked"] = local_context_job["areas_checked"]

            valuation = value_route_sections(
                lighting_analysis=lighting_analysis,
                active_places_analysis=active_places_analysis,
                social_context_analysis=social_context_analysis,
                departure_datetime=request.departure_datetime,
                route_duration_seconds=route_duration_seconds
            )

            data_confidence = _route_data_confidence(lighting_analysis, active_places_analysis)

            # Spread scores over 0-100: rank among walks in this city, with
            # little-data routes kept near the middle (see score_calibration).
            factor = score_calibration.confidence_factor(data_confidence)
            raw_route_score = valuation["route_value_score"]
            raw_scores_to_record.append((route_key, raw_route_score))
            for section in valuation["sections"]:
                section["raw_street_value_score"] = section["street_value_score"]
                shown = score_calibration.calibrate_section(section["street_value_score"], calibration_samples, factor)
                section["street_value_score"] = shown
                section["classification"] = _score_class(shown)
                section["map_colour"] = _score_colour(shown)
            # Local reports that really touch this route (not just a faint edge).
            has_local_reports = any(
                float(item.get("weight", 0.0)) >= 0.15
                for item in (social_context_analysis.get("section_local") or [])
            )
            shown_route = score_calibration.calibrate(raw_route_score, calibration_samples, factor,
                                                      has_local_reports=has_local_reports)
            valuation["raw_route_value_score"] = raw_route_score
            valuation["route_value_score"] = shown_route
            valuation["route_classification"] = _score_class(shown_route)
            valuation["weakest_section_score"] = min(x["street_value_score"] for x in valuation["sections"])
            valuation["score_scale"] = score_calibration.describe(calibration_samples)

            label_inputs.append({
                "route_id": route.route_id,
                "route_value_score": valuation["route_value_score"],
                "estimated_time_minutes": route.estimated_time_minutes
            })

            route_results.append({
                "route_id": route.route_id,
                "estimated_time_minutes": route.estimated_time_minutes,
                "distance_meters": route.distance_meters,
                "source_errors": source_errors,
                **valuation,
                "data_confidence": data_confidence,
                "component_summaries": {
                    "lighting_mean_score": lighting_analysis["combined_score_statistics"]["mean_score"],
                    "route_activity_score": active_places_analysis["route_activity_score"],
                    "social_context_adjustment_points": social_context_analysis["score_adjustment_points"]
                },
                "component_details": {
                    "lighting": {
                        key: value
                        for key, value in lighting_analysis.items()
                        if key != "sample_results"
                    },
                    "active_places": active_places_analysis,
                    "social_context": social_context_analysis
                },
                "source_debug": {
                    "official_valencia_lamps": official_debug,
                    "osm_lit": osm_lit_debug,
                    "osm_individual_lamps": osm_lamp_debug
                }
            })
            # Remember finished, complete analyses only.
            blocking_errors = {"osm_lit", "osm_individual_lamps", "active_places", "official_valencia_lamps"}
            if (not (blocking_errors & set(source_errors))
                    and social_context_analysis.get("local_context_status") != "gathering"):
                ROUTE_RESULT_CACHE[route_key] = (time.time(), copy.deepcopy(route_results[-1]),
                                                 dict(label_inputs[-1]))
                while len(ROUTE_RESULT_CACHE) > 300:
                    ROUTE_RESULT_CACHE.popitem(last=False)

        except Exception as error:
            route_results.append({
                "route_id": route.route_id,
                "estimated_time_minutes": route.estimated_time_minutes,
                "distance_meters": route.distance_meters,
                "status": "analysis_failed",
                "error": f"{type(error).__name__}: {error}",
            })
            continue
    label_assignment = assign_route_labels(label_inputs)
    score_calibration.record(calibration_cell, raw_scores_to_record)

    for route_result in route_results:
        route_result["labels"] = label_assignment["labels_by_route_id"].get(
            route_result["route_id"], []
        )

    display_route_ids = _pick_display_routes(label_inputs, label_assignment)

    scored_count = sum(1 for r in route_results if r.get("status") != "analysis_failed")
    if scored_count == len(route_results) and scored_count > 0:
        overall = "complete"
    elif scored_count > 0:
        overall = "partial"
    else:
        overall = "failed"
    response = {
        "status": "routes_compared_and_scored",
        # complete = every route scored; partial = some failed; failed = none scored.
        "result": overall,
        "scored_route_count": scored_count,
        "route_count": len(route_results),
        "data_complete": area["complete"],
        # True when some map data was still downloading: the app should ask
        # again in a few seconds to get the full-quality scores.
        "refresh_recommended": (not area["complete"]) and area["still_downloading"],
        "data_loading": area["debug"],
        "labels": label_assignment,
        # Up to 5 routes are scored; the app shows these 3 (fastest, safest,
        # best balance, then the next best) in this order.
        "display_route_ids": display_route_ids,
        "routes": route_results,
        "interpretation": (
            "All route candidates were analysed together from cached map "
            "tiles (memory, then Supabase Storage, then the original source "
            "only for tiles never seen before). Safest, Fastest and Balanced "
            "labels are assigned only after every route has a complete score."
        )
    }
    _comparison_cache_put(cache_key, response)
    return response
