from io import BytesIO
import json
import os
import re
import threading
import time
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from fastapi import HTTPException
from supabase import create_client


load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY")

NASA_BUCKET_NAME = "nasa-night-lights"
NASA_STORAGE_FOLDER = "valencian-community/monthly"
NASA_FILE_PREFIX = "vnp46a3_valencian_community_"
NASA_MONTH_PATTERN = re.compile(r"^(\d{4}-\d{2})$")
NASA_FILENAME_PATTERN = re.compile(
    r"^vnp46a3_valencian_community_(\d{4}-\d{2})\.npz$"
)

NASA_MONTH_DATA_CACHE: dict[str, dict] = {}
NASA_AVAILABLE_MONTHS_CACHE = {
    "months": [],
    "checked_at": 0.0,
}
NASA_AVAILABLE_MONTHS_TTL_SECONDS = 900
NASA_CACHE_LOCK = threading.Lock()

supabase_storage = None

if SUPABASE_URL and SUPABASE_SECRET_KEY:
    supabase_storage = create_client(
        SUPABASE_URL,
        SUPABASE_SECRET_KEY,
    )


def _require_storage_client():
    if supabase_storage is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "NASA Storage is not configured. "
                "SUPABASE_SECRET_KEY is missing."
            ),
        )

    return supabase_storage


def list_available_nasa_months(
    force_refresh: bool = False,
) -> list[str]:
    """
    Lists processed NASA months stored in Supabase.

    The folder listing is cached for 15 minutes. That keeps
    normal route requests fast while still allowing newly
    uploaded months to appear without restarting the backend.
    """
    storage_client = _require_storage_client()
    now = time.monotonic()

    with NASA_CACHE_LOCK:
        cached_months = list(
            NASA_AVAILABLE_MONTHS_CACHE["months"]
        )
        checked_at = float(
            NASA_AVAILABLE_MONTHS_CACHE["checked_at"]
        )

    if (
        not force_refresh
        and cached_months
        and now - checked_at
        < NASA_AVAILABLE_MONTHS_TTL_SECONDS
    ):
        return cached_months

    try:
        entries = (
            storage_client.storage
            .from_(NASA_BUCKET_NAME)
            .list(NASA_STORAGE_FOLDER)
        )
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=(
                "NASA month list could not be read "
                f"from Supabase Storage: {error}"
            ),
        )

    months = []

    for entry in entries:
        if isinstance(entry, dict):
            filename = entry.get("name")
        else:
            filename = getattr(entry, "name", None)

        if not filename:
            continue

        match = NASA_FILENAME_PATTERN.fullmatch(
            str(filename)
        )

        if match:
            months.append(match.group(1))

    months = sorted(set(months))

    if not months:
        raise HTTPException(
            status_code=503,
            detail=(
                "No processed NASA monthly caches "
                "were found in Supabase Storage."
            ),
        )

    with NASA_CACHE_LOCK:
        NASA_AVAILABLE_MONTHS_CACHE["months"] = months
        NASA_AVAILABLE_MONTHS_CACHE["checked_at"] = now

    return months


def load_nasa_month_data(
    requested_month: Optional[str] = None,
) -> tuple[str, dict, bool]:
    """
    Loads one processed NASA month from Supabase.

    Returns:
    - the resolved month;
    - the arrays and metadata;
    - whether the data was already cached in backend memory.
    """
    if requested_month is not None:
        requested_month = requested_month.strip()

        if not NASA_MONTH_PATTERN.fullmatch(
            requested_month
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "month must use YYYY-MM format, "
                    "for example 2026-06."
                ),
            )

        resolved_month = requested_month
    else:
        available_months = list_available_nasa_months()
        resolved_month = available_months[-1]

    with NASA_CACHE_LOCK:
        cached_data = NASA_MONTH_DATA_CACHE.get(
            resolved_month
        )

    if cached_data is not None:
        return resolved_month, cached_data, True

    storage_client = _require_storage_client()

    filename = (
        f"{NASA_FILE_PREFIX}"
        f"{resolved_month}.npz"
    )
    storage_path = (
        f"{NASA_STORAGE_FOLDER}/{filename}"
    )

    try:
        file_bytes = (
            storage_client.storage
            .from_(NASA_BUCKET_NAME)
            .download(storage_path)
        )
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=(
                "NASA cache could not be downloaded "
                f"for {resolved_month}: {error}"
            ),
        )

    try:
        with np.load(
            BytesIO(file_bytes),
            allow_pickle=False,
        ) as loaded_file:
            month_data = {
                "latitudes": loaded_file[
                    "latitudes"
                ].copy(),
                "longitudes": loaded_file[
                    "longitudes"
                ].copy(),
                "radiance": loaded_file[
                    "radiance"
                ].copy(),
                "quality": loaded_file[
                    "quality"
                ].copy(),
                "observations": loaded_file[
                    "observations"
                ].copy(),
                "metadata": json.loads(
                    loaded_file["metadata"].item()
                ),
            }
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=(
                "NASA cache was downloaded but could "
                f"not be read: {error}"
            ),
        )

    expected_shape = month_data["radiance"].shape

    if (
        month_data["quality"].shape != expected_shape
        or month_data["observations"].shape
        != expected_shape
    ):
        raise HTTPException(
            status_code=502,
            detail=(
                "NASA cache arrays do not have "
                "matching grid shapes."
            ),
        )

    with NASA_CACHE_LOCK:
        NASA_MONTH_DATA_CACHE[
            resolved_month
        ] = month_data

    return resolved_month, month_data, False


def _nearest_grid_index(
    values: np.ndarray,
    target: float,
) -> int:
    return int(np.abs(values - target).argmin())


def analyse_nasa_route_samples(
    route_samples: list[tuple[float, float]],
    requested_month: Optional[str] = None,
) -> dict:
    """
    Looks up monthly NASA background radiance for route points.

    Coordinate order is (longitude, latitude).
    """
    if not route_samples:
        raise HTTPException(
            status_code=400,
            detail="At least one route sample is required.",
        )

    resolved_month, nasa_data, cache_hit = (
        load_nasa_month_data(
            requested_month=requested_month
        )
    )

    latitudes = nasa_data["latitudes"]
    longitudes = nasa_data["longitudes"]
    radiance = nasa_data["radiance"]
    quality = nasa_data["quality"]
    observations = nasa_data["observations"]
    metadata = nasa_data["metadata"]

    minimum_latitude = float(np.min(latitudes))
    maximum_latitude = float(np.max(latitudes))
    minimum_longitude = float(np.min(longitudes))
    maximum_longitude = float(np.max(longitudes))

    quality_labels = {
        0: "good",
        1: "poor",
        2: "gap_filled",
        255: "missing",
    }

    sample_results = []
    valid_radiance_values = []
    unique_nasa_cells = set()
    good_quality_sample_count = 0
    missing_sample_count = 0
    outside_region_sample_count = 0

    for sample_number, sample in enumerate(
        route_samples,
        start=1,
    ):
        longitude, latitude = sample

        inside_processed_region = (
            minimum_latitude
            <= latitude
            <= maximum_latitude
            and minimum_longitude
            <= longitude
            <= maximum_longitude
        )

        if not inside_processed_region:
            outside_region_sample_count += 1

            sample_results.append({
                "sample_number": sample_number,
                "longitude": round(longitude, 7),
                "latitude": round(latitude, 7),
                "coverage_status": (
                    "outside_processed_region"
                ),
                "nasa_cell_latitude": None,
                "nasa_cell_longitude": None,
                "brightness_radiance": None,
                "quality_code": None,
                "quality": None,
                "supporting_observations": None,
            })
            continue

        latitude_index = _nearest_grid_index(
            latitudes,
            latitude,
        )
        longitude_index = _nearest_grid_index(
            longitudes,
            longitude,
        )

        cell_key = (latitude_index, longitude_index)
        unique_nasa_cells.add(cell_key)

        brightness_value = float(
            radiance[
                latitude_index,
                longitude_index,
            ]
        )
        quality_code = int(
            quality[
                latitude_index,
                longitude_index,
            ]
        )
        observation_count = int(
            observations[
                latitude_index,
                longitude_index,
            ]
        )

        has_valid_radiance = (
            np.isfinite(brightness_value)
            and quality_code != 255
        )

        if has_valid_radiance:
            brightness_output = round(
                brightness_value,
                3,
            )
            valid_radiance_values.append(
                brightness_value
            )

            if quality_code == 0:
                good_quality_sample_count += 1
        else:
            brightness_output = None
            missing_sample_count += 1

        sample_results.append({
            "sample_number": sample_number,
            "longitude": round(longitude, 7),
            "latitude": round(latitude, 7),
            "coverage_status": (
                "available"
                if has_valid_radiance
                else "missing"
            ),
            "nasa_cell_latitude": round(
                float(latitudes[latitude_index]),
                7,
            ),
            "nasa_cell_longitude": round(
                float(longitudes[longitude_index]),
                7,
            ),
            "brightness_radiance": (
                brightness_output
            ),
            "quality_code": quality_code,
            "quality": quality_labels.get(
                quality_code,
                "unknown",
            ),
            "supporting_observations": (
                observation_count
            ),
        })

    if valid_radiance_values:
        brightness_array = np.asarray(
            valid_radiance_values,
            dtype=np.float64,
        )

        brightness_statistics = {
            "minimum_radiance": round(
                float(np.min(brightness_array)),
                3,
            ),
            "median_radiance": round(
                float(np.median(brightness_array)),
                3,
            ),
            "mean_radiance": round(
                float(np.mean(brightness_array)),
                3,
            ),
            "maximum_radiance": round(
                float(np.max(brightness_array)),
                3,
            ),
        }
    else:
        brightness_statistics = {
            "minimum_radiance": None,
            "median_radiance": None,
            "mean_radiance": None,
            "maximum_radiance": None,
        }

    total_samples = len(sample_results)
    valid_sample_count = len(valid_radiance_values)

    return {
        "source": "NASA Black Marble VNP46A3",
        "storage_source": "Supabase Storage",
        "month": resolved_month,
        "month_requested": requested_month,
        "memory_cache_hit": cache_hit,
        "temporal_resolution": metadata.get(
            "temporal_resolution",
            "monthly_composite",
        ),
        "nominal_observation_time": metadata.get(
            "nominal_observation_time",
            "approximately 01:30 local solar time",
        ),
        "radiance_units": metadata.get(
            "radiance_units",
            "nW/(cm^2 sr)",
        ),
        "route_sample_count": total_samples,
        "unique_nasa_cell_count": len(
            unique_nasa_cells
        ),
        "valid_brightness_sample_count": (
            valid_sample_count
        ),
        "valid_brightness_sample_percentage": round(
            100 * valid_sample_count / total_samples
        ),
        "good_quality_sample_count": (
            good_quality_sample_count
        ),
        "missing_sample_count": (
            missing_sample_count
        ),
        "outside_processed_region_sample_count": (
            outside_region_sample_count
        ),
        "brightness_statistics": (
            brightness_statistics
        ),
        "sample_results": sample_results,
        "data_confidence": (
            "regional_background_context"
        ),
        "interpretation": (
            "NASA brightness is monthly background "
            "nighttime radiance at approximately "
            "500-metre resolution. It does not "
            "measure individual street lamps or "
            "street-level illuminance."
        ),
    }
