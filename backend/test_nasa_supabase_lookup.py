from io import BytesIO
import json
import os

import numpy as np
from dotenv import load_dotenv
from supabase import create_client


BUCKET_NAME = "nasa-night-lights"

STORAGE_PATH = (
    "valencian-community/monthly/"
    "vnp46a3_valencian_community_2026-06.npz"
)

TEST_LATITUDE = 39.4699
TEST_LONGITUDE = -0.3763


def nearest_index(
    values: np.ndarray,
    target: float
) -> int:
    return int(
        np.abs(values - target).argmin()
    )


def main():
    load_dotenv()

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_secret_key = os.getenv(
        "SUPABASE_SECRET_KEY"
    )

    if not supabase_url:
        raise RuntimeError(
            "SUPABASE_URL is missing from .env."
        )

    if not supabase_secret_key:
        raise RuntimeError(
            "SUPABASE_SECRET_KEY is missing "
            "from .env."
        )

    supabase = create_client(
        supabase_url,
        supabase_secret_key
    )

    print("Downloading NASA cache from Supabase...")

    file_bytes = (
        supabase.storage
        .from_(BUCKET_NAME)
        .download(STORAGE_PATH)
    )

    print(
        "Downloaded:",
        round(len(file_bytes) / 1_000_000, 2),
        "MB"
    )

    cache = np.load(
        BytesIO(file_bytes),
        allow_pickle=False
    )

    latitudes = cache["latitudes"]
    longitudes = cache["longitudes"]
    radiance = cache["radiance"]
    quality = cache["quality"]
    observations = cache["observations"]

    metadata = json.loads(
        cache["metadata"].item()
    )

    latitude_index = nearest_index(
        latitudes,
        TEST_LATITUDE
    )

    longitude_index = nearest_index(
        longitudes,
        TEST_LONGITUDE
    )

    brightness = float(
        radiance[
            latitude_index,
            longitude_index
        ]
    )

    quality_code = int(
        quality[
            latitude_index,
            longitude_index
        ]
    )

    observation_count = int(
        observations[
            latitude_index,
            longitude_index
        ]
    )

    quality_labels = {
        0: "good",
        1: "poor",
        2: "gap_filled",
        255: "missing"
    }

    print()
    print("Online NASA lookup successful")
    print(
        "Requested coordinate:",
        TEST_LATITUDE,
        TEST_LONGITUDE
    )
    print(
        "Matched NASA cell:",
        float(latitudes[latitude_index]),
        float(longitudes[longitude_index])
    )
    print(
        "Brightness:",
        round(brightness, 3),
        metadata.get("radiance_units")
    )
    print(
        "Quality:",
        quality_labels.get(
            quality_code,
            f"unknown code {quality_code}"
        )
    )
    print(
        "Supporting observations:",
        observation_count
    )
    print(
        "Month:",
        metadata.get("month")
    )


if __name__ == "__main__":
    main()