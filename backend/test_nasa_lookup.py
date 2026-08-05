from pathlib import Path
import json

import numpy as np


CACHE_FILE = Path(
    "nasa_data/processed/"
    "vnp46a3_valencian_community_2026-06.npz"
)

# Central Valencia test coordinate
TEST_LATITUDE = 39.4699
TEST_LONGITUDE = -0.3763


def find_nearest_index(
    values: np.ndarray,
    target: float
) -> int:
    return int(
        np.abs(values - target).argmin()
    )


def main():
    if not CACHE_FILE.exists():
        raise FileNotFoundError(
            f"NASA cache not found: "
            f"{CACHE_FILE.resolve()}"
        )

    data = np.load(
        CACHE_FILE,
        allow_pickle=False
    )

    latitudes = data["latitudes"]
    longitudes = data["longitudes"]
    radiance = data["radiance"]
    quality = data["quality"]
    observations = data["observations"]

    metadata = json.loads(
        data["metadata"].item()
    )

    latitude_index = find_nearest_index(
        latitudes,
        TEST_LATITUDE
    )

    longitude_index = find_nearest_index(
        longitudes,
        TEST_LONGITUDE
    )

    matched_latitude = float(
        latitudes[latitude_index]
    )

    matched_longitude = float(
        longitudes[longitude_index]
    )

    brightness_value = float(
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

    print("NASA lookup successful")
    print()
    print(
        "Requested coordinate:",
        TEST_LATITUDE,
        TEST_LONGITUDE
    )
    print(
        "Matched NASA cell:",
        matched_latitude,
        matched_longitude
    )

    if np.isnan(brightness_value):
        print("Brightness: missing")
    else:
        print(
            "Brightness:",
            round(brightness_value, 2),
            metadata.get(
                "radiance_units",
                "nW/(cm^2 sr)"
            )
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