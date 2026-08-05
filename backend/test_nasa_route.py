from pathlib import Path
import json
import math

import numpy as np


CACHE_FILE = Path(
    "nasa_data/processed/"
    "vnp46a3_valencian_community_2026-06.npz"
)

OUTPUT_FILE = Path(
    "nasa_data/processed/"
    "test_route_nasa_analysis.json"
)

# Coordinate order: [longitude, latitude]
TEST_ROUTE = [
    [-0.3763, 39.4622],
    [-0.3760, 39.4650],
    [-0.3754, 39.4680],
    [-0.3757, 39.4710]
]

SAMPLE_INTERVAL_METERS = 15


def distance_meters(
    point_a: tuple[float, float],
    point_b: tuple[float, float]
) -> float:
    longitude_a, latitude_a = point_a
    longitude_b, latitude_b = point_b

    earth_radius_meters = 6_371_000

    latitude_a_radians = math.radians(
        latitude_a
    )

    latitude_b_radians = math.radians(
        latitude_b
    )

    latitude_delta = math.radians(
        latitude_b - latitude_a
    )

    longitude_delta = math.radians(
        longitude_b - longitude_a
    )

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
    interval_meters: float
) -> list[tuple[float, float]]:
    """
    Adds route checkpoints approximately every 15 metres.
    """

    if len(coordinates) < 2:
        return coordinates

    samples = [coordinates[0]]

    for start, end in zip(
        coordinates,
        coordinates[1:]
    ):
        segment_length = distance_meters(
            start,
            end
        )

        steps = max(
            1,
            math.ceil(
                segment_length
                / interval_meters
            )
        )

        for step in range(1, steps + 1):
            fraction = step / steps

            samples.append((
                start[0]
                + (
                    end[0] - start[0]
                )
                * fraction,
                start[1]
                + (
                    end[1] - start[1]
                )
                * fraction
            ))

    return samples


def nearest_index(
    values: np.ndarray,
    target: float
) -> int:
    return int(
        np.abs(values - target).argmin()
    )


def main():
    if not CACHE_FILE.exists():
        raise FileNotFoundError(
            "NASA cache not found at: "
            f"{CACHE_FILE.resolve()}"
        )

    cache = np.load(
        CACHE_FILE,
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

    route_coordinates = [
        (
            float(coordinate[0]),
            float(coordinate[1])
        )
        for coordinate in TEST_ROUTE
    ]

    route_samples = densify_route(
        coordinates=route_coordinates,
        interval_meters=(
            SAMPLE_INTERVAL_METERS
        )
    )

    quality_labels = {
        0: "good",
        1: "poor",
        2: "gap_filled",
        255: "missing"
    }

    sample_results = []
    valid_brightness_values = []
    unique_cell_keys = set()
    good_quality_count = 0

    for sample_number, (
        longitude,
        latitude
    ) in enumerate(route_samples, start=1):

        latitude_index = nearest_index(
            latitudes,
            latitude
        )

        longitude_index = nearest_index(
            longitudes,
            longitude
        )

        cell_key = (
            latitude_index,
            longitude_index
        )

        unique_cell_keys.add(cell_key)

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

        if np.isnan(brightness_value):
            brightness_output = None
        else:
            brightness_output = round(
                brightness_value,
                3
            )

            valid_brightness_values.append(
                brightness_value
            )

        if quality_code == 0:
            good_quality_count += 1

        sample_results.append({
            "sample_number": sample_number,
            "longitude": round(
                longitude,
                7
            ),
            "latitude": round(
                latitude,
                7
            ),
            "nasa_cell_latitude": round(
                float(
                    latitudes[
                        latitude_index
                    ]
                ),
                7
            ),
            "nasa_cell_longitude": round(
                float(
                    longitudes[
                        longitude_index
                    ]
                ),
                7
            ),
            "brightness_radiance": (
                brightness_output
            ),
            "quality_code": quality_code,
            "quality": quality_labels.get(
                quality_code,
                "unknown"
            ),
            "supporting_observations": (
                observation_count
            )
        })

    if valid_brightness_values:
        brightness_array = np.asarray(
            valid_brightness_values,
            dtype=np.float64
        )

        brightness_statistics = {
            "minimum_radiance": round(
                float(
                    np.min(
                        brightness_array
                    )
                ),
                3
            ),
            "median_radiance": round(
                float(
                    np.median(
                        brightness_array
                    )
                ),
                3
            ),
            "mean_radiance": round(
                float(
                    np.mean(
                        brightness_array
                    )
                ),
                3
            ),
            "maximum_radiance": round(
                float(
                    np.max(
                        brightness_array
                    )
                ),
                3
            )
        }
    else:
        brightness_statistics = {
            "minimum_radiance": None,
            "median_radiance": None,
            "mean_radiance": None,
            "maximum_radiance": None
        }

    result = {
        "status": "nasa_route_analysed",
        "product": metadata.get(
            "product"
        ),
        "month": metadata.get(
            "month"
        ),
        "radiance_units": metadata.get(
            "radiance_units"
        ),
        "sample_interval_meters": (
            SAMPLE_INTERVAL_METERS
        ),
        "route_sample_count": len(
            route_samples
        ),
        "unique_nasa_cell_count": len(
            unique_cell_keys
        ),
        "valid_brightness_sample_count": len(
            valid_brightness_values
        ),
        "good_quality_sample_count": (
            good_quality_count
        ),
        "brightness_statistics": (
            brightness_statistics
        ),
        "sample_results": sample_results,
        "interpretation": (
            "NASA brightness is background "
            "nighttime radiance at approximately "
            "500-metre resolution. It does not "
            "measure individual street lighting."
        )
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as output:
        json.dump(
            result,
            output,
            indent=2
        )

    print("NASA route analysis successful")
    print()
    print(
        "Route checkpoints:",
        result["route_sample_count"]
    )
    print(
        "Unique NASA cells:",
        result["unique_nasa_cell_count"]
    )
    print(
        "Good-quality checkpoints:",
        result["good_quality_sample_count"]
    )
    print(
        "Minimum radiance:",
        brightness_statistics[
            "minimum_radiance"
        ]
    )
    print(
        "Median radiance:",
        brightness_statistics[
            "median_radiance"
        ]
    )
    print(
        "Maximum radiance:",
        brightness_statistics[
            "maximum_radiance"
        ]
    )
    print()
    print(
        "Full result saved to:",
        OUTPUT_FILE.resolve()
    )


if __name__ == "__main__":
    main()