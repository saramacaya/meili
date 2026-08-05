from pathlib import Path
import json
import re

import h5py
import numpy as np

from pathlib import Path
from datetime import datetime

# Folder containing the files you downloaded.
RAW_DIRECTORY = Path("nasa_data/raw")



# Where the smaller processed cache will be saved.
OUTPUT_DIRECTORY = Path("nasa_data/processed")


# Temporary rectangular coverage for the Valencian Community.
# This includes some nearby sea and neighbouring territory.
WEST = -1.6
SOUTH = 37.8
EAST = 0.6
NORTH = 40.9

DATASET_NAMES = {
    "radiance": "AllAngle_Composite_Snow_Free",
    "quality": "AllAngle_Composite_Snow_Free_Quality",
    "observations": "AllAngle_Composite_Snow_Free_Num",
    "latitude": "lat",
    "longitude": "lon",
}


def find_dataset(
    hdf_file: h5py.File,
    dataset_name: str
) -> h5py.Dataset:
    """
    Finds a dataset anywhere inside the NASA HDF5 file.
    """

    matches = []

    def visitor(path, item):
        if (
            isinstance(item, h5py.Dataset)
            and path.split("/")[-1] == dataset_name
        ):
            matches.append(item)

    hdf_file.visititems(visitor)

    if not matches:
        raise KeyError(
            f"Dataset '{dataset_name}' was not found "
            f"inside {hdf_file.filename}"
        )

    if len(matches) > 1:
        raise ValueError(
            f"More than one dataset named '{dataset_name}' "
            f"was found inside {hdf_file.filename}"
        )

    return matches[0]


def identify_tile(filename: str) -> str:
    """
    Extracts a tile code such as h17v04 from the filename.
    """

    match = re.search(r"\.(h\d{2}v\d{2})\.", filename)

    if not match:
        raise ValueError(
            f"Could not identify NASA tile from {filename}"
        )

    return match.group(1)


def create_crop_slice(
    values: np.ndarray,
    minimum: float,
    maximum: float
):
    """
    Finds the continuous part of a latitude or longitude
    array that falls inside the selected regional bounds.
    """

    matching_indexes = np.where(
        (values >= minimum)
        & (values <= maximum)
    )[0]

    if matching_indexes.size == 0:
        return None

    return slice(
        int(matching_indexes.min()),
        int(matching_indexes.max()) + 1
    )


def scaled_radiance(
    dataset: h5py.Dataset,
    row_slice: slice,
    column_slice: slice
) -> np.ndarray:
    """
    Converts NASA's stored integer values into radiance.
    """

    raw_values = dataset[
        row_slice,
        column_slice
    ]

    fill_value = dataset.attrs.get(
        "_FillValue",
        65535
    )

    scale_factor = float(
        dataset.attrs.get(
            "scale_factor",
            0.1
        )
    )

    offset = float(
        dataset.attrs.get(
            "offset",
            0.0
        )
    )

    result = raw_values.astype(np.float32)

    missing_mask = raw_values == fill_value

    result = (
        result * scale_factor
        + offset
    )

    result[missing_mask] = np.nan

    return result


def load_tile(file_path: Path):
    """
    Reads and crops one NASA tile.
    """

    print(f"Reading {file_path.name}")

    with h5py.File(file_path, "r") as hdf_file:
        latitude_dataset = find_dataset(
            hdf_file,
            DATASET_NAMES["latitude"]
        )

        longitude_dataset = find_dataset(
            hdf_file,
            DATASET_NAMES["longitude"]
        )

        latitudes = np.asarray(
            latitude_dataset[:],
            dtype=np.float64
        )

        longitudes = np.asarray(
            longitude_dataset[:],
            dtype=np.float64
        )

        row_slice = create_crop_slice(
            latitudes,
            SOUTH,
            NORTH
        )

        column_slice = create_crop_slice(
            longitudes,
            WEST,
            EAST
        )

        # This tile does not overlap our selected region.
        if row_slice is None or column_slice is None:
            return None

        radiance_dataset = find_dataset(
            hdf_file,
            DATASET_NAMES["radiance"]
        )

        quality_dataset = find_dataset(
            hdf_file,
            DATASET_NAMES["quality"]
        )

        observations_dataset = find_dataset(
            hdf_file,
            DATASET_NAMES["observations"]
        )

        radiance = scaled_radiance(
            radiance_dataset,
            row_slice,
            column_slice
        )

        quality = np.asarray(
            quality_dataset[
                row_slice,
                column_slice
            ],
            dtype=np.uint8
        )

        raw_observations = observations_dataset[
            row_slice,
            column_slice
        ]

        observation_fill_value = (
            observations_dataset.attrs.get(
                "_FillValue",
                65535
            )
        )

        observations = np.asarray(
            raw_observations,
            dtype=np.uint16
        )

        observations[
            raw_observations
            == observation_fill_value
        ] = 0

        return {
            "tile": identify_tile(
                file_path.name
            ),
            "source_file": file_path.name,
            "latitudes": latitudes[row_slice],
            "longitudes": longitudes[column_slice],
            "radiance": radiance,
            "quality": quality,
            "observations": observations,
        }


def combine_tiles(tile_results):
    """
    Combines the cropped pieces into one regional grid.
    """

    all_latitudes = np.unique(
        np.concatenate([
            result["latitudes"]
            for result in tile_results
        ])
    )

    all_longitudes = np.unique(
        np.concatenate([
            result["longitudes"]
            for result in tile_results
        ])
    )

    # Latitude is stored north to south.
    all_latitudes = np.sort(
        all_latitudes
    )[::-1]

    # Longitude is stored west to east.
    all_longitudes = np.sort(
        all_longitudes
    )

    output_shape = (
        len(all_latitudes),
        len(all_longitudes)
    )

    regional_radiance = np.full(
        output_shape,
        np.nan,
        dtype=np.float32
    )

    regional_quality = np.full(
        output_shape,
        255,
        dtype=np.uint8
    )

    regional_observations = np.zeros(
        output_shape,
        dtype=np.uint16
    )

    latitude_indexes = {
        round(float(value), 8): index
        for index, value in enumerate(
            all_latitudes
        )
    }

    longitude_indexes = {
        round(float(value), 8): index
        for index, value in enumerate(
            all_longitudes
        )
    }

    for result in tile_results:
        rows = [
            latitude_indexes[
                round(float(value), 8)
            ]
            for value in result["latitudes"]
        ]

        columns = [
            longitude_indexes[
                round(float(value), 8)
            ]
            for value in result["longitudes"]
        ]

        regional_radiance[
            np.ix_(rows, columns)
        ] = result["radiance"]

        regional_quality[
            np.ix_(rows, columns)
        ] = result["quality"]

        regional_observations[
            np.ix_(rows, columns)
        ] = result["observations"]

    return {
        "latitudes": all_latitudes,
        "longitudes": all_longitudes,
        "radiance": regional_radiance,
        "quality": regional_quality,
        "observations": regional_observations,
    }

def month_label_from_code(
    month_file_code: str
) -> str:
    """
    Converts A2026121 into 2026-05.
    """

    month_date = datetime.strptime(
        month_file_code[1:],
        "%Y%j"
    )

    return month_date.strftime("%Y-%m")


def process_month(
    month_file_code: str,
    files: list[Path]
):
    month_label = month_label_from_code(
        month_file_code
    )

    print()
    print(
        f"Processing {month_label}: "
        f"{len(files)} file(s)"
    )

    tile_results = []

    for file_path in sorted(files):
        result = load_tile(file_path)

        if result is not None:
            tile_results.append(result)

    if not tile_results:
        print(
            f"No overlapping tiles found for "
            f"{month_label}. Skipping."
        )
        return

    regional_data = combine_tiles(
        tile_results
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIRECTORY
        / (
            "vnp46a3_valencian_community_"
            f"{month_label}.npz"
        )
    )

    metadata = {
        "product": "VNP46A3",
        "month": month_label,
        "month_file_code": month_file_code,
        "region": (
            "Valencian Community "
            "initial bounding box"
        ),
        "bounds": {
            "west": WEST,
            "south": SOUTH,
            "east": EAST,
            "north": NORTH
        },
        "source_files": [
            result["source_file"]
            for result in tile_results
        ],
        "tiles": [
            result["tile"]
            for result in tile_results
        ],
        "radiance_units": "nW/(cm^2 sr)",
        "quality_codes": {
            "0": "good",
            "1": "poor",
            "2": "gap_filled",
            "255": "missing"
        },
        "temporal_resolution": (
            "monthly_composite"
        ),
        "nominal_observation_time": (
            "approximately 01:30 "
            "local solar time"
        )
    }

    np.savez_compressed(
        output_file,
        latitudes=regional_data["latitudes"],
        longitudes=regional_data["longitudes"],
        radiance=regional_data["radiance"],
        quality=regional_data["quality"],
        observations=regional_data[
            "observations"
        ],
        metadata=json.dumps(metadata)
    )

    valid_pixels = np.isfinite(
        regional_data["radiance"]
    )

    good_quality_pixels = (
        regional_data["quality"] == 0
    )

    print(
        f"{month_label} processing complete."
    )

    print(
        "Tiles processed:",
        len(tile_results)
    )

    print(
        "Regional grid shape:",
        regional_data["radiance"].shape
    )

    print(
        "Valid brightness pixels:",
        int(valid_pixels.sum())
    )

    print(
        "Good-quality pixels:",
        int(good_quality_pixels.sum())
    )

    print(
        "Processed file:",
        output_file.resolve()
    )

    print(
        "Processed size:",
        round(
            output_file.stat().st_size
            / 1_000_000,
            2
        ),
        "MB"
    )


def main():
    all_files = sorted(
        RAW_DIRECTORY.rglob("*.h5")
    )

    if not all_files:
        raise FileNotFoundError(
            "No NASA HDF5 files were found inside "
            f"{RAW_DIRECTORY.resolve()}"
        )

    files_by_month = {}

    for file_path in all_files:
        match = re.search(
            r"\.(A\d{7})\.",
            file_path.name
        )

        if not match:
            print(
                "Skipping unrecognised filename:",
                file_path.name
            )
            continue

        month_file_code = match.group(1)

        files_by_month.setdefault(
            month_file_code,
            []
        ).append(file_path)

    if not files_by_month:
        raise RuntimeError(
            "NASA files were found, but none had "
            "a recognised monthly date code."
        )

    print(
        "Months detected:",
        len(files_by_month)
    )

    for month_file_code in sorted(
        files_by_month
    ):
        process_month(
            month_file_code=month_file_code,
            files=files_by_month[
                month_file_code
            ]
        )

    print()
    print("All available months processed.")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()