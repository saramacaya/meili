from datetime import date
from pathlib import Path

import earthaccess
from dotenv import load_dotenv


CONCEPT_ID = "C3860061042-LAADS"

VALENCIAN_COMMUNITY_BOUNDS = (
    -1.6,  # west
    37.8,  # south
    0.6,   # east
    40.9   # north
)

MONTHS_TO_DOWNLOAD = [
    (2026, 1),
    (2026, 2),
    (2026, 3),
    (2026, 4)
]

RAW_DIRECTORY = Path("nasa_data/raw")


def nasa_month_code(
    year: int,
    month: int
) -> str:
    """
    Example:
    1 May 2026 becomes A2026121.
    """

    first_day = date(year, month, 1)

    return "A" + first_day.strftime("%Y%j")


def granule_filename(granule) -> str:
    links = granule.data_links()

    if not links:
        return ""

    return links[0].split("/")[-1]


def main():
    load_dotenv()

    authentication = earthaccess.login(
        strategy="environment"
    )

    if not authentication.authenticated:
        raise RuntimeError(
            "NASA authentication failed."
        )

    print("NASA authentication successful.")
    print("Searching NASA catalogue...")

    # The search range is deliberately broader.
    # We filter the returned files by their exact
    # AYYYYDDD monthly code afterward.
    search_results = earthaccess.search_data(
        concept_id=CONCEPT_ID,
        bounding_box=(
            VALENCIAN_COMMUNITY_BOUNDS
        ),
        temporal=(
            "2026-01-01",
            "2026-05-02"
        ),
        count=100
    )

    print(
        "Catalogue results found:",
        len(search_results)
    )

    for year, month in MONTHS_TO_DOWNLOAD:
        month_label = f"{year}-{month:02d}"

        required_code = nasa_month_code(
            year,
            month
        )

        # Deduplicate files using the filename.
        matching_granules = {}

        for granule in search_results:
            filename = granule_filename(
                granule
            )

            if (
                filename
                and f".{required_code}." in filename
            ):
                matching_granules[
                    filename
                ] = granule

        month_granules = list(
            matching_granules.values()
        )

        print()
        print(
            f"{month_label}: "
            f"{len(month_granules)} tile(s) found"
        )

        for filename in sorted(
            matching_granules
        ):
            print(" ", filename)

        if len(month_granules) != 4:
            print(
                "WARNING: Expected four geographic "
                "tiles. This month will not be "
                "downloaded until it is checked."
            )
            continue

        month_directory = (
            RAW_DIRECTORY / month_label
        )

        month_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        existing_filenames = {
            file_path.name
            for file_path in month_directory.glob(
                "*.h5"
            )
        }

        missing_granules = [
            granule
            for granule in month_granules
            if granule_filename(granule)
            not in existing_filenames
        ]

        if not missing_granules:
            print(
                "All four files are already "
                "downloaded."
            )
            continue

        print(
            "Downloading",
            len(missing_granules),
            "file(s)..."
        )

        downloaded_files = earthaccess.download(
            missing_granules,
            str(month_directory)
        )

        print(
            f"{month_label} download complete."
        )

        for downloaded_file in downloaded_files:
            print(" ", downloaded_file)

    print()
    print("Historical download step finished.")


if __name__ == "__main__":
    main()