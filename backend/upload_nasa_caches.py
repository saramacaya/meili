from pathlib import Path
import os

from dotenv import load_dotenv
from supabase import create_client


BUCKET_NAME = "nasa-night-lights"

PROCESSED_DIRECTORY = Path(
    "nasa_data/processed"
)

STORAGE_FOLDER = (
    "valencian-community/monthly"
)


def main():
    load_dotenv()

    supabase_url = os.getenv(
        "SUPABASE_URL"
    )

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

    files = sorted(
        PROCESSED_DIRECTORY.glob(
            "vnp46a3_valencian_community_"
            "*.npz"
        )
    )

    if not files:
        raise FileNotFoundError(
            "No processed NASA monthly files "
            f"were found in "
            f"{PROCESSED_DIRECTORY.resolve()}"
        )

    print(
        f"Found {len(files)} processed "
        "NASA file(s)."
    )

    successful_uploads = 0
    failed_uploads = 0

    for file_path in files:
        storage_path = (
            f"{STORAGE_FOLDER}/"
            f"{file_path.name}"
        )

        file_size_mb = (
            file_path.stat().st_size
            / 1_000_000
        )

        print()
        print(
            f"Uploading {file_path.name} "
            f"({file_size_mb:.2f} MB)"
        )

        try:
            with file_path.open("rb") as file:
                supabase.storage.from_(
                    BUCKET_NAME
                ).upload(
                    path=storage_path,
                    file=file,
                    file_options={
                        "content-type": (
                            "application/octet-stream"
                        ),
                        "cache-control": "3600",
                        "upsert": "true"
                    }
                )

            successful_uploads += 1

            print(
                "Uploaded successfully:"
            )
            print(
                f"  {BUCKET_NAME}/"
                f"{storage_path}"
            )

        except Exception as error:
            failed_uploads += 1

            print(
                "Upload failed:"
            )
            print(
                f"  {type(error).__name__}: "
                f"{error}"
            )

    print()
    print("NASA upload finished.")
    print(
        "Successful uploads:",
        successful_uploads
    )
    print(
        "Failed uploads:",
        failed_uploads
    )

    if failed_uploads:
        raise RuntimeError(
            "One or more NASA files failed "
            "to upload."
        )


if __name__ == "__main__":
    main()