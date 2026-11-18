"""Builds NASA night-light tiles for the whole of Europe into Cloudflare R2.

NASA Black Marble VNP46A3 is a monthly night-time brightness map at about
500 m resolution (15 arc-seconds, 240 cells per degree). Loading all of Europe
at once would need more than 1 GB of memory, so this tool cuts each month into
1 x 1 degree squares (about 110 km x 80 km) and uploads them to R2:

    v1/nasa/<YYYY-MM>/<lat>_<lon>.npz   one square (lat/lon = its south-west corner)
    v1/nasa/index.json                  list of built months + settings

The backend (nasa_route_analysis.py) then loads only the one or two squares a
route touches, and keeps a few dozen in memory.

Each square also stores 1001 "quantiles": the brightness distribution of the
3 x 3 squares around it (about 300 km across). The lighting score uses them to
say how bright a street is compared with its surroundings, the same idea as
the old Valencian Community file, but now for every European city.

Run by .github/workflows/build-nasa-tiles.yml once a month, or by hand:

    EARTHDATA_TOKEN=... R2_ACCOUNT_ID=... R2_ACCESS_KEY_ID=... \
    R2_SECRET_ACCESS_KEY=... python backend/tools/build_nasa_tiles.py

Options:
    --month 2026-09        build this month (repeatable). Default: the last
                           three finished months that are not built yet.
    --force                rebuild months that already exist.
    --keep-months 3        delete older months from R2 (0 = keep all).
    --bounds W S E N       area to build. Default: Europe incl. Canary Islands,
                           Azores, Iceland, Cyprus and Turkey's west.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import r2_storage  # noqa: E402

CONCEPT_ID = "C3860061042-LAADS"          # VNP46A3 (same as download_nasa_history.py)
NASA_PREFIX = "v1/nasa"
INDEX_PATH = f"{NASA_PREFIX}/index.json"
TILE_DEGREES = 1
QUANTILE_COUNT = 1001
EUROPE_BOUNDS = (-32.0, 27.0, 45.0, 72.0)  # west, south, east, north

DATASET_NAMES = {
    "radiance": "AllAngle_Composite_Snow_Free",
    "quality": "AllAngle_Composite_Snow_Free_Quality",
    "observations": "AllAngle_Composite_Snow_Free_Num",
    "latitude": "lat",
    "longitude": "lon",
}


# ------------------------------------------------------------------ helpers

def month_code(month: str) -> str:
    """2026-05 -> A2026121 (NASA's year + day-of-year of the 1st)."""
    year, mon = (int(part) for part in month.split("-"))
    return "A" + date(year, mon, 1).strftime("%Y%j")


def month_end(month: str) -> str:
    year, mon = (int(part) for part in month.split("-"))
    next_first = date(year + (mon == 12), mon % 12 + 1, 1)
    return date.fromordinal(next_first.toordinal() - 1).isoformat()


def default_months(today: Optional[date] = None, count: int = 3) -> list[str]:
    """The last `count` finished months, oldest first."""
    today = today or date.today()
    year, mon = today.year, today.month
    months = []
    for _ in range(count):
        mon -= 1
        if mon == 0:
            year, mon = year - 1, 12
        months.append(f"{year}-{mon:02d}")
    return sorted(months)


def tile_key(lat_floor: int, lon_floor: int) -> str:
    return f"{lat_floor}_{lon_floor}"


def parse_key(key: str) -> tuple[int, int]:
    lat, lon = key.split("_")
    return int(lat), int(lon)


def find_dataset(hdf_file, dataset_name: str):
    import h5py

    matches = []

    def visitor(path, item):
        if isinstance(item, h5py.Dataset) and path.split("/")[-1] == dataset_name:
            matches.append(item)

    hdf_file.visititems(visitor)
    if len(matches) != 1:
        raise KeyError(f"{dataset_name}: found {len(matches)} datasets in {hdf_file.filename}")
    return matches[0]


def _scaled(dataset, raw: np.ndarray) -> np.ndarray:
    fill = dataset.attrs.get("_FillValue", 65535)
    scale = float(dataset.attrs.get("scale_factor", 0.1))
    offset = float(dataset.attrs.get("offset", 0.0))
    out = raw.astype(np.float32) * scale + offset
    out[raw == fill] = np.nan
    return out


# ------------------------------------------------------- pass 1: cut granule

def cut_granule(h5_path: Path, out_dir: Path, bounds: tuple[float, float, float, float]) -> int:
    """Cuts one 10 x 10 degree NASA granule into 1 x 1 degree squares saved in
    out_dir (uncompressed .npz). Returns how many squares were written."""
    import h5py

    west, south, east, north = bounds
    written = 0
    with h5py.File(h5_path, "r") as f:
        lats = np.asarray(find_dataset(f, DATASET_NAMES["latitude"])[:], dtype=np.float64)
        lons = np.asarray(find_dataset(f, DATASET_NAMES["longitude"])[:], dtype=np.float64)
        rad_ds = find_dataset(f, DATASET_NAMES["radiance"])
        qual_ds = find_dataset(f, DATASET_NAMES["quality"])
        obs_ds = find_dataset(f, DATASET_NAMES["observations"])

        lat_floors = np.floor(lats).astype(int)
        lon_floors = np.floor(lons).astype(int)
        for lat_floor in sorted(set(lat_floors.tolist())):
            if lat_floor + 1 <= south or lat_floor >= north:
                continue
            rows = np.where(lat_floors == lat_floor)[0]
            r0, r1 = int(rows.min()), int(rows.max()) + 1
            for lon_floor in sorted(set(lon_floors.tolist())):
                if lon_floor + 1 <= west or lon_floor >= east:
                    continue
                cols = np.where(lon_floors == lon_floor)[0]
                c0, c1 = int(cols.min()), int(cols.max()) + 1

                radiance = _scaled(rad_ds, rad_ds[r0:r1, c0:c1])
                quality = np.asarray(qual_ds[r0:r1, c0:c1], dtype=np.uint8)
                raw_obs = obs_ds[r0:r1, c0:c1]
                observations = np.asarray(raw_obs, dtype=np.uint16)
                observations[raw_obs == obs_ds.attrs.get("_FillValue", 65535)] = 0

                valid = np.isfinite(radiance) & (quality != 255)
                if not valid.any():
                    continue  # open sea / no data: no square needed
                np.savez(
                    out_dir / f"{tile_key(lat_floor, lon_floor)}.npz",
                    latitudes=lats[r0:r1], longitudes=lons[c0:c1],
                    radiance=radiance, quality=quality, observations=observations,
                )
                written += 1
    return written


# --------------------------------------------- pass 2: quantiles + upload

def _valid_values(path: Path) -> np.ndarray:
    with np.load(path) as data:
        radiance, quality = data["radiance"], data["quality"]
        return radiance[np.isfinite(radiance) & (quality != 255)].astype(np.float32)


def finish_tiles(work_dir: Path, month: str) -> dict[str, bytes]:
    """Adds neighbourhood quantiles to every square and returns compressed bytes."""
    paths = {p.stem: p for p in work_dir.glob("*.npz")}
    valid_cache: dict[str, np.ndarray] = {}

    def valid(key: str) -> Optional[np.ndarray]:
        if key not in paths:
            return None
        if key not in valid_cache:
            valid_cache[key] = _valid_values(paths[key])
        return valid_cache[key]

    probabilities = np.linspace(0.0, 1.0, QUANTILE_COUNT)
    out: dict[str, bytes] = {}
    # Walk row by row so the neighbour cache stays small.
    for key in sorted(paths, key=lambda k: parse_key(k)):
        lat, lon = parse_key(key)
        neighbours = [valid(tile_key(lat + dy, lon + dx)) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
        values = np.concatenate([v for v in neighbours if v is not None and v.size])
        quantiles = np.quantile(values, probabilities).astype(np.float32)
        with np.load(paths[key]) as data:
            buffer = io.BytesIO()
            np.savez_compressed(
                buffer,
                latitudes=data["latitudes"], longitudes=data["longitudes"],
                radiance=data["radiance"], quality=data["quality"],
                observations=data["observations"], quantiles=quantiles,
                metadata=np.asarray(json.dumps({
                    "product": "VNP46A3", "month": month, "tile": key,
                    "tile_degrees": TILE_DEGREES,
                    "quantiles_area": "3 x 3 squares around this one",
                    "radiance_units": "nW/(cm^2 sr)",
                    "quality_codes": {"0": "good", "1": "poor", "2": "gap_filled", "255": "missing"},
                    "temporal_resolution": "monthly_composite",
                    "nominal_observation_time": "approximately 01:30 local solar time",
                })),
            )
        out[key] = buffer.getvalue()
        # Drop neighbours two rows below: they are no longer needed.
        for old in [k for k in valid_cache if parse_key(k)[0] < lat - 1]:
            del valid_cache[old]
    return out


def upload(month: str, tiles: dict[str, bytes]) -> int:
    def put(item):
        key, body = item
        return r2_storage.put_bytes(f"{NASA_PREFIX}/{month}/{key}.npz", body,
                                    content_type="application/octet-stream")

    ok = 0
    with ThreadPoolExecutor(max_workers=16) as pool:
        for done in as_completed([pool.submit(put, item) for item in tiles.items()]):
            ok += bool(done.result())
    return ok


# ------------------------------------------------------------- index + prune

def read_index() -> dict:
    raw = r2_storage.get_bytes(INDEX_PATH)
    if not raw:
        return {"months": []}
    try:
        return json.loads(raw)
    except ValueError:
        return {"months": []}


def write_index(index: dict) -> None:
    index["updated_at"] = datetime.now(timezone.utc).isoformat()
    if not r2_storage.put_bytes(INDEX_PATH, json.dumps(index, indent=1).encode(),
                                content_type="application/json"):
        raise RuntimeError("Could not write the NASA index to R2.")


def delete_month(month: str) -> int:
    client = r2_storage.client()
    keys = r2_storage.list_keys(f"{NASA_PREFIX}/{month}/")
    for start in range(0, len(keys), 1000):
        client.delete_objects(Bucket=r2_storage.bucket(), Delete={
            "Objects": [{"Key": k} for k in keys[start:start + 1000]], "Quiet": True})
    return len(keys)


# ---------------------------------------------------------------- download

def find_granules(month: str, bounds) -> list:
    import earthaccess

    code = month_code(month)
    results = earthaccess.search_data(
        concept_id=CONCEPT_ID,
        bounding_box=tuple(bounds),
        temporal=(f"{month}-01", month_end(month)),
        count=500,
    )
    granules = {}
    for granule in results:
        links = granule.data_links()
        name = links[0].split("/")[-1] if links else ""
        if f".{code}." in name:
            granules[name] = granule
    return [granules[name] for name in sorted(granules)]


def build_month(month: str, bounds, work_root: Path) -> Optional[int]:
    import earthaccess

    granules = find_granules(month, bounds)
    print(f"{month}: {len(granules)} NASA granule(s) found")
    if not granules:
        print(f"{month}: not published by NASA yet, skipping.")
        return None

    raw_dir = work_root / "raw"
    tile_dir = work_root / month
    shutil.rmtree(tile_dir, ignore_errors=True)
    tile_dir.mkdir(parents=True)
    squares = 0
    for number, granule in enumerate(granules, start=1):
        shutil.rmtree(raw_dir, ignore_errors=True)
        raw_dir.mkdir(parents=True)
        for attempt in range(3):
            files = earthaccess.download([granule], str(raw_dir))
            if files:
                break
            time.sleep(10 * (attempt + 1))
        else:
            raise RuntimeError(f"Download failed: {granule.data_links()}")
        for path in files:
            written = cut_granule(Path(path), tile_dir, bounds)
            squares += written
            print(f"  [{number}/{len(granules)}] {Path(path).name}: {written} squares")
        shutil.rmtree(raw_dir, ignore_errors=True)  # granules are big; keep disk free

    if squares == 0:
        print(f"{month}: no squares with data, skipping.")
        return None
    tiles = finish_tiles(tile_dir, month)
    size_mb = sum(len(b) for b in tiles.values()) / 1e6
    print(f"{month}: uploading {len(tiles)} squares ({size_mb:.0f} MB) to R2...")
    ok = upload(month, tiles)
    shutil.rmtree(tile_dir, ignore_errors=True)
    if ok != len(tiles):
        raise RuntimeError(f"{month}: only {ok}/{len(tiles)} squares uploaded")
    return ok


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--month", action="append", help="YYYY-MM (repeatable)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-months", type=int, default=3)
    parser.add_argument("--bounds", type=float, nargs=4, metavar=("W", "S", "E", "N"),
                        default=list(EUROPE_BOUNDS))
    parser.add_argument("--work-dir", default="nasa_work")
    args = parser.parse_args(argv)

    for month in args.month or []:
        if not re.fullmatch(r"\d{4}-\d{2}", month):
            parser.error(f"--month must look like 2026-09, got {month}")
    if r2_storage.client() is None:
        print("R2 is not configured (R2_ACCOUNT_ID / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY).")
        return 1

    import earthaccess

    if not earthaccess.login(strategy="environment").authenticated:
        print("NASA Earthdata login failed: set EARTHDATA_TOKEN "
              "(or EARTHDATA_USERNAME + EARTHDATA_PASSWORD).")
        return 1

    index = read_index()
    built = set(index.get("months", []))
    wanted = sorted(set(args.month or default_months()))
    todo = [m for m in wanted if args.force or m not in built]
    print(f"Already in R2: {sorted(built) or 'none'}. To build: {todo or 'nothing'}")

    work_root = Path(args.work_dir)
    for month in todo:
        if build_month(month, args.bounds, work_root):
            built.add(month)
            index.update({
                "months": sorted(built),
                "tile_degrees": TILE_DEGREES,
                "bounds": {"west": args.bounds[0], "south": args.bounds[1],
                           "east": args.bounds[2], "north": args.bounds[3]},
                "quantile_count": QUANTILE_COUNT,
            })
            write_index(index)  # after every month, so a later failure keeps it

    if args.keep_months > 0 and len(built) > args.keep_months:
        for old in sorted(built)[:-args.keep_months]:
            print(f"Removing old month {old}: {delete_month(old)} files")
            built.discard(old)
        index["months"] = sorted(built)
        write_index(index)

    print(f"Done. Months in R2: {sorted(built)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
