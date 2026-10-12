"""Build Meili's map tiles for a whole country from an OpenStreetMap extract.

Replaces live requests to the public Overpass servers (which refuse cloud
servers) with Meili's own data:

  1. The workflow downloads a country file from Geofabrik and filters it with
     osmium to what Meili uses (lit streets, street lamps, relevant places),
     exported as GeoJSON lines.
  2. This script sorts those features into Meili's 0.01-degree tiles, bundles
     5 x 5 tiles into one "block" file (0.05 degree), and uploads the blocks
     to Supabase Storage (bucket meili-tiles, path v1/blocks/<row>_<col>.json.gz).
  3. The backend reads a block when it needs a tile. A tile missing from an
     existing block simply has no mapped features (empty, not unknown).

Blocks on a border are merged with what is already stored, so neighbouring
countries can share a block.

Usage:
  python build_country_tiles.py filtered.geojsonseq --country portugal
Environment: SUPABASE_URL, SUPABASE_SERVICE_KEY (omit both for a dry run).
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

TILE_DEGREES = 0.01
BLOCK_TILES = 5                       # 5 x 5 tiles = 0.05 degree blocks
BUCKET = "meili-tiles"
BLOCK_PREFIX = "v1/blocks"

AMENITY = re.compile(r"^(cafe|restaurant|fast_food|pharmacy|hospital|clinic|doctors|police|"
                     r"fire_station|bar|pub|nightclub)$")
SHOP = re.compile(r"^(convenience|supermarket)$")
TOURISM = re.compile(r"^(hotel|hostel|guest_house)$")


def tile_for(lon: float, lat: float) -> tuple[int, int]:
    return math.floor(lat / TILE_DEGREES), math.floor(lon / TILE_DEGREES)


def block_for(tile: tuple[int, int]) -> tuple[int, int]:
    return tile[0] // BLOCK_TILES, tile[1] // BLOCK_TILES


def tile_key(tile: tuple[int, int]) -> str:
    return f"{tile[0]}_{tile[1]}"


def _centre(geometry: dict) -> tuple[float, float] | None:
    kind, coords = geometry.get("type"), geometry.get("coordinates")
    if kind == "Point":
        return coords[0], coords[1]
    points: list = []
    if kind == "LineString":
        points = coords
    elif kind == "Polygon":
        points = coords[0]
    elif kind == "MultiPolygon":
        points = [p for poly in coords for p in poly[0]]
    if not points:
        return None
    return sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points)


def _osm_id(props: dict) -> tuple[str, int]:
    kind = {"n": "node", "w": "way", "r": "relation"}.get(str(props.get("@type", "n"))[0], "node")
    try:
        return kind, int(props.get("@id"))
    except (TypeError, ValueError):
        return kind, 0


def sort_into_tiles(lines) -> dict[tuple[int, int], dict]:
    """GeoJSON features -> {tile: {"lit_ways": [...], "street_lamps": [...], "places": [...]}}"""
    tiles: dict[tuple[int, int], dict] = {}
    seen_places: set[tuple[str, int]] = set()   # osmium exports closed ways twice

    def bucket(tile):
        return tiles.setdefault(tile, {"lit_ways": [], "street_lamps": [], "places": []})

    for line in lines:
        line = line.strip().lstrip("\x1e")
        if not line:
            continue
        feature = json.loads(line)
        props = feature.get("properties") or {}
        tags = {k: v for k, v in props.items() if not k.startswith("@")}
        geometry = feature.get("geometry") or {}
        kind, osm_id = _osm_id(props)

        if geometry.get("type") == "LineString" and "highway" in tags and "lit" in tags:
            coords = [[round(p[0], 7), round(p[1], 7)] for p in geometry["coordinates"]]
            compact = {"id": osm_id, "tags": {k: tags[k] for k in ("highway", "lit", "name") if k in tags}, "g": coords}
            for tile in {tile_for(lon, lat) for lon, lat in coords}:
                bucket(tile)["lit_ways"].append(compact)
            continue

        if geometry.get("type") == "Point" and tags.get("highway") == "street_lamp":
            lon, lat = geometry["coordinates"][:2]
            bucket(tile_for(lon, lat))["street_lamps"].append(
                {"id": osm_id, "tags": {"highway": "street_lamp"}, "lat": lat, "lon": lon})
            continue

        if (AMENITY.match(tags.get("amenity", "")) or SHOP.match(tags.get("shop", ""))
                or TOURISM.match(tags.get("tourism", ""))):
            if (kind, osm_id) in seen_places:
                continue
            centre = _centre(geometry)
            if centre is None:
                continue
            seen_places.add((kind, osm_id))
            keep = {k: tags[k] for k in ("amenity", "shop", "tourism", "name", "opening_hours") if k in tags}
            bucket(tile_for(*centre))["places"].append(
                {"type": kind, "id": osm_id, "tags": keep, "lat": centre[1], "lon": centre[0]})
    return tiles


def group_blocks(tiles: dict) -> dict[tuple[int, int], dict]:
    blocks: dict[tuple[int, int], dict] = {}
    for tile, data in tiles.items():
        blocks.setdefault(block_for(tile), {})[tile_key(tile)] = data
    return blocks


def _merge(existing: dict, new: dict) -> dict:
    """Union of two tiles' contents, de-duplicated by OSM id."""
    merged = {}
    for layer in ("lit_ways", "street_lamps", "places"):
        seen, items = set(), []
        for item in (existing.get(layer) or []) + (new.get(layer) or []):
            key = (item.get("type", layer), item.get("id"))
            if key not in seen:
                seen.add(key)
                items.append(item)
        merged[layer] = items
    return merged


class Storage:
    def __init__(self, url: str, key: str):
        self.base = url.rstrip("/") + "/storage/v1"
        self.headers = {"apikey": key}

    def ensure_bucket(self) -> None:
        requests.post(f"{self.base}/bucket", headers=self.headers,
                      json={"id": BUCKET, "name": BUCKET, "public": False}, timeout=30)

    def get(self, path: str) -> dict | None:
        r = requests.get(f"{self.base}/object/authenticated/{BUCKET}/{path}", headers=self.headers, timeout=60)
        if r.status_code != 200:
            return None
        try:
            return json.loads(gzip.decompress(r.content))
        except Exception:  # noqa: BLE001
            return None

    uploaded_bytes = 0

    def list_coverage(self) -> list[str]:
        r = requests.post(f"{self.base}/object/list/{BUCKET}", headers=self.headers,
                          json={"prefix": "v1/coverage/", "limit": 1000}, timeout=60)
        if r.status_code != 200:
            return []
        return [item["name"] for item in r.json() if item.get("name", "").endswith(".json.gz")]

    def put(self, path: str, data: dict) -> None:
        body = gzip.compress(json.dumps(data, separators=(",", ":")).encode())
        Storage.uploaded_bytes += len(body)
        for attempt in range(4):
            r = requests.post(f"{self.base}/object/{BUCKET}/{path}",
                              headers={**self.headers, "x-upsert": "true", "Content-Type": "application/gzip"},
                              data=body, timeout=120)
            if r.status_code in (200, 201):
                return
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"upload failed for {path}: HTTP {r.status_code} {r.text[:200]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("geojsonseq")
    parser.add_argument("--country", required=True)
    parser.add_argument("--source-mb", type=float, default=0.0,
                        help="size of the downloaded OpenStreetMap file, to estimate storage use")
    args = parser.parse_args()

    with open(args.geojsonseq, encoding="utf-8") as handle:
        tiles = sort_into_tiles(handle)
    blocks = group_blocks(tiles)
    counts = {layer: sum(len(t[layer]) for t in tiles.values()) for layer in ("lit_ways", "street_lamps", "places")}
    print(f"{args.country}: {len(tiles)} tiles with data in {len(blocks)} blocks; {counts}")

    url, key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY")
    if not (url and key):
        print("Dry run (no SUPABASE_URL / SUPABASE_SERVICE_KEY): nothing uploaded.")
        return 0
    storage = Storage(url, key)
    storage.ensure_bucket()
    now = time.time()

    # ---- storage guard ------------------------------------------------------
    # Meili's own estimate of how much map data is stored, from each country's
    # coverage file. A country already built is always allowed (monthly refresh).
    budget_mb = float(os.getenv("MEILI_STORAGE_BUDGET_MB", "850"))
    used_mb, already_built = 0.0, False
    for name in storage.list_coverage():
        country_name = name.removesuffix(".json.gz")
        info = storage.get(f"v1/coverage/{name}") or {}
        size = info.get("uploaded_mb")
        if size is None:   # older coverage files: about 5 kB per block (Spain: 56.9 MB / 11308)
            size = 0.005 * len(info.get("blocks") or [])
        if country_name == args.country:
            already_built = True
        else:
            used_mb += float(size)
    estimate_mb = len(blocks) * 0.005 if blocks else args.source_mb * 0.045
    print(f"storage: {used_mb:.0f} MB used by other countries, {args.country} needs about "
          f"{estimate_mb:.0f} MB, budget {budget_mb:.0f} MB")
    if not already_built and used_mb + estimate_mb > budget_mb:
        print(f"::warning::Skipping {args.country}: it would bring Meili's map data to about "
              f"{used_mb + estimate_mb:.0f} MB, over the {budget_mb:.0f} MB budget. "
              "Move map storage to Cloudflare R2 or raise MEILI_STORAGE_BUDGET_MB to include it.")
        return 0

    def upload(item) -> int:
        (row, col), block_tiles = item
        path = f"{BLOCK_PREFIX}/{row}_{col}.json.gz"
        existing = storage.get(path) or {}
        merged_tiles = dict(existing.get("tiles") or {})
        for key_, data in block_tiles.items():
            merged_tiles[key_] = _merge(merged_tiles.get(key_, {}), data)
        countries = sorted(set(existing.get("countries") or []) | {args.country})
        storage.put(path, {"fetched_at": now, "source": "extract", "countries": countries, "tiles": merged_tiles})
        return len(block_tiles)

    done = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for n in pool.map(upload, blocks.items()):
            done += 1
            if done % 100 == 0:
                print(f"  uploaded {done}/{len(blocks)} blocks")
    storage.put(f"v1/coverage/{args.country}.json.gz",
                {"country": args.country, "built_at": now, "blocks": [f"{r}_{c}" for r, c in blocks],
                 "uploaded_mb": round(Storage.uploaded_bytes / 1e6, 2)})
    print(f"{args.country}: uploaded {len(blocks)} blocks, {Storage.uploaded_bytes / 1e6:.1f} MB "
          "(Supabase free plan: 1000 MB total storage).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
