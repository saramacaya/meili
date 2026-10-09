"""Tile cache: fast, reusable map data for any city on Earth.

The world is split into small square tiles (TILE_DEGREES on each side,
roughly 1 km). For each tile Meili keeps:

  * "osm"            -- OpenStreetMap lit streets, street lamps and the
                         places used for the activity score;
  * "valencia_lamps" -- official Valencia streetlights (only for tiles
                         inside Valencia's coverage area).

A request only loads the handful of tiles its routes touch, in this order:

  1. backend memory           (microseconds)
  2. Supabase Storage          (tens of milliseconds, survives restarts)
  3. the original source       (seconds, only the first time anyone ever
                                routes through that tile; then saved)

Missing tiles are fetched with ONE Overpass query per small group of
tiles instead of one large query per request, and different tiles load in
parallel. Tiles older than TILE_MAX_AGE_SECONDS are still used, then
refreshed in the background.
"""

from __future__ import annotations

import gzip
import json
import math
import threading
import time
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from typing import Any, Callable, Optional

import requests

TILE_DEGREES = 0.01
TILE_FORMAT_VERSION = 1
TILE_MAX_AGE_SECONDS = 30 * 24 * 3600
MEMORY_TILE_LIMIT = 60
STORAGE_BUCKET = "meili-tiles"
OVERPASS_TIMEOUT_SECONDS = 25
MAX_TILES_PER_OVERPASS_QUERY = 9  # a 3 x 3 block, about 3 km x 3 km

OVERPASS_API_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

PLACE_FILTERS = (
    'nwr["amenity"~"^(cafe|restaurant|fast_food|pharmacy|hospital|clinic|'
    'doctors|police|fire_station|bar|pub|nightclub)$"]({bbox});'
    'nwr["shop"~"^(convenience|supermarket)$"]({bbox});'
    'nwr["tourism"~"^(hotel|hostel|guest_house)$"]({bbox});'
)


# ---------------------------------------------------------------- tile maths

TileId = tuple[int, int]  # (row, column)


def tile_for_point(longitude: float, latitude: float) -> TileId:
    return (
        math.floor(latitude / TILE_DEGREES),
        math.floor(longitude / TILE_DEGREES),
    )


def tile_bounds(tile: TileId) -> tuple[float, float, float, float]:
    """(south, west, north, east) in degrees."""
    row, column = tile
    south = round(row * TILE_DEGREES, 6)
    west = round(column * TILE_DEGREES, 6)
    return south, west, round(south + TILE_DEGREES, 6), round(west + TILE_DEGREES, 6)


def tile_key(tile: TileId) -> str:
    return f"{tile[0]}_{tile[1]}"


def padded_bbox(
    coordinates: list[tuple[float, float]], padding_meters: float
) -> tuple[float, float, float, float]:
    longitudes = [c[0] for c in coordinates]
    latitudes = [c[1] for c in coordinates]
    mean_latitude = sum(latitudes) / len(latitudes)
    lat_pad = padding_meters / 111_000
    lon_pad = padding_meters / (111_000 * max(math.cos(math.radians(mean_latitude)), 0.01))
    return (
        min(latitudes) - lat_pad,
        min(longitudes) - lon_pad,
        max(latitudes) + lat_pad,
        max(longitudes) + lon_pad,
    )


def tiles_for_bbox(bbox: tuple[float, float, float, float]) -> list[TileId]:
    south, west, north, east = bbox
    first_row, first_col = tile_for_point(west, south)
    last_row, last_col = tile_for_point(east, north)
    return [
        (row, col)
        for row in range(first_row, last_row + 1)
        for col in range(first_col, last_col + 1)
    ]


def _in_bbox(lon: float, lat: float, bbox: tuple[float, float, float, float]) -> bool:
    south, west, north, east = bbox
    return south <= lat <= north and west <= lon <= east


def group_tiles(tiles: list[TileId], max_group: int = MAX_TILES_PER_OVERPASS_QUERY) -> list[list[TileId]]:
    """Groups neighbouring tiles into small square blocks (3 x 3)."""
    side = max(1, int(math.sqrt(max_group)))
    groups: dict[tuple[int, int], list[TileId]] = {}
    for row, col in tiles:
        groups.setdefault((row // side, col // side), []).append((row, col))
    return list(groups.values())


def group_bbox(group: list[TileId]) -> tuple[float, float, float, float]:
    bounds = [tile_bounds(t) for t in group]
    return (
        min(b[0] for b in bounds),
        min(b[1] for b in bounds),
        max(b[2] for b in bounds),
        max(b[3] for b in bounds),
    )


# ---------------------------------------------------------------- storage

class _TileStore:
    """Memory LRU in front of Supabase Storage (if configured)."""

    def __init__(self) -> None:
        self._memory: "OrderedDict[str, dict]" = OrderedDict()
        self._lock = threading.Lock()
        self._fetch_locks: dict[str, threading.Lock] = {}
        self._bucket_checked = False

    # -- memory
    def memory_get(self, path: str) -> Optional[dict]:
        with self._lock:
            tile = self._memory.get(path)
            if tile is not None:
                self._memory.move_to_end(path)
            return tile

    def memory_put(self, path: str, tile: dict) -> None:
        with self._lock:
            self._memory[path] = tile
            self._memory.move_to_end(path)
            while len(self._memory) > MEMORY_TILE_LIMIT:
                self._memory.popitem(last=False)

    def fetch_lock(self, path: str) -> threading.Lock:
        with self._lock:
            return self._fetch_locks.setdefault(path, threading.Lock())

    # -- persistent storage
    @staticmethod
    def _client():
        try:
            from nasa_route_analysis import supabase_storage
        except Exception:
            return None
        return supabase_storage

    def _ensure_bucket(self, client) -> None:
        if self._bucket_checked:
            return
        try:
            client.storage.create_bucket(STORAGE_BUCKET, options={"public": False})
        except Exception:
            pass  # already exists, or no permission: uploads will report it
        self._bucket_checked = True

    def storage_get(self, path: str) -> Optional[dict]:
        client = self._client()
        if client is None:
            return None
        try:
            raw = client.storage.from_(STORAGE_BUCKET).download(path)
            return json.loads(gzip.decompress(raw))
        except Exception:
            return None

    def storage_put(self, path: str, tile: dict) -> None:
        client = self._client()
        if client is None:
            return
        self._ensure_bucket(client)
        try:
            client.storage.from_(STORAGE_BUCKET).upload(
                path,
                gzip.compress(json.dumps(tile, separators=(",", ":")).encode()),
                {"content-type": "application/gzip", "upsert": "true"},
            )
        except Exception:
            pass

    def persistent(self) -> bool:
        return self._client() is not None


STORE = _TileStore()
_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="tiles")


def _path(layer: str, tile: TileId) -> str:
    return f"v{TILE_FORMAT_VERSION}/{layer}/{tile_key(tile)}.json.gz"


def _is_stale(tile: dict) -> bool:
    return time.time() - float(tile.get("fetched_at", 0)) > TILE_MAX_AGE_SECONDS


# ---------------------------------------------------------------- sources

def _overpass(query: str) -> dict:
    """Hedged request: asks the first Overpass mirror; if it has not answered
    within a few seconds (or fails), also asks the next one, and so on.
    The first good answer wins, so one slow server never blocks Meili."""
    hedge_after_seconds = 4.0
    errors: list[str] = []

    def ask(url: str) -> dict:
        response = requests.post(
            url,
            data={"data": query},
            headers={"Accept": "application/json", "User-Agent": "Meili safety-routing prototype"},
            timeout=OVERPASS_TIMEOUT_SECONDS + 5,
        )
        response.raise_for_status()
        return response.json()

    pool = ThreadPoolExecutor(max_workers=len(OVERPASS_API_URLS))
    waiting = list(OVERPASS_API_URLS)
    running: dict = {}
    try:
        while waiting or running:
            if waiting:
                url = waiting.pop(0)
                running[pool.submit(ask, url)] = url
            done, _ = wait(list(running), timeout=hedge_after_seconds if waiting else None,
                           return_when=FIRST_COMPLETED)
            for future in done:
                url = running.pop(future)
                try:
                    return future.result()
                except Exception as error:  # noqa: BLE001
                    errors.append(f"{url.split('/')[2]}: {_short_error(error)}")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    raise RuntimeError("All Overpass servers failed: " + " | ".join(errors))


def _short_error(error: Exception) -> str:
    """'429 Too Many Requests', 'timeout', 'connection refused', ..."""
    response = getattr(error, "response", None)
    if response is not None and getattr(response, "status_code", None):
        return f"HTTP {response.status_code}"
    text = str(error)
    for marker, label in (("timed out", "timeout"), ("Read timed out", "timeout"),
                          ("NewConnectionError", "cannot connect"), ("Name or service", "DNS failure")):
        if marker in text:
            return label
    return type(error).__name__ + ": " + text[:80]


def _fetch_osm_group(group: list[TileId]) -> dict[TileId, dict]:
    south, west, north, east = group_bbox(group)
    bbox = f"{south},{west},{north},{east}"
    lighting_query = (
        f"[out:json][timeout:{OVERPASS_TIMEOUT_SECONDS}];"
        f'(way["highway"]["lit"]({bbox});node["highway"="street_lamp"]({bbox}););'
        "out body geom;"
    )
    places_query = (
        f"[out:json][timeout:{OVERPASS_TIMEOUT_SECONDS}];"
        f"({PLACE_FILTERS.format(bbox=bbox)});"
        "out tags center;"
    )
    # Two smaller queries in parallel finish sooner than one big one.
    with ThreadPoolExecutor(max_workers=2) as pool:
        lighting_future = pool.submit(_overpass, lighting_query)
        places_future = pool.submit(_overpass, places_query)
        elements = lighting_future.result().get("elements", []) + places_future.result().get("elements", [])

    now = time.time()
    tiles: dict[TileId, dict] = {
        t: {"fetched_at": now, "lit_ways": [], "street_lamps": [], "places": []} for t in group
    }
    seen: set[tuple[str, str, int]] = set()

    for element in elements:
        etype, eid = element.get("type"), element.get("id")
        tags = element.get("tags") or {}

        if etype == "way" and "lit" in tags and element.get("geometry"):
            if ("lit", etype, eid) in seen:
                continue
            seen.add(("lit", etype, eid))
            coords = [[p["lon"], p["lat"]] for p in element["geometry"] if p]
            compact = {"id": eid, "tags": tags, "g": coords}
            for tile in {tile_for_point(lon, lat) for lon, lat in coords}:
                if tile in tiles:
                    tiles[tile]["lit_ways"].append(compact)
            continue

        if etype == "node" and tags.get("highway") == "street_lamp":
            if ("lamp", etype, eid) in seen:
                continue
            seen.add(("lamp", etype, eid))
            tile = tile_for_point(element["lon"], element["lat"])
            if tile in tiles:
                tiles[tile]["street_lamps"].append(
                    {"id": eid, "tags": tags, "lat": element["lat"], "lon": element["lon"]}
                )
            continue

        lat = element.get("lat", (element.get("center") or {}).get("lat"))
        lon = element.get("lon", (element.get("center") or {}).get("lon"))
        if lat is None or lon is None or ("place", etype, eid) in seen:
            continue
        seen.add(("place", etype, eid))
        tile = tile_for_point(lon, lat)
        if tile in tiles:
            tiles[tile]["places"].append({"type": etype, "id": eid, "tags": tags, "lat": lat, "lon": lon})

    return tiles


# Injected by main.py so this module does not import main (avoids a cycle).
VALENCIA_LAMP_FETCHER: Optional[Callable[[list[tuple[float, float]]], tuple[list, dict]]] = None
VALENCIA_BOUNDS = {"south": 39.35, "west": -0.48, "north": 39.58, "east": -0.28}


def _tile_in_valencia(tile: TileId) -> bool:
    s, w, n, e = tile_bounds(tile)
    b = VALENCIA_BOUNDS
    return not (n < b["south"] or s > b["north"] or e < b["west"] or w > b["east"])


def _fetch_valencia_group(group: list[TileId]) -> dict[TileId, dict]:
    now = time.time()
    tiles = {t: {"fetched_at": now, "points": []} for t in group}
    if VALENCIA_LAMP_FETCHER is None:
        raise RuntimeError("Valencia streetlight fetcher is not configured.")
    south, west, north, east = group_bbox(group)
    points, _debug = VALENCIA_LAMP_FETCHER([(west, south), (east, north)])
    for lon, lat in points:
        tile = tile_for_point(lon, lat)
        if tile in tiles:
            tiles[tile]["points"].append([lon, lat])
    return tiles


FETCHERS: dict[str, Callable[[list[TileId]], dict[TileId, dict]]] = {
    "osm": _fetch_osm_group,
    "valencia_lamps": _fetch_valencia_group,
}


# ---------------------------------------------------------------- loading

def _load_from_storage(layer: str, tiles: list[TileId]) -> dict[TileId, dict]:
    found: dict[TileId, dict] = {}
    pending = []
    for tile in tiles:
        path = _path(layer, tile)
        cached = STORE.memory_get(path)
        if cached is not None:
            found[tile] = cached
        else:
            pending.append(tile)

    futures = {_POOL.submit(STORE.storage_get, _path(layer, t)): t for t in pending}
    for future in as_completed(futures):
        tile = futures[future]
        data = future.result()
        if data is not None:
            STORE.memory_put(_path(layer, tile), data)
            found[tile] = data
    return found


def _fetch_and_save(layer: str, group: list[TileId]) -> dict[TileId, dict]:
    locks = [STORE.fetch_lock(_path(layer, t)) for t in sorted(group)]
    for lock in locks:
        lock.acquire()
    try:
        # Another request may have filled these while we waited.
        already = {t: STORE.memory_get(_path(layer, t)) for t in group}
        if all(v is not None and not _is_stale(v) for v in already.values()):
            return already  # type: ignore[return-value]
        fetched = FETCHERS[layer](group)
        for tile, data in fetched.items():
            STORE.memory_put(_path(layer, tile), data)
            _POOL.submit(STORE.storage_put, _path(layer, tile), data)
        return fetched
    finally:
        for lock in locks:
            lock.release()


def load_layer(layer: str, tiles: list[TileId]) -> tuple[dict[TileId, dict], dict]:
    """Returns tile data for a layer, fetching only what is missing."""
    started = time.perf_counter()
    found = _load_from_storage(layer, tiles)
    missing = [t for t in tiles if t not in found]
    stale = [t for t in found if _is_stale(found[t])]
    errors: list[str] = []

    groups = group_tiles(missing, max_group=4)   # small groups = quick answers
    futures = [_POOL.submit(_fetch_and_save, layer, g) for g in groups]
    for future in as_completed(futures):
        try:
            found.update(future.result())
        except Exception as error:  # noqa: BLE001
            errors.append(str(error))

    for group in group_tiles(stale):  # refresh in the background
        _POOL.submit(_fetch_and_save, layer, group)

    debug = {
        "tiles_needed": len(tiles),
        "tiles_from_cache": len(tiles) - len(missing),
        "tiles_failed": sum(1 for t in missing if t not in found),
        "seconds": round(time.perf_counter() - started, 3),
        "persistent_storage": STORE.persistent(),
    }
    debug["tiles_downloaded"] = sum(1 for t in missing if t in found)
    return found, {"debug": debug, "errors": errors}


# ---------------------------------------------------------------- public API

AREA_DEADLINE_SECONDS = 8.0


def load_area(
    route_geometries: list[list[tuple[float, float]]],
    padding_meters: float,
    deadline_seconds: float = AREA_DEADLINE_SECONDS,
) -> dict[str, Any]:
    """Loads every layer needed for a set of routes, all layers in parallel.

    Never waits longer than `deadline_seconds`: tiles still downloading after
    that keep downloading in the background (and are saved), while this
    request scores with what has arrived and reports itself as partial.
    """
    tiles: set[TileId] = set()
    for geometry in route_geometries:
        tiles.update(tiles_for_bbox(padded_bbox(geometry, padding_meters)))
    tile_list = sorted(tiles)
    valencia_tiles = [t for t in tile_list if _tile_in_valencia(t)]

    futures = {"osm": _POOL.submit(load_layer, "osm", tile_list)}
    if valencia_tiles:
        futures["valencia_lamps"] = _POOL.submit(load_layer, "valencia_lamps", valencia_tiles)
    wait(list(futures.values()), timeout=deadline_seconds)

    def collect(layer: str, wanted: list[TileId]) -> tuple[dict, dict]:
        future = futures.get(layer)
        if future is not None and future.done():
            try:
                return future.result()
            except Exception as error:  # noqa: BLE001
                return {}, {"debug": {}, "errors": [str(error)]}
        # Still downloading: use whatever tiles are already in memory.
        found = {}
        for tile in wanted:
            data = STORE.memory_get(_path(layer, tile))
            if data is not None:
                found[tile] = data
        missing = len(wanted) - len(found)
        return found, {
            "debug": {"tiles_needed": len(wanted), "tiles_ready": len(found),
                      "still_downloading": missing, "deadline_seconds": deadline_seconds},
            "errors": [f"{missing} map tiles still downloading"] if missing else [],
        }

    osm_tiles, osm_info = collect("osm", tile_list)
    if valencia_tiles:
        valencia_tiles_data, valencia_info = collect("valencia_lamps", valencia_tiles)
    else:
        valencia_tiles_data, valencia_info = {}, {"debug": {"skipped": "outside Valencia"}, "errors": []}

    osm_complete = len(osm_tiles) == len(tile_list)
    valencia_complete = len(valencia_tiles_data) == len(valencia_tiles)
    still_downloading = any(not f.done() for f in futures.values())

    return {
        "osm_tiles": osm_tiles,
        "valencia_tiles": valencia_tiles_data,
        "valencia_covered": bool(valencia_tiles),
        "complete": osm_complete and valencia_complete,
        "still_downloading": still_downloading,
        "osm_error": None if osm_complete else "; ".join(osm_info["errors"]) or "Some map tiles could not be loaded.",
        "valencia_error": None if valencia_complete else "; ".join(valencia_info["errors"]) or "Some Valencia streetlight tiles could not be loaded.",
        "debug": {"osm": osm_info["debug"], "valencia_lamps": valencia_info["debug"]},
    }


def route_slice(area: dict[str, Any], geometry: list[tuple[float, float]], padding_meters: float) -> dict[str, list]:
    """Everything near ONE route, in the formats the scoring code expects."""
    bbox = padded_bbox(geometry, padding_meters)
    lit_ways, lamps, places, official = [], [], [], []
    seen_ways: set[int] = set()

    for tile in tiles_for_bbox(bbox):
        osm = area["osm_tiles"].get(tile)
        if osm:
            for way in osm["lit_ways"]:
                if way["id"] in seen_ways:
                    continue
                if any(_in_bbox(lon, lat, bbox) for lon, lat in way["g"]):
                    seen_ways.add(way["id"])
                    lit_ways.append({
                        "type": "way", "id": way["id"], "tags": way["tags"],
                        "geometry": [{"lon": lon, "lat": lat} for lon, lat in way["g"]],
                    })
            for lamp in osm["street_lamps"]:
                if _in_bbox(lamp["lon"], lamp["lat"], bbox):
                    lamps.append({"type": "node", **lamp})
            for place in osm["places"]:
                if _in_bbox(place["lon"], place["lat"], bbox):
                    places.append(place)
        valencia = area["valencia_tiles"].get(tile)
        if valencia:
            official.extend(
                (lon, lat) for lon, lat in valencia["points"] if _in_bbox(lon, lat, bbox)
            )

    return {"lit_ways": lit_ways, "street_lamps": lamps, "places": places, "official_lamps": official}


def corridor_filter(
    points: list[tuple[float, float]], samples: list[tuple[float, float]], radius_meters: float
) -> list[tuple[float, float]]:
    """Keeps only points within roughly radius_meters of the route samples.

    Uses a coarse grid so it is linear in the number of points, which keeps
    nearest-lamp checks fast even in dense city centres.
    """
    if not points or not samples:
        return []
    cell = radius_meters / 111_000
    near: set[tuple[int, int]] = set()
    for lon, lat in samples:
        row, col = math.floor(lat / cell), math.floor(lon / cell)
        for dr in (-1, 0, 1):
            for dc in (-2, -1, 0, 1, 2):  # longitude cells are narrower than latitude cells
                near.add((row + dr, col + dc))
    return [p for p in points if (math.floor(p[1] / cell), math.floor(p[0] / cell)) in near]


# ---------------------------------------------------------------- warm-up

MAX_WARM_TILES = 40


def warm_tiles_for_trip(origin: tuple[float, float], destination: tuple[float, float]) -> dict[str, Any]:
    """Starts downloading the tiles a walk between two points will need.

    Walking routes rarely stray far from the straight line, so we take the
    box around both points padded by 30% of the distance (at least 300 m).
    For long trips we keep only a corridor along the straight line so a
    single warm-up never asks for a whole city.
    """
    (lon1, lat1), (lon2, lat2) = origin, destination
    straight = math.hypot(
        (lon2 - lon1) * 111_000 * math.cos(math.radians((lat1 + lat2) / 2)),
        (lat2 - lat1) * 111_000,
    )
    padding = max(300.0, 0.3 * straight)
    tiles = tiles_for_bbox(padded_bbox([origin, destination], padding))

    if len(tiles) > MAX_WARM_TILES:
        steps = max(2, int(straight / 500))
        line = [
            (lon1 + (lon2 - lon1) * i / steps, lat1 + (lat2 - lat1) * i / steps)
            for i in range(steps + 1)
        ]
        corridor: set[TileId] = set()
        for point in line:
            corridor.update(tiles_for_bbox(padded_bbox([point], 500)))
        tiles = sorted(corridor)[: MAX_WARM_TILES * 2]

    valencia = [t for t in tiles if _tile_in_valencia(t)]

    def run() -> None:
        try:
            load_layer("osm", tiles)
            if valencia:
                load_layer("valencia_lamps", valencia)
        except Exception:
            pass  # warm-up is best effort; the real request will retry

    threading.Thread(target=run, daemon=True, name="tile-warm").start()
    return {"status": "warming", "tiles": len(tiles), "straight_line_meters": round(straight)}


# ---------------------------------------------------------------- prefill

_PREFILL_STATUS: dict[str, Any] = {"state": "idle"}


def prefill_status() -> dict[str, Any]:
    return dict(_PREFILL_STATUS)


def start_prefill(bbox: tuple[float, float, float, float]) -> dict[str, Any]:
    if _PREFILL_STATUS.get("state") == "running":
        return prefill_status()
    tiles = tiles_for_bbox(bbox)
    groups = group_tiles(tiles)
    _PREFILL_STATUS.update(
        state="running", tiles_total=len(tiles), groups_total=len(groups),
        groups_done=0, errors=[], started_at=time.time(), finished_at=None,
        tiles_already_cached=0, tiles_downloaded=0, retries=0,
    )

    def fill(layer: str, group: list[TileId]) -> Optional[str]:
        layer_group = [t for t in group if layer == "osm" or _tile_in_valencia(t)]
        if not layer_group:
            return None
        have = _load_from_storage(layer, layer_group)
        missing = [t for t in layer_group if t not in have or _is_stale(have[t])]
        if not missing:
            _PREFILL_STATUS["tiles_already_cached"] += len(layer_group)
            return None
        last_error = None
        for attempt, pause in enumerate((0, 20, 60, 120)):
            time.sleep(pause)
            try:
                _fetch_and_save(layer, missing)
                _PREFILL_STATUS["tiles_downloaded"] += len(missing)
                return None
            except Exception as error:  # noqa: BLE001
                last_error = str(error)
                _PREFILL_STATUS["retries"] += 1
        return f"{layer} {tile_key(group[0])}: {last_error}"[:400]

    def run() -> None:
        for group in groups:
            for layer in ("osm", "valencia_lamps"):
                problem = fill(layer, group)
                if problem:
                    _PREFILL_STATUS["errors"].append(problem)
            _PREFILL_STATUS["groups_done"] += 1
            time.sleep(2.0)  # be polite to the public Overpass servers
        _PREFILL_STATUS.update(state="finished", finished_at=time.time())

    threading.Thread(target=run, daemon=True, name="tile-prefill").start()
    return prefill_status()
