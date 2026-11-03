"""Areas people searched that Meili has no map data for yet.

When a comparison can't load the map data for its routes, the backend notes
the rough area (one 0.05-degree block, about 5 km, the same grid as the
pre-built map data) in Cloudflare R2 under v1/coverage-requests/. The
monthly country build reads these notes and adds the country (or, for very
large countries, the state or province) that contains each one.

What is stored: the block's centre, how many times it was asked for, and the
first and last day. No account, no exact points, no time of day, so a note
can't be linked to a person.
"""

from __future__ import annotations

import json
import math
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import r2_storage

PREFIX = "v1/coverage-requests"
BLOCK_DEGREES = 0.05
# The same block is written at most once per hour per server process.
THROTTLE_SECONDS = 3600

_recent: dict[str, float] = {}
_lock = threading.Lock()


def block_key(lat: float, lng: float) -> str:
    return f"{math.floor(lat / BLOCK_DEGREES)}_{math.floor(lng / BLOCK_DEGREES)}"


def _centre(key: str) -> tuple[float, float]:
    row, col = (int(x) for x in key.split("_"))
    return round((row + 0.5) * BLOCK_DEGREES, 4), round((col + 0.5) * BLOCK_DEGREES, 4)


def _write(key: str) -> None:
    path = f"{PREFIX}/{key}.json"
    today = datetime.now(timezone.utc).date().isoformat()
    note: Optional[dict] = None
    raw = r2_storage.get_bytes(path)
    if raw:
        try:
            note = json.loads(raw)
        except ValueError:
            note = None
    if not note:
        lat, lng = _centre(key)
        note = {"lat": lat, "lng": lng, "count": 0, "first_seen": today}
    note["count"] = int(note.get("count", 0)) + 1
    note["last_seen"] = today
    r2_storage.put_bytes(path, json.dumps(note).encode(), content_type="application/json")


def record(lat: float, lng: float) -> bool:
    """Notes the block around (lat, lng) in the background. Never raises.

    Returns True when a write was started (False when throttled or R2 is off).
    """
    if r2_storage.client() is None:
        return False
    key = block_key(lat, lng)
    now = time.time()
    with _lock:
        if now - _recent.get(key, 0) < THROTTLE_SECONDS:
            return False
        _recent[key] = now
        if len(_recent) > 5000:
            _recent.clear()

    def run() -> None:
        try:
            _write(key)
        except Exception as error:  # noqa: BLE001
            print(f"coverage request not saved: {error}")

    threading.Thread(target=run, daemon=True).start()
    return True


def list_requests() -> list[dict]:
    """Every stored note (used by the monthly build)."""
    notes = []
    for path in r2_storage.list_keys(PREFIX + "/"):
        raw = r2_storage.get_bytes(path)
        if not raw:
            continue
        try:
            notes.append(json.loads(raw))
        except ValueError:
            continue
    return notes
