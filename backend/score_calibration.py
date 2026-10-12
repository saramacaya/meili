"""Turns Meili's raw route scores into scores that use the whole 0-100 range.

Raw scores bunch up (most walks land between about 50 and 70) because missing
data counts as neutral. The displayed score instead says where a walk ranks
among walks Meili has scored in the same city area:

* Every scored route's raw score is remembered per ~20 km "city cell"
  (Supabase Storage: v1/calibration/<cell>.json.gz, newest 500 kept).
* With enough history (30+ walks), the displayed score is the raw score's
  percentile in that city: 80 means "better than about 80% of walks here".
* Before that, a fixed curve stretches the usual 50-70 band over the range,
  centred on the city's median once a few walks are known.
* Routes with little data are pulled towards the middle (50), so missing data
  never looks alarming or reassuring.

The raw score is kept alongside (raw_route_value_score) for transparency.
"""

from __future__ import annotations

import math
import threading
import time
from bisect import bisect_left, bisect_right
from typing import Any, Optional

import tile_cache

CELL_DEGREES = 0.2
DEFAULT_CENTER = 62.0     # typical raw score across cities
DEFAULT_SCALE = 9.0       # how strongly the curve stretches (gentle: keeps top scores distinguishable)
MIN_SAMPLES = 30          # walks needed before the city's own ranking counts
FULL_SAMPLES = 100        # walks at which the city's ranking fully takes over
MAX_SAMPLES = 500
CACHE_SECONDS = 600

_cache: dict[str, tuple[float, list[float]]] = {}
_lock = threading.Lock()


def cell_for(lon: float, lat: float) -> str:
    return f"{math.floor(lat / CELL_DEGREES)}_{math.floor(lon / CELL_DEGREES)}"


def _path(cell: str) -> str:
    return f"v1/calibration/{cell}.json.gz"


def load_samples(cell: str) -> list[float]:
    with _lock:
        cached = _cache.get(cell)
        if cached and time.time() - cached[0] < CACHE_SECONDS:
            return cached[1]
    try:
        data = tile_cache.STORE.storage_get(_path(cell)) or {}
    except Exception:  # noqa: BLE001
        data = {}
    samples = sorted(float(x) for x in (data.get("samples") or []))
    with _lock:
        _cache[cell] = (time.time(), samples)
    return samples


def record(cell: str, raw_scores: list[float]) -> None:
    """Remember new raw scores for this city cell (background, best effort)."""
    if not raw_scores:
        return

    def run() -> None:
        try:
            data = tile_cache.STORE.storage_get(_path(cell)) or {}
            recent = list(data.get("recent") or []) + [round(float(s), 2) for s in raw_scores]
            recent = recent[-MAX_SAMPLES:]
            tile_cache.STORE.storage_put(_path(cell), {"recent": recent, "samples": recent,
                                                       "updated_at": time.time()})
            with _lock:
                _cache[cell] = (time.time(), sorted(recent))
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(target=run, daemon=True, name="score-calibration").start()


def _curve(raw: float, center: float) -> float:
    return 100.0 / (1.0 + math.exp(-(raw - center) / DEFAULT_SCALE))


def _median(values: list[float]) -> float:
    n = len(values)
    return values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2


def confidence_factor(data_confidence: Optional[dict[str, Any]]) -> float:
    """1.0 with good data; down to 0.25 when the score rests on very little."""
    if not data_confidence:
        return 1.0
    lighting = float(data_confidence.get("lighting_evidence_coverage_percentage") or 0)
    activity = float(data_confidence.get("activity_data_coverage_percentage") or 0)
    combined = 0.6 * lighting + 0.4 * activity
    return max(0.25, min(1.0, combined / 60.0))


def calibrate(raw: float, samples: list[float], factor: float = 1.0) -> float:
    center = _median(samples) if len(samples) >= 10 else DEFAULT_CENTER
    score = _curve(raw, center)
    n = len(samples)
    if n >= MIN_SAMPLES:
        below = bisect_left(samples, raw)
        equal = bisect_right(samples, raw) - below
        percentile = (below + 0.5 * equal) / n
        city_score = 5.0 + 90.0 * percentile
        weight = min(1.0, (n - MIN_SAMPLES) / (FULL_SAMPLES - MIN_SAMPLES))
        score = (1 - weight) * score + weight * city_score
    score = 50.0 + (score - 50.0) * factor
    # Never claim perfection or total danger: the evidence can't support it.
    return round(max(5.0, min(95.0, score)), 1)


def describe(samples: list[float]) -> dict[str, Any]:
    n = len(samples)
    if n >= FULL_SAMPLES:
        kind = "city_ranking"
    elif n >= MIN_SAMPLES:
        kind = "city_ranking_blended"
    else:
        kind = "default_curve"
    return {
        "type": kind,
        "walks_in_city_history": n,
        "explanation": (
            "Shows how this walk ranks among walks scored in this city "
            "(50 = typical here). Routes with little data stay near 50."
        ),
    }
