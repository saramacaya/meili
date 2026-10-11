"""Automatic local context, anywhere on Earth.

The first time a route passes through an area Meili has no local context
for, this module gathers it in the background (it never slows a search):

  1. Neighbourhoods along the route   -> OpenStreetMap Nominatim (free)
  2. Recent local news about each one -> GDELT news index (free, no key)
  3. Article -> structured records    -> AI model (GitHub Models by default:
                                         free, rate-limited)
  4. Records saved in Supabase         -> reused by every later route

Automatically gathered records carry verification_weight = 0.7, so they
count less than reviewed records (1.0). Approving a record in Supabase
(origin = 'reviewed') or, later, confirmations from user ratings can raise
that weight; setting status = 'removed' hides a record.
"""

from __future__ import annotations

import html
import json
import math
import os
import re
import threading
import time
import unicodedata
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import requests

USER_AGENT = "Meili safety-routing prototype (contact: github.com/saramacaya/meili)"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

AUTO_VERIFICATION_WEIGHT = 0.7
REGATHER_AFTER_DAYS = 90
MAX_ARTICLES_PER_AREA = 4
MIN_RADIUS_M, MAX_RADIUS_M = 300.0, 1500.0
RECORDS_TABLE = "area_context_records"
AREAS_TABLE = "area_context_areas"

AI_PROVIDER = os.getenv("MEILI_AI_PROVIDER", "github").lower()
AI_MODELS = {"github": "openai/gpt-4o-mini", "ollama": "llama3.1:8b", "anthropic": "claude-haiku-5-5"}
AI_MODEL = os.getenv("MEILI_CONTEXT_MODEL", AI_MODELS.get(AI_PROVIDER, ""))

ALLOWED_SENTIMENTS = {"negative", "positive", "mixed"}
ALLOWED_TIERS = {"confirmed_incident", "resident_complaint", "perception_only", "municipal_data"}
ALLOWED_TIMES = {"daytime", "evening", "late_night", "weekend", "unspecified"}

EXTRACTION_PROMPT = (
    "You extract walking-safety evidence for one specific neighbourhood from one news article. "
    "Use ONLY what the article states. Never add facts, numbers, causes or opinions. "
    "Ignore anything not clearly about street safety (crime, harassment, lighting, feeling safe or "
    "unsafe on the street, police presence) IN the named neighbourhood. Do not name private individuals. "
    "Return JSON only, no prose: {\"records\": [ {\"sentiment\": \"negative|positive|mixed\", "
    "\"evidence_tier\": \"confirmed_incident|resident_complaint|perception_only|municipal_data\", "
    "\"time_context\": \"daytime|evening|late_night|weekend|unspecified\", "
    "\"summary\": \"one neutral English sentence, max 35 words\", "
    "\"about_this_neighbourhood\": true } ] }. "
    "Return at most 2 records. Return {\"records\": []} if the article has nothing relevant."
)


# --------------------------------------------------------------------- log

from collections import deque

GATHER_LOG: deque = deque(maxlen=40)   # shown on /diagnose


def _log(event: str, **details: Any) -> None:
    GATHER_LOG.appendleft({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event, **details})


# --------------------------------------------------------------------- utils

def _plain(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    ).lower().strip()


def area_key(city: str, name: str) -> str:
    return f"{_plain(city)}|{_plain(name)}"


def _distance_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat = math.radians((a[1] + b[1]) / 2)
    return math.hypot((a[0] - b[0]) * 111_320 * math.cos(lat), (a[1] - b[1]) * 110_540)


def _db():
    try:
        from nasa_route_analysis import supabase_storage  # service-role client
    except Exception:  # noqa: BLE001
        return None
    return supabase_storage


# -------------------------------------------------- 1. neighbourhood lookup

_reverse_cache: dict[tuple[float, float], Optional[dict]] = {}
_nominatim_lock = threading.Lock()
_last_nominatim_call = [0.0]


def reverse_neighbourhood(lon: float, lat: float) -> Optional[dict]:
    """Neighbourhood name, city and a centre/radius for a point (cached)."""
    key = (round(lon, 3), round(lat, 3))
    if key in _reverse_cache:
        return _reverse_cache[key]
    with _nominatim_lock:  # Nominatim policy: at most 1 request per second
        wait = 1.1 - (time.monotonic() - _last_nominatim_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_nominatim_call[0] = time.monotonic()
        try:
            response = requests.get(
                NOMINATIM_URL,
                params={"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 15, "addressdetails": 1},
                headers={"User-Agent": USER_AGENT, "Accept-Language": "en"},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as error:  # noqa: BLE001
            _log("place lookup failed", error=str(error)[:150])
            return None
    address = data.get("address") or {}
    name = (address.get("neighbourhood") or address.get("suburb") or address.get("quarter")
            or address.get("city_district"))
    city = address.get("city") or address.get("town") or address.get("village") or address.get("municipality")
    result = None
    if name and city:
        try:
            south, north, west, east = (float(v) for v in data["boundingbox"])
            centre = ((west + east) / 2, (south + north) / 2)
            radius = _distance_m((west, south), (east, north)) / 2
        except Exception:  # noqa: BLE001
            centre, radius = (lon, lat), 600.0
        result = {
            "name": name, "city": city, "country_code": (address.get("country_code") or "").upper(),
            "longitude": centre[0], "latitude": centre[1],
            "radius_meters": max(MIN_RADIUS_M, min(MAX_RADIUS_M, radius)),
        }
    _reverse_cache[key] = result
    return result


def neighbourhoods_along(route_geometries: list[list[tuple[float, float]]]) -> list[dict]:
    """Up to ~4 distinct neighbourhoods per route (start, middle points, end)."""
    found: dict[str, dict] = {}
    for geometry in route_geometries:
        if not geometry:
            continue
        picks = sorted({0, len(geometry) // 3, 2 * len(geometry) // 3, len(geometry) - 1})
        for index in picks:
            lon, lat = geometry[index]
            area = reverse_neighbourhood(lon, lat)
            if area:
                found.setdefault(area_key(area["city"], area["name"]), area)
    return list(found.values())


# ---------------------------------------------------------- 2. news search

def search_news(name: str, city: str) -> list[dict]:
    """Recent articles mentioning the neighbourhood and its city (GDELT)."""
    query = f'"{name}" "{city}"'
    try:
        response = requests.get(
            GDELT_URL,
            params={"query": query, "mode": "artlist", "format": "json",
                    "maxrecords": 25, "sort": "datedesc"},
            headers={"User-Agent": USER_AGENT},
            timeout=20,
        )
        response.raise_for_status()
        articles = response.json().get("articles", []) if response.text.strip().startswith("{") else []
    except Exception as error:  # noqa: BLE001
        _log("news search failed", area=name, error=str(error)[:150])
        return []
    seen, unique = set(), []
    for article in articles:
        title = _plain(article.get("title") or "")
        if title and title not in seen:
            seen.add(title)
            unique.append(article)
    return unique


def fetch_article_text(url: str, limit: int = 5000) -> str:
    """Readable text of an article page (best effort, first `limit` chars)."""
    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=12)
        response.raise_for_status()
    except Exception:  # noqa: BLE001
        return ""
    page = re.sub(r"(?is)<(script|style|nav|header|footer|aside|form)[^>]*>.*?</\1>", " ", response.text)
    paragraphs = re.findall(r"(?is)<p[^>]*>(.*?)</p>", page)
    text = " ".join(re.sub(r"<[^>]+>", " ", p) for p in paragraphs) or re.sub(r"<[^>]+>", " ", page)
    return " ".join(html.unescape(text).split())[:limit]


# --------------------------------------------------------- 3. AI extraction

def _ai_complete(system: str, user: str) -> str:
    if AI_PROVIDER == "anthropic":
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": os.getenv("ANTHROPIC_API_KEY", ""), "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": AI_MODEL, "max_tokens": 500, "system": system,
                  "messages": [{"role": "user", "content": user}]},
            timeout=60,
        )
        response.raise_for_status()
        return "".join(b.get("text", "") for b in response.json().get("content", []) if b.get("type") == "text")
    url = ("https://models.github.ai/inference/chat/completions" if AI_PROVIDER == "github"
           else os.getenv("OLLAMA_URL", "http://localhost:11434") + "/v1/chat/completions")
    token = os.getenv("GITHUB_MODELS_TOKEN") or os.getenv("GITHUB_TOKEN") or "local"
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"model": AI_MODEL, "max_tokens": 500, "temperature": 0.1,
              "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
        timeout=90,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def extract_records(area: dict, article: dict, text: str) -> list[dict]:
    """Structured, validated records from one article (may be empty)."""
    user = (f"Neighbourhood: {area['name']}, {area['city']}\n"
            f"Article title: {article.get('title')}\nPublished: {article.get('seendate')}\n"
            f"Article text: {text}")
    raw = _ai_complete(EXTRACTION_PROMPT, user)
    try:
        data = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
    except ValueError:
        return []
    records = []
    for item in (data.get("records") or [])[:2]:
        summary = " ".join(str(item.get("summary") or "").split())
        if (item.get("about_this_neighbourhood") is not True
                or item.get("sentiment") not in ALLOWED_SENTIMENTS
                or item.get("evidence_tier") not in ALLOWED_TIERS
                or not summary or len(summary.split()) > 45):
            continue
        records.append({
            "sentiment": item["sentiment"],
            "evidence_tier": item["evidence_tier"],
            "time_context": item.get("time_context") if item.get("time_context") in ALLOWED_TIMES else "unspecified",
            "summary": summary,
        })
    return records


def _seen_date(article: dict) -> Optional[str]:
    raw = str(article.get("seendate") or "")
    return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}" if len(raw) >= 8 and raw[:8].isdigit() else None


# ------------------------------------------------------- 4. store and load

def gather_area(area: dict) -> dict:
    """News -> AI -> records for one neighbourhood; saves to Supabase."""
    db = _db()
    key = area_key(area["city"], area["name"])
    rows: list[dict] = []
    articles = search_news(area["name"], area["city"])
    for article in articles[: MAX_ARTICLES_PER_AREA * 3]:
        if len({r["source_url"] for r in rows}) >= MAX_ARTICLES_PER_AREA:
            break
        text = fetch_article_text(article.get("url") or "")
        if len(text) < 300 or _plain(area["name"]) not in _plain(text):
            continue  # paywalled, empty, or doesn't actually mention the area
        try:
            extracted = extract_records(area, article, text)
        except Exception as error:  # noqa: BLE001  (rate limit, bad JSON...)
            _log("AI extraction failed", area=area["name"], error=str(error)[:150])
            continue
        for record in extracted:
            rows.append({
                "id": str(uuid.uuid4()),
                "area_key": key, "area_name": area["name"], "city": area["city"],
                "country_code": area.get("country_code"),
                "latitude": area["latitude"], "longitude": area["longitude"],
                "radius_meters": area["radius_meters"],
                "sentiment": record["sentiment"], "evidence_tier": record["evidence_tier"],
                "time_context": record["time_context"], "summary": record["summary"],
                "source_url": article.get("url"), "source_name": article.get("domain"),
                "post_date": _seen_date(article),
                "origin": "auto", "status": "active",
                "verification_weight": AUTO_VERIFICATION_WEIGHT,
                "model": f"{AI_PROVIDER}:{AI_MODEL}",
            })
        time.sleep(2)  # stay inside free AI rate limits
    if db is not None:
        if rows:
            db.table(RECORDS_TABLE).insert(rows).execute()
        db.table(AREAS_TABLE).upsert({
            "area_key": key, "area_name": area["name"], "city": area["city"],
            "latitude": area["latitude"], "longitude": area["longitude"],
            "radius_meters": area["radius_meters"],
            "last_gathered_at": datetime.now(timezone.utc).isoformat(),
            "status": "done", "record_count": len(rows), "articles_found": len(articles),
        }).execute()
    _area_cache.pop(key, None)
    _records_cache.clear()
    result = {"area": area["name"], "records": len(rows), "articles_found": len(articles)}
    _log("area gathered", **result, saved=db is not None)
    return result


_records_cache: dict[tuple, tuple[float, list[dict]]] = {}
_area_cache: dict[str, tuple[float, Optional[dict]]] = {}


def load_records_near(bbox: tuple[float, float, float, float]) -> list[dict]:
    """Active stored records around a route, in the analysis record format."""
    south, west, north, east = bbox
    pad = 0.02
    key = tuple(round(v, 2) for v in bbox)
    cached = _records_cache.get(key)
    if cached and time.time() - cached[0] < 600:
        return cached[1]
    db = _db()
    rows: list[dict] = []
    if db is not None:
        try:
            rows = (db.table(RECORDS_TABLE).select("*").eq("status", "active")
                    .gte("latitude", south - pad).lte("latitude", north + pad)
                    .gte("longitude", west - pad).lte("longitude", east + pad)
                    .limit(300).execute().data) or []
        except Exception:  # noqa: BLE001  (table missing, network...)
            rows = []
    records = [{
        "record_id": f"auto-{row['id']}",
        "rating_eligible": True,
        "area": row.get("area_name"), "neighbourhood": row.get("area_name"),
        "street_or_place": None,
        "overall_sentiment": row.get("sentiment"),
        "evidence_tier": row.get("evidence_tier"),
        "time_context": row.get("time_context") or "unspecified",
        "opinion_summary": row.get("summary"),
        "source_platform": row.get("source_name"),
        "source_url": row.get("source_url"),
        "post_date": row.get("post_date"),
        "confidence": 2,
        "origin": row.get("origin") or "auto",
        "verification_weight": float(row.get("verification_weight") or AUTO_VERIFICATION_WEIGHT),
        "locations": [{
            "type": "circle", "latitude": row["latitude"], "longitude": row["longitude"],
            "radius_meters": row.get("radius_meters") or 600, "precision": "neighbourhood",
        }],
    } for row in rows]
    _records_cache[key] = (time.time(), records)
    return records


# ------------------------------------------------- background orchestration

_trip_jobs: dict[tuple, dict] = {}
_worker_lock = threading.Lock()   # one gathering job at a time (polite + free tiers)
GATHERING: set[str] = set()


def _area_record_count(key: str) -> int:
    cached = _area_cache.get(key)
    row = cached[1] if cached else None
    return int((row or {}).get("record_count") or 0)


def _area_known(key: str) -> bool:
    cached = _area_cache.get(key)
    if cached and time.time() - cached[0] < 600:
        row = cached[1]
    else:
        row = None
        db = _db()
        if db is not None:
            try:
                data = db.table(AREAS_TABLE).select("last_gathered_at,record_count").eq("area_key", key).limit(1).execute().data
                row = data[0] if data else None
            except Exception:  # noqa: BLE001
                row = None
        _area_cache[key] = (time.time(), row)
    if not row or not row.get("last_gathered_at"):
        return False
    gathered = datetime.fromisoformat(str(row["last_gathered_at"]).replace("Z", "+00:00"))
    return datetime.now(timezone.utc) - gathered < timedelta(days=REGATHER_AFTER_DAYS)


def _trip_key(route_geometries) -> tuple:
    points = [p for g in route_geometries for p in g]
    return (round(min(p[1] for p in points), 2), round(min(p[0] for p in points), 2),
            round(max(p[1] for p in points), 2), round(max(p[0] for p in points), 2))


def ensure_context(route_geometries: list[list[tuple[float, float]]]) -> dict:
    """Starts background gathering for this trip's area if needed (never blocks).

    Returns {"status": "gathering" | "ready" | "none_found", "areas_checked": [...]}.
    The app asks again every few seconds while the status is "gathering".
    """
    if not route_geometries or not any(route_geometries):
        return {"status": "ready", "areas_checked": []}
    key = _trip_key(route_geometries)
    job = _trip_jobs.get(key)
    if job and (job["state"] == "running" or time.time() - job.get("finished_at", 0) < 3600):
        return {"status": job["state"] if job["state"] == "running" else job["result"],
                "areas_checked": list(job["areas"])}

    job = {"state": "running", "areas": [], "result": "ready"}
    _trip_jobs[key] = job

    def run() -> None:
        with _worker_lock:
            found_any = False
            try:
                for area in neighbourhoods_along(route_geometries):
                    akey = area_key(area["city"], area["name"])
                    if _area_known(akey):
                        count = _area_record_count(akey)
                        job["areas"].append({"area": area["name"], "status": f"checked earlier: {count} notes"})
                        found_any = found_any or count > 0
                        continue
                    GATHERING.add(akey)
                    try:
                        result = gather_area(area)
                        found_any = found_any or result["records"] > 0
                        job["areas"].append({
                            "area": area["name"],
                            "status": f"{result['records']} notes from {result['articles_found']} articles",
                        })
                    finally:
                        GATHERING.discard(akey)
                if not job["areas"]:
                    _log("no neighbourhoods found along route")
            except Exception as error:  # noqa: BLE001
                _log("gathering crashed", error=str(error)[:200])
            job["result"] = "ready" if found_any else "none_found"
            job["finished_at"] = time.time()
            job["state"] = "done"

    threading.Thread(target=run, daemon=True, name="local-context").start()
    return {"status": "gathering", "areas_checked": []}


def diagnose() -> dict:
    """Checks every outside service local context depends on, right now."""
    def timed(fn):
        start = time.monotonic()
        try:
            detail = fn()
            return f"ok in {time.monotonic() - start:.1f} s" + (f" ({detail})" if detail else "")
        except Exception as error:  # noqa: BLE001
            return f"FAILED after {time.monotonic() - start:.1f} s: {str(error)[:160]}"

    def nominatim():
        r = requests.get(NOMINATIM_URL, params={"lat": 46.0511, "lon": 14.5051, "format": "jsonv2", "zoom": 15},
                         headers={"User-Agent": USER_AGENT}, timeout=10)
        r.raise_for_status()
        return (r.json().get("address") or {}).get("suburb") or "answered"

    def gdelt():
        r = requests.get(GDELT_URL, params={"query": '"Ljubljana"', "mode": "artlist", "format": "json", "maxrecords": 5},
                         headers={"User-Agent": USER_AGENT}, timeout=20)
        r.raise_for_status()
        return f"{len(r.json().get('articles', []))} articles" if r.text.strip().startswith("{") else r.text[:80]

    def ai():
        if AI_PROVIDER == "github" and not (os.getenv("GITHUB_MODELS_TOKEN") or os.getenv("GITHUB_TOKEN")):
            raise RuntimeError("GITHUB_MODELS_TOKEN is not set on Render")
        return _ai_complete("Reply with the single word: ok", "ping").strip()[:20]

    def supabase():
        db = _db()
        if db is None:
            raise RuntimeError("Supabase client not configured (SUPABASE_URL / SUPABASE_SECRET_KEY)")
        rows = db.table(AREAS_TABLE).select("area_key").limit(1).execute().data
        records = db.table(RECORDS_TABLE).select("id").limit(1000).execute().data
        return f"tables exist, {len(records)} notes stored"

    return {
        "place_names_nominatim": timed(nominatim),
        "news_gdelt": timed(gdelt),
        "ai_model": f"{AI_PROVIDER}:{AI_MODEL} -> " + timed(ai),
        "database_supabase": timed(supabase),
        "recent_activity": list(GATHER_LOG)[:15],
    }
