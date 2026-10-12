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

import outbound

USER_AGENT = "Meili safety-routing prototype (contact: github.com/saramacaya/meili)"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

AUTO_VERIFICATION_WEIGHT = 0.7
REGATHER_AFTER_DAYS = 90
MAX_ARTICLES_PER_AREA = 6
MIN_RADIUS_M, MAX_RADIUS_M = 300.0, 1500.0
RECORDS_TABLE = "area_context_records"
AREAS_TABLE = "area_context_areas"

# AI provider. All except "anthropic" use the same OpenAI-style chat format.
#   gemini  Google AI Studio free tier  (key: GEMINI_API_KEY)
#   groq    Groq free tier              (key: GROQ_API_KEY)
#   github  GitHub Models               (key: GITHUB_MODELS_TOKEN)
#   ollama  a model on your own computer
#   anthropic  Claude (paid)            (key: ANTHROPIC_API_KEY)
# Default: the first provider whose key is set on the server.
def _default_provider() -> str:
    for name, env in (("gemini", "GEMINI_API_KEY"), ("groq", "GROQ_API_KEY"),
                      ("anthropic", "ANTHROPIC_API_KEY"), ("github", "GITHUB_MODELS_TOKEN")):
        if os.getenv(env):
            return name
    return "github"


AI_PROVIDER = os.getenv("MEILI_AI_PROVIDER", _default_provider()).lower()
AI_MODELS = {
    "gemini": "gemini-3.8-flash",
    "groq": "llama-3.3-70b-versatile",
    "github": "openai/gpt-4o-mini",
    "ollama": "llama3.1:8b",
    "anthropic": "claude-haiku-5-5",
}
AI_MODEL = os.getenv("MEILI_CONTEXT_MODEL", AI_MODELS.get(AI_PROVIDER, ""))
AI_URLS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "github": "https://models.github.ai/inference/chat/completions",
}
AI_KEYS = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY", "github": "GITHUB_MODELS_TOKEN"}

ALLOWED_SENTIMENTS = {"negative", "positive", "mixed"}
ALLOWED_TIERS = {"confirmed_incident", "resident_complaint", "perception_only", "municipal_data"}
ALLOWED_TIMES = {"daytime", "evening", "late_night", "weekend", "unspecified"}

EXTRACTION_PROMPT = (
    "You extract walking-safety evidence about one specific neighbourhood from one web page "
    "(a news article, forum or Reddit thread, travel guide, review or blog). "
    "Use ONLY what the page states. Never add facts, numbers, causes or opinions of your own. "
    "Count: reported incidents (crime, harassment, assaults, robberies), police activity, lighting, "
    "and clearly stated impressions from residents or visitors about how safe the streets feel "
    "(e.g. 'fine during the day, I avoid the side streets at night'). "
    "Ignore anything not about street safety IN the named neighbourhood, generic city-wide claims, "
    "advertising, and property prices. Do not name private individuals. "
    "Evidence tiers: confirmed_incident = a specific reported event; municipal_data = official statistics; "
    "resident_complaint = residents or local groups describing problems they experience; "
    "perception_only = general impressions or visitor opinions. "
    "Return JSON only, no prose: {\"records\": [ {\"sentiment\": \"negative|positive|mixed\", "
    "\"evidence_tier\": \"confirmed_incident|resident_complaint|perception_only|municipal_data\", "
    "\"time_context\": \"daytime|evening|late_night|weekend|unspecified\", "
    "\"summary\": \"one neutral English sentence, max 35 words, saying who said it (e.g. 'Residents on a forum say…')\", "
    "\"about_this_neighbourhood\": true } ] }. "
    "Return at most 2 records. Return {\"records\": []} if the page has nothing relevant."
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


SEARCH_VERSION = 3   # bump when the search improves: every area gets re-checked once


def area_key(city: str, name: str) -> str:
    return f"v{SEARCH_VERSION}|{_plain(city)}|{_plain(name)}"


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


MAX_AREAS_PER_TRIP = 5
LOOKUP_EVERY_METERS = 600
MAX_LOOKUPS = 7


def neighbourhoods_along(route_geometries: list[list[tuple[float, float]]]) -> list[dict]:
    """Distinct neighbourhoods along the trip: a lookup about every 600 m of the
    longest route (start and end always included), up to 5 areas."""
    main = max((g for g in route_geometries if g), key=len, default=[])
    if not main:
        return []
    picks, walked, last_pick = [main[0]], 0.0, 0.0
    for a, b in zip(main, main[1:]):
        walked += _distance_m(a, b)
        if walked - last_pick >= LOOKUP_EVERY_METERS:
            picks.append(b)
            last_pick = walked
    picks.append(main[-1])
    if len(picks) > MAX_LOOKUPS:   # keep them spread evenly
        step = (len(picks) - 1) / (MAX_LOOKUPS - 1)
        picks = [picks[round(i * step)] for i in range(MAX_LOOKUPS)]
    found: dict[str, dict] = {}
    for lon, lat in picks:
        area = reverse_neighbourhood(lon, lat)
        if area:
            found.setdefault(area_key(area["city"], area["name"]), area)
        if len(found) >= MAX_AREAS_PER_TRIP:
            break
    return list(found.values())


# ---------------------------------------------------------- 2. news search

def search_news(name: str, city: str) -> list[dict]:
    """Recent articles mentioning the neighbourhood and its city (GDELT)."""
    query = f'"{name}" "{city}"'
    params = {"query": query, "mode": "artlist", "format": "json", "maxrecords": 25, "sort": "datedesc"}
    try:
        # Render's shared IP is often over GDELT's limit: prefer Meili's proxy.
        if outbound.proxy_available():
            response = outbound.via_proxy(GDELT_URL, params=params, timeout=30)
        else:
            response = requests.get(GDELT_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=20)
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


GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"


# Local language per country, and safety words in that language, so the
# search finds local news about street safety rather than concerts and markets.
LANGUAGE_BY_COUNTRY = {
    "ES": "es", "MX": "es", "AR": "es", "CO": "es", "CL": "es", "PE": "es",
    "PT": "pt", "BR": "pt", "IT": "it", "FR": "fr", "BE": "fr", "DE": "de", "AT": "de",
    "CH": "de", "NL": "nl", "SI": "sl", "HR": "hr", "PL": "pl", "CZ": "cs", "RO": "ro",
    "GR": "el", "TR": "tr", "CN": "zh", "JP": "ja", "KR": "ko",
}
SAFETY_WORDS = {
    "es": "robo OR atraco OR agresión OR inseguridad OR policía OR vandalismo OR seguridad",
    "pt": "assalto OR roubo OR agressão OR insegurança OR polícia OR segurança",
    "it": "furto OR rapina OR aggressione OR sicurezza OR polizia OR degrado",
    "fr": "vol OR agression OR insécurité OR police OR sécurité",
    "de": "Diebstahl OR Überfall OR Polizei OR Sicherheit OR Gewalt",
    "nl": "diefstal OR overval OR politie OR veiligheid",
    "sl": "kraja OR rop OR policija OR varnost OR napad",
    "en": "robbery OR theft OR assault OR police OR safety OR crime",
}


def _name_variants(name: str) -> list[str]:
    """'La Antigua - Santa Cruz' -> ['La Antigua - Santa Cruz', 'La Antigua', 'Santa Cruz']"""
    parts = [p.strip() for p in re.split(r"\s+-\s+|/|,|\(|\)", name) if p.strip()]
    variants = [name] + [p for p in parts if p != name and len(p) > 3]
    return list(dict.fromkeys(variants))[:3]


def search_news_google(name: str, city: str, country_code: str = "") -> list[dict]:
    """News search (Google News RSS) in the local language, steered to safety topics."""
    import xml.etree.ElementTree as ET
    country = (country_code or "US").upper()
    language = LANGUAGE_BY_COUNTRY.get(country, "en")
    words = SAFETY_WORDS.get(language, SAFETY_WORDS["en"])
    names = " OR ".join(f'"{variant}"' for variant in _name_variants(name))
    params = {"q": f"({names}) {city} ({words}) when:1y", "hl": language, "gl": country,
              "ceid": f"{country}:{language}"}
    try:
        response = requests.get(GOOGLE_NEWS_RSS, params=params, headers={"User-Agent": USER_AGENT}, timeout=15)
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except Exception as error:  # noqa: BLE001
        _log("news search failed", area=name, error=str(error)[:150])
        return []
    articles = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        description = re.sub(r"<[^>]+>", " ", item.findtext("description") or "")
        source = item.find("source")
        pub = item.findtext("pubDate") or ""
        try:
            seen = datetime.strptime(pub[:16], "%a, %d %b %Y").strftime("%Y%m%dT000000Z")
        except ValueError:
            seen = ""
        articles.append({
            "title": title, "url": item.findtext("link"), "seendate": seen,
            "domain": (source.text if source is not None else "news"),
            "snippet": " ".join(html.unescape(description).split()),
        })
    _log("news found", area=name, language=language, articles=len(articles))
    return articles[:25]


TAVILY_URL = "https://api.tavily.com/search"
OPINION_PHRASES = {
    "es": "{name} {city} es seguro de noche opiniones vivir en el barrio",
    "pt": "{name} {city} é seguro à noite opiniões viver no bairro",
    "it": "{name} {city} è sicuro di notte opinioni vivere nel quartiere",
    "fr": "{name} {city} quartier sûr la nuit avis habitants",
    "de": "{name} {city} sicher nachts Erfahrungen wohnen im Viertel",
    "nl": "{name} {city} veilig 's nachts ervaringen wonen in de wijk",
    "sl": "{name} {city} varno ponoči mnenja življenje v soseski",
}


def search_opinions(name: str, city: str, country_code: str = "") -> list[dict]:
    """Opinions about walking safety in an area from the whole web (Tavily):
    resident and visitor discussions, forums, guides, local news. Returns the
    page text too, so the AI reads the actual discussion, not just a headline."""
    key = os.getenv("TAVILY_API_KEY", "").strip()
    if not key:
        return []
    language = LANGUAGE_BY_COUNTRY.get((country_code or "").upper(), "en")
    queries = [f'is {name} in {city} safe to walk at night? residents and visitors opinions']
    if language in OPINION_PHRASES:
        queries.append(OPINION_PHRASES[language].format(name=name, city=city))
    pages: dict[str, dict] = {}
    for query in queries:
        try:
            response = requests.post(
                TAVILY_URL,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"query": query, "search_depth": "basic", "max_results": 8,
                      "include_raw_content": "text", "time_range": "year"},
                timeout=20,
            )
            response.raise_for_status()
            results = response.json().get("results", [])
        except Exception as error:  # noqa: BLE001
            _log("opinion search failed", area=name, error=str(error)[:150])
            continue
        for item in results:
            url = item.get("url")
            if not url or url in pages:
                continue
            raw = " ".join(str(item.get("raw_content") or "").split())
            pages[url] = {
                "title": item.get("title") or "",
                "url": url,
                "domain": url.split("/")[2] if "//" in url else "web",
                "seendate": (str(item.get("published_date") or "")[:10].replace("-", "") + "T000000Z")
                            if item.get("published_date") else "",
                "snippet": " ".join(str(item.get("content") or "").split()),
                "text": raw[:6000],   # already-read page text: no need to fetch it again
                "kind": "opinion_search",
            }
    _log("opinions found", area=name, pages=len(pages), searches=len(queries))
    return list(pages.values())


from html.parser import HTMLParser


class _ParagraphText(HTMLParser):
    """Collects text inside <p> tags, skipping scripts and styles.
    A streaming parser: unlike regular expressions over the whole page it never
    blocks the server for long, even on huge pages."""

    def __init__(self, limit: int):
        super().__init__(convert_charrefs=True)
        self.limit, self.parts, self.size = limit, [], 0
        self.in_p = 0
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript", "nav", "header", "footer", "aside", "form"):
            self.skip += 1
        elif tag == "p":
            self.in_p += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "nav", "header", "footer", "aside", "form"):
            self.skip = max(0, self.skip - 1)
        elif tag == "p":
            self.in_p = max(0, self.in_p - 1)

    def handle_data(self, data):
        if self.in_p and not self.skip and self.size < self.limit:
            text = " ".join(data.split())
            if text:
                self.parts.append(text)
                self.size += len(text) + 1


MAX_PAGE_BYTES = 400_000   # never read more than this of any page


def fetch_article_text(url: str, limit: int = 5000) -> str:
    """Readable text of an article page (best effort, first `limit` chars)."""
    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=6, stream=True)
        response.raise_for_status()
        raw = response.raw.read(MAX_PAGE_BYTES, decode_content=True) or b""
        response.close()
    except Exception:  # noqa: BLE001
        return ""
    page = raw.decode(response.encoding or "utf-8", errors="replace")
    parser = _ParagraphText(limit)
    # Feed in chunks and pause briefly so search requests always get CPU time
    # on Render's small instance.
    for start in range(0, len(page), 20_000):
        parser.feed(page[start:start + 20_000])
        if parser.size >= limit:
            break
        time.sleep(0.005)
    return " ".join(parser.parts)[:limit]


# --------------------------------------------------------- 3. AI extraction

def _ai_complete(system: str, user: str) -> str:
    if AI_PROVIDER == "anthropic":
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": os.getenv("ANTHROPIC_API_KEY", ""), "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": AI_MODEL, "max_tokens": 2048, "system": system,
                  "messages": [{"role": "user", "content": user}]},
            timeout=60,
        )
        response.raise_for_status()
        return "".join(b.get("text", "") for b in response.json().get("content", []) if b.get("type") == "text")
    url = AI_URLS.get(AI_PROVIDER) or (os.getenv("OLLAMA_URL", "http://localhost:11434") + "/v1/chat/completions")
    token = os.getenv(AI_KEYS.get(AI_PROVIDER, ""), "") or os.getenv("GITHUB_TOKEN") or "local"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    if AI_PROVIDER == "github":
        # Required by GitHub's models API (without them it answers "OK").
        headers["Accept"] = "application/vnd.github+json"
        headers["X-GitHub-Api-Version"] = "2022-11-28"
    payload = {"model": AI_MODEL, "max_tokens": 2048, "temperature": 0.1,
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if AI_PROVIDER == "gemini":
        # Thinking models are slow; extracting facts from an article needs little reasoning.
        payload["reasoning_effort"] = "low"
    # Follow redirects ourselves and keep it a POST: the default behaviour
    # turns a redirected POST into a GET, which never reaches the model.
    # Busy models answer 429/503: wait a moment and retry, then try the
    # fallback model (MEILI_CONTEXT_FALLBACK_MODEL) if one is configured.
    models = [AI_MODEL] + [m for m in [os.getenv("MEILI_CONTEXT_FALLBACK_MODEL", "").strip()] if m]
    for model_index, model_name in enumerate(models):
        payload["model"] = model_name
        for attempt, pause in enumerate((0, 2, 5)):
            time.sleep(pause)
            response = _post_following_redirects(url, headers, payload)
            if response.status_code not in (429, 500, 502, 503, 504):
                break
            _log("AI busy, retrying", model=model_name, status=response.status_code, attempt=attempt + 1)
        if response.status_code not in (429, 500, 502, 503, 504):
            break
    if response.status_code != 200 or "json" not in response.headers.get("Content-Type", ""):
        raise RuntimeError(
            f"HTTP {response.status_code} from {url} "
            f"({response.headers.get('Content-Type', '?')}): {response.text[:300]!r}"
        )
    return response.json()["choices"][0]["message"]["content"]


def _post_following_redirects(url: str, headers: dict, payload: dict):
    hops = []
    for _ in range(4):
        response = requests.post(url, headers=headers, json=payload, timeout=45, allow_redirects=False)
        if response.status_code == 400 and "reasoning_effort" in payload and "reason" in response.text.lower():
            payload.pop("reasoning_effort")   # setting not supported by this model: retry without it
            response = requests.post(url, headers=headers, json=payload, timeout=45, allow_redirects=False)
        location = response.headers.get("Location")
        if response.status_code in (301, 302, 303, 307, 308) and location:
            hops.append(f"{response.status_code} -> {location}")
            url = requests.compat.urljoin(url, location)
            continue
        break
    return response


def extract_records(area: dict, article: dict, text: str) -> list[dict]:
    """Structured, validated records from one article (may be empty)."""
    user = (f"Neighbourhood: {area['name']}, {area['city']} "
            f"(also written as: {', '.join(_name_variants(area['name']))})\n"
            f"Page title: {article.get('title')}\nSource: {article.get('domain')}\n"
            f"Published: {article.get('seendate') or 'unknown'}\nPage text: {text}")
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

def gather_area(area: dict, budget_seconds: float = 40.0) -> dict:
    """News -> AI -> records for one neighbourhood; saves to Supabase.
    Fast path: quick news source first, articles fetched in parallel, two AI
    calls at a time. Typically 10-30 seconds per area."""
    from concurrent.futures import ThreadPoolExecutor as _Pool

    db = _db()
    key = area_key(area["city"], area["name"])
    rows: list[dict] = []
    name_variants = [_plain(v) for v in _name_variants(area["name"])]

    def mentions(text: str) -> bool:
        plain = _plain(text or "")
        return any(v in plain for v in name_variants)

    opinions = search_opinions(area["name"], area["city"], area.get("country_code") or "")
    news = search_news_google(area["name"], area["city"], area.get("country_code") or "")
    if not news and not opinions:
        news = search_news(area["name"], area["city"])
    articles = opinions + news
    # Keep only articles whose headline or snippet mentions the area at all.
    candidates = [a for a in articles
                  if mentions(a.get("title")) or mentions(a.get("snippet")) or mentions(a.get("text"))][:10] \
        or articles[:6]

    def readable(article: dict) -> tuple[dict, str]:
        if len(article.get("text") or "") >= 300:
            return article, article["text"]          # page text already provided by the search
        text = fetch_article_text(article.get("url") or "")
        if len(text) < 300 and article.get("snippet"):
            text = f"{article.get('title', '')}. {article['snippet']}"
        return article, text

    deadline = time.monotonic() + budget_seconds
    pool = _Pool(max_workers=6)
    futures = [pool.submit(readable, c) for c in candidates]
    texts = []
    for future in futures:
        try:
            article, text = future.result(timeout=max(0.5, deadline - time.monotonic() - 15))
        except Exception:  # noqa: BLE001  (slow page: skip it)
            continue
        if mentions(text) or mentions(article.get("title")):
            texts.append((article, text))
    pool.shutdown(wait=False, cancel_futures=True)

    def extract(item: tuple[dict, str]) -> tuple[dict, list[dict]]:
        article, text = item
        try:
            return article, extract_records(area, article, text)
        except Exception as error:  # noqa: BLE001  (rate limit, bad JSON...)
            _log("AI extraction failed", area=area["name"], error=str(error)[:150])
            return article, []

    pool = _Pool(max_workers=2)   # stay inside free AI rate limits
    futures = [pool.submit(extract, item) for item in texts[:MAX_ARTICLES_PER_AREA]]
    results = []
    for future in futures:
        try:
            results.append(future.result(timeout=max(0.5, deadline - time.monotonic())))
        except Exception:  # noqa: BLE001  (AI too slow: keep what we have)
            _log("AI step skipped: time limit", area=area["name"])
    pool.shutdown(wait=False, cancel_futures=True)

    for article, extracted in results:
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
# Up to two trips gather at once, so one slow trip never blocks everyone.
_worker_slots = threading.BoundedSemaphore(2)
JOB_BUDGET_SECONDS = 60      # a trip's gathering stops after this
JOB_WATCHDOG_SECONDS = 80    # never report "gathering" for longer than this
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
    # Areas with no news are checked again sooner: news appears all the time.
    days = REGATHER_AFTER_DAYS if int(row.get("record_count") or 0) > 0 else 14
    return datetime.now(timezone.utc) - gathered < timedelta(days=days)


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
    if job and job["state"] == "running" and time.time() - job["started_at"] > JOB_WATCHDOG_SECONDS:
        # Watchdog: whatever is still running, stop telling the app to wait.
        _log("gathering watchdog stopped a slow trip", areas=len(job["areas"]))
        job["state"] = "done"
        job["finished_at"] = time.time()
        job["result"] = "ready" if job.get("found_any") else "none_found"
    if job and (job["state"] == "running" or time.time() - job.get("finished_at", 0) < 3600):
        return {"status": "gathering" if job["state"] == "running" else job["result"],
                "areas_checked": list(job["areas"])}

    job = {"state": "running", "areas": [], "result": "ready", "started_at": time.time(), "found_any": False}
    _trip_jobs[key] = job

    def run() -> None:
        if not _worker_slots.acquire(timeout=20):
            job["areas"].append({"area": "-", "status": "skipped: server busy, try again later"})
            job.update(state="done", finished_at=time.time(), result="none_found")
            return
        try:
            started = time.monotonic()
            found_any = False
            try:
                for area in neighbourhoods_along(route_geometries):
                    if time.monotonic() - started > JOB_BUDGET_SECONDS:
                        job["areas"].append({"area": area["name"], "status": "skipped: time limit"})
                        continue
                    akey = area_key(area["city"], area["name"])
                    if _area_known(akey):
                        count = _area_record_count(akey)
                        job["areas"].append({"area": area["name"], "status": f"checked earlier: {count} notes"})
                        found_any = found_any or count > 0
                        continue
                    GATHERING.add(akey)
                    try:
                        remaining = max(10.0, JOB_BUDGET_SECONDS - (time.monotonic() - started))
                        result = gather_area(area, budget_seconds=remaining)
                        found_any = found_any or result["records"] > 0
                        job["found_any"] = found_any
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
            if job["state"] == "running":
                job["result"] = "ready" if found_any else "none_found"
                job["finished_at"] = time.time()
                job["state"] = "done"
        finally:
            _worker_slots.release()

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
        params = {"query": '"Ljubljana"', "mode": "artlist", "format": "json", "maxrecords": 5}
        if outbound.proxy_available():
            r = outbound.via_proxy(GDELT_URL, params=params, timeout=30)
        else:
            r = requests.get(GDELT_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=20)
        r.raise_for_status()
        return f"{len(r.json().get('articles', []))} articles" if r.text.strip().startswith("{") else r.text[:80]

    def ai():
        key_name = AI_KEYS.get(AI_PROVIDER)
        if key_name and not os.getenv(key_name):
            raise RuntimeError(f"{key_name} is not set on Render")
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
        "news_gdelt": ("via proxy -> " if outbound.proxy_available() else "direct -> ") + timed(gdelt),
        "news_backup_google": timed(lambda: f"{len(search_news_google('Center', 'Ljubljana', 'SI'))} articles"),
        "opinions_tavily": (timed(lambda: f"{len(search_opinions('Gràcia', 'Barcelona', 'ES'))} pages")
                            if os.getenv("TAVILY_API_KEY") else "not configured (set TAVILY_API_KEY on Render)"),
        "ai_model": f"{AI_PROVIDER}:{AI_MODEL} -> " + timed(ai),
        "database_supabase": timed(supabase),
        "recent_activity": list(GATHER_LOG)[:15],
    }
