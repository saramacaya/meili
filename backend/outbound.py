"""Outbound requests to free public data services, with a proxy fallback.

Some public services (OpenStreetMap Overpass, GDELT news) block or rate-limit
Render's shared IP addresses. If MEILI_PROXY_URL and MEILI_PROXY_SECRET are
set, requests can go through Meili's own Supabase Edge Function instead,
which forwards them from Supabase's servers.
"""

from __future__ import annotations

import os
from typing import Optional

import requests

PROXY_URL = os.getenv("MEILI_PROXY_URL", "").strip()
PROXY_SECRET = os.getenv("MEILI_PROXY_SECRET", "").strip()


def proxy_available() -> bool:
    return bool(PROXY_URL and PROXY_SECRET)


def via_proxy(url: str, *, form: Optional[dict] = None, params: Optional[dict] = None,
              timeout: float = 60) -> requests.Response:
    """Same request, sent through the Supabase Edge Function."""
    if not proxy_available():
        raise RuntimeError("proxy not configured (MEILI_PROXY_URL / MEILI_PROXY_SECRET)")
    full_url = requests.Request("GET", url, params=params).prepare().url if params else url
    response = requests.post(
        PROXY_URL,
        json={"url": full_url, "method": "POST" if form else "GET", "form": form},
        headers={"x-meili-secret": PROXY_SECRET, "Content-Type": "application/json"},
        timeout=timeout,
    )
    response.raise_for_status()
    return response


def nominatim_get(url: str, params: dict, user_agent: str, timeout: float = 12) -> requests.Response:
    """Nominatim GET: through the proxy when configured (Render's shared IPs
    get HTTP 429), falling back to a direct request if the proxy fails.
    The caller keeps Nominatim's 1 request/second pacing."""
    if proxy_available():
        try:
            return via_proxy(url, params=params, timeout=timeout + 5)
        except Exception:  # noqa: BLE001 - fall back to a direct request
            pass
    response = requests.get(url, params=params, headers={"User-Agent": user_agent}, timeout=timeout)
    response.raise_for_status()
    return response
