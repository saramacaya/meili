"""Requires a signed-in meili user on backend calls.

The app already sends the user's Supabase access token on every request
(Authorization: Bearer ...). This middleware checks that token with the app's
Supabase Auth server, so the Google/OpenRouteService-backed endpoints can't be
used by anyone who just finds the Render URL.

Off until both AUTH_SUPABASE_URL and AUTH_SUPABASE_ANON_KEY are set. They must
be the Supabase project the app signs users in with (the Lovable frontend's),
which may differ from SUPABASE_URL used for storage here.
"""

from __future__ import annotations

import hashlib
import os
import time

import httpx
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

# Paths that stay open: health checks for the keepalive workflow, API docs,
# diagnostics opened in a browser, and prefill (it has its own token).
PUBLIC_PATHS = {
    "/health",
    "/system/keepalive",
    "/docs",
    "/docs/oauth2-redirect",
    "/openapi.json",
    "/redoc",
    "/diagnose",
    "/tiles/diagnose",
    "/tiles/prefill",
    "/tiles/prefill/status",
}

VALID_TOKEN_SECONDS = 300
MAX_CACHED_TOKENS = 5000

_valid_tokens: dict[str, float] = {}  # sha256(token) -> when it was confirmed


def enabled() -> bool:
    return bool(os.getenv("AUTH_SUPABASE_URL") and os.getenv("AUTH_SUPABASE_ANON_KEY"))


async def _token_is_valid(token: str) -> bool:
    digest = hashlib.sha256(token.encode()).hexdigest()
    now = time.time()
    confirmed = _valid_tokens.get(digest)
    if confirmed and now - confirmed < VALID_TOKEN_SECONDS:
        return True

    url = os.getenv("AUTH_SUPABASE_URL", "").rstrip("/") + "/auth/v1/user"
    headers = {
        "apikey": os.getenv("AUTH_SUPABASE_ANON_KEY", ""),
        "Authorization": f"Bearer {token}",
    }
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(url, headers=headers)
    if response.status_code != 200:
        return False

    if len(_valid_tokens) >= MAX_CACHED_TOKENS:
        for key, stamp in list(_valid_tokens.items()):
            if now - stamp >= VALID_TOKEN_SECONDS:
                del _valid_tokens[key]
        if len(_valid_tokens) >= MAX_CACHED_TOKENS:
            _valid_tokens.clear()
    _valid_tokens[digest] = now
    return True


class SupabaseAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if (
            not enabled()
            or request.method == "OPTIONS"  # CORS preflight carries no token
            or request.url.path in PUBLIC_PATHS
        ):
            return await call_next(request)

        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            return JSONResponse({"detail": "Sign in required."}, status_code=401)
        try:
            valid = await _token_is_valid(token)
        except httpx.HTTPError:
            # Supabase Auth unreachable: a 503 lets the app retry instead of
            # signing the user out.
            return JSONResponse({"detail": "Sign-in check unavailable, try again."}, status_code=503)
        if not valid:
            return JSONResponse({"detail": "Sign in required."}, status_code=401)
        return await call_next(request)
