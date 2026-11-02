"""Fill Meili's map tile cache from this computer.

The public Overpass servers now refuse cloud servers (Render, Supabase,
Cloudflare), but answer normal connections. This runs the backend's own
prefill (tile_cache.start_prefill) here, downloads directly from Overpass,
and saves the tiles to the shared Supabase Storage cache that the backend
reads, so routes in these cities are fast and complete for everyone.

Usage (from the meili folder):
    python3 backend/tools/prefill_from_mac.py             # every city
    python3 backend/tools/prefill_from_mac.py paris rome  # only these
    python3 backend/tools/prefill_from_mac.py --list

Needs backend/.env with SUPABASE_URL and SUPABASE_SECRET_KEY. Tiles already
cached and fresher than 30 days are skipped, so re-runs are cheap.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
CITIES_FILE = Path(__file__).with_name("prefill_cities.txt")


def load_cities() -> dict[str, tuple[float, float, float, float]]:
    cities = {}
    for line in CITIES_FILE.read_text().splitlines():
        parts = line.split("#", 1)[0].split()
        if len(parts) == 5:
            name, *numbers = parts
            cities[name] = tuple(float(n) for n in numbers)  # south, west, north, east
    return cities


def main(argv: list[str]) -> int:
    cities = load_cities()
    if "--list" in argv:
        for name, box in cities.items():
            print(f"{name:12} {box}")
        return 0
    chosen = [a.lower() for a in argv if not a.startswith("-")] or list(cities)
    unknown = [c for c in chosen if c not in cities]
    if unknown:
        print(f"Unknown city: {', '.join(unknown)}. Add it to {CITIES_FILE.name} or run with --list.")
        return 2

    sys.path.insert(0, str(BACKEND))
    from dotenv import load_dotenv
    load_dotenv(BACKEND / ".env")
    # Download directly from this computer, not through the cloud proxy.
    os.environ.pop("MEILI_PROXY_URL", None)
    os.environ.pop("MEILI_PROXY_SECRET", None)
    if not (os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SECRET_KEY")):
        print("backend/.env needs SUPABASE_URL and SUPABASE_SECRET_KEY to save tiles.")
        return 2

    import tile_cache
    if not tile_cache.STORE.persistent():
        print("Can't reach Supabase Storage with the keys in backend/.env.")
        return 2

    failed = []
    for name in chosen:
        box = cities[name]
        status = tile_cache.start_prefill(box)
        print(f"\n{name}: {status['tiles_total']} tiles in {status['groups_total']} groups")
        while status.get("state") == "running":
            time.sleep(10)
            status = tile_cache.prefill_status()
            print(f"  {status['groups_done']}/{status['groups_total']} groups · "
                  f"{status['tiles_downloaded']} downloaded · {status['tiles_already_cached']} already cached · "
                  f"{len(status['errors'])} errors", flush=True)
        for error in status.get("errors", []):
            print(f"  ! {error}")
        if status.get("errors"):
            failed.append(name)
        # A fresh status for the next city.
        tile_cache._PREFILL_STATUS["state"] = "idle"

    print("\nDone." if not failed else f"\nFinished with errors in: {', '.join(failed)}. Re-run them; finished tiles are kept.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
