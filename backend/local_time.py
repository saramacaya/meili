"""The local time where a walk happens, whatever time zone the phone is in.

The app sends the departure time with the phone's own offset (e.g. +08:00 when
Sara tests a Spanish route from China). Opening hours and "at night" reports
are about the clock time AT THE ROUTE, so convert there first.

Uses timezonefinder (offline, no network). If it isn't installed or finds
nothing (open sea), falls back to Europe/Madrid inside Spain's rough box and to
a fixed offset from the longitude elsewhere.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone, tzinfo
from functools import lru_cache
from zoneinfo import ZoneInfo

DEFAULT_ZONE = ZoneInfo("Europe/Madrid")

try:
    from timezonefinder import TimezoneFinder
    _finder = TimezoneFinder()
except Exception:  # noqa: BLE001  (package missing: fall back below)
    _finder = None


@lru_cache(maxsize=512)
def _zone_for_cell(lat_cell: float, lon_cell: float) -> tzinfo:
    if _finder is not None:
        try:
            name = _finder.timezone_at(lng=lon_cell, lat=lat_cell)
            if name:
                return ZoneInfo(name)
        except Exception:  # noqa: BLE001
            pass
    if 35.0 <= lat_cell <= 44.5 and -10.0 <= lon_cell <= 4.5:
        return DEFAULT_ZONE
    return timezone(timedelta(hours=round(lon_cell / 15)))


def zone_at(longitude: float, latitude: float) -> tzinfo:
    """Time zone at a point (cached per ~1 km)."""
    return _zone_for_cell(round(float(latitude), 2), round(float(longitude), 2))


def to_local(moment: datetime, longitude: float, latitude: float) -> datetime:
    """The same instant, expressed in the time zone at (longitude, latitude).
    A time without a zone is taken as UTC."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(zone_at(longitude, latitude))
