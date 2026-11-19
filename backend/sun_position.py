"""Whether it is daylight at a place and time (no network, no extra packages).

Uses the standard NOAA solar-position approximation (accurate to well under a
degree, i.e. a few minutes around sunset), which is plenty for deciding how
much street lighting matters on a walk.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

# Sun altitude (degrees) at which it counts as full daylight / full darkness.
# Between the two (sunset + civil twilight, about 30 minutes) it fades.
DAY_ALTITUDE = 0.0
NIGHT_ALTITUDE = -6.0


def sun_altitude(latitude: float, longitude: float, moment: datetime) -> float:
    """Sun height above the horizon in degrees (negative = below)."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    utc = moment.astimezone(timezone.utc)
    day_of_year = utc.timetuple().tm_yday
    hours = utc.hour + utc.minute / 60 + utc.second / 3600
    gamma = 2 * math.pi / 365 * (day_of_year - 1 + (hours - 12) / 24)
    equation_of_time = 229.18 * (
        0.000075 + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma) - 0.040849 * math.sin(2 * gamma)
    )
    declination = (
        0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma) + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma) + 0.00148 * math.sin(3 * gamma)
    )
    true_solar_minutes = hours * 60 + equation_of_time + 4 * longitude
    hour_angle = math.radians(true_solar_minutes / 4 - 180)
    lat = math.radians(latitude)
    cos_zenith = (math.sin(lat) * math.sin(declination)
                  + math.cos(lat) * math.cos(declination) * math.cos(hour_angle))
    return 90 - math.degrees(math.acos(max(-1.0, min(1.0, cos_zenith))))


def daylight_factor(latitude: float, longitude: float, moment: datetime) -> float:
    """1.0 in daylight, 0.0 at night, in between during twilight."""
    altitude = sun_altitude(latitude, longitude, moment)
    if altitude >= DAY_ALTITUDE:
        return 1.0
    if altitude <= NIGHT_ALTITUDE:
        return 0.0
    return (altitude - NIGHT_ALTITUDE) / (DAY_ALTITUDE - NIGHT_ALTITUDE)
