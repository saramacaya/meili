import math
import threading
from typing import Optional

import numpy as np

from nasa_route_analysis import load_nasa_month_data


NASA_PERCENTILE_CACHE: dict[str, np.ndarray] = {}
NASA_PERCENTILE_CACHE_LOCK = threading.Lock()


LIGHTING_SOURCE_WEIGHTS = {
    "official_valencia_lamps": 0.45,
    "osm_lit_tag": 0.35,
    "osm_individual_lamps": 0.10,
    "nasa_background": 0.10,
}

# A source that has no usable observation contributes nothing.
# Available sources move the score away from this neutral point.
NEUTRAL_LIGHTING_SCORE = 50.0


def classify_nasa_brightness(
    percentile: Optional[float],
) -> str:
    """Classifies only valid NASA observations; missing data is unknown."""
    if percentile is None:
        return "unknown"
    if percentile < 25:
        return "dark"
    if percentile < 45:
        return "dim"
    if percentile <= 55:
        return "neutral"
    if percentile <= 75:
        return "bright"
    return "very_bright"


def distance_meters(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
) -> float:
    longitude_a, latitude_a = point_a
    longitude_b, latitude_b = point_b
    earth_radius_meters = 6_371_000

    latitude_a_radians = math.radians(latitude_a)
    latitude_b_radians = math.radians(latitude_b)
    latitude_delta = math.radians(latitude_b - latitude_a)
    longitude_delta = math.radians(longitude_b - longitude_a)

    haversine = (
        math.sin(latitude_delta / 2) ** 2
        + math.cos(latitude_a_radians)
        * math.cos(latitude_b_radians)
        * math.sin(longitude_delta / 2) ** 2
    )

    return (
        2
        * earth_radius_meters
        * math.asin(math.sqrt(haversine))
    )


def positive_distance_score(
    distance: Optional[float],
    coverage_radius_meters: float,
) -> Optional[float]:
    """
    Converts distance to a positive lighting-evidence score.

    - Inside the coverage radius: full positive evidence.
    - Between one and two radii: evidence fades linearly.
    - Beyond two radii: no usable positive evidence.
    """
    if distance is None:
        return None

    if distance <= coverage_radius_meters:
        return 1.0

    maximum_evidence_distance = (
        coverage_radius_meters * 2
    )

    if distance >= maximum_evidence_distance:
        return 0.0

    return max(
        0.0,
        1.0
        - (
            distance - coverage_radius_meters
        )
        / coverage_radius_meters,
    )


def classify_combined_score(score: Optional[float]) -> str:
    if score is None:
        return "unknown"

    if score >= 75:
        return "strong_lighting_evidence"

    if score >= 55:
        return "moderate_lighting_evidence"

    if score >= 35:
        return "limited_or_mixed_lighting_evidence"

    return "weak_lighting_evidence"


def calculate_confidence(
    official_available: bool,
    osm_lit_available: bool,
    osm_lamp_positive_evidence: bool,
    nasa_available: bool,
) -> str:
    if official_available and osm_lit_available:
        return "high"

    street_level_source_count = sum([
        official_available,
        osm_lit_available,
        osm_lamp_positive_evidence,
    ])

    if street_level_source_count >= 2:
        return "medium_high"

    if street_level_source_count >= 1 and nasa_available:
        return "medium"

    if street_level_source_count >= 1:
        return "low_medium"

    if nasa_available:
        return "low"

    return "none"


def _get_sorted_nasa_radiance(
    month: str,
) -> np.ndarray:
    with NASA_PERCENTILE_CACHE_LOCK:
        cached_values = NASA_PERCENTILE_CACHE.get(month)

    if cached_values is not None:
        return cached_values

    _, nasa_data, _ = load_nasa_month_data(
        requested_month=month,
    )

    radiance = nasa_data["radiance"]
    quality = nasa_data["quality"]

    valid_mask = (
        np.isfinite(radiance)
        & (quality != 255)
    )

    sorted_values = np.sort(
        radiance[valid_mask].astype(
            np.float32,
            copy=False,
        )
    )

    with NASA_PERCENTILE_CACHE_LOCK:
        NASA_PERCENTILE_CACHE[month] = sorted_values

    return sorted_values


def nasa_brightness_percentile(
    brightness_value: Optional[float],
    sorted_regional_values: np.ndarray,
) -> Optional[float]:
    if (
        brightness_value is None
        or not np.isfinite(brightness_value)
        or sorted_regional_values.size == 0
    ):
        return None

    position = np.searchsorted(
        sorted_regional_values,
        brightness_value,
        side="right",
    )

    return round(
        100
        * float(position)
        / float(sorted_regional_values.size),
        1,
    )


def nearest_official_lamp_result(
    sample: tuple[float, float],
    streetlights: list[
        tuple[float, float]
    ],
    coverage_radius_meters: float,
) -> dict:
    """
    Finds the nearest official Valencia streetlight.

    A nearby official lamp provides positive evidence.

    A very distant lamp does not prove that the route point
    is dark, so it is treated as unavailable evidence rather
    than receiving a score of zero.
    """
    if not streetlights:
        return {
            "available": False,
            "positive_evidence_available": False,
            "nearest_distance_meters": None,
            "covered": None,
            "source_score": None,
        }

    nearest_distance = min(
        distance_meters(
            sample,
            streetlight
        )
        for streetlight in streetlights
    )

    maximum_evidence_distance = (
        coverage_radius_meters * 2
    )

    # Beyond twice the requested radius, the official
    # dataset provides no useful evidence for this point.
    # Do not treat this as confirmed darkness.
    if nearest_distance > maximum_evidence_distance:
        return {
            "available": True,
            "positive_evidence_available": False,
            "nearest_distance_meters": round(
                nearest_distance,
                1,
            ),
            "covered": False,
            "source_score": None,
        }

    distance_score = positive_distance_score(
        nearest_distance,
        coverage_radius_meters,
    )

    return {
        "available": True,
        "positive_evidence_available": True,
        "nearest_distance_meters": round(
            nearest_distance,
            1,
        ),
        "covered": (
            nearest_distance
            <= coverage_radius_meters
        ),
        "source_score": round(
            100 * distance_score,
            1,
        ),
    }

def nearest_osm_lit_result(
    sample: tuple[float, float],
    lit_ways: list[dict],
    match_radius_meters: float,
) -> dict:
    nearest_way = None
    nearest_distance = None

    for way in lit_ways:
        sampled_geometry = way.get(
            "sampled_geometry",
            [],
        )

        if not sampled_geometry:
            continue

        distance_to_way = min(
            distance_meters(sample, way_sample)
            for way_sample in sampled_geometry
        )

        if (
            nearest_distance is None
            or distance_to_way < nearest_distance
        ):
            nearest_distance = distance_to_way
            nearest_way = way

    if (
        nearest_way is None
        or nearest_distance is None
        or nearest_distance > match_radius_meters
    ):
        return {
            "available": False,
            "lighting_evidence": "unknown",
            "lit_value": None,
            "matched_osm_way_id": None,
            "distance_to_way_meters": None,
            "source_score": None,
        }

    classification = nearest_way.get(
        "lit_classification",
        "unknown",
    )

    classification_scores = {
        "lit": 1.0,
        "unlit": 0.0,
        "conditional_or_other": 0.5,
    }

    source_score = classification_scores.get(
        classification
    )

    return {
        "available": source_score is not None,
        "lighting_evidence": classification,
        "lit_value": nearest_way.get(
            "lit_value"
        ),
        "matched_osm_way_id": nearest_way.get(
            "osm_id"
        ),
        "distance_to_way_meters": round(
            nearest_distance,
            1,
        ),
        "source_score": (
            round(100 * source_score, 1)
            if source_score is not None
            else None
        ),
    }


def nearest_osm_lamp_result(
    sample: tuple[float, float],
    street_lamps: list[dict],
    coverage_radius_meters: float,
) -> dict:
    if not street_lamps:
        return {
            "positive_evidence_available": False,
            "nearest_osm_lamp_id": None,
            "nearest_distance_meters": None,
            "covered": None,
            "source_score": None,
        }

    nearest_lamp = min(
        street_lamps,
        key=lambda lamp: distance_meters(
            sample,
            (
                lamp["longitude"],
                lamp["latitude"],
            ),
        ),
    )

    nearest_distance = distance_meters(
        sample,
        (
            nearest_lamp["longitude"],
            nearest_lamp["latitude"],
        ),
    )

    source_score = positive_distance_score(
        nearest_distance,
        coverage_radius_meters,
    )

    # OSM individual-lamp mapping is incomplete. A mapped
    # nearby lamp is positive evidence, but a distant lamp is
    # not treated as proof that the route point is dark.
    positive_evidence_available = (
        source_score is not None
        and source_score > 0
    )

    return {
        "positive_evidence_available": (
            positive_evidence_available
        ),
        "nearest_osm_lamp_id": nearest_lamp.get(
            "osm_id"
        ),
        "nearest_distance_meters": round(
            nearest_distance,
            1,
        ),
        "covered": (
            nearest_distance
            <= coverage_radius_meters
        ),
        "source_score": (
            round(100 * source_score, 1)
            if positive_evidence_available
            else None
        ),
    }


def combine_lighting_sources(
    route_samples: list[tuple[float, float]],
    official_streetlights: list[tuple[float, float]],
    osm_lit_ways: list[dict],
    osm_street_lamps: list[dict],
    nasa_analysis: Optional[dict],
    official_coverage_radius_meters: float,
    osm_lit_match_radius_meters: float,
    osm_lamp_coverage_radius_meters: float,
) -> dict:
    nasa_sample_results = (
        nasa_analysis.get("sample_results", [])
        if nasa_analysis
        else []
    )

    nasa_month = (
        nasa_analysis.get("month")
        if nasa_analysis
        else None
    )

    sorted_regional_radiance = (
        _get_sorted_nasa_radiance(nasa_month)
        if nasa_month
        else np.asarray([], dtype=np.float32)
    )

    sample_results = []
    combined_scores = []

    classification_counts = {
        "strong_lighting_evidence": 0,
        "moderate_lighting_evidence": 0,
        "limited_or_mixed_lighting_evidence": 0,
        "weak_lighting_evidence": 0,
        "unknown": 0,
    }

    confidence_counts = {
        "high": 0,
        "medium_high": 0,
        "medium": 0,
        "low_medium": 0,
        "low": 0,
        "none": 0,
    }

    conflict_count = 0
    official_covered_count = 0
    osm_lit_mapped_count = 0
    osm_lit_positive_count = 0
    osm_lit_negative_count = 0
    osm_lamp_positive_count = 0
    nasa_available_count = 0

    for index, sample in enumerate(route_samples):
        official_result = nearest_official_lamp_result(
            sample=sample,
            streetlights=official_streetlights,
            coverage_radius_meters=(
                official_coverage_radius_meters
            ),
        )

        osm_lit_result = nearest_osm_lit_result(
            sample=sample,
            lit_ways=osm_lit_ways,
            match_radius_meters=(
                osm_lit_match_radius_meters
            ),
        )

        osm_lamp_result = nearest_osm_lamp_result(
            sample=sample,
            street_lamps=osm_street_lamps,
            coverage_radius_meters=(
                osm_lamp_coverage_radius_meters
            ),
        )

        nasa_result = (
            nasa_sample_results[index]
            if index < len(nasa_sample_results)
            else None
        )

        nasa_brightness = (
            nasa_result.get("brightness_radiance")
            if nasa_result
            else None
        )

        nasa_percentile = nasa_brightness_percentile(
            brightness_value=nasa_brightness,
            sorted_regional_values=(
                sorted_regional_radiance
            ),
        )

        nasa_available = (
            nasa_result is not None
            and nasa_result.get("coverage_status")
            == "available"
            and nasa_percentile is not None
        )

        source_values = []
        source_weights = []

        official_score = official_result.get(
            "source_score"
        )

        if official_score is not None:
            source_values.append(official_score)
            source_weights.append(
                LIGHTING_SOURCE_WEIGHTS[
                    "official_valencia_lamps"
                ]
            )

        osm_lit_score = osm_lit_result.get(
            "source_score"
        )

        if osm_lit_score is not None:
            source_values.append(osm_lit_score)
            source_weights.append(
                LIGHTING_SOURCE_WEIGHTS[
                    "osm_lit_tag"
                ]
            )

        osm_lamp_score = osm_lamp_result.get(
            "source_score"
        )

        if osm_lamp_score is not None:
            source_values.append(osm_lamp_score)
            source_weights.append(
                LIGHTING_SOURCE_WEIGHTS[
                    "osm_individual_lamps"
                ]
            )

        if nasa_available:
            source_values.append(nasa_percentile)
            source_weights.append(
                LIGHTING_SOURCE_WEIGHTS[
                    "nasa_background"
                ]
            )

        if source_weights:
            # Centre every available source on 50. A valid low
            # NASA percentile therefore reduces the score, while
            # a missing NASA/OSM/official observation is neutral
            # because it is not added to this calculation at all.
            weighted_effect = sum(
                (value - NEUTRAL_LIGHTING_SCORE) * weight
                for value, weight in zip(
                    source_values,
                    source_weights,
                )
            )

            combined_score = round(
                max(
                    0.0,
                    min(
                        100.0,
                        NEUTRAL_LIGHTING_SCORE
                        + weighted_effect
                    )
                ),
                1,
            )

            combined_scores.append(
                combined_score
            )
        else:
            combined_score = None

        classification = classify_combined_score(
            combined_score
        )

        confidence = calculate_confidence(
            official_available=(
                official_score is not None
            ),
            osm_lit_available=(
                osm_lit_score is not None
            ),
            osm_lamp_positive_evidence=(
                osm_lamp_score is not None
            ),
            nasa_available=nasa_available,
        )

        source_conflict = (
            osm_lit_result.get(
                "lighting_evidence"
            ) == "unlit"
            and (
                official_result.get("covered") is True
                or osm_lamp_result.get("covered") is True
            )
        )

        if source_conflict:
            conflict_count += 1

        if official_result.get("covered") is True:
            official_covered_count += 1

        if osm_lit_score is not None:
            osm_lit_mapped_count += 1

        if (
            osm_lit_result.get("lighting_evidence")
            == "lit"
        ):
            osm_lit_positive_count += 1

        if (
            osm_lit_result.get("lighting_evidence")
            == "unlit"
        ):
            osm_lit_negative_count += 1

        if osm_lamp_score is not None:
            osm_lamp_positive_count += 1

        if nasa_available:
            nasa_available_count += 1

        classification_counts[classification] += 1
        confidence_counts[confidence] += 1

        sample_results.append({
            "sample_number": index + 1,
            "longitude": round(sample[0], 7),
            "latitude": round(sample[1], 7),
            "official_valencia_lamps": (
                official_result
            ),
            "osm_lit": osm_lit_result,
            "osm_individual_lamps": (
                osm_lamp_result
            ),
            "nasa_background": {
                "available": nasa_available,
                "month": nasa_month,
                "brightness_radiance": (
                    nasa_brightness
                ),
                "brightness_percentile_within_processed_region": (
                    nasa_percentile
                ),
                "lighting_evidence": (
                    classify_nasa_brightness(
                        nasa_percentile
                        if nasa_available
                        else None
                    )
                ),
                "quality": (
                    nasa_result.get("quality")
                    if nasa_result
                    else None
                ),
                "supporting_observations": (
                    nasa_result.get(
                        "supporting_observations"
                    )
                    if nasa_result
                    else None
                ),
            },
            "combined_lighting_evidence": {
                "score": combined_score,
                "classification": classification,
                "confidence": confidence,
                "source_conflict": source_conflict,
                "available_weight": round(
                    sum(source_weights),
                    2,
                ),
            },
        })

    total_samples = len(route_samples)

    if combined_scores:
        score_array = np.asarray(
            combined_scores,
            dtype=np.float64,
        )

        score_statistics = {
            "minimum_score": round(
                float(np.min(score_array)),
                1,
            ),
            "median_score": round(
                float(np.median(score_array)),
                1,
            ),
            "mean_score": round(
                float(np.mean(score_array)),
                1,
            ),
            "maximum_score": round(
                float(np.max(score_array)),
                1,
            ),
        }
    else:
        score_statistics = {
            "minimum_score": None,
            "median_score": None,
            "mean_score": None,
            "maximum_score": None,
        }

        

    positive_evidence_count = (
        classification_counts[
            "strong_lighting_evidence"
        ]
        + classification_counts[
            "moderate_lighting_evidence"
        ]
    )

    return {
        "route_sample_count": total_samples,
        "combined_score_statistics": score_statistics,
        "combined_positive_evidence_sample_count": (
            positive_evidence_count
        ),
        "combined_positive_evidence_percentage": round(
            100
            * positive_evidence_count
            / total_samples
        ) if total_samples else 0,
        "classification_counts": classification_counts,
        "confidence_counts": confidence_counts,
        "source_conflict_sample_count": conflict_count,
        "source_coverage": {
            "official_lamp_covered_sample_percentage": round(
                100
                * official_covered_count
                / total_samples
            ) if total_samples else 0,
            "osm_lit_mapped_sample_percentage": round(
                100
                * osm_lit_mapped_count
                / total_samples
            ) if total_samples else 0,
            "osm_lit_positive_sample_percentage": round(
                100
                * osm_lit_positive_count
                / total_samples
            ) if total_samples else 0,
            "osm_lit_negative_sample_percentage": round(
                100
                * osm_lit_negative_count
                / total_samples
            ) if total_samples else 0,
            "osm_individual_lamp_positive_evidence_percentage": round(
                100
                * osm_lamp_positive_count
                / total_samples
            ) if total_samples else 0,
            "nasa_available_sample_percentage": round(
                100
                * nasa_available_count
                / total_samples
            ) if total_samples else 0,
        },
        "source_weights": {
            key: round(value, 2)
            for key, value in (
                LIGHTING_SOURCE_WEIGHTS.items()
            )
        },
        "sample_results": sample_results,
        "interpretation": (
            "The score starts at a neutral 50 and available "
            "sources move it up or down at each "
            "approximately equal-distance route point. "
            "A valid low NASA radiance percentile is negative "
            "lighting evidence. Missing NASA observations, "
            "missing OSM lit tags and absent mapped lamps are "
            "neutral, not proof of darkness. NASA contributes "
            "regional background context. This is "
            "a lighting-evidence score, not a lux value."
        ),
    }
