"""Combine Meili's existing evidence into route-section and route scores.

This module performs no network requests.  ``main.py`` obtains the lighting,
active-place and social-context analyses, then passes their results here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


LIGHTING_WEIGHT = 0.60
ACTIVITY_WEIGHT = 0.40
SECTION_COUNT = 10


def _clamp(value: float, minimum: float = 0.0, maximum: float = 100.0) -> float:
    return max(minimum, min(maximum, value))


def _classification(score: float) -> str:
    if score >= 70:
        return "stronger_evidence"
    if score >= 55:
        return "moderate_evidence"
    if score >= 40:
        return "limited_or_mixed_evidence"
    return "weaker_evidence"


def _colour(score: float) -> str:
    if score >= 70:
        return "green"
    if score >= 55:
        return "yellow"
    if score >= 40:
        return "orange"
    return "red"


def _lighting_by_section(lighting_analysis: dict[str, Any]) -> list[float]:
    samples = lighting_analysis.get("sample_results") or []
    buckets: list[list[float]] = [[] for _ in range(SECTION_COUNT)]

    for index, sample in enumerate(samples):
        evidence = sample.get("combined_lighting_evidence") or {}
        score = evidence.get("score")
        if score is None:
            continue
        # Lighting samples are already approximately equally spaced.
        progress = index / max(1, len(samples) - 1)
        section_index = min(SECTION_COUNT - 1, int(progress * SECTION_COUNT))
        buckets[section_index].append(float(score))

    route_mean = (
        lighting_analysis.get("combined_score_statistics", {}).get("mean_score")
    )
    fallback = 50.0 if route_mean is None else float(route_mean)
    return [
        sum(bucket) / len(bucket) if bucket else fallback
        for bucket in buckets
    ]


def _activity_by_section(active_places_analysis: dict[str, Any]) -> list[float]:
    raw_segments = active_places_analysis.get("activity_segments") or []
    scores = [float(segment.get("activity_score", 0.0)) for segment in raw_segments]
    fallback = float(active_places_analysis.get("route_activity_score", 0.0))

    if len(scores) == SECTION_COUNT:
        return scores
    if not scores:
        return [fallback] * SECTION_COUNT

    # Defensive resampling in case the activity component changes its segment count.
    return [
        scores[min(len(scores) - 1, int(i * len(scores) / SECTION_COUNT))]
        for i in range(SECTION_COUNT)
    ]


def _social_effects_by_section(
    social_context_analysis: dict[str, Any],
    departure_datetime: datetime,
    route_duration_seconds: int,
) -> tuple[list[float], list[list[str]]]:
    effects = [0.0] * SECTION_COUNT
    record_ids: list[list[str]] = [[] for _ in range(SECTION_COUNT)]

    for record in social_context_analysis.get("matched_records") or []:
        try:
            arrival = datetime.fromisoformat(record["arrival_time"])
            elapsed = arrival.timestamp() - departure_datetime.timestamp()
            progress = _clamp(elapsed / route_duration_seconds, 0.0, 1.0)
            section_index = min(SECTION_COUNT - 1, int(progress * SECTION_COUNT))
            effects[section_index] += float(record.get("effect_points", 0.0))
            if record.get("record_id"):
                record_ids[section_index].append(str(record["record_id"]))
        except (KeyError, TypeError, ValueError, OverflowError):
            continue

    # A local opinion effect is intentionally small: at most -3/+2 points.
    effects = [_clamp(effect, -3.0, 2.0) for effect in effects]
    return effects, record_ids


def value_route_sections(
    *,
    lighting_analysis: dict[str, Any],
    active_places_analysis: dict[str, Any],
    social_context_analysis: dict[str, Any],
    departure_datetime: datetime,
    route_duration_seconds: int,
) -> dict[str, Any]:
    """Return ten section scores and one distance-weighted route score."""
    if route_duration_seconds <= 0:
        raise ValueError("route_duration_seconds must be positive")

    lighting_scores = _lighting_by_section(lighting_analysis)
    activity_scores = _activity_by_section(active_places_analysis)
    social_effects, social_record_ids = _social_effects_by_section(
        social_context_analysis,
        departure_datetime,
        route_duration_seconds,
    )

    sections = []
    for index in range(SECTION_COUNT):
        base_score = (
            LIGHTING_WEIGHT * lighting_scores[index]
            + ACTIVITY_WEIGHT * activity_scores[index]
        )
        final_score = _clamp(base_score + social_effects[index])
        sections.append({
            "section_index": index,
            "start_progress_percentage": index * 10,
            "end_progress_percentage": (index + 1) * 10,
            "lighting_score": round(lighting_scores[index], 1),
            "active_places_score": round(activity_scores[index], 1),
            "social_context_adjustment_points": round(social_effects[index], 3),
            "matched_social_record_ids": social_record_ids[index],
            "street_value_score": round(final_score, 1),
            "classification": _classification(final_score),
            "map_colour": _colour(final_score),
        })

    route_score = sum(section["street_value_score"] for section in sections) / SECTION_COUNT
    weakest = min(sections, key=lambda section: section["street_value_score"])

    return {
        "route_value_score": round(route_score, 1),
        "route_classification": _classification(route_score),
        "section_count": SECTION_COUNT,
        "weakest_section_index": weakest["section_index"],
        "weakest_section_score": weakest["street_value_score"],
        "component_weights": {
            "lighting": LIGHTING_WEIGHT,
            "active_places": ACTIVITY_WEIGHT,
            "social_context": "local adjustment capped at -3/+2 points per section",
        },
        "sections": sections,
        "method_note": (
            "The route is divided into ten approximately equal-distance sections. "
            "Missing lighting observations remain neutral inside the lighting component; "
            "they are not treated as evidence that a section is dark."
        ),
    }
