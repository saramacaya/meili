import json
import math
import time
from pathlib import Path


import requests


BASE_URL = "https://meili-backend.onrender.com"

OUTPUT_FILE = Path(
    "route_lighting_comparison.json"
)

# Ruzafa area to Torres de Serranos.
# This is long enough to have a better chance of producing
# meaningfully different walking alternatives.
ROUTE_REQUEST = {
    "origin": "Ruzafa",
    "destination": "Torres de Serranos",
    "origin_latitude": 39.4622,
    "origin_longitude": -0.3763,
    "destination_latitude": 39.4793,
    "destination_longitude": -0.3764,
    "initial_preference": "balanced",
    "user_type": "test_user"
}

LIGHTING_SETTINGS = {
    "sample_interval_meters": 15,
    "official_lamp_radius_meters": 25,
    "osm_lit_match_radius_meters": 15,
    "osm_lamp_radius_meters": 25,
    "month": None
}


def post_json(
    endpoint: str,
    payload: dict,
    timeout_seconds: int = 180,
    attempts: int = 2
) -> dict:
    """
    Sends one POST request.

    A second attempt is made when Render or an external
    data source temporarily returns a server error.
    """
    url = f"{BASE_URL}{endpoint}"
    last_error = None

    for attempt_number in range(
        1,
        attempts + 1
    ):
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=timeout_seconds
            )

            if (
                response.status_code >= 500
                and attempt_number < attempts
            ):
                print(
                    f"Temporary {response.status_code} "
                    f"error. Retrying in 5 seconds..."
                )

                time.sleep(5)
                continue

            response.raise_for_status()

            return response.json()

        except (
            requests.RequestException,
            ValueError
        ) as error:
            last_error = error

            if attempt_number < attempts:
                print(
                    "Request failed temporarily. "
                    "Retrying in 5 seconds..."
                )

                time.sleep(5)

    raise RuntimeError(
        f"Request to {endpoint} failed: "
        f"{last_error}"
    )


def distance_meters(
    point_a: tuple[float, float],
    point_b: tuple[float, float]
) -> float:
    longitude_a, latitude_a = point_a
    longitude_b, latitude_b = point_b

    earth_radius_meters = 6_371_000

    latitude_a_radians = math.radians(
        latitude_a
    )

    latitude_b_radians = math.radians(
        latitude_b
    )

    latitude_delta = math.radians(
        latitude_b - latitude_a
    )

    longitude_delta = math.radians(
        longitude_b - longitude_a
    )

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


def longest_consecutive_stretch_meters(
    sample_results: list[dict],
    target_classification: str
) -> float:
    """
    Measures the actual geographical length of the
    longest consecutive stretch with one classification.
    """
    longest_stretch = 0.0
    current_stretch = 0.0

    previous_point = None
    previous_was_target = False

    for sample in sample_results:
        classification = sample.get(
            "combined_lighting_evidence",
            {}
        ).get(
            "classification",
            "unknown"
        )

        current_point = (
            float(sample["longitude"]),
            float(sample["latitude"])
        )

        current_is_target = (
            classification
            == target_classification
        )

        if current_is_target:
            if (
                previous_was_target
                and previous_point is not None
            ):
                current_stretch += distance_meters(
                    previous_point,
                    current_point
                )
            else:
                current_stretch = 0.0

            longest_stretch = max(
                longest_stretch,
                current_stretch
            )
        else:
            current_stretch = 0.0

        previous_point = current_point
        previous_was_target = current_is_target

    return round(longest_stretch, 1)

def percentage(
    count: int,
    total: int
) -> float:
    if total <= 0:
        return 0.0

    return round(
        100 * count / total,
        1
    )


def summarise_route(
    route: dict,
    lighting_analysis: dict
) -> dict:
    sample_results = lighting_analysis.get(
        "sample_results",
        []
    )

    sample_count = len(sample_results)

    classifications = [
        sample.get(
            "combined_lighting_evidence",
            {}
        ).get(
            "classification",
            "unknown"
        )
        for sample in sample_results
    ]

    confidence_values = [
        sample.get(
            "combined_lighting_evidence",
            {}
        ).get(
            "confidence",
            "none"
        )
        for sample in sample_results
    ]

    weak_count = classifications.count(
        "weak_lighting_evidence"
    )

    limited_count = classifications.count(
        "limited_or_mixed_lighting_evidence"
    )

    moderate_count = classifications.count(
        "moderate_lighting_evidence"
    )

    strong_count = classifications.count(
        "strong_lighting_evidence"
    )

    unknown_count = classifications.count(
        "unknown"
    )

    longest_weak_stretch_meters = (
        longest_consecutive_stretch_meters(
            sample_results=sample_results,
            target_classification=(
                "weak_lighting_evidence"
            )
    )
)

    source_errors = lighting_analysis.get(
        "source_errors",
        {}
    )

    critical_source_names = {
        "official_valencia_lamps",
        "osm_lit",
        "nasa_background"
    }

    critical_source_errors = {
        source_name: error
        for source_name, error
        in source_errors.items()
        if source_name
        in critical_source_names
    }

    comparison_complete = (
        len(critical_source_errors) == 0
    )

    score_statistics = lighting_analysis.get(
        "combined_score_statistics",
        {}
    )


    high_confidence_count = (
        confidence_values.count("high")
        + confidence_values.count(
            "medium_high"
        )
    )

    return {
        "route_id": route.get("route_id"),
        "estimated_time_minutes": route.get(
            "estimated_time_minutes"
        ),
        "distance_meters": route.get(
            "distance_meters"
        ),
        "distance_kilometers": round(
            route.get(
                "distance_meters",
                0
            ) / 1000,
            2
        ),
        "route_sample_count": sample_count,
        "minimum_lighting_score": (
            score_statistics.get(
                "minimum_score"
            )
        ),
        "median_lighting_score": (
            score_statistics.get(
                "median_score"
            )
        ),
        "mean_lighting_score": (
            score_statistics.get(
                "mean_score"
            )
        ),
        "maximum_lighting_score": (
            score_statistics.get(
                "maximum_score"
            )
        ),
        "strong_sample_percentage": percentage(
            strong_count,
            sample_count
        ),
        "moderate_sample_percentage": percentage(
            moderate_count,
            sample_count
        ),
        "limited_sample_percentage": percentage(
            limited_count,
            sample_count
        ),
        "weak_sample_percentage": percentage(
            weak_count,
            sample_count
        ),
        "low_evidence_sample_percentage": percentage(
            weak_count + limited_count,
            sample_count
        ),
        "unknown_sample_percentage": percentage(
            unknown_count,
            sample_count
        ),
        "strong_or_moderate_percentage": (
            percentage(
                strong_count + moderate_count,
                sample_count
            )
        ),
        "high_confidence_sample_percentage": (
            percentage(
                high_confidence_count,
                sample_count
            )
        ),
        "comparison_complete": comparison_complete,
        "critical_source_errors": (
            critical_source_errors
        ),
        "approximate_longest_weak_stretch_meters": (
            longest_weak_stretch_meters
        ),
        "source_conflict_sample_count": (
            lighting_analysis.get(
                "source_conflict_sample_count",
                0
            )
        ),
        "source_errors": source_errors
    }


def print_summary(
    route_number: int,
    summary: dict
) -> None:
    print()
    print(
        f"ROUTE {route_number}: "
        f"{summary['route_id']}"
    )

    print(
        "Time:",
        summary["estimated_time_minutes"],
        "minutes"
    )

    print(
        "Distance:",
        summary["distance_kilometers"],
        "km"
    )

    print(
        "Mean lighting score:",
        summary["mean_lighting_score"]
    )

    print(
        "Median lighting score:",
        summary["median_lighting_score"]
    )

    print(
        "Minimum lighting score:",
        summary["minimum_lighting_score"]
    )

    print(
        "Strong or moderate points:",
        f"{summary['strong_or_moderate_percentage']}%"
    )

    print(
        "Weak points:",
        f"{summary['weak_sample_percentage']}%"
    )

    print(
        "Longest weak stretch:",
        "approximately",
        summary[
            "approximate_longest_weak_stretch_meters"
        ],
        "meters"
    )

    print(
        "High-confidence points:",
        f"{summary['high_confidence_sample_percentage']}%"
    )

    print(
        "Source conflicts:",
        summary["source_conflict_sample_count"]
    )

    if summary["source_errors"]:
        print(
            "Source errors:",
            summary["source_errors"]
        )
    else:
        print("Source errors: none")


def main():
    print(
        "Requesting real walking alternatives..."
    )

    route_response = post_json(
        endpoint="/routes/test-real",
        payload=ROUTE_REQUEST,
        timeout_seconds=90
    )

    route_container = route_response.get(
        "route",
        {}
    )

    routes = route_container.get(
        "routes",
        []
    )

    if not routes:
        raise RuntimeError(
            "The routing endpoint returned no routes."
        )

    print(
        "Walking alternatives returned:",
        len(routes)
    )

    comparison_payload = {
        "routes": [
            {
                "route_id": route["route_id"],
                "geometry": route["geometry"],
                "estimated_time_minutes": route.get(
                    "estimated_time_minutes"
                ),
                "distance_meters": route.get(
                    "distance_meters"
                )
            }
            for route in routes
        ],
        **LIGHTING_SETTINGS
    }

    print()
    print(
        "Requesting one shared lighting comparison..."
    )

    comparison_response = post_json(
        endpoint="/safety/lighting/compare",
        payload=comparison_payload,
        timeout_seconds=300
    )

    batch_route_results = (
        comparison_response.get(
            "routes",
            []
        )
    )

    shared_osm_failed = any(
        "osm_lit"
        in route_result.get(
            "source_errors",
            {}
        )
        for route_result
        in batch_route_results
    )

    if shared_osm_failed:
        print(
            "The shared OSM request failed. "
            "Waiting 30 seconds and retrying "
            "the whole comparison..."
        )

        time.sleep(30)

        comparison_response = post_json(
            endpoint="/safety/lighting/compare",
            payload=comparison_payload,
            timeout_seconds=300
        )

        batch_route_results = (
            comparison_response.get(
                "routes",
                []
            )
        )

    shared_osm_debug = (
        comparison_response.get(
            "shared_osm_download",
            {}
        )
    )

    print()
    print(
        "One shared OSM download used:",
        shared_osm_debug.get(
            "one_download_used_for_all_routes"
        )
    )

    print(
        "Shared OSM cache hit:",
        shared_osm_debug.get(
            "cache_hit"
        )
    )

    print(
        "Shared lit ways downloaded:",
        shared_osm_debug.get(
            "raw_lit_ways_found"
        )
    )

    print(
        "Shared street lamps downloaded:",
        shared_osm_debug.get(
            "raw_street_lamps_found"
        )
    )

    route_results_by_id = {
        route_result.get("route_id"):
        route_result
        for route_result
        in batch_route_results
    }

    comparison_results = []

    for route_number, route in enumerate(
        routes,
        start=1
    ):
        route_id = route["route_id"]

        lighting_analysis = (
            route_results_by_id.get(
                route_id
            )
        )

        if lighting_analysis is None:
            error_message = (
                "The batch endpoint returned no "
                f"lighting result for {route_id}."
            )

            print()
            print(error_message)

            comparison_results.append({
                "route": route,
                "error": error_message
            })

            continue

        summary = summarise_route(
            route=route,
            lighting_analysis=(
                lighting_analysis
            )
        )

        comparison_results.append({
            "route": route,
            "lighting_summary": summary,
            "lighting_analysis": (
                lighting_analysis
            )
        })

        print_summary(
            route_number=route_number,
            summary=summary
        )

    successful_results = [
        result
        for result in comparison_results
        if "lighting_summary" in result
    ]

    complete_results = [
        result
        for result in successful_results
        if result[
            "lighting_summary"
        ].get(
            "comparison_complete",
            False
        )
    ]

    incomplete_results = [
        result
        for result in successful_results
        if not result[
            "lighting_summary"
        ].get(
            "comparison_complete",
            False
        )
    ]
    # This is only a descriptive lighting comparison.
    # It is not yet the final Meili route-ranking formula.
    lighting_order = sorted(
        complete_results,
        key=lambda result: (
            result[
                "lighting_summary"
            ][
                "low_evidence_sample_percentage"
            ],
            result[
                "lighting_summary"
            ][
                "approximate_longest_weak_stretch_meters"
            ],
            -(
                result[
                    "lighting_summary"
                ][
                    "mean_lighting_score"
                ]
                or 0
            ),
            result[
                "lighting_summary"
            ][
                "estimated_time_minutes"
            ]
            or 0
        )
    )

    if incomplete_results:
        print()
        print(
            "ROUTES EXCLUDED FROM FAIR COMPARISON"
        )

        for result in incomplete_results:
            summary = result["lighting_summary"]

            print(
                "-",
                summary["route_id"],
                "because critical source data failed:",
                summary["critical_source_errors"]
            )

    print()
    print("LIGHTING-ONLY COMPARISON ORDER")

    for position, result in enumerate(
        lighting_order,
        start=1
    ):
        summary = result["lighting_summary"]

        print(
            f"{position}. "
            f"{summary['route_id']} — "
            f"{summary['low_evidence_sample_percentage']}% "
            f"low evidence, "
            f"{summary['weak_sample_percentage']}% weak, "
            f"longest weak stretch "
            f"{summary['approximate_longest_weak_stretch_meters']} m, "
            f"mean score "
            f"{summary['mean_lighting_score']}, "
            f"{summary['estimated_time_minutes']} min"
        )

    output = {
        "route_request": ROUTE_REQUEST,
        "lighting_settings": LIGHTING_SETTINGS,
        "route_count": len(routes),
        "results": comparison_results,
        "lighting_only_order": [
            result[
                "lighting_summary"
            ][
                "route_id"
            ]
            for result in lighting_order
        ]
    }

    OUTPUT_FILE.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print()
    print(
        "Complete comparison saved to:",
        OUTPUT_FILE.resolve()
    )


if __name__ == "__main__":
    main()