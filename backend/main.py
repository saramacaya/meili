from typing import Literal, Optional
import os
import unicodedata

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = None

if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(
    title="Meili Backend",
    description="Backend for the Meili behavioural route-choice prototype.",
    version="0.1.0"
)


class RouteRequest(BaseModel):
    origin: str
    destination: str
    initial_preference: Literal["fastest", "balanced", "safest"]
    user_type: str

class RouteSelectionRequest(BaseModel):
    user_id: str
    origin: str
    destination: str
    initial_preference: Literal["fastest", "balanced", "safest"]
    final_choice_type: Literal["fastest", "balanced", "safest"]
    user_type: str
    fastest_route_id: str
    balanced_route_id: str
    safest_route_id: str

class FeedbackRequest(BaseModel):
    user_id: str
    chosen_route_id: str
    origin: str
    destination: str
    final_choice_type: Literal["fastest", "balanced", "safest"]
    user_type: str
    perceived_safety_rating: int = Field(..., ge=1, le=5)
    would_choose_again: bool
    comment: Optional[str] = None

class UserCreateRequest(BaseModel):
    user_type: str

def normalise_text(value: str) -> str:
    """
    Makes text easier to match.

    Example:
    'Plaça de la Reina' becomes 'placa de la reina'
    """
    value = value.lower().strip()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = " ".join(value.split())
    return value


MOCK_JOURNEYS = {
    ("ruzafa", "placa de la reina"): {
        "display_origin": "Ruzafa",
        "display_destination": "Plaça de la Reina",
        "routes": [
            {
                "id": "ruzafa_reina_fastest",
                "route_type": "fastest",
                "route_name": "Fastest Route",
                "estimated_time_minutes": 13,
                "distance_meters": 1300,
                "safety_score": 60,
                "explanation": "Shortest walking path through side streets, but with a lower safety score.",
                "tradeoff_summary": "Saves time, but has the lowest safety score."
            },
            {
                "id": "ruzafa_reina_balanced",
                "route_type": "balanced",
                "route_name": "Balanced Route",
                "estimated_time_minutes": 16,
                "distance_meters": 1400,
                "safety_score": 74,
                "explanation": "Mixes lit avenues with shorter connectors to balance time and perceived safety.",
                "tradeoff_summary": "Takes 3 extra minutes for a 14-point safety improvement."
            },
            {
                "id": "ruzafa_reina_safest",
                "route_type": "safest",
                "route_name": "Safest Route",
                "estimated_time_minutes": 19,
                "distance_meters": 1600,
                "safety_score": 85,
                "explanation": "Routes along better-lit main streets and busier public areas.",
                "tradeoff_summary": "Takes 6 extra minutes for a 25-point safety improvement."
            }
        ]
    },

    ("valencia nord", "mercado central"): {
        "display_origin": "Valencia Nord",
        "display_destination": "Mercado Central",
        "routes": [
            {
                "id": "nord_mercado_fastest",
                "route_type": "fastest",
                "route_name": "Fastest Route",
                "estimated_time_minutes": 11,
                "distance_meters": 950,
                "safety_score": 65,
                "explanation": "A direct route through central streets with moderate safety characteristics.",
                "tradeoff_summary": "Fastest option, but includes a few busier crossings."
            },
            {
                "id": "nord_mercado_balanced",
                "route_type": "balanced",
                "route_name": "Balanced Route",
                "estimated_time_minutes": 14,
                "distance_meters": 1100,
                "safety_score": 78,
                "explanation": "Uses wider streets and avoids some lower-rated segments.",
                "tradeoff_summary": "Adds 3 minutes for a 13-point safety improvement."
            },
            {
                "id": "nord_mercado_safest",
                "route_type": "safest",
                "route_name": "Safest Route",
                "estimated_time_minutes": 17,
                "distance_meters": 1300,
                "safety_score": 87,
                "explanation": "Prioritises better-lit, more active streets near central public areas.",
                "tradeoff_summary": "Adds 6 minutes for a 22-point safety improvement."
            }
        ]
    },

    ("turia gardens", "placa de la reina"): {
        "display_origin": "Turia Gardens",
        "display_destination": "Plaça de la Reina",
        "routes": [
            {
                "id": "turia_reina_fastest",
                "route_type": "fastest",
                "route_name": "Fastest Route",
                "estimated_time_minutes": 15,
                "distance_meters": 1250,
                "safety_score": 67,
                "explanation": "A shorter route using smaller connecting streets.",
                "tradeoff_summary": "Saves time, but has more uneven segment scores."
            },
            {
                "id": "turia_reina_balanced",
                "route_type": "balanced",
                "route_name": "Balanced Route",
                "estimated_time_minutes": 18,
                "distance_meters": 1450,
                "safety_score": 80,
                "explanation": "Balances directness with more visible and active streets.",
                "tradeoff_summary": "Adds 3 minutes for a 13-point safety improvement."
            },
            {
                "id": "turia_reina_safest",
                "route_type": "safest",
                "route_name": "Safest Route",
                "estimated_time_minutes": 22,
                "distance_meters": 1700,
                "safety_score": 90,
                "explanation": "Uses the most active and better-lit route segments.",
                "tradeoff_summary": "Adds 7 minutes for a 23-point safety improvement."
            }
        ]
    }
}


@app.get("/health")
def health_check():
    return {
        "status": "Meili backend is running"
    }


@app.post("/routes/generate")
def generate_routes(request: RouteRequest):
    origin_key = normalise_text(request.origin)
    destination_key = normalise_text(request.destination)

    journey_key = (origin_key, destination_key)

    if journey_key in MOCK_JOURNEYS:
        journey = MOCK_JOURNEYS[journey_key]
        matched_demo_route = True
    else:
        journey = MOCK_JOURNEYS[("ruzafa", "placa de la reina")]
        matched_demo_route = False

    if supabase is None:
        return {
            "status": "generated_not_saved",
            "message": "Supabase is not configured, so routes were generated but not saved.",
            "requested_origin": request.origin,
            "requested_destination": request.destination,
            "matched_origin": journey["display_origin"],
            "matched_destination": journey["display_destination"],
            "matched_demo_route": matched_demo_route,
            "user_type": request.user_type,
            "initial_preference": request.initial_preference,
            "recommended_route_type": request.initial_preference,
            "routes": journey["routes"],
            "note": "Routes are mock data for MVP testing, not live navigation results."
        }

    try:
        user_response = supabase.table("users").insert({
            "user_type": request.user_type
        }).execute()

        if not user_response.data:
            return {
                "status": "error",
                "message": "The user could not be created."
            }

        created_user = user_response.data[0]
        user_id = created_user["id"]

        route_rows = []

        for route in journey["routes"]:
            route_rows.append({
                "user_id": user_id,
                "origin_text": journey["display_origin"],
                "destination_text": journey["display_destination"],
                "initial_preference": request.initial_preference,
                "route_type": route["route_type"],
                "estimated_time_minutes": route["estimated_time_minutes"],
                "distance_meters": route["distance_meters"],
                "safety_score": route["safety_score"],
                "route_geometry_json": None,
                "explanation": route["explanation"]
            })

        routes_response = (
            supabase
            .table("routes")
            .insert(route_rows)
            .execute()
        )

        saved_routes = routes_response.data or []

        database_id_by_route_type = {
            route["route_type"]: route["id"]
            for route in saved_routes
        }

        routes_with_database_ids = []

        for route in journey["routes"]:
            route_copy = route.copy()
            route_copy["database_id"] = database_id_by_route_type.get(
                route["route_type"]
            )
            routes_with_database_ids.append(route_copy)

        return {
            "status": "routes_generated_and_saved",
            "user_id": user_id,
            "requested_origin": request.origin,
            "requested_destination": request.destination,
            "matched_origin": journey["display_origin"],
            "matched_destination": journey["display_destination"],
            "matched_demo_route": matched_demo_route,
            "user_type": request.user_type,
            "initial_preference": request.initial_preference,
            "recommended_route_type": request.initial_preference,
            "routes": routes_with_database_ids,
            "note": "Routes are mock MVP data and have been saved to Supabase."
        }

    except Exception as error:
        return {
            "status": "error",
            "message": str(error)
        }

@app.post("/routes/select")
def select_route(request: RouteSelectionRequest):
    origin_key = normalise_text(request.origin)
    destination_key = normalise_text(request.destination)

    journey_key = (origin_key, destination_key)

    if journey_key in MOCK_JOURNEYS:
        journey = MOCK_JOURNEYS[journey_key]
        matched_demo_route = True
    else:
        journey = MOCK_JOURNEYS[("ruzafa", "placa de la reina")]
        matched_demo_route = False

    routes = journey["routes"]

    fastest_route = next(route for route in routes if route["route_type"] == "fastest")
    chosen_route = next(
        route for route in routes
        if route["route_type"] == request.final_choice_type
    )

    extra_time_minutes = (
        chosen_route["estimated_time_minutes"]
        - fastest_route["estimated_time_minutes"]
    )

    safety_gain = (
        chosen_route["safety_score"]
        - fastest_route["safety_score"]
    )

    changed_preference = request.initial_preference != request.final_choice_type

    route_id_by_type = {
        "fastest": request.fastest_route_id,
        "balanced": request.balanced_route_id,
        "safest": request.safest_route_id
    }

    chosen_route_id = route_id_by_type[request.final_choice_type]

    if supabase is None:
        return {
            "status": "selection_calculated_not_saved",
            "message": "Supabase is not configured, so the route choice was calculated but not saved.",
            "requested_origin": request.origin,
            "requested_destination": request.destination,
            "matched_origin": journey["display_origin"],
            "matched_destination": journey["display_destination"],
            "matched_demo_route": matched_demo_route,
            "user_id": request.user_id,
            "user_type": request.user_type,
            "initial_preference": request.initial_preference,
            "final_choice_type": request.final_choice_type,
            "changed_preference": changed_preference,
            "chosen_route": chosen_route,
            "chosen_route_id": chosen_route_id,
            "extra_time_minutes": extra_time_minutes,
            "safety_gain": safety_gain
        }

    try:
        choice_response = supabase.table("route_choices").insert({
            "user_id": request.user_id,
            "chosen_route_id": chosen_route_id,
            "fastest_route_id": request.fastest_route_id,
            "balanced_route_id": request.balanced_route_id,
            "safest_route_id": request.safest_route_id,
            "initial_preference": request.initial_preference,
            "final_choice_type": request.final_choice_type,
            "extra_time_minutes": extra_time_minutes,
            "safety_gain": safety_gain,
            "framing_group": None
        }).execute()

        saved_choice = choice_response.data[0] if choice_response.data else None

        return {
            "status": "route_choice_saved",
            "route_choice": saved_choice,
            "requested_origin": request.origin,
            "requested_destination": request.destination,
            "matched_origin": journey["display_origin"],
            "matched_destination": journey["display_destination"],
            "matched_demo_route": matched_demo_route,
            "user_id": request.user_id,
            "user_type": request.user_type,
            "initial_preference": request.initial_preference,
            "final_choice_type": request.final_choice_type,
            "changed_preference": changed_preference,
            "chosen_route": chosen_route,
            "chosen_route_id": chosen_route_id,
            "extra_time_minutes": extra_time_minutes,
            "safety_gain": safety_gain,
            "note": "The route choice has been saved to Supabase."
        }

    except Exception as error:
        return {
            "status": "error",
            "message": str(error)
        }

@app.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    origin_key = normalise_text(request.origin)
    destination_key = normalise_text(request.destination)

    journey_key = (origin_key, destination_key)

    if journey_key in MOCK_JOURNEYS:
        journey = MOCK_JOURNEYS[journey_key]
        matched_demo_route = True
    else:
        journey = MOCK_JOURNEYS[("ruzafa", "placa de la reina")]
        matched_demo_route = False

    routes = journey["routes"]

    chosen_route = next(
        route for route in routes
        if route["route_type"] == request.final_choice_type
    )

    if supabase is None:
        return {
            "status": "feedback_validated_not_saved",
            "message": "Supabase is not configured, so feedback was validated but not saved.",
            "requested_origin": request.origin,
            "requested_destination": request.destination,
            "matched_origin": journey["display_origin"],
            "matched_destination": journey["display_destination"],
            "matched_demo_route": matched_demo_route,
            "user_id": request.user_id,
            "chosen_route_id": request.chosen_route_id,
            "user_type": request.user_type,
            "final_choice_type": request.final_choice_type,
            "chosen_route_name": chosen_route["route_name"],
            "perceived_safety_rating": request.perceived_safety_rating,
            "would_choose_again": request.would_choose_again,
            "comment": request.comment
        }

    try:
        feedback_response = supabase.table("feedback").insert({
            "user_id": request.user_id,
            "route_id": request.chosen_route_id,
            "perceived_safety_rating": request.perceived_safety_rating,
            "would_choose_again": request.would_choose_again,
            "comment": request.comment
        }).execute()

        saved_feedback = feedback_response.data[0] if feedback_response.data else None

        return {
            "status": "feedback_saved",
            "feedback": saved_feedback,
            "requested_origin": request.origin,
            "requested_destination": request.destination,
            "matched_origin": journey["display_origin"],
            "matched_destination": journey["display_destination"],
            "matched_demo_route": matched_demo_route,
            "user_id": request.user_id,
            "chosen_route_id": request.chosen_route_id,
            "user_type": request.user_type,
            "final_choice_type": request.final_choice_type,
            "chosen_route_name": chosen_route["route_name"],
            "perceived_safety_rating": request.perceived_safety_rating,
            "would_choose_again": request.would_choose_again,
            "comment": request.comment,
            "note": "Feedback has been saved to Supabase."
        }

    except Exception as error:
        return {
            "status": "error",
            "message": str(error)
        }
        
@app.get("/database/health")
def database_health():
    if supabase is None:
        return {
            "status": "not_configured",
            "message": "Supabase URL or key is missing from the backend .env file."
        }

    try:
        response = supabase.table("users").select("id").limit(1).execute()

        return {
            "status": "connected",
            "table_checked": "users",
            "rows_returned": len(response.data)
        }

    except Exception as error:
        return {
            "status": "error",
            "message": str(error)
        }

@app.post("/users/create")
def create_user(request: UserCreateRequest):
    if supabase is None:
        return {
            "status": "not_configured",
            "message": "Supabase URL or key is missing from the backend .env file."
        }

    try:
        response = supabase.table("users").insert({
            "user_type": request.user_type
        }).execute()

        created_user = response.data[0] if response.data else None

        return {
            "status": "user_created",
            "user": created_user
        }

    except Exception as error:
        return {
            "status": "error",
            "message": str(error)
        }