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
    origin: str
    destination: str
    initial_preference: Literal["fastest", "balanced", "safest"]
    final_choice_type: Literal["fastest", "balanced", "safest"]
    user_type: str

class FeedbackRequest(BaseModel):
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

    return {
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

    return {
        "requested_origin": request.origin,
        "requested_destination": request.destination,
        "matched_origin": journey["display_origin"],
        "matched_destination": journey["display_destination"],
        "matched_demo_route": matched_demo_route,
        "user_type": request.user_type,
        "initial_preference": request.initial_preference,
        "final_choice_type": request.final_choice_type,
        "changed_preference": changed_preference,
        "chosen_route": chosen_route,
        "fastest_route_time_minutes": fastest_route["estimated_time_minutes"],
        "chosen_route_time_minutes": chosen_route["estimated_time_minutes"],
        "extra_time_minutes": extra_time_minutes,
        "safety_gain": safety_gain,
        "note": "This selection summary is calculated from mock route data and is not yet saved to Supabase."
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

    return {
        "status": "feedback_received",
        "requested_origin": request.origin,
        "requested_destination": request.destination,
        "matched_origin": journey["display_origin"],
        "matched_destination": journey["display_destination"],
        "matched_demo_route": matched_demo_route,
        "user_type": request.user_type,
        "final_choice_type": request.final_choice_type,
        "chosen_route_id": chosen_route["id"],
        "chosen_route_name": chosen_route["route_name"],
        "perceived_safety_rating": request.perceived_safety_rating,
        "would_choose_again": request.would_choose_again,
        "comment": request.comment,
        "note": "Feedback is validated by the backend but not yet saved to Supabase."
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