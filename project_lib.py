"""
project_lib.py — Helper library for the AgentsVille Trip Planner project.

Contains:
- Pydantic models (VacationInfo, TravelPlan, Activity, DayPlan, Traveler, ToolCall, WeatherForecast)
- Mock weather + activity data for the fictional city "AgentsVille"
- Mock tool functions: calculator_tool, get_activities_by_date_tool, run_evals_tool, final_answer_tool
- `available_tools` registry mapping tool name -> {function, description, parameters}
- print_in_box utility
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numexpr
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class Traveler(BaseModel):
    """A single traveler going on the trip."""
    name: str = Field(..., description="Full name of the traveler")
    age: int = Field(..., ge=0, description="Age of the traveler in years")
    interests: List[str] = Field(
        default_factory=list,
        description="Personal interests for this traveler (e.g., 'art', 'food')",
    )


class VacationInfo(BaseModel):
    """Travelers' vacation preferences and constraints.

    Used as the structured input to the trip planner.
    """
    destination: str = Field(..., description="Destination city (e.g., 'AgentsVille')")
    date_of_arrival: date = Field(..., description="ISO date when the trip starts")
    date_of_departure: date = Field(..., description="ISO date when the trip ends (inclusive)")
    budget: float = Field(..., ge=0, description="Total budget in USD for the whole trip")
    travelers: List[Traveler] = Field(..., min_length=1, description="List of travelers")
    interests: List[str] = Field(
        default_factory=list,
        description="Group-level interests shared by all travelers",
    )

    @property
    def num_days(self) -> int:
        """Inclusive number of days from arrival to departure."""
        return (self.date_of_departure - self.date_of_arrival).days + 1

    def date_range(self) -> List[date]:
        """All dates from arrival to departure, inclusive."""
        return [
            self.date_of_arrival + timedelta(days=i)
            for i in range(self.num_days)
        ]


class WeatherForecast(BaseModel):
    """Weather forecast for a single date in AgentsVille."""
    date: date
    high_temperature_celsius: float
    low_temperature_celsius: float
    condition: str = Field(..., description="e.g., 'sunny', 'cloudy', 'rainy', 'stormy'")
    precipitation_chance: float = Field(..., ge=0, le=1, description="0..1")
    description: str = Field(..., description="Short human-readable summary")


class Activity(BaseModel):
    """A single bookable activity in AgentsVille."""
    activity_id: str = Field(..., description="Stable unique identifier, e.g. 'ART-001'")
    name: str
    description: str
    location: str
    start_time: str = Field(..., description="ISO 8601 datetime, e.g. '2025-06-10T10:00:00'")
    end_time: str = Field(..., description="ISO 8601 datetime, e.g. '2025-06-10T12:00:00'")
    price: float = Field(..., ge=0, description="Price per person in USD")
    related_interests: List[str] = Field(default_factory=list)
    indoor: bool = Field(..., description="True if held indoors (weather-resistant)")

    @property
    def activity_date(self) -> date:
        return datetime.fromisoformat(self.start_time).date()


class DayPlan(BaseModel):
    """A single day's plan within the itinerary."""
    date: date
    activities: List[Activity]
    notes: Optional[str] = None


class TravelPlan(BaseModel):
    """The full multi-day itinerary returned by the ItineraryAgent."""
    city: str
    start_date: date
    end_date: date
    travelers: List[Traveler]
    total_cost: float = Field(..., ge=0, description="Sum of all activity prices x number of travelers")
    days: List[DayPlan]
    summary: Optional[str] = Field(None, description="Short narrative summary of the itinerary")


class ToolCall(BaseModel):
    """A structured tool invocation produced by the ReAct agent."""
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Mock data: weather forecast and activities for AgentsVille
# ---------------------------------------------------------------------------
# Default trip window used by the starter notebook: 2025-06-10 -> 2025-06-13.

_WEATHER_DATA: List[WeatherForecast] = [
    WeatherForecast(
        date=date(2025, 6, 10),
        high_temperature_celsius=28.0,
        low_temperature_celsius=18.0,
        condition="sunny",
        precipitation_chance=0.05,
        description="Sunny, warm, and dry. Great day for outdoor activities.",
    ),
    WeatherForecast(
        date=date(2025, 6, 11),
        high_temperature_celsius=26.0,
        low_temperature_celsius=17.0,
        condition="partly cloudy",
        precipitation_chance=0.20,
        description="Mostly pleasant with some clouds. Light breeze in the afternoon.",
    ),
    WeatherForecast(
        date=date(2025, 6, 12),
        high_temperature_celsius=21.0,
        low_temperature_celsius=15.0,
        condition="rainy",
        precipitation_chance=0.85,
        description="Heavy rain expected throughout the day. Outdoor activities not advised.",
    ),
    WeatherForecast(
        date=date(2025, 6, 13),
        high_temperature_celsius=30.0,
        low_temperature_celsius=20.0,
        condition="sunny",
        precipitation_chance=0.00,
        description="Hot and clear. Excellent for any outdoor plans, bring sunscreen.",
    ),
]


_ACTIVITY_DATA: List[Activity] = [
    # --- 2025-06-10 ---
    Activity(
        activity_id="ART-001",
        name="AgentsVille Modern Art Gallery Tour",
        description="Guided walk through AgentsVille's most renowned modern art gallery, featuring local AI-inspired sculpture.",
        location="Downtown Art District",
        start_time="2025-06-10T10:00:00",
        end_time="2025-06-10T12:00:00",
        price=25.0,
        related_interests=["art", "culture"],
        indoor=True,
    ),
    Activity(
        activity_id="TECH-001",
        name="AI & Robotics Meetup",
        description="Informal meetup with local AI engineers; lightning talks and demos.",
        location="AgentsVille Innovation Hub",
        start_time="2025-06-10T18:00:00",
        end_time="2025-06-10T20:30:00",
        price=0.0,
        related_interests=["technology", "ai", "networking"],
        indoor=True,
    ),
    Activity(
        activity_id="FOOD-001",
        name="Street Food Walking Tour",
        description="Sample seven AgentsVille street food specialties across the night market.",
        location="Night Market District",
        start_time="2025-06-10T19:00:00",
        end_time="2025-06-10T21:30:00",
        price=45.0,
        related_interests=["food", "culture"],
        indoor=False,
    ),
    Activity(
        activity_id="PARK-001",
        name="Sunrise Hike at Algorithm Peak",
        description="Easy 2-hour hike with panoramic views of AgentsVille.",
        location="Algorithm Peak Trailhead",
        start_time="2025-06-10T06:30:00",
        end_time="2025-06-10T09:00:00",
        price=15.0,
        related_interests=["outdoors", "hiking", "nature"],
        indoor=False,
    ),

    # --- 2025-06-11 ---
    Activity(
        activity_id="MUS-001",
        name="History of Computing Museum Visit",
        description="Self-guided tour of the History of Computing Museum with augmented-reality exhibits.",
        location="Museum Quarter",
        start_time="2025-06-11T09:30:00",
        end_time="2025-06-11T12:00:00",
        price=20.0,
        related_interests=["technology", "history", "culture"],
        indoor=True,
    ),
    Activity(
        activity_id="FOOD-002",
        name="Hands-on Pasta Cooking Class",
        description="Make fresh pasta from scratch with a local chef. All ingredients included.",
        location="Chef's Studio, Old Town",
        start_time="2025-06-11T14:00:00",
        end_time="2025-06-11T17:00:00",
        price=65.0,
        related_interests=["food", "cooking"],
        indoor=True,
    ),
    Activity(
        activity_id="OUT-001",
        name="River Cruise & Picnic",
        description="Two-hour cruise along the AgentsVille River with a packed picnic lunch.",
        location="Riverfront Pier 4",
        start_time="2025-06-11T12:30:00",
        end_time="2025-06-11T14:30:00",
        price=40.0,
        related_interests=["outdoors", "relaxation"],
        indoor=False,
    ),
    Activity(
        activity_id="MUSIC-001",
        name="Jazz Night at the Cipher Club",
        description="Live local jazz quartet at AgentsVille's premier basement jazz club.",
        location="Cipher Club, Old Town",
        start_time="2025-06-11T20:00:00",
        end_time="2025-06-11T23:00:00",
        price=30.0,
        related_interests=["music", "culture", "nightlife"],
        indoor=True,
    ),

    # --- 2025-06-12 (rainy) ---
    Activity(
        activity_id="ART-002",
        name="Interactive Digital Art Exhibition",
        description="Immersive room-scale digital art installations curated by AgentsVille creators.",
        location="Digital Arts Center",
        start_time="2025-06-12T10:00:00",
        end_time="2025-06-12T12:30:00",
        price=22.0,
        related_interests=["art", "technology"],
        indoor=True,
    ),
    Activity(
        activity_id="TECH-002",
        name="Build-Your-Own-Agent Workshop",
        description="Three-hour hands-on workshop building a simple LLM agent. Laptops provided.",
        location="AgentsVille Innovation Hub",
        start_time="2025-06-12T13:00:00",
        end_time="2025-06-12T16:00:00",
        price=85.0,
        related_interests=["technology", "ai"],
        indoor=True,
    ),
    Activity(
        activity_id="FOOD-003",
        name="Tea House Afternoon Tasting",
        description="Sample twelve regional teas with pairings at AgentsVille's oldest tea house.",
        location="Tea Lane, Old Town",
        start_time="2025-06-12T15:30:00",
        end_time="2025-06-12T17:00:00",
        price=35.0,
        related_interests=["food", "culture", "relaxation"],
        indoor=True,
    ),
    Activity(
        activity_id="OUT-002",
        name="Botanic Garden Outdoor Walk",
        description="Self-guided walking tour through the open-air botanic gardens.",
        location="AgentsVille Botanic Gardens",
        start_time="2025-06-12T10:00:00",
        end_time="2025-06-12T12:00:00",
        price=12.0,
        related_interests=["outdoors", "nature"],
        indoor=False,
    ),

    # --- 2025-06-13 ---
    Activity(
        activity_id="OUT-003",
        name="Kayaking on Lake Lambda",
        description="Guided morning kayaking session with equipment included.",
        location="Lake Lambda Boathouse",
        start_time="2025-06-13T08:00:00",
        end_time="2025-06-13T10:30:00",
        price=55.0,
        related_interests=["outdoors", "adventure", "sports"],
        indoor=False,
    ),
    Activity(
        activity_id="TECH-003",
        name="AI Ethics Panel Discussion",
        description="Panel of researchers discussing responsible AI deployment, with audience Q&A.",
        location="Innovation Hub Auditorium",
        start_time="2025-06-13T14:00:00",
        end_time="2025-06-13T16:00:00",
        price=10.0,
        related_interests=["technology", "ai"],
        indoor=True,
    ),
    Activity(
        activity_id="FOOD-004",
        name="Sunset Rooftop Dinner",
        description="Five-course tasting menu on the rooftop of Tower 42 with views over AgentsVille.",
        location="Tower 42 Rooftop",
        start_time="2025-06-13T19:00:00",
        end_time="2025-06-13T22:00:00",
        price=95.0,
        related_interests=["food", "romance"],
        indoor=False,
    ),
    Activity(
        activity_id="MUSIC-002",
        name="Outdoor Symphony in the Park",
        description="Free open-air symphony performance in Central Park, AgentsVille.",
        location="Central Park Bandshell",
        start_time="2025-06-13T17:00:00",
        end_time="2025-06-13T19:00:00",
        price=0.0,
        related_interests=["music", "outdoors", "culture"],
        indoor=False,
    ),
]


# ---------------------------------------------------------------------------
# Mock "API" functions (used both by the notebook directly and as tools)
# ---------------------------------------------------------------------------

def get_weather_forecast(target_date: date | str) -> Dict[str, Any]:
    """Return the weather forecast for a single date in AgentsVille.

    Args:
        target_date: a `datetime.date` or ISO-format string ('YYYY-MM-DD').
    """
    if isinstance(target_date, str):
        target_date = date.fromisoformat(target_date)
    for w in _WEATHER_DATA:
        if w.date == target_date:
            return w.model_dump(mode="json")
    raise ValueError(f"No weather forecast available for {target_date}")


def get_weather_forecast_range(start: date, end: date) -> List[Dict[str, Any]]:
    """Return weather forecasts for every date from start to end (inclusive)."""
    out = []
    cur = start
    while cur <= end:
        out.append(get_weather_forecast(cur))
        cur += timedelta(days=1)
    return out


def get_activities_for_range(start: date, end: date) -> List[Dict[str, Any]]:
    """Return all available activities whose start date falls in [start, end]."""
    return [
        a.model_dump(mode="json")
        for a in _ACTIVITY_DATA
        if start <= a.activity_date <= end
    ]


def _get_activity_by_id(activity_id: str) -> Optional[Activity]:
    for a in _ACTIVITY_DATA:
        if a.activity_id == activity_id:
            return a
    return None


def get_all_activity_ids() -> List[str]:
    return [a.activity_id for a in _ACTIVITY_DATA]


# ---------------------------------------------------------------------------
# Tool functions exposed to the ReAct agent
# ---------------------------------------------------------------------------

def calculator_tool(expression: str) -> str:
    """Evaluate a numeric expression and return the result as a string.

    Args:
        expression (str): A pure-arithmetic expression, e.g. "25 + 65*2 + 40".

    Returns:
        str: The numeric result, or an error message string.
    """
    try:
        # numexpr.evaluate returns a numpy scalar; .item() unwraps to Python float/int.
        result = numexpr.evaluate(expression).item()
        return f"{result}"
    except Exception as exc:  # pragma: no cover — surfaces error string to LLM
        return f"ERROR: could not evaluate '{expression}': {exc}"


SUPPORTED_CITIES = {"agentsville"}


def get_activities_by_date_tool(date: str, city: str) -> List[Dict[str, Any]]:
    """Return the list of activities available in a given city on a given date.

    Use this tool when you need to look up which activities are bookable on a
    specific day — for example, to find a replacement activity after another
    one was discarded due to bad weather, or to verify that an activity_id you
    plan to schedule actually exists on the date you intend.

    Args:
        date (str): The date to look up, formatted as an ISO 8601 calendar
            date string in the form 'YYYY-MM-DD' (e.g., '2025-06-11').
        city (str): The destination city to search, e.g. 'AgentsVille'. Case
            insensitive. Only AgentsVille has activity data available; any
            other city will return an empty list.

    Returns:
        list[dict]: A list of activity records. Each record is a dict with the
            keys: 'activity_id' (str), 'name' (str), 'description' (str),
            'location' (str), 'start_time' (str, ISO 8601 datetime),
            'end_time' (str, ISO 8601 datetime), 'price' (float, per person USD),
            'related_interests' (list[str]), and 'indoor' (bool).
            Returns an empty list if no activities are available, if the city
            is not supported, or if the date format is invalid.
    """
    if not isinstance(city, str) or city.strip().lower() not in SUPPORTED_CITIES:
        return []
    try:
        target = datetime.fromisoformat(date).date() if "T" in date else datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError as exc:
        return [{"error": f"Invalid date '{date}'. Expected YYYY-MM-DD. ({exc})"}]
    return [
        a.model_dump(mode="json")
        for a in _ACTIVITY_DATA
        if a.activity_date == target
    ]


# `run_evals_tool` and `final_answer_tool` are bound at notebook level so they
# can close over the evaluator functions defined there (which need access to
# an OpenAI client). We expose factory functions here.

def make_run_evals_tool(eval_fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
    """Bind a user-supplied evaluation function to the standard tool interface."""

    def run_evals_tool(travel_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Run all evaluation criteria against a proposed travel plan.

        Args:
            travel_plan (dict): A JSON-serializable TravelPlan object to evaluate.

        Returns:
            dict: { "all_passed": bool, "results": [ {"name": str, "passed": bool, "message": str}, ... ] }
        """
        return eval_fn(travel_plan)

    return run_evals_tool


def final_answer_tool(travel_plan: Dict[str, Any]) -> Dict[str, Any]:
    """Submit the final, revised itinerary. Calling this ends the ReAct loop.

    Args:
        travel_plan (dict): The final TravelPlan as a JSON-serializable dict.

    Returns:
        dict: { "status": "final", "travel_plan": <the submitted plan> }
    """
    return {"status": "final", "travel_plan": travel_plan}


# ---------------------------------------------------------------------------
# available_tools registry (filled at notebook level with bound run_evals_tool)
# ---------------------------------------------------------------------------

def build_available_tools(run_evals_tool: Callable[..., Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Build the tool registry the ReAct agent will use.

    Each entry is: tool_name -> { 'function': callable, 'description': str, 'parameters': {name: type_str} }
    """
    return {
        "calculator_tool": {
            "function": calculator_tool,
            "description": (
                "Evaluate a pure-arithmetic expression (e.g. '25 + 65*2 + 40'). "
                "Use this for totaling activity prices or multiplying by number of travelers. "
                "Returns the numeric result as a string."
            ),
            "parameters": {"expression": "str"},
        },
        "get_activities_by_date_tool": {
            "function": get_activities_by_date_tool,
            "description": (
                "Look up available activities in a given city on a single date. "
                "Returns a list of activity records (activity_id, name, description, location, "
                "start_time, end_time, price, related_interests, indoor). "
                "Only AgentsVille has activity data; other cities return an empty list."
            ),
            "parameters": {
                "date": "str (YYYY-MM-DD)",
                "city": "str (e.g. 'AgentsVille')",
            },
        },
        "run_evals_tool": {
            "function": run_evals_tool,
            "description": (
                "Run all evaluation criteria against a proposed travel plan. "
                "Returns {'all_passed': bool, 'results': [...]} so you can see which checks fail."
            ),
            "parameters": {"travel_plan": "dict (TravelPlan JSON)"},
        },
        "final_answer_tool": {
            "function": final_answer_tool,
            "description": (
                "Submit the final, revised TravelPlan. Calling this signals the end of the ReAct loop. "
                "Only call this AFTER run_evals_tool reports all checks pass."
            ),
            "parameters": {"travel_plan": "dict (TravelPlan JSON)"},
        },
    }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def print_in_box(text: str, title: Optional[str] = None, width: int = 88) -> None:
    """Pretty-print a block of text inside an ASCII box for notebook readability."""
    border = "+" + "-" * (width - 2) + "+"
    print(border)
    if title:
        head = f"| {title.upper()}".ljust(width - 1) + "|"
        print(head)
        print(border)
    for raw_line in text.splitlines() or [""]:
        # word-wrap each line to the inner width
        inner = width - 4
        line = raw_line
        if not line:
            print("| " + " " * (width - 4) + " |")
            continue
        while len(line) > inner:
            cut = line.rfind(" ", 0, inner)
            if cut <= 0:
                cut = inner
            print("| " + line[:cut].ljust(inner) + " |")
            line = line[cut:].lstrip()
        if line:
            print("| " + line.ljust(inner) + " |")
    print(border)


def extract_json_block(text: str) -> Optional[str]:
    """Best-effort extraction of a JSON object from an LLM response.

    Handles plain JSON, fenced ```json blocks, and the THOUGHT/ACTION format
    where ACTION contains a JSON object.
    """
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1)

    # Try to find the largest balanced {...} block.
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_thought_action(text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Parse a ReAct message of the form:

        THOUGHT: <free text reasoning>
        ACTION: {"tool_name": "...", "arguments": {...}}

    Returns (thought_text, action_dict_or_None).
    """
    thought_match = re.search(r"THOUGHT\s*:\s*(.*?)(?=\bACTION\s*:|\Z)", text, flags=re.DOTALL | re.IGNORECASE)
    action_match = re.search(r"ACTION\s*:\s*(.*)", text, flags=re.DOTALL | re.IGNORECASE)

    thought = thought_match.group(1).strip() if thought_match else ""
    action = None
    if action_match:
        action_block = action_match.group(1).strip()
        blob = extract_json_block(action_block) or action_block
        try:
            action = json.loads(blob)
        except json.JSONDecodeError:
            try:
                from json_repair import repair_json  # type: ignore
                action = json.loads(repair_json(blob))
            except Exception:
                action = None
    return thought, action
