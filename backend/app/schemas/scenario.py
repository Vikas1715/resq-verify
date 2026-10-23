"""Scenario data model: what the planner (LLM or solver) is given."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

VehicleType = Literal["bus", "truck", "ambulance", "boat"]
Hazard = Literal["flood", "earthquake", "landslide", "building_fire"]
Size = Literal["small", "medium", "large"]
Scarcity = Literal["low", "medium", "high"]


class Incident(BaseModel):
    id: str
    name: str
    hazard: Hazard
    people: int = Field(ge=0, description="People who must be evacuated")
    priority: int = Field(ge=1, le=5, description="5 = most urgent")
    road_id: str
    waterway: bool = Field(description="True if the site can be reached by water")
    narrow_access: bool = Field(description="True if large vehicles (bus) cannot enter")
    allowed_vehicle_types: list[VehicleType]


class Road(BaseModel):
    id: str
    blocked: bool


class Team(BaseModel):
    id: str
    available: bool
    max_incidents: int = Field(ge=0)


class Vehicle(BaseModel):
    id: str
    type: VehicleType
    capacity: int = Field(ge=0, description="People per trip")
    max_trips: int = Field(ge=0, description="Trips available in the planning window")


class Shelter(BaseModel):
    id: str
    name: str
    capacity: int = Field(ge=0)


class Difficulty(BaseModel):
    total_demand: int
    total_transport_capacity: int
    total_shelter_capacity: int
    demand_to_transport_ratio: float
    demand_to_shelter_ratio: float
    blocked_road_fraction: float
    unreachable_incidents: int
    available_team_slots: int
    team_load_ratio: float
    feasible_vehicle_incident_pairs: int
    difficulty_score: float
    # filled only for scenarios built from a difficulty level (None otherwise, and left out of the content hash when None)
    servable_incidents: Optional[int] = Field(None, description="incidents with at least one usable vehicle type and some team capacity")
    servable_demand_fraction: Optional[float] = None
    boat_count: Optional[int] = None
    restricted_incidents: Optional[int] = Field(None, description="incidents whose allowed vehicle types were narrowed by the level")


class Scenario(BaseModel):
    scenario_id: str
    generator_version: str
    seed: int
    size: Size
    scarcity: Scarcity
    params: dict[str, Any] = Field(default_factory=dict)
    difficulty_level: Optional[str] = Field(None, description="L0..L4 when built from a difficulty level")
    incidents: list[Incident]
    roads: list[Road]
    teams: list[Team]
    vehicles: list[Vehicle]
    shelters: list[Shelter]
    difficulty: Optional[Difficulty] = None
    content_hash: Optional[str] = None
    created_at: Optional[str] = None

    # lookup helpers (methods, not fields)
    def incident_map(self) -> dict[str, Incident]:
        return {x.id: x for x in self.incidents}

    def road_map(self) -> dict[str, Road]:
        return {x.id: x for x in self.roads}

    def team_map(self) -> dict[str, Team]:
        return {x.id: x for x in self.teams}

    def vehicle_map(self) -> dict[str, Vehicle]:
        return {x.id: x for x in self.vehicles}

    def shelter_map(self) -> dict[str, Shelter]:
        return {x.id: x for x in self.shelters}

    def prompt_dict(self) -> dict[str, Any]:
        """Compact view shown to the LLM (no hidden metadata such as difficulty or content hash)."""
        return {
            "scenario_id": self.scenario_id,
            "incidents": [x.model_dump() for x in self.incidents],
            "roads": [x.model_dump() for x in self.roads],
            "teams": [x.model_dump() for x in self.teams],
            "vehicles": [x.model_dump() for x in self.vehicles],
            "shelters": [x.model_dump() for x in self.shelters],
        }
