"""Allocation plan schema. The LLM must return exactly this structure; the validator reads only this."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TeamAssignment(BaseModel):
    model_config = ConfigDict(extra="ignore")
    team_id: str
    incident_id: str


class VehicleAssignment(BaseModel):
    model_config = ConfigDict(extra="ignore")
    vehicle_id: str
    incident_id: str
    trips: int
    people: int


class ShelterAllocation(BaseModel):
    model_config = ConfigDict(extra="ignore")
    incident_id: str
    shelter_id: str
    people: int


class UnservedDemand(BaseModel):
    model_config = ConfigDict(extra="ignore")
    incident_id: str
    people: int


class Plan(BaseModel):
    model_config = ConfigDict(extra="ignore")
    team_assignments: list[TeamAssignment] = Field(default_factory=list)
    vehicle_assignments: list[VehicleAssignment] = Field(default_factory=list)
    shelter_allocations: list[ShelterAllocation] = Field(default_factory=list)
    unserved_demand: list[UnservedDemand] = Field(default_factory=list)
    reasoning_summary: str = ""

    def record_set(self) -> set[tuple]:
        """Comparable set of all records (used to measure how much a repair changed the plan)."""
        s: set[tuple] = set()
        for t in self.team_assignments:
            s.add(("T", t.team_id, t.incident_id))
        for v in self.vehicle_assignments:
            s.add(("V", v.vehicle_id, v.incident_id, v.trips, v.people))
        for a in self.shelter_allocations:
            s.add(("S", a.incident_id, a.shelter_id, a.people))
        for u in self.unserved_demand:
            s.add(("U", u.incident_id, u.people))
        return s


def plan_json_schema_text() -> str:
    """Human-readable schema shown to the LLM inside prompts."""
    return (
        "{\n"
        '  "team_assignments":    [ {"team_id": "TEAM-01", "incident_id": "INC-01"} ],\n'
        '  "vehicle_assignments": [ {"vehicle_id": "VEH-01", "incident_id": "INC-01", "trips": 2, "people": 40} ],\n'
        '  "shelter_allocations": [ {"incident_id": "INC-01", "shelter_id": "SHL-01", "people": 40} ],\n'
        '  "unserved_demand":     [ {"incident_id": "INC-02", "people": 15} ],\n'
        '  "reasoning_summary":   "one or two sentences"\n'
        "}"
    )


def gemini_response_schema() -> dict[str, Any]:
    """OpenAPI-subset schema for Gemini's optional `responseSchema` (GEMINI_USE_RESPONSE_SCHEMA=true)."""
    def arr(props: dict[str, str]) -> dict[str, Any]:
        return {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {k: {"type": v} for k, v in props.items()},
                "required": list(props.keys()),
            },
        }

    return {
        "type": "OBJECT",
        "properties": {
            "team_assignments": arr({"team_id": "STRING", "incident_id": "STRING"}),
            "vehicle_assignments": arr({"vehicle_id": "STRING", "incident_id": "STRING", "trips": "INTEGER", "people": "INTEGER"}),
            "shelter_allocations": arr({"incident_id": "STRING", "shelter_id": "STRING", "people": "INTEGER"}),
            "unserved_demand": arr({"incident_id": "STRING", "people": "INTEGER"}),
            "reasoning_summary": {"type": "STRING"},
        },
        "required": ["team_assignments", "vehicle_assignments", "shelter_allocations", "unserved_demand"],
    }
