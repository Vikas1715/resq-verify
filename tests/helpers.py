"""Hand-built scenario with a known optimum, plus a scripted LLM for deterministic method tests."""
from __future__ import annotations

import json
from typing import Optional

from backend.app.schemas.plan import Plan
from backend.app.schemas.scenario import Scenario
from backend.app.services.llm.base import LLMClient, LLMResult


def tiny_scenario() -> Scenario:
    """2 incidents (50 people), everyone can be served: weighted optimum = 5*30 + 3*20 = 210."""
    return Scenario.model_validate({
        "scenario_id": "tiny", "generator_version": "test", "seed": 0, "size": "small", "scarcity": "low",
        "incidents": [
            {"id": "I1", "name": "School", "hazard": "flood", "people": 30, "priority": 5, "road_id": "R1", "waterway": False,
             "narrow_access": False, "allowed_vehicle_types": ["bus", "truck", "ambulance"]},
            {"id": "I2", "name": "Village", "hazard": "flood", "people": 20, "priority": 3, "road_id": "R2", "waterway": True,
             "narrow_access": True, "allowed_vehicle_types": ["boat", "truck"]},
        ],
        "roads": [{"id": "R1", "blocked": False}, {"id": "R2", "blocked": True}],
        "teams": [{"id": "T1", "available": True, "max_incidents": 2}, {"id": "T2", "available": False, "max_incidents": 1}],
        "vehicles": [
            {"id": "V1", "type": "bus", "capacity": 20, "max_trips": 2},
            {"id": "V2", "type": "boat", "capacity": 10, "max_trips": 3},
            {"id": "V3", "type": "truck", "capacity": 15, "max_trips": 2},
        ],
        "shelters": [{"id": "S1", "name": "Hall", "capacity": 40}, {"id": "S2", "name": "Church", "capacity": 25}],
    })


def valid_plan_dict() -> dict:
    return {
        "team_assignments": [{"team_id": "T1", "incident_id": "I1"}, {"team_id": "T1", "incident_id": "I2"}],
        "vehicle_assignments": [{"vehicle_id": "V1", "incident_id": "I1", "trips": 2, "people": 30},
                                {"vehicle_id": "V2", "incident_id": "I2", "trips": 2, "people": 20}],
        "shelter_allocations": [{"incident_id": "I1", "shelter_id": "S1", "people": 30},
                                {"incident_id": "I2", "shelter_id": "S2", "people": 20}],
        "unserved_demand": [], "reasoning_summary": "test",
    }


def valid_plan() -> Plan:
    return Plan.model_validate(valid_plan_dict())


def broken_plan_dict() -> dict:
    d = valid_plan_dict()
    d["vehicle_assignments"][0]["people"] = 45     # V1: 2 trips x 20 = 40 < 45  -> vehicle_capacity (and coverage)
    return d


class ScriptedLLM(LLMClient):
    """Returns canned replies in order and records every prompt, so tests can assert on what was sent."""
    provider = "scripted"
    is_simulated = False
    model = "scripted-model"

    def __init__(self, replies: list[str]):
        self.replies, self.prompts = list(replies), []

    def generate(self, prompt: str, **kw) -> LLMResult:
        self.prompts.append(prompt)
        text = self.replies.pop(0) if self.replies else self.replies_last
        self.replies_last = text
        return LLMResult(text=text, model=self.model, input_tokens=100, output_tokens=50, total_tokens=150, latency_s=0.01)

    replies_last = "{}"


def j(d: dict) -> str:
    return json.dumps(d)
