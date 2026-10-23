"""Validation report schema (machine-readable output of the deterministic validator)."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

ViolationType = Literal[
    "vehicle_capacity", "team_availability", "shelter_capacity",
    "road_accessibility", "coverage", "compatibility", "schema_integrity",
]

VIOLATION_TYPES: list[str] = [
    "vehicle_capacity", "team_availability", "shelter_capacity",
    "road_accessibility", "coverage", "compatibility", "schema_integrity",
]

RULE_FOR_TYPE: dict[str, str] = {
    "vehicle_capacity": "V1", "team_availability": "V2", "shelter_capacity": "V3",
    "road_accessibility": "V4", "coverage": "V5", "compatibility": "V6", "schema_integrity": "V7",
}


class Violation(BaseModel):
    rule: str = Field(description="V1..V7")
    type: ViolationType
    severity: Literal["hard"] = "hard"
    entity: str
    incident_id: Optional[str] = None
    expected: Any = None
    actual: Any = None
    message: str


class ValidationReport(BaseModel):
    valid: bool
    n_violations: int
    violations: list[Violation]
    counts_by_type: dict[str, int]
    parse_failed: bool = False
    parse_error: Optional[str] = None
    validator_version: str = "1.0"
