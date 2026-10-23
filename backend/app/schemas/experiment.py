"""Experiment configuration (what to run). Loadable from YAML files in experiments/configs/."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

METHOD_NAMES = ["llm_only", "self_critique", "resq_verify", "resample_verify", "ortools"]
LEVEL_NAMES = ["L0", "L1", "L2", "L3", "L4"]


class ExperimentConfig(BaseModel):
    name: str = "experiment"
    description: str = ""
    sizes: list[Literal["small", "medium", "large"]] = ["small"]
    scarcity: list[Literal["low", "medium", "high"]] = ["high"]
    difficulty_levels: list[Literal["L0", "L1", "L2", "L3", "L4"]] = Field(
        default_factory=list, description="when set, scenarios come from these difficulty levels and sizes/scarcity are ignored")
    num_scenarios: int = Field(5, ge=1, le=500, description="scenarios per (size, scarcity) cell, or per difficulty level")
    base_seed: int = 42
    repetitions: int = Field(1, ge=1, le=20, description="LLM repetitions per scenario (LLM output varies run to run)")
    methods: list[str] = Field(default_factory=lambda: list(METHOD_NAMES))
    max_repair_rounds: int = Field(3, ge=0, le=10, description="K-based budget, used only when call_budgets is empty")
    call_budgets: list[int] = Field(default_factory=list, description="LLM-call budgets to compare, e.g. [1, 2, 3, 5]; empty = use K")
    nested_budgets: bool = Field(True, description="run each (scenario, method, repetition) once at the largest budget and derive the "
                                                   "smaller budgets from the same trajectory; false = independent run per budget")
    self_critique_rounds: Optional[int] = Field(None, ge=0, le=10, description="defaults to max_repair_rounds")
    shared_initial_plan: bool = Field(True, description="M1, M2, M3 and M5 start from the same first LLM plan (paired design)")
    feedback_format: Literal["structured", "free_text"] = "structured"
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_output_tokens: Optional[int] = Field(None, ge=256)
    request_delay_s: Optional[float] = Field(None, ge=0.0)
    solver_time_limit_s: Optional[float] = Field(None, gt=0.0)
    compute_reference: bool = Field(True, description="always solve with CP-SAT to get the reference objective, even if M4 is not selected")
    overrides: dict[str, Any] = Field(default_factory=dict, description="scenario generator overrides, e.g. {n_incidents: 6}")

    @field_validator("call_budgets")
    @classmethod
    def _check_budgets(cls, v: list[int]) -> list[int]:
        if any((not isinstance(b, int)) or b < 1 or b > 20 for b in v):
            raise ValueError("call_budgets must be integers between 1 and 20")
        return sorted(set(v))

    @field_validator("methods")
    @classmethod
    def _check_methods(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("methods must not be empty")
        bad = [m for m in v if m not in METHOD_NAMES]
        if bad:
            raise ValueError(f"unknown method(s) {bad}; allowed: {METHOD_NAMES}")
        return [m for m in METHOD_NAMES if m in v]  # canonical order

    @property
    def critique_rounds(self) -> int:
        return self.max_repair_rounds if self.self_critique_rounds is None else self.self_critique_rounds

    @property
    def uses_budgets(self) -> bool:
        return bool(self.call_budgets)
