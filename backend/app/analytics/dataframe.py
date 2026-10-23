"""Flatten stored run records into a tidy pandas DataFrame (one row per run)."""
from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from backend.app.schemas.validation import VIOLATION_TYPES

BUDGETED = ("self_critique", "resq_verify", "resample_verify")   # methods whose result depends on the call budget
FLAT_COLUMNS = [
    "experiment_id", "run_id", "status", "scenario_id", "scenario_seed", "scenario_size", "scarcity_level",
    "difficulty_level", "difficulty_score", "method", "method_code", "run_index", "call_budget", "calls_used", "budget_mode",
    "derived_from_budget", "first_feasible_call", "provider", "is_simulated", "model", "model_version",
    "temperature", "prompt_version", "prompt_hash", "planner_template_hash", "revision_template_hash",
    "feedback_format", "max_repair_rounds",
    "initial_feasible", "final_feasibility", "repair_success", "repair_rounds_used",
    "n_violations_initial", "n_violations_final", "plan_quality", "plan_quality_raw", "optimality_gap",
    "weighted_served", "people_served", "latency_s", "latency_initial_s", "latency_repair_s", "wall_time_s",
    "tokens_input", "tokens_output", "tokens_total", "estimated_cost_usd", "retry_count", "stop_reason",
    "failure_reason", "solver_objective", "solver_reference_status", "persistence_final_fraction",
]


def runs_to_frame(runs: Iterable[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in runs:
        row = {k: r.get(k) for k in FLAT_COLUMNS if not k.startswith("tokens_") and k != "persistence_final_fraction"}
        row["persistence_final_fraction"] = (r.get("failure_persistence") or {}).get("final_fraction")
        usage = r.get("token_usage") or {}
        row["tokens_input"] = usage.get("input")
        row["tokens_output"] = (usage.get("output") or 0) + (usage.get("thoughts") or 0) if usage else None
        row["tokens_total"] = usage.get("total")
        for t in VIOLATION_TYPES:
            row[f"viol_final_{t}"] = (r.get("violation_counts") or {}).get(t)
            row[f"viol_initial_{t}"] = (r.get("violation_counts_initial") or {}).get(t)
        rows.append(row)
    cols = FLAT_COLUMNS + [f"viol_final_{t}" for t in VIOLATION_TYPES] + [f"viol_initial_{t}" for t in VIOLATION_TYPES]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df
    for c in ("initial_feasible", "final_feasibility", "repair_success", "is_simulated"):
        df[c] = df[c].astype("object")
    df["effective_quality"] = [q if (f is True and q is not None) else 0.0 for q, f in zip(df["plan_quality"], df["final_feasibility"])]
    df.loc[df["status"] != "ok", "effective_quality"] = float("nan")
    return df


def ok_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Runs that completed (infrastructure/API errors are excluded from all rates)."""
    return df[df["status"] == "ok"] if not df.empty else df


def budgets_in(df: pd.DataFrame) -> list[int]:
    """Call budgets that were actually compared (budget-mode runs of M2/M3/M5), sorted. Empty for K-based (legacy) experiments."""
    if df.empty or "budget_mode" not in df:
        return []
    sub = df[df["method"].isin(BUDGETED) & df["budget_mode"].isin(["nested", "independent"]) & df["call_budget"].notna()]
    return sorted({int(b) for b in sub["call_budget"]})


def budget_view(df: pd.DataFrame, budget: int | None) -> pd.DataFrame:
    """The runs to compare at ONE call budget.

    M2, M3 and M5 rows are kept only for that budget. M1 (always one call) and M4 (no LLM calls) are kept as they are, so they
    appear as reference lines at every budget. K-based (legacy) rows are never filtered.
    """
    if df.empty or budget is None:
        return df
    budgeted_mode = df["budget_mode"].isin(["nested", "independent"]) & df["method"].isin(BUDGETED)
    keep = ~budgeted_mode | (df["call_budget"] == budget)
    return df[keep]


def headline_budget(df: pd.DataFrame, requested: int | None = None) -> int | None:
    bs = budgets_in(df)
    if not bs:
        return None
    return requested if requested in bs else bs[-1]
