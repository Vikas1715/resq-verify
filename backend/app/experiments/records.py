"""Turn a method Outcome into the full, flat-plus-nested run record that is stored and exported."""
from __future__ import annotations

from typing import Any, Optional

from backend.app.core.config import Settings
from backend.app.schemas.scenario import Scenario
from backend.app.services.methods import BUDGETED_METHODS, METHOD_CODES, Outcome
from backend.app.utils.ids import new_run_id, utcnow_iso
from backend.app.validators.metrics import plan_quality
from backend.app.schemas.validation import VIOLATION_TYPES


def estimate_cost(settings: Settings, usage: dict) -> Optional[float]:
    """USD cost from user-supplied prices; None when prices are not configured (never guessed)."""
    if settings.price_input_per_m is None or settings.price_output_per_m is None:
        return None
    inp = usage.get("input") or 0
    out = (usage.get("output") or 0) + (usage.get("thoughts") or 0)
    return round(inp / 1e6 * settings.price_input_per_m + out / 1e6 * settings.price_output_per_m, 6)


def _sum_usage(rounds: list[dict]) -> dict:
    """Total tokens over rounds. A field stays None when the API reported nothing for it in any round (never filled with 0)."""
    tot: dict[str, Any] = {"input": None, "output": None, "thoughts": None, "total": None, "estimated": False}
    for r in rounds:
        u = r.get("token_usage") or {}
        for k in ("input", "output", "thoughts", "total"):
            if u.get(k) is not None:
                tot[k] = (tot[k] or 0) + u[k]
        tot["estimated"] = tot["estimated"] or bool(u.get("estimated"))
    return tot


def _violation_keys(round_: dict) -> set[tuple]:
    return {(v.get("type"), v.get("entity")) for v in ((round_.get("validation") or {}).get("violations") or [])}


def failure_persistence(rounds: list[dict]) -> dict[str, Any]:
    """How many of the initial plan's violations (type, entity) are still present later.

    `final_fraction`: share of the initial violations still present in the final plan (None if the initial plan was valid).
    `by_round`: for each later round, the share of the PREVIOUS round's violations that are still present.
    For M5 (independent resamples) this is how often a fresh draw repeats the same mistake; for M3 it is how often a mistake
    survives feedback that named it.
    """
    llm_rounds = [r for r in rounds if r.get("kind") != "solver"]
    if not llm_rounds:
        return {"initial_violations": None, "persisted_in_final": None, "final_fraction": None, "by_round": [], "by_type": {}}
    keys = [_violation_keys(r) for r in llm_rounds]
    first, last = keys[0], keys[-1]
    by_round = []
    for prev, cur in zip(keys, keys[1:]):
        by_round.append(round(len(prev & cur) / len(prev), 4) if prev else None)
    by_type: dict[str, dict[str, int]] = {}
    for t, _e in first:
        by_type.setdefault(t, {"initial": 0, "persisted": 0})["initial"] += 1
    for t, e in first & last:
        by_type[t]["persisted"] += 1
    return {"initial_violations": len(first), "persisted_in_final": len(first & last),
            "final_fraction": round(len(first & last) / len(first), 4) if first else None, "by_round": by_round, "by_type": by_type}


def build_run_record(*, experiment_id: str, scenario: Scenario, outcome: Outcome, run_index: int, start_time: str,
                     end_time: str, wall_time_s: float, settings: Settings, llm_meta: dict, method_cfg: Optional[dict],
                     prompt_meta: Optional[dict], reference: Optional[dict], shared_initial_plan: Optional[bool],
                     budget_mode: Optional[str] = None, derived_from_budget: Optional[int] = None) -> dict[str, Any]:
    rounds = outcome.rounds
    ok = outcome.status == "ok"
    first = rounds[0] if rounds else None
    usage = _sum_usage(rounds)
    init_usage = _sum_usage(rounds[:1])
    latency_total = round(sum(r.get("latency_s") or 0 for r in rounds), 4)
    latency_initial = (rounds[0].get("latency_s") or 0) if rounds else 0
    quality = plan_quality(scenario, outcome.final_plan) if outcome.final_plan is not None else None
    final_feasible = outcome.final_report.valid if (ok and outcome.final_report is not None) else None
    initial_feasible = first["is_valid"] if (ok and first) else None
    repair_kind = {"resq_verify": "repair", "self_critique": "critique_revise", "resample_verify": "resample"}.get(outcome.method)
    n_repairs = sum(1 for r in rounds if r["kind"] == repair_kind) if repair_kind else 0

    ref_obj = (reference or {}).get("objective_weighted_served")
    gap = None
    if outcome.method != "ortools" and final_feasible and quality and ref_obj:
        gap = round((ref_obj - quality["weighted_served"]) / ref_obj, 6)  # negative = better than the solver's (time-limited) reference

    counts_final = outcome.final_report.counts_by_type if outcome.final_report else {t: 0 for t in VIOLATION_TYPES}
    counts_init = first["violation_counts"] if first else {t: 0 for t in VIOLATION_TYPES}
    usage_list = [(r.get("model_version")) for r in rounds if r.get("model_version")]
    api_errors = [e for r in rounds for e in (r.get("api_errors") or [])]
    llm_rounds = [r for r in rounds if r.get("kind") != "solver"]
    first_ok, cum = None, 0
    for r in llm_rounds:                      # cumulative LLM calls at the first valid plan (None = never feasible within the run)
        cum += int(r.get("calls") or 0)
        if r.get("is_valid") and first_ok is None and ok:
            first_ok = cum
    templ = (prompt_meta or {}).get("template_hashes") or {}

    rec: dict[str, Any] = {
        "run_id": new_run_id(), "experiment_id": experiment_id, "status": outcome.status,
        "scenario_id": scenario.scenario_id, "scenario_seed": scenario.seed, "scenario_size": scenario.size,
        "scarcity_level": scenario.scarcity, "scenario_content_hash": scenario.content_hash,
        "difficulty_score": scenario.difficulty.difficulty_score if scenario.difficulty else None,
        "difficulty_level": scenario.difficulty_level,
        "method": outcome.method, "method_code": METHOD_CODES[outcome.method], "run_index": run_index,
        "provider": llm_meta.get("provider"), "is_simulated": bool(llm_meta.get("is_simulated")),
        "model": llm_meta.get("model"), "model_version": usage_list[-1] if usage_list else None,
        "temperature": (method_cfg or {}).get("temperature"), "max_output_tokens": (method_cfg or {}).get("max_output_tokens"),
        "prompt_version": (prompt_meta or {}).get("version"), "prompt_hash": (prompt_meta or {}).get("hash"),
        "prompt_template_hashes": templ or None,
        "planner_template_hash": templ.get("planner.txt") if outcome.method != "ortools" else None,
        "revision_template_hash": templ.get("revision.txt") if outcome.method in ("self_critique", "resq_verify") else None,
        "call_budget": outcome.call_budget, "calls_used": outcome.calls_used if outcome.method != "ortools" else 0,
        "budget_mode": budget_mode if outcome.method in BUDGETED_METHODS else None,
        "derived_from_budget": derived_from_budget, "first_feasible_call": first_ok,
        "n_violations_by_round": [r.get("n_violations") for r in llm_rounds],
        "violation_types_by_round": [sorted(t for t, n in (r.get("violation_counts") or {}).items() if n) for r in llm_rounds],
        "failure_persistence": failure_persistence(rounds) if ok else None,
        "feedback_format": (method_cfg or {}).get("feedback_format") if outcome.method == "resq_verify" else None,
        "max_repair_rounds": (method_cfg or {}).get("max_repair_rounds") if outcome.method in ("resq_verify", "self_critique", "resample_verify") else None,
        "shared_initial_plan": shared_initial_plan if outcome.method in ("llm_only", "self_critique", "resq_verify", "resample_verify") else None,
        "start_time": start_time, "end_time": end_time, "wall_time_s": round(wall_time_s, 4),
        "latency_s": latency_total, "latency_initial_s": latency_initial, "latency_repair_s": round(latency_total - latency_initial, 4),
        "token_usage": usage, "initial_token_usage": init_usage, "estimated_cost_usd": estimate_cost(settings, usage) if usage["total"] else None,
        "retry_count": sum(r.get("retry_count") or 0 for r in rounds), "api_errors": api_errors,
        "rounds": rounds,
        "parsed_plan": outcome.final_plan.model_dump() if outcome.final_plan else None,
        "raw_llm_output": outcome.final_raw, "validator_report": outcome.final_report.model_dump() if outcome.final_report else None,
        "repair_rounds_used": n_repairs, "stop_reason": outcome.stop_reason,
        "initial_feasible": initial_feasible, "final_feasibility": final_feasible,
        "repair_success": (True if final_feasible else False) if (ok and initial_feasible is False) else None,
        "n_violations_initial": first["n_violations"] if (ok and first) else None,
        "n_violations_final": outcome.final_report.n_violations if (ok and outcome.final_report) else None,
        "violation_counts_initial": counts_init, "violation_counts": counts_final,
        "plan_quality": (quality["quality_norm"] if (quality and final_feasible) else None),
        "plan_quality_raw": quality["quality_norm"] if quality else None,
        "weighted_served": quality["weighted_served"] if quality else None,
        "people_served": quality["people_served"] if quality else None,
        "solver": outcome.solver,
        "solver_objective": (outcome.solver or {}).get("objective_weighted_served") if outcome.method == "ortools" else ref_obj,
        "solver_reference_status": (reference or {}).get("status"),
        "optimality_gap": gap, "failure_reason": outcome.failure_reason,
    }
    return rec
