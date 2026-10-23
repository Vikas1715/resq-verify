"""Aggregate tables built from the run DataFrame. Every function returns plain JSON-able structures."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from backend.app.analytics import stats as S
from backend.app.analytics.dataframe import BUDGETED, budget_view, budgets_in, headline_budget, ok_runs, runs_to_frame
from backend.app.schemas.validation import VIOLATION_TYPES
from backend.app.services.methods import METHODS, METHOD_LABELS

NO_CONCLUSION = "No experimental conclusion available yet."

# (metric column, label, is_binary, higher_is_better)
COMPARE_METRICS = [
    ("final_feasibility", "Feasibility rate", True, True),
    ("effective_quality", "Effective plan quality (0 if infeasible)", False, True),
    ("n_violations_final", "Violations in final plan", False, False),
    ("latency_s", "Latency (s)", False, False),
    ("tokens_total", "Total tokens", False, False),
]


def _num(df: pd.DataFrame, col: str) -> list:
    return pd.to_numeric(df[col].astype(float), errors="coerce").dropna().tolist() if col in df else []


def method_summary(df: pd.DataFrame) -> list[dict[str, Any]]:
    """One row per method with headline metrics."""
    rows = []
    for m in METHODS:
        sub_all = df[df["method"] == m] if not df.empty else df
        if sub_all.empty:
            continue
        sub = sub_all[sub_all["status"] == "ok"]
        feas = S.proportion(sub["final_feasibility"].tolist())
        init = S.proportion(sub["initial_feasible"].tolist()) if m != "ortools" else {"rate": None, "n": 0, "k": 0, "ci_low": None, "ci_high": None}
        rep_src = sub[sub["initial_feasible"] == False]  # noqa: E712 (object column)
        rep = S.proportion(rep_src["repair_success"].tolist()) if m in BUDGETED else {"rate": None, "n": 0, "k": 0, "ci_low": None, "ci_high": None}
        feasible_only = sub[sub["final_feasibility"] == True]  # noqa: E712
        rows.append({
            "method": m, "label": METHOD_LABELS[m], "n_runs": int(len(sub_all)), "n_ok": int(len(sub)), "n_errors": int(len(sub_all) - len(sub)),
            "call_budget": (int(sub["call_budget"].dropna().iloc[0]) if ("call_budget" in sub and sub["call_budget"].notna().any()
                                                                        and m in BUDGETED and sub["call_budget"].nunique() == 1) else None),
            "calls_used": S.describe(_num(sub, "calls_used")) if m != "ortools" else None,
            "feasibility": feas, "initial_feasibility": init, "violation_rate": (1 - feas["rate"]) if feas["rate"] is not None else None,
            "repair_success": rep,
            "repair_rounds_used": S.describe(_num(sub[sub["initial_feasible"] == False], "repair_rounds_used")) if m in BUDGETED else None,  # noqa: E712
            "quality_feasible_only": S.describe(_num(feasible_only, "plan_quality")),
            "effective_quality": S.describe(_num(sub, "effective_quality")),
            "optimality_gap": S.describe(_num(feasible_only, "optimality_gap")) if m != "ortools" else None,
            "violations_final": S.describe(_num(sub, "n_violations_final")),
            "latency_s": S.describe(_num(sub, "latency_s")),
            "tokens_total": S.describe(_num(sub, "tokens_total")) if m != "ortools" else None,
            "cost_usd_total": (float(pd.to_numeric(sub["estimated_cost_usd"].astype(float), errors="coerce").sum())
                               if m != "ortools" and sub["estimated_cost_usd"].notna().any() else None),
            "is_simulated": bool(sub["is_simulated"].astype(bool).any()) if m != "ortools" and len(sub) else False,
        })
    return rows


def breakdown(df: pd.DataFrame, by: str) -> list[dict[str, Any]]:
    """Feasibility / quality / latency per method for each level of `by` (scenario_size or scarcity_level)."""
    rows = []
    if df.empty:
        return rows
    ok = ok_runs(df)
    for level, grp in ok.groupby(by):
        for m in METHODS:
            sub = grp[grp["method"] == m]
            if sub.empty:
                continue
            f = S.proportion(sub["final_feasibility"].tolist())
            rows.append({by: level, "method": m, "n": f["n"], "feasibility": f["rate"], "ci_low": f["ci_low"], "ci_high": f["ci_high"],
                         "effective_quality": float(np.mean(_num(sub, "effective_quality"))) if _num(sub, "effective_quality") else None,
                         "latency_s": float(np.mean(_num(sub, "latency_s"))) if _num(sub, "latency_s") else None,
                         "mean_violations": float(np.mean(_num(sub, "n_violations_final"))) if _num(sub, "n_violations_final") else None})
    return rows


def violation_taxonomy(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Mean violation count per run, by type, for the initial plan and the final plan of each method."""
    rows = []
    ok = ok_runs(df) if not df.empty else df
    for m in METHODS:
        sub = ok[ok["method"] == m] if not ok.empty else ok
        if sub.empty or m == "ortools":
            continue
        for t in VIOLATION_TYPES:
            rows.append({"method": m, "violation_type": t,
                         "initial_total": int(pd.to_numeric(sub[f"viol_initial_{t}"], errors="coerce").fillna(0).sum()),
                         "final_total": int(pd.to_numeric(sub[f"viol_final_{t}"], errors="coerce").fillna(0).sum()),
                         "initial_per_run": float(pd.to_numeric(sub[f"viol_initial_{t}"], errors="coerce").fillna(0).mean()),
                         "final_per_run": float(pd.to_numeric(sub[f"viol_final_{t}"], errors="coerce").fillna(0).mean())})
    return rows


def comparisons(df: pd.DataFrame, pairs: Optional[list[tuple[str, str]]] = None) -> list[dict[str, Any]]:
    """Paired scenario-level tests, run separately at every call budget that was compared (equal-budget comparison).

    Default pairs: ResQ-Verify vs every other method present (this includes M5 = blind resampling, and M2 = self-critique).
    """
    budgets = budgets_in(df)
    if budgets:
        out: list[dict[str, Any]] = []
        for b in budgets:
            for r in _comparisons_one(budget_view(df, b), pairs):
                out.append({**r, "call_budget": b})
        return out
    return [{**r, "call_budget": None} for r in _comparisons_one(df, pairs)]


def _comparisons_one(df: pd.DataFrame, pairs: Optional[list[tuple[str, str]]] = None) -> list[dict[str, Any]]:
    if df.empty:
        return []
    present = [m for m in METHODS if m in set(df["method"])]
    if pairs is None:
        pairs = [("resq_verify", o) for o in present if o != "resq_verify"] if "resq_verify" in present else []
    rows: list[dict[str, Any]] = []
    for a, b in pairs:
        if a not in present or b not in present:
            continue
        group = []
        for metric, label, binary, higher_better in COMPARE_METRICS:
            if "ortools" in (a, b) and metric in ("tokens_total",):
                continue
            r = S.paired_compare(df, metric, a, b, binary=binary)
            r["label"], r["higher_is_better"] = label, higher_better
            group.append(r)
        rows.extend(S.apply_holm(group))  # Holm within each method pair
    return rows


def scalability(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Latency, tokens and feasibility versus scenario size, per method."""
    return [r for r in breakdown(df, "scenario_size")]


def by_scenario(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Per-scenario wide table: feasibility rate per method (useful to find hard scenarios)."""
    if df.empty:
        return []
    ok = ok_runs(df)
    rows = []
    for sid, grp in ok.groupby("scenario_id"):
        row = {"scenario_id": sid, "size": grp["scenario_size"].iloc[0], "scarcity": grp["scarcity_level"].iloc[0],
               "difficulty_level": grp["difficulty_level"].iloc[0], "difficulty_score": grp["difficulty_score"].iloc[0]}
        for m in METHODS:
            sub = grp[grp["method"] == m]
            row[f"feasible_{m}"] = (sum(1 for v in sub["final_feasibility"] if v is True) / len(sub)) if len(sub) else None
        rows.append(row)
    return rows


def _row_stats(sub: pd.DataFrame, method: Optional[str] = None) -> dict[str, Any]:
    """Metrics for one group of ok runs (shared by the budget and difficulty-level tables)."""
    f = S.proportion(sub["final_feasibility"].tolist())
    init_false = sub[sub["initial_feasible"] == False]  # noqa: E712 (object column)
    rep = S.proportion(init_false["repair_success"].tolist())
    eq, lat, cu, nv = _num(sub, "effective_quality"), _num(sub, "latency_s"), _num(sub, "calls_used"), _num(sub, "n_violations_final")
    tok = _num(sub, "tokens_total")
    return {"n": f["n"], "feasibility": f["rate"], "ci_low": f["ci_low"], "ci_high": f["ci_high"],
            "initial_feasibility": S.proportion(sub["initial_feasible"].tolist())["rate"],
            # repair success only means something for methods that get further attempts (M2, M3, M5)
            "repair_success": rep["rate"] if method in BUDGETED else None, "n_repair_cases": rep["n"] if method in BUDGETED else 0,
            "effective_quality": float(np.mean(eq)) if eq else None, "calls_used": float(np.mean(cu)) if cu else None,
            "latency_s": float(np.mean(lat)) if lat else None, "mean_violations": float(np.mean(nv)) if nv else None,
            "tokens_total": float(np.mean(tok)) if tok else None}


def budget_curve(df: pd.DataFrame, group_by: Optional[str] = None) -> list[dict[str, Any]]:
    """Feasibility, repair success, quality, calls and latency versus call budget, per method.

    M1 can only ever use one call, so its single result is repeated at every budget (`broadcast`=True). M4 (OR-Tools) is a
    reference with no LLM calls and is repeated too (`reference`=True). Rows with K-based (legacy) budgets are not included.
    """
    budgets = budgets_in(df)
    if not budgets:
        return []
    ok = ok_runs(df)
    rows: list[dict[str, Any]] = []
    groups = [(None, ok)] if group_by is None else list(ok.groupby(group_by))
    for gval, g in groups:
        for b in budgets:
            view = budget_view(g, b)
            for m in METHODS:
                sub = view[view["method"] == m]
                if sub.empty:
                    continue
                row = {"call_budget": b, "method": m, "label": METHOD_LABELS[m], **_row_stats(sub, m),
                       "broadcast": m == "llm_only", "reference": m == "ortools"}
                if group_by:
                    row[group_by] = gval
                rows.append(row)
    return rows


def by_difficulty_level(df: pd.DataFrame, call_budget: Optional[int] = None) -> list[dict[str, Any]]:
    """Per difficulty level and method, at one call budget (default: the largest budget compared)."""
    if df.empty or "difficulty_level" not in df or df["difficulty_level"].notna().sum() == 0:
        return []
    b = headline_budget(df, call_budget)
    ok = budget_view(ok_runs(df), b)
    rows = []
    for lv, g in sorted(ok.groupby("difficulty_level")):
        for m in METHODS:
            sub = g[g["method"] == m]
            if sub.empty:
                continue
            rows.append({"difficulty_level": lv, "call_budget": b if m in BUDGETED else (1 if m == "llm_only" else None),
                         "method": m, "label": METHOD_LABELS[m], **_row_stats(sub, m)})
    return rows


def failure_persistence(runs: list[dict], df: pd.DataFrame, call_budget: Optional[int] = None) -> list[dict[str, Any]]:
    """How often violations survive later attempts, per method (M2, M3, M5), at one call budget.

    For every run whose initial plan was infeasible: the share of the initial violations (type, entity) still present in the
    final plan, and per round the share of the previous round's violations still present. M5 draws independent plans, so its
    numbers show how often a fresh draw repeats the same mistake; M3 shows how often a mistake survives named feedback.
    """
    b = headline_budget(df, call_budget)
    rows = []
    for m in BUDGETED:
        sel = [r for r in runs if r.get("method") == m and r.get("status") == "ok" and (r.get("failure_persistence") or {}).get("initial_violations")
               and (b is None or r.get("budget_mode") not in ("nested", "independent") or r.get("call_budget") == b)]
        if not sel:
            continue
        fin = [r["failure_persistence"]["final_fraction"] for r in sel if r["failure_persistence"].get("final_fraction") is not None]
        width = max(len(r["failure_persistence"]["by_round"]) for r in sel)
        per_round = []
        for k in range(width):
            vals = [r["failure_persistence"]["by_round"][k] for r in sel if len(r["failure_persistence"]["by_round"]) > k
                    and r["failure_persistence"]["by_round"][k] is not None]
            per_round.append(float(np.mean(vals)) if vals else None)
        by_type: dict[str, dict[str, int]] = {}
        for r in sel:
            for t, c in r["failure_persistence"]["by_type"].items():
                d = by_type.setdefault(t, {"initial": 0, "persisted": 0})
                d["initial"] += c["initial"]; d["persisted"] += c["persisted"]
        rows.append({"method": m, "label": METHOD_LABELS[m], "call_budget": b, "n_runs_with_initial_violations": len(sel),
                     "mean_final_fraction_persisting": float(np.mean(fin)) if fin else None, "mean_fraction_by_round": per_round,
                     "by_type": by_type})
    return rows


def build_summary(runs: list[dict], experiment: Optional[dict] = None, call_budget: Optional[int] = None) -> dict[str, Any]:
    """Everything the UI and exports show. Budget-dependent tables use `call_budget` (default: the largest budget compared);
    `budget_curve` and the equal-budget `comparisons` cover every budget."""
    df = runs_to_frame(runs)
    if df.empty:
        return {"has_results": False, "message": NO_CONCLUSION, "n_runs": 0, "methods": [], "comparisons": [], "violation_taxonomy": [],
                "by_size": [], "by_scarcity": [], "by_scenario": [], "is_simulated": False,
                "call_budgets": [], "call_budget_selected": None, "budget_curve": [], "budget_curve_by_level": [], "by_difficulty_level": [],
                "failure_persistence": []}
    b = headline_budget(df, call_budget)
    hv = budget_view(df, b)          # one budget for the headline tables, so budgets are never mixed in one row
    methods = method_summary(hv)
    return {
        "call_budgets": budgets_in(df), "call_budget_selected": b,
        "budget_curve": budget_curve(df), "budget_curve_by_level": budget_curve(df, "difficulty_level") if df["difficulty_level"].notna().any() else [],
        "by_difficulty_level": by_difficulty_level(df, b), "failure_persistence": failure_persistence(runs, df, b),
        "has_results": bool((df["status"] == "ok").any()),
        "message": None if (df["status"] == "ok").any() else NO_CONCLUSION,
        "n_runs": int(len(df)), "n_ok": int((df["status"] == "ok").sum()), "n_errors": int((df["status"] != "ok").sum()),
        "n_scenarios": int(df["scenario_id"].nunique()),
        "is_simulated": bool(df.loc[df["method"] != "ortools", "is_simulated"].astype(bool).any()),
        "providers": sorted({str(p) for p in df["provider"].dropna().unique()}),
        "models": sorted({str(p) for p in df["model"].dropna().unique()}),
        "methods": methods, "by_size": breakdown(hv, "scenario_size"), "by_scarcity": breakdown(hv, "scarcity_level"),
        "violation_taxonomy": violation_taxonomy(hv), "comparisons": comparisons(df), "by_scenario": by_scenario(hv),
    }
