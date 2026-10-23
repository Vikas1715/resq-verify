"""Markdown research summary and the export bundle. Contains only numbers computed from stored runs."""
from __future__ import annotations

import csv
import io
import json
import zipfile
from typing import Any, Optional

from backend.app.analytics.dataframe import runs_to_frame
from backend.app.analytics.plots import all_plots
from backend.app.analytics.summary import NO_CONCLUSION, build_summary
from backend.app.utils.ids import utcnow_iso

SIM_BANNER = ("> **SIMULATED LLM — NOT EXPERIMENTAL RESULT.** These runs used the built-in mock LLM (a test double). "
              "The numbers below exercise the pipeline only and must not be reported as findings.\n")


def _f(x: Optional[float], d: int = 3) -> str:
    return "—" if x is None else f"{x:.{d}f}"


def _pct(p: dict) -> str:
    if not p or p.get("rate") is None: return "—"
    return f"{p['rate']*100:.1f}% [{p['ci_low']*100:.1f}, {p['ci_high']*100:.1f}] (n={p['n']})"


def _ms(d: Optional[dict], digits: int = 3) -> str:
    if not d or d.get("mean") is None: return "—"
    sd = f" ± {d['sd']:.{digits}f}" if d.get("sd") is not None else ""
    return f"{d['mean']:.{digits}f}{sd} (med {d['median']:.{digits}f}, n={d['n']})"


def research_summary_md(runs: list[dict], experiment: Optional[dict] = None, experiments: Optional[list[dict]] = None,
                        call_budget: Optional[int] = None) -> str:
    s = build_summary(runs, experiment, call_budget)
    lines = ["# ResQ-Verify — Research Summary", "", f"_Generated: {utcnow_iso()}_", ""]
    if not s["has_results"]:
        return "\n".join(lines + [f"**{NO_CONCLUSION}**", "", "No completed runs were found. Run an experiment first (Experiment Lab or `experiments/scripts/run_experiment.py`).", ""])
    if s["is_simulated"]:
        lines += [SIM_BANNER]
    exps = experiments or ([experiment] if experiment else [])
    lines += ["## Setup", ""]
    lines += [f"- Runs: {s['n_runs']} ({s['n_ok']} ok, {s['n_errors']} errored — errored runs are excluded from all rates)",
              f"- Scenarios: {s['n_scenarios']}", f"- Providers: {', '.join(s['providers']) or '—'}; models: {', '.join(s['models']) or '—'}"]
    for e in exps:
        c = e.get("config") or {}
        lines.append(f"- Experiment `{e.get('experiment_id')}` ({e.get('status')}): sizes={c.get('sizes')}, scarcity={c.get('scarcity')}, "
                     f"levels={c.get('difficulty_levels') or '—'}, scenarios/cell={c.get('num_scenarios')}, repetitions={c.get('repetitions')}, "
                     f"call budgets={c.get('call_budgets') or 'K-based (K=' + str(c.get('max_repair_rounds')) + ')'}, "
                     f"temperature={c.get('temperature') if c.get('temperature') is not None else 'env default'}, prompt={e.get('prompt_version')}")
    if s.get("call_budget_selected") is not None:
        lines += ["", f"_Tables below that depend on the call budget use budget = {s['call_budget_selected']} LLM calls "
                      f"(budgets compared: {s['call_budgets']}). M1 always uses 1 call and M4 uses none; both are shown as references._"]
    lines += ["", "## Headline metrics by method", "",
              "| Method | Feasibility (95% Wilson CI) | Initial feasibility | Repair success | LLM calls used | Effective quality | Quality (feasible only) | Opt. gap vs CP-SAT | Latency s |",
              "|---|---|---|---|---|---|---|---|---|"]
    for m in s["methods"]:
        lines.append(f"| {m['label']} | {_pct(m['feasibility'])} | {_pct(m['initial_feasibility'])} | {_pct(m['repair_success'])} | "
                     f"{_ms(m.get('calls_used'), 2)} | {_ms(m['effective_quality'])} | {_ms(m['quality_feasible_only'])} | {_ms(m['optimality_gap'])} | {_ms(m['latency_s'], 2)} |")
    lines += ["", "Effective quality counts an infeasible plan as 0. Optimality gap = (CP-SAT objective − plan objective) / CP-SAT objective; "
              "the CP-SAT reference is time-limited, so a negative gap is possible and a positive gap is relative to the solver's best found solution, not a proven optimum unless its status is OPTIMAL.", ""]
    if s.get("budget_curve"):
        lines += ["## Feasibility vs LLM-call budget", "",
                  "| Budget | Method | n | Feasibility [95% CI] | Repair success | Eff. quality | Calls used (mean) | Latency s | Note |", "|---|---|---|---|---|---|---|---|---|"]
        for r in s["budget_curve"]:
            note = "M1 uses 1 call (repeated at each budget)" if r["broadcast"] else ("reference, no LLM calls" if r["reference"] else "")
            ci = f" [{_f(r['ci_low'], 2)}, {_f(r['ci_high'], 2)}]"
            lines.append(f"| {r['call_budget']} | {r['label']} | {r['n']} | {_f(r['feasibility'], 2)}{ci} | {_f(r['repair_success'], 2)} (n={r['n_repair_cases']}) | "
                         f"{_f(r['effective_quality'])} | {_f(r['calls_used'], 2)} | {_f(r['latency_s'], 2)} | {note} |")
        lines += ["", "_Budgets are nested when `nested_budgets` is true: the result for a smaller budget is the first part of the same run, "
                  "so budgets are not independent samples._", ""]
    if s.get("by_difficulty_level"):
        lines += [f"## By difficulty level (budget = {s['call_budget_selected']})", "",
                  "| Level | Method | n | Feasibility [95% CI] | Eff. quality | Mean violations | Calls used | Latency s |", "|---|---|---|---|---|---|---|---|"]
        for r in s["by_difficulty_level"]:
            lines.append(f"| {r['difficulty_level']} | {r['label']} | {r['n']} | {_f(r['feasibility'], 2)} [{_f(r['ci_low'], 2)}, {_f(r['ci_high'], 2)}] | "
                         f"{_f(r['effective_quality'])} | {_f(r['mean_violations'], 2)} | {_f(r['calls_used'], 2)} | {_f(r['latency_s'], 2)} |")
        lines.append("")
    if s.get("failure_persistence"):
        lines += ["## Failure persistence (initial plan infeasible)", "",
                  "Share of the initial plan's violations (same type and entity) still present later. M5 draws independent plans, so its value "
                  "shows how often a fresh draw repeats a mistake; M3 shows how often a mistake survives feedback that named it.", "",
                  "| Method | Budget | Runs | Final plan: share persisting | Share persisting by round |", "|---|---|---|---|---|"]
        for r in s["failure_persistence"]:
            lines.append(f"| {r['label']} | {r['call_budget']} | {r['n_runs_with_initial_violations']} | {_f(r['mean_final_fraction_persisting'], 3)} | "
                         f"{', '.join(_f(x, 2) for x in r['mean_fraction_by_round']) or '—'} |")
        lines.append("")
    lines += ["## Paired comparisons (scenario-level, equal call budget, Holm-adjusted within each method pair and budget)", "",
              "| Budget | A vs B | Metric | n pairs | mean A | mean B | mean diff [95% boot CI] | Test | p | p (Holm) | Effect |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in s["comparisons"]:
        lines.append(f"| {c.get('call_budget') if c.get('call_budget') is not None else '—'} | {c['method_a']} vs {c['method_b']} | {c['label']} | {c['n_pairs']} | {_f(c['mean_a'])} | {_f(c['mean_b'])} | "
                     f"{_f(c['mean_diff'])} [{_f(c['ci_low'])}, {_f(c['ci_high'])}] | {c['test']} | {_f(c['p_value'], 4)} | {_f(c.get('p_holm'), 4)} | "
                     f"{_f(c['effect_size'])} ({c.get('effect_name') or '—'}) |")
    if any(c.get("note") for c in s["comparisons"]):
        lines += ["", "_Notes:_ " + "; ".join(sorted({c['note'] for c in s['comparisons'] if c.get('note')}))]
    lines += ["", "## Violation taxonomy (total violations over all runs)", "", "| Method | Type | Initial | Final |", "|---|---|---|---|"]
    for v in s["violation_taxonomy"]:
        if v["initial_total"] or v["final_total"]:
            lines.append(f"| {v['method']} | {v['violation_type']} | {v['initial_total']} | {v['final_total']} |")
    lines += ["", "## Breakdown by scenario size", "", "| Size | Method | n | Feasibility | Eff. quality | Latency s |", "|---|---|---|---|---|---|"]
    for r in s["by_size"]:
        lines.append(f"| {r['scenario_size']} | {r['method']} | {r['n']} | {_f(r['feasibility'], 2)} | {_f(r['effective_quality'])} | {_f(r['latency_s'], 2)} |")
    lines += ["", "## Breakdown by scarcity", "", "| Scarcity | Method | n | Feasibility | Eff. quality | Latency s |", "|---|---|---|---|---|---|"]
    for r in s["by_scarcity"]:
        lines.append(f"| {r['scarcity_level']} | {r['method']} | {r['n']} | {_f(r['feasibility'], 2)} | {_f(r['effective_quality'])} | {_f(r['latency_s'], 2)} |")
    cost = [m for m in s["methods"] if m.get("cost_usd_total") is not None]
    lines += ["", "## Tokens and cost", ""]
    for m in s["methods"]:
        if m["method"] != "ortools":
            lines.append(f"- {m['label']}: tokens per run {_ms(m['tokens_total'], 0)}")
    lines.append("- Cost: " + ("; ".join(f"{m['label']} ${m['cost_usd_total']:.4f}" for m in cost) if cost else
                               "not computed (no token prices configured; set PRICE_INPUT_PER_M_TOKENS / PRICE_OUTPUT_PER_M_TOKENS)."))
    lines += ["", "## Interpretation guardrails", "",
              "- This file reports measurements only; it makes no claim beyond the stored runs.",
              "- Scenarios are synthetic and seeded; results do not transfer to real incidents without validation on real data.",
              "- The verifier-feedback idea is not claimed as novel; the contribution is the controlled paired comparison under hard constraints.", ""]
    return "\n".join(lines)


def runs_csv(runs: list[dict]) -> str:
    df = runs_to_frame(runs)
    return df.to_csv(index=False)


def _rows_csv(rows: list[dict]) -> str:
    if not rows:
        return ""
    cols = sorted({k for r in rows for k in r})
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols)
    w.writeheader()
    w.writerows({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()} for r in rows)
    return buf.getvalue()


def comparisons_csv(runs: list[dict]) -> str:
    return _rows_csv(build_summary(runs)["comparisons"])


def attempts_csv(runs: list[dict]) -> str:
    """One row per LLM attempt/round, so every run can be reconstructed. Only real trajectories are listed: records that were
    cut from a larger-budget run (derived_from_budget set) share their rounds with that run and are skipped to avoid duplicates."""
    rows = []
    for r in runs:
        if r.get("derived_from_budget") is not None or r.get("method") == "ortools":
            continue
        cum = 0
        for x in r.get("rounds") or []:
            cum += int(x.get("calls") or 0)
            u = x.get("token_usage") or {}
            rows.append({"experiment_id": r.get("experiment_id"), "run_id": r.get("run_id"), "scenario_id": r.get("scenario_id"),
                         "difficulty_level": r.get("difficulty_level"), "method": r.get("method"), "run_index": r.get("run_index"),
                         "call_budget": r.get("call_budget"), "round": x.get("round"), "kind": x.get("kind"), "calls_this_round": x.get("calls"),
                         "calls_cumulative": cum, "is_valid": x.get("is_valid"), "n_violations": x.get("n_violations"),
                         "violation_types": ";".join(t for t, n in (x.get("violation_counts") or {}).items() if n),
                         "parse_error": bool(x.get("parse_error")), "edit_distance": x.get("edit_distance"),
                         "latency_s": x.get("latency_s"), "tokens_total": u.get("total"), "tokens_estimated": u.get("estimated"),
                         "model_version": x.get("model_version")})
    return _rows_csv(rows)


def build_bundle(runs: list[dict], experiments: list[dict], scenarios: list[dict]) -> bytes:
    """Zip with CSVs, JSON, markdown summary, plots, config and scenarios. Always consistent with the stored runs."""
    df = runs_to_frame(runs)
    sim = bool(not df.empty and df.loc[df["method"] != "ortools", "is_simulated"].astype(bool).any())
    summary = build_summary(runs)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("research_summary.md", research_summary_md(runs, experiments=experiments))
        z.writestr("runs.csv", runs_csv(runs))
        z.writestr("runs_full.json", json.dumps(runs, indent=1, default=str))
        z.writestr("summary.json", json.dumps(summary, indent=1, default=str))
        z.writestr("paired_comparisons.csv", comparisons_csv(runs))
        z.writestr("attempts.csv", attempts_csv(runs))
        z.writestr("budget_curve.csv", _rows_csv(summary.get("budget_curve") or []))
        z.writestr("by_difficulty_level.csv", _rows_csv(summary.get("by_difficulty_level") or []))
        z.writestr("failure_persistence.csv", _rows_csv(summary.get("failure_persistence") or []))
        z.writestr("experiments.json", json.dumps(experiments, indent=1, default=str))
        z.writestr("scenarios.json", json.dumps(scenarios, indent=1, default=str))
        for name, data in all_plots(df, runs, sim).items():
            z.writestr(f"plots/{name}", data)
        if sim:
            z.writestr("README_SIMULATED.txt", "These results come from the mock LLM (test double). SIMULATED — NOT EXPERIMENTAL RESULT.\n")
    return buf.getvalue()
