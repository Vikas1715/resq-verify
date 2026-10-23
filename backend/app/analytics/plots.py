"""Publication-style matplotlib figures (PNG). Used for exports; the UI uses Plotly with the same palette."""
from __future__ import annotations

import io
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backend.app.analytics.dataframe import budget_view, budgets_in, headline_budget, ok_runs  # noqa: E402
from backend.app.schemas.validation import VIOLATION_TYPES  # noqa: E402
from backend.app.services.methods import METHODS, METHOD_LABELS  # noqa: E402

PALETTE = {"llm_only": "#9aa5b1", "self_critique": "#e0a030", "resq_verify": "#1f6feb", "resample_verify": "#a371f7", "ortools": "#2da44e"}
WATERMARK = "SIMULATED LLM (mock) — NOT EXPERIMENTAL RESULT"


def _fig_bytes(fig, simulated: bool) -> bytes:
    if simulated:
        fig.text(0.5, 0.5, WATERMARK, ha="center", va="center", rotation=20, fontsize=13, color="red", alpha=0.18)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    return buf.getvalue()


def _methods(df): return [m for m in METHODS if m in set(df["method"])]


def feasibility_bar(df: pd.DataFrame, sim: bool) -> bytes:
    from backend.app.analytics.stats import proportion
    ok, ms = ok_runs(df), _methods(df)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, m in enumerate(ms):
        p = proportion(ok[ok["method"] == m]["final_feasibility"].tolist())
        if p["rate"] is None: continue
        ax.bar(i, p["rate"], color=PALETTE[m], yerr=[[p["rate"] - p["ci_low"]], [p["ci_high"] - p["rate"]]], capsize=4)
        ax.text(i, min(p["ci_high"] + 0.02, 1.02), f"{p['rate']:.0%}\n(n={p['n']})", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(ms)), [METHOD_LABELS[m] for m in ms], rotation=15, ha="right", fontsize=8)
    ax.set_ylim(0, 1.2); ax.set_ylabel("Final feasibility rate (95% Wilson CI)"); ax.set_title("Feasibility by method")
    return _fig_bytes(fig, sim)


def quality_box(df: pd.DataFrame, sim: bool) -> bytes:
    ok, ms = ok_runs(df), _methods(df)
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [pd.to_numeric(ok[ok["method"] == m]["effective_quality"].astype(float), errors="coerce").dropna() for m in ms]
    bp = ax.boxplot(data, patch_artist=True, tick_labels=[METHOD_LABELS[m] for m in ms]) if data else None
    if bp:
        for patch, m in zip(bp["boxes"], ms): patch.set_facecolor(PALETTE[m]); patch.set_alpha(0.7)
    ax.tick_params(axis="x", labelrotation=15, labelsize=8)
    ax.set_ylabel("Effective quality (0 if infeasible)"); ax.set_title("Plan quality by method")
    return _fig_bytes(fig, sim)


def violation_bar(df: pd.DataFrame, sim: bool) -> bytes:
    ok = ok_runs(df)
    ms = [m for m in _methods(df) if m != "ortools"]
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.8 / max(len(ms), 1)
    for j, m in enumerate(ms):
        sub = ok[ok["method"] == m]
        vals = [pd.to_numeric(sub[f"viol_initial_{t}"], errors="coerce").fillna(0).mean() for t in VIOLATION_TYPES]
        ax.bar(np.arange(len(VIOLATION_TYPES)) + j * w, vals, w, label=METHOD_LABELS[m], color=PALETTE[m])
    ax.set_xticks(np.arange(len(VIOLATION_TYPES)) + w * (len(ms) - 1) / 2, VIOLATION_TYPES, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean violations per initial plan"); ax.set_title("Violation taxonomy (initial plans)"); ax.legend(fontsize=8)
    return _fig_bytes(fig, sim)


def feasibility_by_size(df: pd.DataFrame, sim: bool) -> bytes:
    ok, ms = ok_runs(df), _methods(df)
    order = [s for s in ("small", "medium", "large") if s in set(ok["scenario_size"])]
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in ms:
        ys = []
        for s in order:
            v = ok[(ok["method"] == m) & (ok["scenario_size"] == s)]["final_feasibility"].tolist()
            ys.append(np.mean([bool(x) for x in v]) if v else np.nan)
        ax.plot(order, ys, marker="o", label=METHOD_LABELS[m], color=PALETTE[m])
    ax.set_ylim(0, 1.05); ax.set_ylabel("Feasibility rate"); ax.set_xlabel("Scenario size"); ax.set_title("Scalability: feasibility vs size"); ax.legend(fontsize=8)
    return _fig_bytes(fig, sim)


def latency_by_size(df: pd.DataFrame, sim: bool) -> bytes:
    ok, ms = ok_runs(df), _methods(df)
    order = [s for s in ("small", "medium", "large") if s in set(ok["scenario_size"])]
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in ms:
        ys = [pd.to_numeric(ok[(ok["method"] == m) & (ok["scenario_size"] == s)]["latency_s"].astype(float), errors="coerce").mean() for s in order]
        ax.plot(order, ys, marker="o", label=METHOD_LABELS[m], color=PALETTE[m])
    ax.set_ylabel("Mean latency (s)"); ax.set_xlabel("Scenario size"); ax.set_title("Scalability: latency vs size"); ax.legend(fontsize=8)
    return _fig_bytes(fig, sim)


def feasibility_vs_budget(df: pd.DataFrame, sim: bool) -> bytes:
    """Feasibility versus LLM-call budget. M1 (one call) and M4 (no LLM calls) are drawn as flat reference lines."""
    from backend.app.analytics.stats import proportion
    ok, budgets = ok_runs(df), budgets_in(df)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for m in _methods(df):
        ys, lo, hi = [], [], []
        for b in budgets:
            v = budget_view(ok, b)
            p = proportion(v[v["method"] == m]["final_feasibility"].tolist())
            ys.append(p["rate"]); lo.append(p["ci_low"]); hi.append(p["ci_high"])
        if all(y is None for y in ys):
            continue
        style = "--" if m in ("llm_only", "ortools") else "-"
        ax.plot(budgets, ys, style, marker="o", color=PALETTE[m], label=METHOD_LABELS[m] + (" (1 call)" if m == "llm_only" else " (reference)" if m == "ortools" else ""))
        if m not in ("llm_only", "ortools"):
            ax.fill_between(budgets, [x if x is not None else np.nan for x in lo], [x if x is not None else np.nan for x in hi], color=PALETTE[m], alpha=0.12)
    ax.set_xticks(budgets); ax.set_ylim(0, 1.05)
    ax.set_xlabel("LLM-call budget"); ax.set_ylabel("Final feasibility rate (95% Wilson CI band)")
    ax.set_title("Feasibility vs call budget"); ax.legend(fontsize=8)
    return _fig_bytes(fig, sim)


def feasibility_by_level(df: pd.DataFrame, sim: bool) -> bytes:
    from backend.app.analytics.stats import proportion
    b = headline_budget(df)
    ok = budget_view(ok_runs(df), b)
    levels = sorted(l for l in ok["difficulty_level"].dropna().unique())
    ms = _methods(ok)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    w = 0.8 / max(len(ms), 1)
    for j, m in enumerate(ms):
        ys = [proportion(ok[(ok["difficulty_level"] == l) & (ok["method"] == m)]["final_feasibility"].tolist())["rate"] for l in levels]
        ax.bar(np.arange(len(levels)) + j * w, [y if y is not None else 0 for y in ys], w, label=METHOD_LABELS[m], color=PALETTE[m])
    ax.set_xticks(np.arange(len(levels)) + w * (len(ms) - 1) / 2, levels); ax.set_ylim(0, 1.05)
    ax.set_xlabel("Difficulty level"); ax.set_ylabel("Final feasibility rate")
    ax.set_title(f"Feasibility by difficulty level" + (f" (budget {b})" if b else "")); ax.legend(fontsize=7)
    return _fig_bytes(fig, sim)


def repair_trajectory(runs: list[dict], sim: bool, budget: int | None = None) -> bytes:
    """Mean violations after each round for ResQ-Verify runs whose initial plan was infeasible."""
    series = []
    for r in runs:
        if budget is not None and r.get("budget_mode") in ("nested", "independent") and r.get("call_budget") != budget:
            continue
        if r.get("method") == "resq_verify" and r.get("status") == "ok" and r.get("initial_feasible") is False:
            series.append([x["n_violations"] for x in r["rounds"]])
    fig, ax = plt.subplots(figsize=(6, 4))
    if series:
        L = max(len(s) for s in series)
        mat = np.array([s + [s[-1]] * (L - len(s)) for s in series], dtype=float)
        ax.plot(range(L), mat.mean(axis=0), marker="o", color=PALETTE["resq_verify"], label=f"mean (n={len(series)})")
        for s in series[:30]: ax.plot(range(len(s)), s, color=PALETTE["resq_verify"], alpha=0.12)
        ax.set_xticks(range(L)); ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No repaired runs yet", ha="center", transform=ax.transAxes)
    ax.set_xlabel("Round (0 = initial plan)"); ax.set_ylabel("Violations"); ax.set_title("Repair trajectory")
    return _fig_bytes(fig, sim)


def all_plots(df: pd.DataFrame, runs: list[dict], sim: bool) -> dict[str, bytes]:
    if df.empty or not (df["status"] == "ok").any():
        return {}
    b = headline_budget(df)
    hv = budget_view(df, b)   # tables that depend on the budget use one budget, so budgets are never mixed in one bar
    figs: dict[str, Callable[[], bytes]] = {
        "feasibility_by_method.png": lambda: feasibility_bar(hv, sim), "quality_by_method.png": lambda: quality_box(hv, sim),
        "violation_taxonomy.png": lambda: violation_bar(hv, sim), "feasibility_by_size.png": lambda: feasibility_by_size(hv, sim),
        "latency_by_size.png": lambda: latency_by_size(hv, sim), "repair_trajectory.png": lambda: repair_trajectory(runs, sim, b)}
    if budgets_in(df):
        figs["feasibility_vs_budget.png"] = lambda: feasibility_vs_budget(df, sim)
    if df["difficulty_level"].notna().any():
        figs["feasibility_by_difficulty_level.png"] = lambda: feasibility_by_level(df, sim)
    out = {}
    for name, fn in figs.items():
        try: out[name] = fn()
        except Exception as e:  # a single bad figure must not break an export
            plt.close("all"); out[name + ".error.txt"] = str(e).encode()
    return out
