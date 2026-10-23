"""Plotly figures (interactive counterparts of backend/app/analytics/plots.py)."""
from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go

from frontend.ui import METHOD_COLORS, METHOD_LABELS

LAYOUT = dict(template="plotly_white", margin=dict(l=40, r=20, t=50, b=40), font=dict(size=13), height=380)


def feasibility_bar(methods: list[dict]) -> go.Figure:
    fig = go.Figure()
    for m in methods:
        f = m["feasibility"]
        if f["rate"] is None:
            continue
        fig.add_bar(x=[m["label"]], y=[f["rate"]], marker_color=METHOD_COLORS[m["method"]], name=m["label"],
                    error_y=dict(type="data", symmetric=False, array=[f["ci_high"] - f["rate"]], arrayminus=[f["rate"] - f["ci_low"]]),
                    text=[f"{f['rate']:.0%} (n={f['n']})"], textposition="outside", hovertemplate="%{x}: %{y:.1%}<extra></extra>")
    fig.update_layout(title="Final feasibility rate (95% Wilson CI)", yaxis=dict(range=[0, 1.15], tickformat=".0%"), showlegend=False, **LAYOUT)
    return fig


def quality_box(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for m, g in df[df["status"] == "ok"].groupby("method"):
        fig.add_box(y=pd.to_numeric(g["effective_quality"], errors="coerce").dropna(), name=METHOD_LABELS[m], marker_color=METHOD_COLORS[m], boxmean=True)
    fig.update_layout(title="Effective plan quality (0 if infeasible)", yaxis_title="quality", showlegend=False, **LAYOUT)
    return fig


def violation_bars(taxonomy: list[dict], which: str = "initial_per_run") -> go.Figure:
    fig = go.Figure()
    df = pd.DataFrame(taxonomy)
    for m, g in df.groupby("method"):
        fig.add_bar(x=g["violation_type"], y=g[which], name=METHOD_LABELS[m], marker_color=METHOD_COLORS[m])
    fig.update_layout(title="Mean violations per run by type " + ("(initial plan)" if which.startswith("initial") else "(final plan)"),
                      barmode="group", yaxis_title="violations / run", **LAYOUT)
    return fig


def by_level(rows: list[dict], level: str, metric: str, title: str, pct: bool = False) -> go.Figure:
    fig = go.Figure()
    df = pd.DataFrame(rows)
    order = {"small": 0, "medium": 1, "large": 2, "low": 0, "high": 2}
    for m, g in df.groupby("method"):
        g = g.sort_values(level, key=lambda s: s.map(order).fillna(1))
        fig.add_scatter(x=g[level], y=g[metric], mode="lines+markers", name=METHOD_LABELS[m], line=dict(color=METHOD_COLORS[m]))
    fig.update_layout(title=title, yaxis=dict(tickformat=".0%") if pct else {}, xaxis_title=level.replace("_", " "), **LAYOUT)
    return fig


def budget_curve(rows: list[dict], metric: str = "feasibility") -> go.Figure:
    """Feasibility (or another metric) versus LLM-call budget. M1 and M4 are flat reference lines."""
    fig = go.Figure()
    df = pd.DataFrame(rows)
    for m, g in df.groupby("method"):
        g = g.sort_values("call_budget")
        ref = m in ("llm_only", "ortools")
        fig.add_scatter(x=g["call_budget"], y=g[metric], mode="lines+markers", name=METHOD_LABELS[m] + (" (reference)" if ref else ""),
                        line=dict(color=METHOD_COLORS[m], dash="dash" if ref else "solid"))
    fig.update_layout(title=f"{metric.replace('_', ' ').capitalize()} vs LLM-call budget",
                      xaxis=dict(title="LLM-call budget", dtick=1), yaxis=dict(tickformat=".0%") if metric in ("feasibility", "repair_success") else {}, **LAYOUT)
    return fig


def repair_trajectory(runs: list[dict]) -> go.Figure:
    fig = go.Figure()
    series = [[r["n_violations"] for r in x["rounds"]] for x in runs if x.get("method") == "resq_verify" and x.get("initial_feasible") is False]
    for s in series[:40]:
        fig.add_scatter(x=list(range(len(s))), y=s, mode="lines", line=dict(color=METHOD_COLORS["resq_verify"], width=1), opacity=0.2, showlegend=False)
    if series:
        L = max(map(len, series))
        mean = [sum((s + [s[-1]] * (L - len(s)))[i] for s in series) / len(series) for i in range(L)]
        fig.add_scatter(x=list(range(L)), y=mean, mode="lines+markers", name=f"mean (n={len(series)})", line=dict(color=METHOD_COLORS["resq_verify"], width=3))
    fig.update_layout(title="Repair trajectory: violations per round", xaxis_title="round (0 = initial plan)", yaxis_title="violations", **LAYOUT)
    return fig


def scenario_map(sc: dict) -> go.Figure:
    """Schematic (not geographic) view: demand per incident and capacities."""
    inc = pd.DataFrame(sc["incidents"])
    roads = {r["id"]: r["blocked"] for r in sc["roads"]}
    inc["road"] = inc["road_id"].map(lambda r: "blocked" if roads.get(r) else "open")
    fig = go.Figure(go.Bar(x=inc["id"], y=inc["people"], marker_color=["#c62828" if b == "blocked" else "#1f6feb" for b in inc["road"]],
                           text=[f"P{p}" for p in inc["priority"]], textposition="outside",
                           customdata=inc[["name", "hazard", "road", "waterway", "narrow_access"]],
                           hovertemplate="<b>%{x}</b> %{customdata[0]}<br>%{customdata[1]} · road %{customdata[2]}<br>waterway=%{customdata[3]} narrow=%{customdata[4]}<br>%{y} people<extra></extra>"))
    fig.update_layout(title="People per incident (label = priority; red = blocked road)", yaxis_title="people", showlegend=False, **LAYOUT)
    return fig


def capacity_bars(sc: dict) -> go.Figure:
    veh = pd.DataFrame(sc["vehicles"])
    veh["trip_capacity"] = veh["capacity"] * veh["max_trips"]
    fig = go.Figure()
    for t, g in veh.groupby("type"):
        fig.add_bar(x=g["id"], y=g["trip_capacity"], name=t)
    fig.update_layout(title="Vehicle capacity over all trips (people)", barmode="group", **LAYOUT)
    return fig
