import pandas as pd
import streamlit as st

from frontend import charts, ui


def render(api):
    st.title("Results & Analytics")
    exps = ui.safe(lambda: api.experiments(include_adhoc=False), [])
    opts = ["All experiments"] + [e["experiment_id"] for e in exps]
    pick = st.selectbox("Scope", opts)
    adhoc = st.checkbox("Include interactive Planning Lab runs", False)
    eid = None if pick == "All experiments" else pick
    s = ui.safe(lambda: api.summary(eid, adhoc))
    if not s or not s.get("has_results"):
        ui.no_results()
        return
    if s.get("call_budgets"):
        pick_b = st.select_slider("LLM-call budget for the tables below", options=s["call_budgets"], value=s["call_budget_selected"],
                                  help="Budget-dependent tables compare methods at this budget. M1 always uses 1 call; M4 uses none.")
        if pick_b != s["call_budget_selected"]:
            s = ui.safe(lambda: api.summary(eid, adhoc, call_budget=int(pick_b))) or s
    if s["is_simulated"]:
        ui.sim_banner()
    c = st.columns(4)
    with c[0]: ui.kpi("Runs (ok / errors)", f"{s['n_ok']} / {s['n_errors']}")
    with c[1]: ui.kpi("Scenarios", s["n_scenarios"])
    with c[2]: ui.kpi("Models", ", ".join(s["models"]) or "—")
    with c[3]: ui.kpi("Providers", ", ".join(s["providers"]) or "—")
    st.caption("Errored runs (API/infrastructure failures) are excluded from every rate. Unparseable LLM output counts as infeasible.")
    rows = ui.safe(lambda: api.analytics_runs(eid, adhoc), [])
    df = pd.DataFrame(rows)
    st.subheader("Headline metrics")
    table = [{"Method": m["label"], "Runs": m["n_ok"], "Feasibility": ui.fmt_pct(m["feasibility"]["rate"]),
              "95% CI": f"[{ui.fmt_pct(m['feasibility']['ci_low'])}, {ui.fmt_pct(m['feasibility']['ci_high'])}]" if m["feasibility"]["rate"] is not None else "—",
              "Initial feasibility": ui.fmt_pct(m["initial_feasibility"]["rate"]), "Repair success": ui.fmt_pct(m["repair_success"]["rate"]),
              "LLM calls/run": ui.fmt_num((m.get("calls_used") or {}).get("mean"), 2), "Eff. quality (mean)": ui.fmt_num(m["effective_quality"]["mean"]), "Quality | feasible": ui.fmt_num(m["quality_feasible_only"]["mean"]),
              "Gap vs CP-SAT": ui.fmt_num((m["optimality_gap"] or {}).get("mean")), "Latency s": ui.fmt_num(m["latency_s"]["mean"], 2),
              "Tokens/run": ui.fmt_num((m["tokens_total"] or {}).get("mean"), 0)} for m in s["methods"]]
    st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)
    t = st.tabs(["Feasibility & quality", "Violations", "Scalability", "Repair", "Paired statistics", "By scenario", "Call budget & difficulty"])
    with t[0]:
        a, b = st.columns(2)
        a.plotly_chart(charts.feasibility_bar(s["methods"]), use_container_width=True)
        if not df.empty:
            b.plotly_chart(charts.quality_box(df), use_container_width=True)
        st.plotly_chart(charts.by_level(s["by_scarcity"], "scarcity_level", "feasibility", "Feasibility by scarcity", True), use_container_width=True)
    with t[1]:
        if s["violation_taxonomy"]:
            a, b = st.columns(2)
            a.plotly_chart(charts.violation_bars(s["violation_taxonomy"], "initial_per_run"), use_container_width=True)
            b.plotly_chart(charts.violation_bars(s["violation_taxonomy"], "final_per_run"), use_container_width=True)
    with t[2]:
        a, b = st.columns(2)
        a.plotly_chart(charts.by_level(s["by_size"], "scenario_size", "feasibility", "Feasibility vs scenario size", True), use_container_width=True)
        b.plotly_chart(charts.by_level(s["by_size"], "scenario_size", "latency_s", "Latency (s) vs scenario size"), use_container_width=True)
    with t[3]:
        if eid:
            full = ui.safe(lambda: api.runs(eid, full=True, method="resq_verify"), [])
            st.plotly_chart(charts.repair_trajectory(full), use_container_width=True)
        else:
            st.info("Select a single experiment to see repair trajectories.")
    with t[4]:
        comp = s["comparisons"]
        if not comp:
            st.info("Paired comparisons need ResQ-Verify and at least one other method.")
        else:
            st.caption("Scenario-level pairing (repetitions averaged). Binary outcomes: exact McNemar. Continuous: Wilcoxon signed-rank, or paired t only if n ≥ 20 and differences pass Shapiro–Wilk. Holm-adjusted within each method pair and budget. Each budget is compared on its own (equal LLM calls).")
            st.dataframe(pd.DataFrame([{ "Budget": c.get("call_budget"), "A": c["method_a"], "B": c["method_b"], "Metric": c["label"], "n": c["n_pairs"], "mean A": ui.fmt_num(c["mean_a"]),
                                         "mean B": ui.fmt_num(c["mean_b"]), "diff": ui.fmt_num(c["mean_diff"]), "Test": c["test"], "p": ui.fmt_num(c["p_value"], 4),
                                         "p Holm": ui.fmt_num(c.get("p_holm"), 4), "Effect": f"{ui.fmt_num(c['effect_size'])} ({c.get('effect_name') or '—'})",
                                         "Note": c.get("note", "")} for c in comp]), hide_index=True, use_container_width=True)
    with t[5]:
        st.dataframe(pd.DataFrame(s["by_scenario"]), hide_index=True, use_container_width=True)
    with t[6]:
        if s.get("budget_curve"):
            a, b = st.columns(2)
            a.plotly_chart(charts.budget_curve(s["budget_curve"], "feasibility"), use_container_width=True)
            b.plotly_chart(charts.budget_curve(s["budget_curve"], "effective_quality"), use_container_width=True)
            st.caption("M1 (one call) and M4 (no LLM calls) are references, drawn flat. Budgets are nested when the experiment used nested budgets, so "
                       "the points are not independent samples.")
            st.dataframe(pd.DataFrame(s["budget_curve"]).drop(columns=["broadcast", "reference"]), hide_index=True, use_container_width=True)
        else:
            st.info("No budget comparison in this scope. Run an experiment with LLM-call budgets (Experiment Lab).")
        if s.get("by_difficulty_level"):
            st.markdown(f"**By difficulty level** (budget {s['call_budget_selected']})")
            st.dataframe(pd.DataFrame(s["by_difficulty_level"]), hide_index=True, use_container_width=True)
        if s.get("failure_persistence"):
            st.markdown("**Failure persistence** — share of the initial plan's violations still present later")
            st.dataframe(pd.DataFrame([{k: v for k, v in r.items() if k != "by_type"} for r in s["failure_persistence"]]), hide_index=True,
                         use_container_width=True)
