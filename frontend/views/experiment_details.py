import pandas as pd
import streamlit as st

from frontend import ui
from frontend.views._common import render_run


def render(api):
    st.title("Experiment Details")
    exps = ui.safe(api.experiments, [])
    if not exps:
        st.info("No experiments or runs yet.")
        return
    eid = st.selectbox("Experiment", [e["experiment_id"] for e in exps],
                       format_func=lambda i: next((f"{e['name']} · {i}" + (" (interactive)" if e.get("kind") == "adhoc" else "") for e in exps if e["experiment_id"] == i), i))
    e = ui.safe(lambda: api.experiment(eid))
    if not e:
        return
    if e.get("is_simulated"):
        ui.sim_banner()
    c = st.columns(4)
    with c[0]: ui.kpi("Status", e["status"])
    with c[1]: ui.kpi("Progress", f"{e['progress']['done']}/{e['progress']['total']}")
    with c[2]: ui.kpi("Model", (e.get("llm") or {}).get("model") or "—")
    with c[3]: ui.kpi("Prompt", f"{e.get('prompt_version') or '—'} · {e.get('prompt_hash') or '—'}")
    if e.get("message"):
        st.info(e["message"])
    with st.expander("Configuration, LLM settings and environment"):
        st.json({"config": e["config"], "llm": e.get("llm"), "storage": e.get("storage_backend"), "settings": e.get("settings"), "references": e.get("references")}, expanded=False)
    f = st.columns(3)
    method = f[0].selectbox("Method", ["all", "llm_only", "self_critique", "resq_verify", "resample_verify", "ortools"])
    status = f[1].selectbox("Status", ["all", "ok", "error"])
    only_bad = f[2].checkbox("Only infeasible", False)
    params = {k: v for k, v in (("method", method), ("status", status)) if v != "all"}
    runs = ui.safe(lambda: api.runs(eid, **params), [])
    if only_bad:
        runs = [r for r in runs if r.get("final_feasibility") is False]
    if not runs:
        st.info("No runs match.")
        return
    cols = ["run_id", "scenario_id", "method", "run_index", "status", "initial_feasible", "final_feasibility", "repair_rounds_used", "n_violations_final", "plan_quality_raw", "optimality_gap", "latency_s"]
    st.dataframe(pd.DataFrame(runs)[[c for c in cols if c in runs[0]]], hide_index=True, use_container_width=True)
    rid = st.selectbox("Open a run", [r["run_id"] for r in runs], format_func=lambda i: next(f"{r['scenario_id']} · {r['method']} · #{r['run_index']} ({i[-6:]})" for r in runs if r["run_id"] == i))
    run = ui.safe(lambda: api.run(rid))
    if run:
        render_run(run, show_raw=True)
