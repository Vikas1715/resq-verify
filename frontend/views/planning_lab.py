import streamlit as st

from frontend import ui
from frontend.views._common import render_run, scenario_picker

KINDS = {"M1 · LLM-only": "generate", "M2 · Self-critique": "self-critique", "M3 · ResQ-Verify": "verify",
         "M5 · Resample + Verify": "resample-verify"}


def render(api):
    st.title("Planning Lab")
    st.caption("Run the methods on one scenario. Every run is stored (as an interactive run) and never mixed into experiment statistics.")
    sid = scenario_picker(api, "pl_sc")
    if not sid:
        return
    c = st.columns(5)
    k = c[0].number_input("Max repair / critique rounds (K)", 0, 10, 3)
    budget = c[1].number_input("LLM-call budget (0 = use K)", 0, 20, 0,
                               help="If set, M2, M3 and M5 each get at most this many LLM calls (K is ignored). M1 always uses 1 call.")
    fmt = c[2].selectbox("Feedback format", ["structured", "free_text"])
    temp = c[3].number_input("Temperature (0 = env default)", 0.0, 2.0, 0.0, 0.1)
    ref = c[4].checkbox("Also compute CP-SAT reference", True)
    opts = dict(max_repair_rounds=int(k), feedback_format=fmt, temperature=(temp or None), compute_reference=ref,
                call_budget=(int(budget) or None))
    b = st.columns(6)
    clicked = None
    for col, (label, kind) in zip(b, KINDS.items()):
        if col.button(label, use_container_width=True):
            clicked = kind
    if b[4].button("Compare all (M1–M5)", type="primary", use_container_width=True):
        clicked = "compare"
    if b[5].button("M4 · OR-Tools only", use_container_width=True):
        clicked = "solve"
    if clicked:
        with st.spinner("Running... LLM calls can take a while, especially with repair rounds."):
            if clicked == "compare":
                res = ui.safe(lambda: api.compare(sid, max_repair_rounds=int(k), feedback_format=fmt, temperature=(temp or None), call_budget=(int(budget) or None)))
                st.session_state["pl_result"] = res["runs"] if res else None
            elif clicked == "solve":
                res = ui.safe(lambda: api.solve(sid))
                st.session_state["pl_solve"] = res
                st.session_state["pl_result"] = None
            else:
                res = ui.safe(lambda: api.plan(clicked, sid, **opts))
                st.session_state["pl_result"] = [res] if res else None
    runs = st.session_state.get("pl_result")
    if runs:
        if len(runs) > 1:
            rows = [{"Method": ui.METHOD_LABELS[r["method"]], "Feasible": r.get("final_feasibility"), "Violations (initial → final)":
                     f"{r.get('n_violations_initial', '—')} → {r.get('n_violations_final', '—')}", "Quality": ui.fmt_num(r.get("plan_quality_raw")),
                     "LLM calls": r.get("calls_used"), "Latency s": round(r.get("latency_s", 0), 2),
                     "Tokens": (r.get("token_usage") or {}).get("total") or "—"} for r in runs]
            st.dataframe(rows, hide_index=True, use_container_width=True)
            st.caption("One scenario is an illustration, not evidence. Use the Experiment Lab for conclusions.")
        for r in runs:
            with st.container(border=True):
                render_run(r)
    sol = st.session_state.get("pl_solve")
    if sol and not runs:
        st.markdown(f"**CP-SAT** status {sol['status']} · quality {ui.fmt_num(sol.get('quality', {}).get('quality_norm'))} · {sol['runtime_s']:.2f}s")
        st.markdown(ui.violations_html(sol.get("report", {})), unsafe_allow_html=True)
        st.json(sol["plan"], expanded=False)
