from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from frontend import ui


def scenario_picker(api, key: str):
    """Select-box of stored scenarios; returns scenario_id or None (with a hint when there are none)."""
    scs = ui.safe(api.scenarios, [])
    if not scs:
        st.info("No scenarios yet. Create one in the Scenario Lab.")
        return None
    ids = [s["scenario_id"] for s in scs]
    return st.selectbox("Scenario", ids, key=key)


def plan_tables(plan: dict | None) -> None:
    if not plan:
        st.caption("No parsed plan (the output could not be parsed).")
        return
    for title, k in (("Team assignments", "team_assignments"), ("Vehicle assignments", "vehicle_assignments"),
                     ("Shelter allocations", "shelter_allocations"), ("Unserved demand", "unserved_demand")):
        rows = plan.get(k) or []
        st.markdown(f"**{title}** ({len(rows)})")
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    if plan.get("reasoning_summary"):
        st.caption("Model's reasoning summary: " + plan["reasoning_summary"])


def render_run(run: dict, *, show_raw: bool = False) -> None:
    """Full view of one run record: outcome, per-round violations, plan, validator report."""
    if run.get("is_simulated"):
        ui.sim_banner()
    label = ui.METHOD_LABELS.get(run["method"], run["method"])
    if run["status"] != "ok":
        st.error(f"{label}: infrastructure/API error — excluded from rates. {run.get('failure_reason') or ''}")
        for e in run.get("api_errors", [])[:5]:
            st.caption(e)
        return
    feas = run.get("final_feasibility")
    st.markdown(f"#### {label} " + ui.pill("FEASIBLE", "ok") if feas else f"#### {label} " + ui.pill("INFEASIBLE", "bad"), unsafe_allow_html=True)
    c = st.columns(5)
    q = run.get("plan_quality_raw")
    c[0].metric("Final violations", run.get("n_violations_final"))
    c[1].metric("Initial violations", run.get("n_violations_initial") if run.get("n_violations_initial") is not None else "—")
    c[2].metric("Plan quality", ui.fmt_num(q))
    c[3].metric("Gap vs CP-SAT", ui.fmt_pct(run.get("optimality_gap")))
    c[4].metric("Latency", f"{run.get('latency_s', 0):.2f}s")
    tok = (run.get("token_usage") or {}).get("total")
    st.caption(f"Stop reason: {run.get('stop_reason')} · rounds: {len(run.get('rounds', [])) - 1 if run['method'] != 'ortools' else 0}"
               f" · LLM calls: {run.get('calls_used', '—')} of budget {run.get('call_budget') if run.get('call_budget') is not None else '—'}"
               f" · tokens: {tok if tok else '—'} · model: {run.get('model') or '—'} · prompt {run.get('prompt_version') or '—'}")
    if run.get("solver"):
        s = run["solver"]
        st.caption(f"CP-SAT status {s.get('status')} · relative gap {s.get('relative_gap')} · runtime {s.get('runtime_s'):.2f}s")
    if run["method"] in ("resq_verify", "self_critique", "resample_verify") and run.get("rounds"):
        st.markdown("**Rounds**")
        for r in run["rounds"]:
            icon = "✅" if r["is_valid"] else "❌"
            with st.expander(f"{icon} Round {r['round']} · {r['kind']} · {r['n_violations']} violation(s) · {r.get('latency_s', 0):.2f}s"):
                if r.get("parse_error"):
                    st.warning("Parse error: " + r["parse_error"])
                st.markdown(ui.violations_html(r["validation"]), unsafe_allow_html=True)
                if r.get("feedback_sent"):
                    st.markdown("Feedback sent to the LLM for this round:")
                    st.code(r["feedback_sent"], language="json" if run.get("feedback_format") == "structured" else "text")
                if r.get("critique"):
                    st.markdown("Self-critique (no validator involved):")
                    st.json(r["critique"])
                if r.get("edit_distance") is not None and r["round"] > 0:
                    st.caption(f"Records changed vs previous plan: {r['edit_distance']}")
    st.markdown("**Final validator report**")
    st.markdown(ui.violations_html(run.get("validator_report") or {}), unsafe_allow_html=True)
    with st.expander("Final plan"):
        plan_tables(run.get("parsed_plan"))
    if show_raw:
        with st.expander("Raw LLM output"):
            st.code(run.get("raw_llm_output") or "", language="json")
    with st.expander("Full record (JSON)"):
        st.json({k: v for k, v in run.items() if k not in ("rounds", "raw_llm_output")})


def json_download(label: str, obj, filename: str) -> None:
    st.download_button(label, json.dumps(obj, indent=1, default=str), file_name=filename, mime="application/json")
