import json

import streamlit as st

from frontend import ui
from frontend.views._common import render_run, scenario_picker

RULES = [("V1", "Vehicle capacity", "people ≤ trips × capacity; total trips ≤ vehicle's max trips"),
         ("V2", "Team availability", "only available teams; ≤ max incidents per team"),
         ("V3", "Shelter capacity", "people placed in a shelter ≤ its capacity"),
         ("V4", "Road accessibility", "only boats may go to a blocked road"),
         ("V5", "Coverage & bookkeeping", "transported = sheltered ≤ people; unserved reported exactly; transported sites need a team"),
         ("V6", "Compatibility", "vehicle type allowed at the incident"),
         ("V7", "Schema integrity", "known ids, no duplicates, no negatives, parseable output")]


def render(api):
    st.title("Verification Lab")
    st.caption("The validator is deterministic code, not an LLM. Paste any plan (yours, an LLM's, or an edited one) and see which hard constraints it breaks.")
    with st.expander("The seven hard-constraint rules"):
        st.table({"Rule": [r[0] for r in RULES], "Name": [r[1] for r in RULES], "Checks": [r[2] for r in RULES]})
    sid = scenario_picker(api, "vl_sc")
    if not sid:
        return
    c = st.columns(3)
    if c[0].button("Load CP-SAT plan"):
        sol = ui.safe(lambda: api.solve(sid))
        if sol and sol.get("plan"):
            st.session_state["vl_text"] = json.dumps(sol["plan"], indent=1)
    if c[1].button("Break the loaded plan (demo)"):
        try:
            p = json.loads(st.session_state.get("vl_text", ""))
            if p.get("vehicle_assignments"):
                p["vehicle_assignments"][0]["people"] = p["vehicle_assignments"][0]["people"] + 500
                st.session_state["vl_text"] = json.dumps(p, indent=1)
            else:
                st.info("The loaded plan has no vehicle assignments to break.")
        except ValueError:
            st.info("Load a plan first.")
    text = st.text_area("Plan JSON", key="vl_text", height=300, placeholder='{"team_assignments": [], "vehicle_assignments": [], "shelter_allocations": [], "unserved_demand": []}')
    if st.button("Validate", type="primary") and text.strip():
        res = ui.safe(lambda: api.validate(sid, raw_text=text))
        if res:
            st.session_state["vl_res"] = res
    res = st.session_state.get("vl_res")
    if res:
        if not res["parsed"]:
            st.warning("Could not parse the plan: " + str(res["parse_error"]))
        st.markdown(ui.violations_html(res["report"]), unsafe_allow_html=True)
        if res.get("quality"):
            q = res["quality"]
            st.caption(f"Plan quality {ui.fmt_num(q['quality_norm'])} — {q['people_served']} of {q['people_total']} people served (priority-weighted).")
        counts = {k: v for k, v in res["report"]["counts_by_type"].items() if v}
        if counts:
            st.bar_chart(counts)
        if not res["report"]["valid"]:
            st.divider()
            st.markdown("**Repair with the verifier loop** — sends these exact violations to the LLM, re-validates, and repeats.")
            k = st.number_input("Max rounds (K)", 0, 10, 3, key="vl_k")
            if st.button("Repair with LLM"):
                with st.spinner("Repairing..."):
                    rec = ui.safe(lambda: api.repair(sid, json.loads(text) if res["parsed"] else text, max_repair_rounds=int(k)))
                if rec:
                    render_run(rec)
