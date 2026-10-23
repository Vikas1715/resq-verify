import pandas as pd
import streamlit as st

from frontend import charts, ui
from frontend.views._common import json_download


def render(api):
    st.title("Scenario Lab")
    st.caption("Synthetic scenarios are generated from a seed, so the same inputs always give the same scenario.")
    with st.form("gen"):
        c = st.columns(5)
        size = c[0].selectbox("Size", ["small", "medium", "large"], index=0)
        scarcity = c[1].selectbox("Scarcity", ["low", "medium", "high"], index=2)
        level = c[2].selectbox("Difficulty level", ["(none)", "L0", "L1", "L2", "L3", "L4"],
                               help="If a level is chosen it sets size, scarcity and extra constraints; Size and Scarcity are then ignored.")
        seed = c[3].number_input("Seed", min_value=0, value=42, step=1)
        go = c[4].form_submit_button("Generate & save", use_container_width=True)
    if go:
        sc = ui.safe(lambda: api.generate_scenario(size, scarcity, int(seed), difficulty_level=None if level == "(none)" else level))
        if sc:
            st.session_state["scenario_view"] = sc["scenario_id"]
            st.success(f"Saved scenario {sc['scenario_id']} (hash {sc['content_hash']})")
    scs = ui.safe(api.scenarios, [])
    if not scs:
        st.info("No scenarios saved yet.")
        return
    st.subheader("Saved scenarios")
    st.dataframe(pd.DataFrame(scs), hide_index=True, use_container_width=True)
    ids = [s["scenario_id"] for s in scs]
    cur = st.session_state.get("scenario_view")
    sid = st.selectbox("Inspect scenario", ids, index=ids.index(cur) if cur in ids else 0)
    sc = ui.safe(lambda: api.scenario(sid))
    if not sc:
        return
    d = sc.get("difficulty") or {}
    c = st.columns(5)
    for col, (k, lab) in zip(c, [("total_demand", "People at risk"), ("total_transport_capacity", "Transport capacity"), ("total_shelter_capacity", "Shelter capacity"),
                                  ("demand_to_transport_ratio", "Demand / transport"), ("difficulty_score", "Difficulty score")]):
        with col:
            v = d.get(k)
            ui.kpi(lab, f"{v:.2f}" if isinstance(v, float) else v)
    st.caption(f"Seed {sc['seed']} · generator v{sc['generator_version']} · hash {sc['content_hash']}. Difficulty score is a heuristic for comparing scenarios, not a ground truth.")
    t = st.tabs(["Overview", "Incidents", "Resources", "Raw JSON"])
    with t[0]:
        l, r = st.columns(2)
        l.plotly_chart(charts.scenario_map(sc), use_container_width=True)
        r.plotly_chart(charts.capacity_bars(sc), use_container_width=True)
    with t[1]:
        st.dataframe(pd.DataFrame(sc["incidents"]), hide_index=True, use_container_width=True)
        st.dataframe(pd.DataFrame(sc["roads"]), hide_index=True)
    with t[2]:
        a, b, c3 = st.columns(3)
        a.markdown("**Teams**"); a.dataframe(pd.DataFrame(sc["teams"]), hide_index=True)
        b.markdown("**Vehicles**"); b.dataframe(pd.DataFrame(sc["vehicles"]), hide_index=True)
        c3.markdown("**Shelters**"); c3.dataframe(pd.DataFrame(sc["shelters"]), hide_index=True)
    with t[3]:
        st.json(sc, expanded=False)
        json_download("Download scenario JSON", sc, f"{sc['scenario_id']}.json")
