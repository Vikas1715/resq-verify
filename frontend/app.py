"""ResQ-Verify console. Start with:  streamlit run frontend/app.py"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from frontend import ui  # noqa: E402
from frontend.api_client import Api, ApiError  # noqa: E402
from frontend.views import (experiment_details, experiment_lab, export_page, overview, planning_lab, results,  # noqa: E402
                            scenario_lab, verification_lab)

PAGES = {
    "Overview": overview, "Scenario Lab": scenario_lab, "Planning Lab": planning_lab, "Verification Lab": verification_lab,
    "Experiment Lab": experiment_lab, "Results & Analytics": results, "Experiment Details": experiment_details,
    "Research Export": export_page,
}


def main() -> None:
    st.set_page_config(page_title="ResQ-Verify", page_icon="🛟", layout="wide")
    ui.load_css()
    api = Api(os.getenv("BACKEND_URL", "http://127.0.0.1:8000"))
    with st.sidebar:
        st.markdown("### 🛟 ResQ-Verify")
        st.caption("Verifier-augmented LLM planning under hard constraints")
        page = st.radio("Navigate", list(PAGES), label_visibility="collapsed", key="page")
        try:
            h = api.health()
            llm = h["llm"]
            if llm["is_simulated"]:
                st.warning("LLM: SIMULATED (mock)")
            elif llm["configured"]:
                st.success(f"LLM: {llm['model']}")
            else:
                st.error("LLM not configured: " + ", ".join(llm.get("missing", [])))
            st.caption(f"Storage: {h['storage']['active']} · Solver: {'ok' if h['solver']['ortools_available'] else 'missing'} · Prompts {h['prompt_version']}")
            if h["storage"].get("note"):
                st.caption(h["storage"]["note"])
        except ApiError as e:
            st.error(e.message)
    PAGES[page].render(api)


main()
