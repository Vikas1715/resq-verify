import streamlit as st

from frontend import ui


def render(api):
    st.title("ResQ-Verify")
    st.markdown("**A verifier-augmented LLM framework for reliable emergency resource planning under hard constraints.**")
    st.markdown(
        "An LLM drafts an allocation plan. A deterministic validator checks seven hard-constraint rules. Exact, structured violations go back "
        "to the LLM, which repairs the plan for up to *K* rounds. The framework compares this with an LLM-only baseline, an LLM self-critique "
        "baseline and an OR-Tools CP-SAT reference, all on the same paired, seeded scenarios.")
    st.caption("The idea of feeding verifier feedback to an LLM is established in prior work; this project does not claim it as new. "
               "The contribution is a controlled, reproducible comparison on constrained emergency planning.")
    h = ui.safe(api.health)
    if h:
        if h["llm"]["is_simulated"]:
            ui.sim_banner()
        elif not h["llm"]["configured"]:
            ui.banner("<b>Gemini is not configured.</b> Add GEMINI_API_KEY and GEMINI_MODEL to your .env file and restart the backend. "
                      "The scenario generator, validator and OR-Tools solver still work.", "info")
        c = st.columns(4)
        with c[0]: ui.kpi("LLM provider", h["llm"]["provider"])
        with c[1]: ui.kpi("Model", h["llm"].get("model") or "—")
        with c[2]: ui.kpi("Storage", h["storage"]["active"])
        with c[3]: ui.kpi("Prompt version", h["prompt_version"])
    exps = ui.safe(lambda: api.experiments(include_adhoc=False), [])
    s = ui.safe(api.summary)
    st.subheader("Research status")
    if not exps or not s or not s.get("has_results"):
        ui.no_results()
    else:
        if s["is_simulated"]:
            ui.sim_banner()
        c = st.columns(3)
        with c[0]: ui.kpi("Experiments", len(exps))
        with c[1]: ui.kpi("Completed runs", s["n_ok"])
        with c[2]: ui.kpi("Scenarios", s["n_scenarios"])
        st.caption("Open **Results & Analytics** for the numbers. This page never interprets them.")
    st.subheader("Workflow")
    st.markdown(
        "1. **Scenario Lab** — create seeded synthetic scenarios.\n"
        "2. **Planning Lab** — try the four methods on one scenario and watch the repair rounds.\n"
        "3. **Verification Lab** — paste or edit any plan and see exactly which rules it breaks.\n"
        "4. **Experiment Lab** — run a configured, resumable, paired experiment.\n"
        "5. **Results & Analytics** / **Experiment Details** — inspect metrics, statistics and every run.\n"
        "6. **Research Export** — download CSV, JSON, plots and `research_summary.md` for the paper.")
    st.subheader("Methods compared")
    st.table({"Code": ["M1", "M2", "M3", "M5", "M4"], "Method": ["LLM-only", "LLM self-critique", "ResQ-Verify (validator loop)", "Resample + Verify (no feedback)",
                                                                     "OR-Tools CP-SAT"],
              "Uses validator to fix the plan?": ["No (evaluated afterwards only)", "No (evaluated afterwards only)", "Yes, feedback to the LLM",
                                                  "Only to decide when to stop; the LLM sees no feedback", "n/a (exact model)"]})
