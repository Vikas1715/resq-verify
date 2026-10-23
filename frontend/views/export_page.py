import streamlit as st

from frontend import ui

FILES = [("research_summary.md", "text/markdown", "Markdown summary with tables (paste-ready for the paper)"),
         ("runs.csv", "text/csv", "One row per run, flat columns"),
         ("runs.json", "application/json", "Complete run records incl. rounds, prompts metadata, raw output"),
         ("comparisons.csv", "text/csv", "Paired tests with Holm-adjusted p-values and effect sizes"),
         ("plots.zip", "application/zip", "PNG figures"),
         ("bundle.zip", "application/zip", "Everything above + experiments + scenarios")]


def render(api):
    st.title("Research Export")
    st.caption("Exports are generated from the stored run records only. Give `bundle.zip` (or `research_summary.md` + `runs.csv`) to your writing assistant.")
    exps = ui.safe(lambda: api.experiments(include_adhoc=False), [])
    pick = st.selectbox("Scope", ["All experiments"] + [e["experiment_id"] for e in exps])
    adhoc = st.checkbox("Include interactive Planning Lab runs", False)
    eid = None if pick == "All experiments" else pick
    s = ui.safe(lambda: api.summary(eid, adhoc))
    if not s or not s.get("has_results"):
        ui.no_results()
    elif s["is_simulated"]:
        ui.sim_banner()
        st.warning("These exports contain simulated-LLM runs. Do not report them as findings.")
    for name, mime, desc in FILES:
        c = st.columns([3, 5, 2])
        c[0].markdown(f"**{name}**")
        c[1].caption(desc)
        data = ui.safe(lambda n=name: api.export(n, eid, adhoc))
        if data is not None:
            c[2].download_button("Download", data, file_name=name, mime=mime, key=f"dl_{name}")
    md = ui.safe(lambda: api.export("research_summary.md", eid, adhoc))
    if md:
        st.subheader("Preview: research_summary.md")
        st.markdown(md.decode("utf-8"))
