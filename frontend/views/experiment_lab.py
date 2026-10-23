import pandas as pd
import streamlit as st

from frontend import ui

METHODS = {"llm_only": "M1 LLM-only", "self_critique": "M2 Self-critique", "resq_verify": "M3 ResQ-Verify", "resample_verify": "M5 Resample + Verify",
           "ortools": "M4 OR-Tools"}


def _progress_table(exps):
    rows = []
    for e in exps:
        p = e.get("progress") or {}
        rows.append({"id": e["experiment_id"], "name": e.get("name"), "status": e.get("status"), "done": f"{p.get('done', 0)}/{p.get('total', 0)}",
                     "errors": p.get("errors", 0), "simulated": e.get("is_simulated"), "created": e.get("created_at")})
    return pd.DataFrame(rows)


def render(api):
    st.title("Experiment Lab")
    st.caption("Experiments are defined by configuration (YAML files in experiments/configs or the form below), not by editing code. "
               "Every scenario goes through every selected method (paired design).")
    h = ui.safe(api.health)
    if h and h["llm"]["is_simulated"]:
        ui.sim_banner()
    elif h and not h["llm"]["configured"]:
        st.warning("Gemini is not configured. LLM methods will fail until GEMINI_API_KEY and GEMINI_MODEL are set; OR-Tools-only experiments still work.")

    files = ui.safe(api.config_files, [])
    src = st.radio("Configuration source", ["Form", "YAML file"], horizontal=True)
    cfg, cfg_file = None, None
    if src == "YAML file" and files:
        cfg_file = st.selectbox("Config file", files)
        loaded = ui.safe(lambda: api.config_file(cfg_file))
        if loaded:
            st.json(loaded, expanded=False)
            cfg = loaded
    elif src == "YAML file":
        st.info("No YAML configs found in experiments/configs.")
    else:
        c = st.columns(3)
        name = c[0].text_input("Name", "my-experiment")
        sizes = c[1].multiselect("Sizes", ["small", "medium", "large"], ["small"])
        scarcity = c[2].multiselect("Scarcity", ["low", "medium", "high"], ["high"])
        c = st.columns(4)
        n = c[0].number_input("Scenarios per cell", 1, 500, 5)
        seed = c[1].number_input("Base seed", 0, 10**6, 42)
        reps = c[2].number_input("LLM repetitions", 1, 20, 1)
        k = c[3].number_input("Repair rounds K", 0, 10, 3)
        methods = st.multiselect("Methods", list(METHODS), list(METHODS), format_func=METHODS.get)
        c = st.columns(3)
        fmt = c[0].selectbox("Feedback format", ["structured", "free_text"])
        shared = c[1].checkbox("Share the first plan across the LLM methods (paired)", True)
        temp = c[2].number_input("Temperature (0 = env default)", 0.0, 2.0, 0.0, 0.1)
        c = st.columns(2)
        budgets = c[0].multiselect("LLM-call budgets (empty = use K above)", [1, 2, 3, 4, 5, 6, 8, 10], [1, 2, 3, 5],
                                   help="Each of M2, M3 and M5 is compared at each of these budgets. M1 always uses 1 call, M4 none.")
        levels = c[1].multiselect("Difficulty levels (empty = sizes × scarcity above)", ["L0", "L1", "L2", "L3", "L4"], [])
        cfg = dict(name=name, sizes=sizes, scarcity=scarcity, num_scenarios=int(n), base_seed=int(seed), repetitions=int(reps), methods=methods,
                   max_repair_rounds=int(k), feedback_format=fmt, shared_initial_plan=shared, temperature=(temp or None),
                   call_budgets=budgets, difficulty_levels=levels)
    if cfg:
        lv = list(cfg.get("difficulty_levels") or [])
        n_cells = (len(lv) if lv else len(cfg["sizes"]) * len(cfg["scarcity"])) * cfg["num_scenarios"]
        n_llm = sum(m not in ("ortools",) for m in cfg["methods"])
        calls = n_cells * cfg["repetitions"] * n_llm
        bud = list(cfg.get("call_budgets") or [])
        per = f"at most {max(bud)} LLM calls per run (budgets nested in one trajectory)" if bud else f"1 + up to {cfg['max_repair_rounds']} further calls per run"
        st.caption(f"{n_cells} scenarios · {calls} LLM-method runs · {per}. Real API calls cost time and may cost money.")
        if st.button("Start experiment", type="primary", disabled=(not (cfg.get("difficulty_levels") or (cfg["sizes"] and cfg["scarcity"]))) or not cfg["methods"]):
            e = ui.safe(lambda: api.run_experiment(cfg if src == "Form" else None, cfg_file if src == "YAML file" else None))
            if e:
                st.success(f"Started {e['experiment_id']}. It runs in the background; refresh below.")
    st.subheader("Experiments")
    c = st.columns([1, 5])
    c[0].button("Refresh")
    exps = ui.safe(lambda: api.experiments(include_adhoc=False), [])
    if not exps:
        st.info("No experiments yet.")
        return
    st.dataframe(_progress_table(exps), hide_index=True, use_container_width=True)
    eid = st.selectbox("Manage experiment", [e["experiment_id"] for e in exps])
    e = next(x for x in exps if x["experiment_id"] == eid)
    p = e.get("progress") or {}
    st.progress(min(1.0, p.get("done", 0) / p["total"]) if p.get("total") else 0.0, text=f"{e['status']} · {p.get('done', 0)}/{p.get('total', 0)} runs · {p.get('errors', 0)} errors")
    if e.get("message"):
        st.info(e["message"])
    b = st.columns(3)
    if b[0].button("Cancel", disabled=e["status"] not in ("running", "queued")):
        ui.safe(lambda: api.cancel(eid)); st.rerun()
    if b[1].button("Resume / retry failed", disabled=e["status"] in ("running", "queued", "completed") and not p.get("errors")):
        ui.safe(lambda: api.resume(eid)); st.rerun()
