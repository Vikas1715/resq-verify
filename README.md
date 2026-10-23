# ResQ-Verify

**A Verifier-Augmented LLM Framework for Reliable Emergency Resource Planning Under Hard Constraints**

An LLM drafts an emergency allocation plan (teams, vehicles, shelters). A deterministic validator checks seven hard-constraint rules. The platform asks one question: **does feedback from a deterministic verifier help more than other ways of spending the same number of LLM calls?** It compares, at equal LLM-call budgets and on identical, seeded, synthetic scenarios:

| | Method | What the LLM gets after a failed attempt |
|---|---|---|
| M1 | LLM-only | nothing (one call) |
| M2 | Self-critique | its own critique of its plan |
| M3 | ResQ-Verify | the validator's exact violation list |
| M5 | Resample + Verify | nothing: a fresh independent attempt from the same prompt (validator only decides when to stop) |
| M4 | OR-Tools CP-SAT | (classical reference, no LLM) |

M2 and M3 use the **same revision prompt** (`prompts/revision.txt`); only the feedback text differs. Plans are generated at five configurable **difficulty levels** (L0–L4). Every run is logged and exported as tables, statistics and plots.

> **Honesty notes**
> - Feeding verifier feedback to an LLM is established in prior work. This project does not claim the mechanism is new; it provides a controlled, reproducible comparison for constrained emergency planning.
> - The repository ships **no experimental results**. Until you run an experiment the app says: *No experimental conclusion available yet.*
> - `LLM_PROVIDER=mock` is a **simulated LLM** for trying the software without an API key. Everything it produces is stamped "SIMULATED" and must never be reported as a finding.
> - Scenarios are synthetic.

---

## 1. What you need (Windows)

| Need | Notes |
|---|---|
| Python 3.10, 3.11 or 3.12 | python.org → during install tick **Add Python to PATH**. (Very new Python versions may lack OR-Tools wheels.) |
| A Gemini API key | Google AI Studio → "Get API key". Free tier works but is rate-limited. |
| MongoDB | **Optional.** The default file storage needs nothing. |

## 2. First-time setup

Open **PowerShell** or Command Prompt in the project folder (unzip `resq-verify.zip`, then `cd` into it):

```bat
setup.bat
```

This creates `.venv`, installs everything and copies `.env.example` to `.env`. Manual equivalent:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
copy .env.example .env
```

Now open `.env` in Notepad and set:

```
GEMINI_API_KEY=your-key
GEMINI_MODEL=the-exact-model-name-from-AI-Studio
```

(The model name is deliberately not pre-filled; it is recorded in every experiment.) Never commit `.env`.

macOS/Linux: same steps with `source .venv/bin/activate` and `cp .env.example .env`.

## 3. Run it (two terminals)

Terminal 1 — backend:
```bat
start_backend.bat
```
(= `uvicorn backend.app.api.app:app --host 127.0.0.1 --port 8000`; API docs at http://127.0.0.1:8000/docs)

Terminal 2 — frontend:
```bat
start_frontend.bat
```
(= `streamlit run frontend\app.py`; opens http://localhost:8501)

The sidebar shows whether the LLM is configured, which storage is active and whether the solver loaded.

## 4. Run the tests
```bat
run_tests.bat
```
(= `python -m pytest tests -q`). Tests never call the real Gemini API or need MongoDB.

## 5. Your first experiment
1. **Scenario Lab** → generate `small / high / seed 42` → inspect it.
2. **Verification Lab** → "Load CP-SAT plan" → Validate (valid) → "Break the loaded plan" → Validate (violations listed by rule).
3. **Planning Lab** → pick the scenario → "Compare all (M1–M5)" → open each round to see what the validator found and what the LLM changed. One scenario is an illustration, not evidence.
4. **Experiment Lab** → YAML file → `quick_check.yaml` → Start. Watch progress; if it stops, press **Resume**.
5. **Experiment Details** → open runs, read raw LLM output, check parse errors and API errors.
6. Run `pilot.yaml`, then your frozen final config (see `docs/experiment_protocol.md`).
7. **Results & Analytics** → metrics, CIs, paired tests. **Research Export** → download `bundle.zip`.

Command-line alternative (no server needed):
```bat
python experiments\scripts\run_experiment.py --config experiments\configs\quick_check.yaml
python experiments\scripts\run_experiment.py --resume EXP-...
python experiments\scripts\export_results.py --experiment EXP-...
```

## 6. Where results are
- `data/results/<experiment_id>/runs.jsonl` and `experiment.json` — raw, complete records.
- `data/scenarios/` — scenarios used.
- `data/exports/` — CLI exports; the UI exports download from the browser.
- Export contents: `research_summary.md`, `runs.csv`, `runs_full.json`, `paired_comparisons.csv`, `attempts.csv` (one row per LLM attempt), `budget_curve.csv`, `by_difficulty_level.csv`, `failure_persistence.csv`, `summary.json`, `plots/*.png` (including `feasibility_vs_budget.png`), `experiments.json`, `scenarios.json`.

**For paper writing, give Claude:** `bundle.zip` (or at least `research_summary.md` + `runs.csv` + `paired_comparisons.csv`), the YAML config used, the model name/version from `experiments.json`, and `prompts/` (with the prompt hash).

## 7. Experiment configuration (no code edits)
Edit/copy a file in `experiments/configs/`: `sizes`, `scarcity`, `difficulty_levels` (L0–L4; replaces sizes × scarcity when set), `num_scenarios`, `base_seed`, `repetitions`, `methods` (`llm_only`, `self_critique`, `resq_verify`, `resample_verify`, `ortools`), `call_budgets` (e.g. `[1, 2, 3, 5]`), `nested_budgets`, `max_repair_rounds` (only used when `call_budgets` is empty), `feedback_format` (structured | free_text), `shared_initial_plan`, `temperature`, `solver_time_limit_s`. Defaults for everything else come from `.env`.

**Call budgets.** A budget B is the maximum number of LLM calls per run (the validator and the solver are not LLM calls). M1 always uses 1 call. M3 and M5 use the first plan plus up to B−1 more. M2 spends 2 calls per round (critique + revision), so it gets ⌊(B−1)/2⌋ rounds. With `nested_budgets: true` each (scenario, method, repetition) is run once at the largest budget and the smaller budgets are cut from the same run, which saves API calls but means the budgets are not independent samples; set it to `false` to run every budget separately. `calls_used` is logged for every run.

**Difficulty levels** (design choices, not measured difficulty): L0 small and ample; L1 medium; L2 large + many blocked roads; L3 large + vehicle-type restrictions + boat/team bottlenecks; L4 large + high scarcity + interacting constraints (boat-only sites, few teams, one big shelter). `python experiments/scripts/probe_levels.py` prints what each level looks like and how a simple greedy heuristic does against CP-SAT, without any LLM.

## 8. Optional: MongoDB
Set `STORAGE_BACKEND=mongo` and `MONGODB_URI` in `.env` (e.g. a local MongoDB or Atlas). If it cannot be reached the app falls back to files and says so in the sidebar. `docker compose up` starts MongoDB + both servers (untested template).

## 9. Project layout
```
backend/app/{api,analytics,core,experiments,schemas,services,solvers,validators,utils}
frontend/{app.py,views/,charts.py,style.css}
prompts/            versioned prompts (VERSION + content hash stored per run)
experiments/{configs,scripts}
data/{scenarios,results,exports,sample}   (sample = labelled DEMO files)
docs/               architecture, methodology, experiment_protocol, database
tests/
```

## 10. Troubleshooting
| Symptom | Fix |
|---|---|
| UI says "Cannot reach the backend" | Start `start_backend.bat` first; check port 8000 is free. |
| "GEMINI_API_KEY is not set" | Fill `.env`, restart the backend (settings are read at startup). |
| HTTP 429 / 503 in runs | Free-tier limits. Set `LLM_REQUEST_DELAY_S=4` (or higher) in `.env`; failed runs are retried by **Resume**. |
| `pip install ortools` fails | Use Python 3.10–3.12 (64-bit). Without OR-Tools the LLM methods still work but M4/reference is unavailable. |
| MongoDB errors | Leave `STORAGE_BACKEND=file`; MongoDB is optional. |
| Runs marked "interrupted" | The backend was stopped mid-run. Press **Resume**. |

## 11. Limitations
Synthetic scenarios · difficulty levels are not calibrated until you run the pilot · under nested budgets the budgets are not independent · single LLM family via REST · Gemini is not strictly deterministic even with a seed · CP-SAT is time-limited (status and gap recorded) · cost is estimated only from prices you supply · the UI was not browser-tested by the author of this package in the build environment (see the hand-over notes).

See `docs/` for architecture, methodology (rules, metrics, statistics), the experiment protocol and the data layout.
