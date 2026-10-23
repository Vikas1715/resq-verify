# Architecture

```
 Streamlit console (frontend/)  --HTTP-->  FastAPI (backend/app/api)
                                               |
        +--------------------------------------+------------------------------------+
        |                    |                 |                  |                  |
  scenario generator   LLM clients        validator (V1-V7)   CP-SAT solver     analytics/exports
  services/            services/llm       validators/         solvers/           analytics/
        \___________________ methods.py (M1-M5) + experiments/runner.py ___________/
                                               |
                                 Store: files (default) | MongoDB (optional)
```

| Layer | Folder | Role |
|---|---|---|
| UI | `frontend/` | Streamlit pages + custom CSS + Plotly. Only talks to the API. |
| API | `backend/app/api/` | FastAPI routes, uniform error format `{"error": {code, message, details}}`. |
| Domain schemas | `backend/app/schemas/` | Pydantic models: Scenario, Plan, ValidationReport, ExperimentConfig. |
| Generator | `services/scenario_generator.py` | Seeded synthetic scenarios, difficulty metadata, content hash. |
| Validator | `validators/validator.py` | Deterministic rules V1–V7; the only judge of feasibility. |
| LLM | `services/llm/` | `GeminiClient` (REST, retries, usage), `MockLLM` (labelled test double). |
| Methods | `services/methods.py` | M1 LLM-only, M2 self-critique, M3 ResQ-Verify, M4 CP-SAT, M5 resample + verify; call budgets and nested-budget truncation. |
| Runner | `experiments/` | Paired, resumable, cancellable experiment execution + run records. |
| Storage | `services/store/` | `FileStore` (JSON/JSONL) and `MongoStore`; mongo mirrors writes to files. |
| Analytics | `analytics/` | Flattening, statistics, summary tables, plots, Markdown report, bundle. |
| Prompts | `prompts/` | Versioned plain-text prompts (`VERSION` + content hash recorded per run). |

## Request flow for one ResQ-Verify run
1. `generate_initial`: planner prompt → LLM → `parse_plan` (never raises).
2. `validate_plan` → `ValidationReport` (parse failures become a V7 violation).
3. If invalid and rounds remain: `build_feedback` (structured JSON or free text) → repair prompt → LLM → back to 2.
4. Stop when valid or after K rounds. Each round is logged (plan, raw output, report, latency, tokens, edit distance).

M2 (self-critique) uses the same LLM to critique and revise **without** validator output; the validator only scores the result afterwards. M1 is a single call scored afterwards. M1, M2, M3 and M5 can share the first plan (`shared_initial_plan`, default on) so differences come from the method, not sampling luck.

## Design decisions
- **The validator is code, not an LLM**: results do not depend on a model judging a model.
- **Two failure classes**: unparseable/invalid plans are *infeasible* results; API/infrastructure failures are `status="error"`, excluded from rates and retried on resume.
- **Plan quality is computed from the plan's records**, never from the LLM's own claims.
- **No silent fallbacks for research data**: a missing Gemini key makes the experiment fail with a clear message; the mock LLM must be selected explicitly and stamps `is_simulated=true` on everything it touches.
