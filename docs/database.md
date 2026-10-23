# Data storage

Default is plain files (no database needed). MongoDB is optional (`STORAGE_BACKEND=mongo`); when it is down and `MONGO_FALLBACK_TO_FILE=true`, the app falls back to files and says so in the sidebar / `/health`. With MongoDB active, every write is also mirrored to files, so exports never depend on the database.

## Files (`data/`)
| Path | Content |
|---|---|
| `data/scenarios/<scenario_id>.json` | scenario + difficulty metadata + content hash |
| `data/results/<experiment_id>/experiment.json` | config, LLM block, prompt version/hash, status, progress, solver references |
| `data/results/<experiment_id>/runs.jsonl` | one JSON run record per line (append-only; the latest record per scenario/method/run_index wins) |
| `data/results/ADHOC-interactive/` | runs started from the Planning Lab / Verification Lab |
| `data/exports/` | output of `export_results.py` |

## MongoDB collections
`scenarios` (unique `scenario_id`), `experiments` (unique `experiment_id`), `experiment_runs` (unique `run_id`, index on experiment/scenario/method/run_index). Documents have the same shape as the files.

## Run record (main fields)
`run_id, experiment_id, status (ok|error), scenario_id/seed/size, scarcity_level, method (+code M1–M5), run_index, difficulty_level, call_budget, calls_used, budget_mode (nested|independent|legacy_k), derived_from_budget, first_feasible_call, n_violations_by_round, violation_types_by_round, failure_persistence{initial_violations, persisted_in_final, final_fraction, by_round, by_type}, prompt_template_hashes, planner_template_hash, revision_template_hash, provider, is_simulated, model, model_version, temperature, prompt_version, prompt_hash, feedback_format, max_repair_rounds, shared_initial_plan, start_time, end_time, latency_s, token_usage, estimated_cost_usd, retry_count, api_errors, rounds[] (plan, raw_llm_output, parse_error, validation, n_violations, violation_counts, latency, tokens, edit_distance, feedback_sent), parsed_plan, raw_llm_output, validator_report, initial_feasible, final_feasibility, repair_success, repair_rounds_used, violation_counts_initial/final, plan_quality, plan_quality_raw, optimality_gap, solver, failure_reason`.

Corrupted JSONL lines (e.g. after a crash mid-write) are skipped and counted in `/health`.

Run identity is `(scenario_id, method, run_index[, call_budget])`: with call budgets, one scenario holds one run per budget. Each `rounds[]` entry also has `calls` (LLM calls in that round: initial 1, repair 1, resample 1, critique+revision 2, approved critique 1). Token fields are `null` when the API reported nothing. Records for a smaller budget cut from a larger run have `derived_from_budget` set and share their rounds with that run (`attempts.csv` lists each real trajectory once).
