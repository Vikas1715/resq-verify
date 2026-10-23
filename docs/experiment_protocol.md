# Experiment protocol

1. **Check the setup** — `.env` has `GEMINI_API_KEY` and `GEMINI_MODEL`; the sidebar shows the model name (not "SIMULATED").
2. **Quick check** — run `experiments/configs/quick_check.yaml` (Experiment Lab → YAML file, or CLI). Open Experiment Details and read a few runs, including raw output, to confirm parsing works.
3. **Pilot** — `pilot.yaml` (levels L0–L4, budgets 1/2/3/5, development seeds 1000–1004). Look at variance, error counts and cost, and at whether each level is neither trivially easy nor hopeless for the LLM. Decide budgets, repetitions, levels and any prompt changes *before* the main study, and write the decision rules down first (for example "drop a level if M1 is ≥95 % or ≤5 % feasible on it").
4. **Freeze** — copy the chosen YAML to a new name (e.g. `final_v1.yaml`). Do not edit prompts after this point (the prompt hash is stored with each run; a change shows up as a different hash).
5. **Main study** — run the frozen config once (`full_study.yaml` uses the test seeds 2000–2009 per level). If it stops, **resume** (completed runs are skipped; failed API calls are retried).
6. **Ablations** (optional) — `ablation_feedback_format.yaml` twice (structured / free_text); the call-budget sweep is built into `call_budgets`.
7. **Analyse** — Results & Analytics; Research Export → `bundle.zip`.
8. **Report** — copy numbers from `research_summary.md`; always report n, CI, test used and effect size; report errored runs and infeasible-by-parse runs separately.

## Reproducibility checklist (all stored automatically)
Scenario seed + content hash · generator version · difficulty level · model name and returned model version · temperature, max tokens, call budget, calls used, K, feedback format · prompt version + overall hash + per-template hashes · per-round raw output, parsed plan, validator report · timestamps, latency, tokens, retries · storage backend · solver status/gap.

## Pairing and seeds
Scenario *k* of each (size, scarcity) cell, or of each difficulty level, uses seed `base_seed + k`, so changing methods, K or feedback format keeps scenarios identical across experiments. Use different `base_seed` ranges for pilot and main study to avoid tuning on test scenarios.
