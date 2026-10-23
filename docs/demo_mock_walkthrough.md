# Live demo with the SIMULATED (mock) LLM — zero Gemini calls

What it proves: the software pipeline end to end (scenario -> flawed plan -> deterministic validator -> verifier feedback -> repair loop -> re-validation -> metrics -> OR-Tools reference -> exports).
What it does NOT prove: that an LLM is good or bad at repairing plans. The mock repairs a violation with probability 0.8 when it receives the validator report and notices only ~30% of problems in its own critique. Any "improvement" shown with it is built into the simulator.

1. `start_demo_backend.bat` (project folder) — SIMULATED LLM, data in `demo_data\`, `.env` untouched.
2. `start_frontend.bat` in a second terminal -> http://localhost:8501 (sidebar must show the yellow "LLM: SIMULATED (mock)").
3. Scenario Lab: small / high / seed 42 -> Generate & save.
4. Verification Lab: paste `data\sample\demo_multi_violation_plan.json` -> Validate (14 violations, V1-V6) -> Repair with LLM (K = 3).
5. Planning Lab: same scenario -> Compare all (M1-M5).
6. Experiment Lab: YAML file `demo_mock_levels.yaml` -> Start (about 20 s; 88 runs) -> Results & Analytics -> Research Export.
Everything lands in `demo_data\`, never in `data\`.
