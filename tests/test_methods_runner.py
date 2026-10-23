import json

from backend.app.core.config import PROJECT_ROOT
from backend.app.core.errors import LLMCallError
from backend.app.experiments.manager import ExperimentManager
from backend.app.experiments.runner import ExperimentRunner, planned_total
from backend.app.schemas.experiment import ExperimentConfig
from backend.app.services.llm.base import LLMClient
from backend.app.services.methods import (
    MethodConfig, MethodContext, generate_initial, run_llm_only, run_resq_verify, run_self_critique,
)
from backend.app.services.prompts import PromptBook
from backend.app.services.store.file_store import FileStore
from tests.helpers import ScriptedLLM, broken_plan_dict, j, tiny_scenario, valid_plan_dict

SC = tiny_scenario()


def ctx(llm, **cfg):
    return MethodContext(llm, PromptBook(PROJECT_ROOT / "prompts"), MethodConfig(**cfg), sleep=lambda s: None)


def test_llm_only_never_repairs_and_makes_one_call():
    llm = ScriptedLLM([j(broken_plan_dict())])
    c = ctx(llm)
    out = run_llm_only(c, SC, generate_initial(c, SC))
    assert len(llm.prompts) == 1 and not out.final_report.valid and out.stop_reason == "single_shot"


def test_resq_verify_repairs_with_validator_feedback():
    llm = ScriptedLLM([j(broken_plan_dict()), j(valid_plan_dict())])
    c = ctx(llm, max_repair_rounds=3)
    out = run_resq_verify(c, SC, generate_initial(c, SC))
    assert out.final_report.valid and out.stop_reason == "valid" and len(out.rounds) == 2
    assert len(llm.prompts) == 2
    assert "vehicle_capacity" in llm.prompts[1] and "V1" in llm.prompts[1]   # exact validator facts reached the LLM
    assert out.rounds[1]["kind"] == "repair" and out.rounds[1]["edit_distance"] >= 1


def test_resq_verify_stops_at_k_rounds_and_reports_failure():
    llm = ScriptedLLM([j(broken_plan_dict())])        # LLM keeps returning the same broken plan
    out = run_resq_verify(ctx(llm, max_repair_rounds=2), SC, generate_initial(ctx(llm), SC))
    assert not out.final_report.valid and out.stop_reason == "max_rounds" and len(llm.prompts) == 3


def test_resq_verify_k_zero_equals_no_repair():
    llm = ScriptedLLM([j(broken_plan_dict())])
    c = ctx(llm, max_repair_rounds=0)
    out = run_resq_verify(c, SC, generate_initial(c, SC))
    assert len(llm.prompts) == 1 and not out.final_report.valid


def test_resq_verify_recovers_from_unparseable_repair():
    llm = ScriptedLLM([j(broken_plan_dict()), "I cannot comply {", j(valid_plan_dict())])
    c = ctx(llm, max_repair_rounds=3)
    out = run_resq_verify(c, SC, generate_initial(c, SC))
    assert out.final_report.valid and len(out.rounds) == 3 and out.rounds[1]["parse_error"]


def test_unparseable_initial_counts_as_infeasible_not_error():
    llm = ScriptedLLM(["totally not json"])
    c = ctx(llm)
    out = run_llm_only(c, SC, generate_initial(c, SC))
    assert out.status == "ok" and not out.final_report.valid and out.final_report.parse_failed


def test_self_critique_can_approve_an_invalid_plan_because_it_has_no_validator():
    """The baseline must not be rescued by the validator: if the model says 'valid', the plan stays as it is."""
    llm = ScriptedLLM([j(broken_plan_dict()), j({"plan_is_valid": True, "issues": []})])
    c = ctx(llm, self_critique_rounds=3)
    out = run_self_critique(c, SC, generate_initial(c, SC))
    assert out.stop_reason == "self_approved" and not out.final_report.valid
    assert all("vehicle_capacity" not in p for p in llm.prompts)


def test_self_critique_revision_is_evaluated_afterwards():
    llm = ScriptedLLM([j(broken_plan_dict()), j({"plan_is_valid": False, "issues": ["capacity"]}), j(valid_plan_dict()),
                       j({"plan_is_valid": True, "issues": []})])
    c = ctx(llm, self_critique_rounds=3)
    out = run_self_critique(c, SC, generate_initial(c, SC))
    assert out.final_report.valid and out.stop_reason == "self_approved"


def test_api_failure_is_an_error_run_not_an_infeasible_plan():
    class Down(LLMClient):
        provider, model = "down", "x"
        def generate(self, prompt, **kw):
            raise LLMCallError("HTTP 503", retry_count=4, errors=["503"])
    c = ctx(Down())
    att = generate_initial(c, SC)
    out = run_resq_verify(c, SC, att)
    assert out.status == "error" and out.failure_reason


# ---------------------------------------------------------------- runner (uses the labelled mock LLM)
def make_runner(settings, **kw):
    store = FileStore(settings.data_dir)
    return store, ExperimentRunner(store, settings, sleep=lambda s: None, **kw)


def small_cfg(**kw):
    base = dict(name="t", sizes=["small"], scarcity=["high"], num_scenarios=2, repetitions=2, max_repair_rounds=3)
    base.update(kw)
    return ExperimentConfig(**base)


def test_full_run_with_mock_is_complete_paired_and_flagged_simulated(settings):
    store, runner = make_runner(settings)
    cfg = small_cfg(methods=["llm_only", "self_critique", "resq_verify", "ortools"])   # the original four methods (M5 has its own tests)
    doc = runner.create(cfg)
    assert doc["is_simulated"] and doc["progress"]["total"] == planned_total(cfg) == 2 * (1 + 2 * 3)
    res = runner.run(doc["experiment_id"])
    assert res["status"] == "completed" and res["progress"]["done"] == res["progress"]["total"]
    runs = store.list_runs(doc["experiment_id"])
    assert len(runs) == 14
    for r in runs:
        for key in ("run_id", "scenario_id", "scenario_seed", "method", "start_time", "end_time", "latency_s", "token_usage", "validator_report",
                    "final_feasibility", "violation_counts", "prompt_version", "prompt_hash", "provider", "is_simulated"):
            assert key in r, key
    llm_runs = [r for r in runs if r["method"] != "ortools"]
    assert all(r["is_simulated"] and r["provider"] == "mock" for r in llm_runs)
    # paired design: M1, M2, M3 of one (scenario, repetition) share the same initial plan
    by = {}
    for r in llm_runs:
        by.setdefault((r["scenario_id"], r["run_index"]), []).append(json.dumps(r["rounds"][0]["plan"], sort_keys=True))
    assert all(len(set(v)) == 1 and len(v) == 3 for v in by.values())
    ref = [r for r in runs if r["method"] == "ortools"]
    assert len(ref) == 2 and all(r["final_feasibility"] for r in ref)


def test_resume_does_not_repeat_completed_runs(settings):
    store, runner = make_runner(settings)
    doc = runner.create(small_cfg(num_scenarios=1, repetitions=1))
    runner.run(doc["experiment_id"])
    n = len(store.list_runs(doc["experiment_id"]))
    again = runner.run(doc["experiment_id"])
    assert again["status"] == "completed" and len(store.list_runs(doc["experiment_id"])) == n


def test_failed_runs_are_logged_and_retried_on_resume(settings):
    calls = {"n": 0}

    class Flaky(LLMClient):
        provider, model, is_simulated = "mock", "flaky", True
        def generate(self, prompt, **kw):
            calls["n"] += 1
            if calls["n"] <= 1:
                raise LLMCallError("HTTP 503", retry_count=4, errors=["503"])
            from backend.app.services.llm.mock import MockLLM
            return MockLLM(settings).generate(prompt, **kw)
    store, runner = make_runner(settings, llm_factory=lambda s: Flaky())
    cfg = small_cfg(num_scenarios=1, repetitions=1, methods=["llm_only", "resq_verify"])
    doc = runner.create(cfg)
    res = runner.run(doc["experiment_id"])
    assert res["progress"]["errors"] >= 1
    assert any(r["status"] == "error" for r in store.list_runs(doc["experiment_id"]))
    runner.run(doc["experiment_id"])
    assert all(r["status"] == "ok" for r in store.list_runs(doc["experiment_id"]))


def test_missing_gemini_key_fails_cleanly_without_running(settings):
    import dataclasses
    s = dataclasses.replace(settings, llm_provider="gemini", gemini_api_key=None, gemini_model="m")
    store, runner = make_runner(s)
    doc = runner.create(small_cfg())
    res = runner.run(doc["experiment_id"])
    assert res["status"] == "failed" and "GEMINI_API_KEY" in res["message"] and store.list_runs(doc["experiment_id"]) == []


def test_ortools_only_experiment_needs_no_llm(settings):
    import dataclasses
    s = dataclasses.replace(settings, llm_provider="gemini", gemini_api_key=None, gemini_model=None)
    store, runner = make_runner(s)
    doc = runner.create(small_cfg(methods=["ortools"], num_scenarios=2))
    assert doc["llm"]["provider"] is None
    assert runner.run(doc["experiment_id"])["status"] == "completed"


def test_manager_runs_in_background_and_can_cancel(settings):
    store, runner = make_runner(settings)
    mgr = ExperimentManager(store, runner)
    doc = mgr.start(small_cfg(num_scenarios=1, repetitions=1))
    mgr.join(doc["experiment_id"], 60)
    assert store.get_experiment(doc["experiment_id"])["status"] == "completed"


def test_interrupted_experiments_are_marked_on_startup(settings):
    store, runner = make_runner(settings)
    doc = runner.create(small_cfg())
    store.update_experiment(doc["experiment_id"], {"status": "running"})
    assert ExperimentManager(store, runner).mark_interrupted() == 1
    assert store.get_experiment(doc["experiment_id"])["status"] == "interrupted"


def test_config_validation_rejects_bad_input():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ExperimentConfig(methods=["magic"])
    with pytest.raises(ValidationError):
        ExperimentConfig(num_scenarios=0)
    assert ExperimentConfig(methods=["ortools", "llm_only"]).methods == ["llm_only", "ortools"]


def test_cost_only_from_user_supplied_prices(settings):
    import dataclasses
    from backend.app.experiments.records import estimate_cost
    u = {"input": 1_000_000, "output": 500_000, "thoughts": 0}
    assert estimate_cost(settings, u) is None
    s = dataclasses.replace(settings, price_input_per_m=1.0, price_output_per_m=2.0)
    assert estimate_cost(s, u) == 2.0


def test_all_shipped_yaml_configs_are_valid():
    from backend.app.experiments.configs import list_config_files, load_config_file
    files = list_config_files()
    assert {"quick_check.yaml", "pilot.yaml", "full_study.yaml"} <= set(files)
    for f in files:
        assert load_config_file(f).methods
