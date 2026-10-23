"""Tests for the research-validity changes: M5 (blind resample + verify), the shared M2/M3 revision template, equal LLM-call
budgets, difficulty levels, richer run logging, and the budget-aware analytics/export. No real LLM is used: replies are
scripted, or come from the labelled mock (a test double)."""
import io
import json
import zipfile

import pytest

from backend.app.analytics.dataframe import budgets_in, runs_to_frame
from backend.app.analytics.report import build_bundle, research_summary_md
from backend.app.analytics.summary import build_summary
from backend.app.core.config import PROJECT_ROOT
from backend.app.experiments.records import failure_persistence
from backend.app.experiments.runner import ExperimentRunner, method_budgets, planned_total
from backend.app.schemas.experiment import ExperimentConfig
from backend.app.services.llm.base import LLMClient, LLMResult
from backend.app.services.llm.mock import MockLLM
from backend.app.services.methods import (
    MethodConfig, MethodContext, effective_budget, generate_initial, run_llm_only, run_resample_verify, run_resq_verify,
    run_self_critique, truncate_outcome,
)
from backend.app.services.prompts import PromptBook
from backend.app.services.scenario_generator import (
    DIFFICULTY_LEVELS, generate_level_scenario, generate_scenario,
)
from backend.app.services.store.base import dedupe_runs
from backend.app.services.store.file_store import FileStore
from backend.app.solvers.cpsat_solver import solve_scenario, solver_available
from tests.helpers import ScriptedLLM, broken_plan_dict, j, tiny_scenario, valid_plan_dict

SC = tiny_scenario()
BAD_CRITIQUE = j({"plan_is_valid": False, "issues": [{"rule": "R1", "description": "V1 overloaded"}]})


def ctx(llm, **cfg):
    return MethodContext(llm, PromptBook(PROJECT_ROOT / "prompts"), MethodConfig(**cfg), sleep=lambda s: None)


class SeedLLM(ScriptedLLM):
    """ScriptedLLM that also records the seed of every call."""
    def __init__(self, replies):
        super().__init__(replies)
        self.seeds = []

    def generate(self, prompt, **kw):
        self.seeds.append(kw.get("seed"))
        return super().generate(prompt, **kw)


# ------------------------------------------------------------------ M5
def test_m5_resamples_with_identical_prompt_and_no_feedback():
    llm = SeedLLM([j(broken_plan_dict()), j(broken_plan_dict()), j(valid_plan_dict())])
    c = ctx(llm, call_budget=5)
    out = run_resample_verify(c, SC, generate_initial(c, SC))
    assert out.final_report.valid and out.stop_reason == "valid" and out.calls_used == 3
    assert llm.prompts[0] == llm.prompts[1] == llm.prompts[2]          # the SAME planner prompt every time
    assert all("vehicle_capacity" not in p and "Feedback" not in p and "violation" not in p.lower() for p in llm.prompts)
    assert len(set(llm.seeds)) == 3                                     # but independent draws (different seeds)
    assert [r["kind"] for r in out.rounds] == ["initial", "resample", "resample"]


def test_m5_accepts_a_feasible_first_plan_without_further_calls():
    llm = ScriptedLLM([j(valid_plan_dict())])
    c = ctx(llm, call_budget=5)
    out = run_resample_verify(c, SC, generate_initial(c, SC))
    assert out.calls_used == 1 and len(llm.prompts) == 1 and out.final_report.valid


def test_m5_stops_at_budget_and_reports_infeasible():
    llm = ScriptedLLM([j(broken_plan_dict())])
    c = ctx(llm, call_budget=3)
    out = run_resample_verify(c, SC, generate_initial(c, SC))
    assert not out.final_report.valid and out.calls_used == 3 == len(llm.prompts) and out.stop_reason == "budget_exhausted"


# ------------------------------------------------------------------ M2 / M3 share one revision template
def _strip(prompt: str, feedback: str) -> str:
    return prompt.replace(feedback, "<<FB>>")


def test_m2_and_m3_revision_prompts_differ_only_in_the_feedback_text():
    book = PromptBook(PROJECT_ROOT / "prompts")
    plan = None
    from backend.app.schemas.plan import Plan
    plan = Plan.model_validate(broken_plan_dict())
    a = book.revision(SC, plan, None, "SELF-CRITIQUE-TEXT")
    b = book.revision(SC, plan, None, "VALIDATOR-TEXT")
    assert a.replace("SELF-CRITIQUE-TEXT", "<<FB>>") == b.replace("VALIDATOR-TEXT", "<<FB>>")
    assert book.revise(SC, plan, None, "x") == book.revision(SC, plan, None, "x") == book.repair(SC, plan, None, "x")


def test_real_m2_and_m3_runs_send_the_same_revision_template():
    l2 = ScriptedLLM([j(broken_plan_dict()), BAD_CRITIQUE, j(valid_plan_dict())])
    c2 = ctx(l2, call_budget=3)
    out2 = run_self_critique(c2, SC, generate_initial(c2, SC))
    l3 = ScriptedLLM([j(broken_plan_dict()), j(valid_plan_dict())])
    c3 = ctx(l3, call_budget=3)
    out3 = run_resq_verify(c3, SC, generate_initial(c3, SC))
    rev2, rev3 = l2.prompts[2], l3.prompts[1]
    head = lambda p: p.split("Feedback on the current plan:")[0]
    tail = lambda p: p.split("<<<END_FEEDBACK>>>")[1]
    assert head(rev2) == head(rev3) and tail(rev2) == tail(rev3)
    assert "V1 overloaded" in rev2 and "vehicle_capacity" in rev3 and "vehicle_capacity" not in rev2
    assert out2.calls_used == 3 and out3.calls_used == 2


def test_prompt_template_hashes_are_logged_and_stable():
    book = PromptBook(PROJECT_ROOT / "prompts")
    assert set(book.template_hashes) == {"_rules.txt", "planner.txt", "self_critique.txt", "revision.txt"}
    assert book.template_hashes == PromptBook(PROJECT_ROOT / "prompts").template_hashes


# ------------------------------------------------------------------ equal call budgets
@pytest.mark.parametrize("budget", [1, 2, 3, 5])
def test_every_method_respects_the_call_budget_exactly(budget):
    expected = {"llm_only": 1, "resq_verify": budget, "resample_verify": budget,
                "self_critique": 1 + 2 * ((budget - 1) // 2)}
    for method, fn in (("llm_only", run_llm_only), ("self_critique", run_self_critique), ("resq_verify", run_resq_verify),
                       ("resample_verify", run_resample_verify)):
        # always-infeasible plans, critiques that always find a problem -> every method spends its whole budget
        llm = ScriptedLLM([j(broken_plan_dict())])
        llm.replies_last = j(broken_plan_dict())
        if method == "self_critique":
            llm.replies = [j(broken_plan_dict())] + [BAD_CRITIQUE, j(broken_plan_dict())] * 5
        c = ctx(llm, call_budget=budget)
        out = fn(c, SC, generate_initial(c, SC))
        assert out.calls_used == len(llm.prompts) == expected[method], (method, budget)
        assert out.call_budget == (1 if method == "llm_only" else budget)
        assert sum(r["calls"] for r in out.rounds) == out.calls_used


def test_effective_budget_in_legacy_k_mode():
    mc = MethodConfig(max_repair_rounds=3, self_critique_rounds=3)
    assert [effective_budget(m, mc) for m in ("llm_only", "resq_verify", "resample_verify", "self_critique", "ortools")] == [1, 4, 4, 7, None]


def test_api_failure_counts_the_failed_call_and_is_an_error():
    class Down(LLMClient):
        provider, model = "down", "x"
        n = 0
        def generate(self, prompt, **kw):
            Down.n += 1
            if Down.n == 1:
                return LLMResult(text=j(broken_plan_dict()), model="x")
            from backend.app.core.errors import LLMCallError
            raise LLMCallError("HTTP 503", retry_count=3, errors=["503"])
    c = ctx(Down(), call_budget=4)
    out = run_resample_verify(c, SC, generate_initial(c, SC))
    assert out.status == "error" and out.calls_used == 2


@pytest.mark.parametrize("method", ["self_critique", "resq_verify", "resample_verify"])
def test_smaller_budget_cut_from_one_trajectory_equals_an_independent_run(method):
    """Nested budgets are only valid if cutting a big run gives exactly what a small run would have produced."""
    mock = MockLLM()
    book = PromptBook(PROJECT_ROOT / "prompts")
    from backend.app.services.methods import RUNNERS
    for seed in range(6):
        sc = generate_scenario("medium", "high", 500 + seed)
        big = MethodContext(mock, book, MethodConfig(call_budget=5, base_seed=7), sleep=lambda s: None)
        init = generate_initial(big, sc)
        full = RUNNERS[method](big, sc, init)
        for b in (1, 2, 3, 4):
            small_ctx = MethodContext(mock, book, MethodConfig(call_budget=b, base_seed=7), sleep=lambda s: None)
            indep = RUNNERS[method](small_ctx, sc, init)
            cut = truncate_outcome(full, sc, b)
            assert cut.calls_used == indep.calls_used, (method, seed, b)
            assert cut.final_report.valid == indep.final_report.valid
            assert json.dumps(cut.final_plan.model_dump() if cut.final_plan else None, sort_keys=True) == \
                   json.dumps(indep.final_plan.model_dump() if indep.final_plan else None, sort_keys=True)
            assert len(cut.rounds) == len(indep.rounds) and cut.call_budget == b


# ------------------------------------------------------------------ difficulty levels
def test_level_scenarios_are_deterministic_and_use_their_own_ids():
    a, b = generate_level_scenario("L3", 11), generate_level_scenario("L3", 11)
    assert a.content_hash == b.content_hash and a.scenario_id == "L3-s11" and a.difficulty_level == "L3"
    assert generate_level_scenario("L3", 12).content_hash != a.content_hash
    assert len({generate_level_scenario(l, 11).scenario_id for l in DIFFICULTY_LEVELS}) == 5


def test_scenarios_without_a_level_keep_their_old_hash():
    # hash recorded from the previous generator version (mentor guide, small / high / seed 42)
    assert generate_scenario("small", "high", 42).content_hash == "12c0950f4923d148"


def test_levels_get_harder_in_the_intended_directions():
    def avg(level, f, n=30):
        return sum(f(generate_level_scenario(level, 4000 + k)) for k in range(n)) / n
    inc = {l: avg(l, lambda s: len(s.incidents)) for l in DIFFICULTY_LEVELS}
    assert inc["L0"] < inc["L1"] < inc["L2"] == inc["L3"] == inc["L4"]
    assert avg("L2", lambda s: s.difficulty.blocked_road_fraction) > avg("L1", lambda s: s.difficulty.blocked_road_fraction)
    assert avg("L0", lambda s: s.difficulty.demand_to_transport_ratio) < avg("L4", lambda s: s.difficulty.demand_to_transport_ratio)
    assert avg("L3", lambda s: s.difficulty.restricted_incidents) > 0 == avg("L2", lambda s: s.difficulty.restricted_incidents or 0)
    assert all(generate_level_scenario("L3", 4000 + k).difficulty.boat_count <= 2 for k in range(30))
    assert avg("L4", lambda s: s.difficulty.available_team_slots) < avg("L2", lambda s: s.difficulty.available_team_slots)
    # one big shelter in L4
    big = [max(x.capacity for x in s.shelters) / sum(x.capacity for x in s.shelters) for s in (generate_level_scenario("L4", 4000 + k) for k in range(10))]
    assert min(big) > 0.4


def test_level_scenarios_stay_servable_for_most_incidents():
    for level in DIFFICULTY_LEVELS:
        for k in range(25):
            s = generate_level_scenario(level, 5000 + k)
            assert s.difficulty.servable_incidents >= 0.7 * len(s.incidents) - 1e-9, (level, k)
            assert s.difficulty.servable_demand_fraction > 0.5


@pytest.mark.skipif(not solver_available(), reason="ortools is not installed")
def test_solver_finds_a_non_trivial_plan_on_every_level():
    for level in DIFFICULTY_LEVELS:
        r = solve_scenario(generate_level_scenario(level, 6000), time_limit_s=10, workers=1, seed=0)
        assert r["status"] in ("OPTIMAL", "FEASIBLE") and r["objective_weighted_served"] > 0


# ------------------------------------------------------------------ config, keys
def test_config_validates_budgets_and_levels():
    cfg = ExperimentConfig(call_budgets=[5, 1, 3, 3], difficulty_levels=["L0", "L2"])
    assert cfg.call_budgets == [1, 3, 5] and cfg.uses_budgets
    with pytest.raises(Exception):
        ExperimentConfig(call_budgets=[0])
    with pytest.raises(Exception):
        ExperimentConfig(difficulty_levels=["L9"])
    assert "resample_verify" in ExperimentConfig().methods


def test_run_key_includes_the_call_budget():
    base = {"scenario_id": "s", "method": "resq_verify", "run_index": 0, "status": "ok"}
    runs = [{**base, "call_budget": 1}, {**base, "call_budget": 3}, {**base, "call_budget": 3, "x": 1}]
    assert len(dedupe_runs(runs)) == 2
    assert len(dedupe_runs([{k: v for k, v in base.items()}, dict(base)])) == 1     # no budget: 3-part key as before


# ------------------------------------------------------------------ runner integration + logging
def make_runner(settings):
    store = FileStore(settings.data_dir)
    return store, ExperimentRunner(store, settings, sleep=lambda s: None)


def levels_cfg(**kw):
    base = dict(name="lv", difficulty_levels=["L0", "L2"], num_scenarios=2, repetitions=2, base_seed=9100,
                methods=["llm_only", "self_critique", "resq_verify", "resample_verify", "ortools"], call_budgets=[1, 2, 3, 5])
    base.update(kw)
    return ExperimentConfig(**base)


def test_planned_total_counts_one_run_per_method_and_budget():
    cfg = levels_cfg()
    assert method_budgets(cfg, "llm_only") == [1] and method_budgets(cfg, "resq_verify") == [1, 2, 3, 5]
    assert planned_total(cfg) == 4 * (1 + 2 * (1 + 3 * 4))


@pytest.fixture()
def finished(settings):
    store, runner = make_runner(settings)
    cfg = levels_cfg()
    doc = runner.create(cfg)
    res = runner.run(doc["experiment_id"])
    return store, runner, cfg, doc["experiment_id"], res


def test_experiment_with_levels_and_budgets_is_complete_and_unique(finished):
    store, runner, cfg, eid, res = finished
    assert res["status"] == "completed" and res["progress"]["done"] == res["progress"]["total"] == planned_total(cfg)
    runs = store.list_runs(eid)
    assert len(runs) == planned_total(cfg)
    assert res["prompt_template_hashes"]["revision.txt"]
    keys = {(r["scenario_id"], r["method"], r["run_index"], r["call_budget"]) for r in runs}
    assert len(keys) == len(runs)
    assert {r["difficulty_level"] for r in runs} == {"L0", "L2"}
    assert {r["scenario_id"] for r in runs} == {"L0-s9100", "L0-s9101", "L2-s9100", "L2-s9101"}


def test_every_run_logs_what_is_needed_to_reconstruct_it(finished):
    store, runner, cfg, eid, _ = finished
    for r in store.list_runs(eid):
        for k in ("scenario_id", "scenario_seed", "difficulty_level", "method", "run_index", "call_budget", "calls_used", "model", "provider",
                  "prompt_version", "prompt_hash", "final_feasibility", "n_violations_final", "latency_s", "token_usage", "stop_reason",
                  "rounds", "validator_report", "failure_persistence", "first_feasible_call"):
            assert k in r, k
        if r["method"] == "ortools":
            assert r["calls_used"] == 0 and r["call_budget"] is None
            continue
        assert r["calls_used"] <= r["call_budget"] and r["calls_used"] == sum(x["calls"] for x in r["rounds"])
        assert r["revision_template_hash"] if r["method"] in ("self_critique", "resq_verify") else r["revision_template_hash"] is None
        assert r["planner_template_hash"] and r["is_simulated"]
        if r["method"] in ("resq_verify", "resample_verify"):
            assert r["budget_mode"] == "nested" or r["method"] == "llm_only"
            assert r["calls_used"] == (r["rounds"][-1]["round"] + 1)


def test_larger_budget_never_does_worse_on_the_same_trajectory(finished):
    store, _, _, eid, _ = finished
    by = {}
    for r in store.list_runs(eid):
        if r["method"] in ("resq_verify", "resample_verify", "self_critique") and r["status"] == "ok":
            by.setdefault((r["scenario_id"], r["method"], r["run_index"]), {})[r["call_budget"]] = r
    for (sid, m, _), runs in by.items():
        if m == "self_critique":   # M2 never looks at the validator, so more rounds can make a plan worse
            continue
        feas = [bool(runs[b]["final_feasibility"]) for b in sorted(runs)]
        assert feas == sorted(feas), (sid, m)


def test_budget_1_equals_m1_for_every_method(finished):
    store, _, _, eid, _ = finished
    runs = store.list_runs(eid)
    m1 = {(r["scenario_id"], r["run_index"]): r["final_feasibility"] for r in runs if r["method"] == "llm_only"}
    for r in runs:
        if r["method"] in ("resq_verify", "resample_verify", "self_critique") and r["call_budget"] == 1:
            assert r["final_feasibility"] == m1[(r["scenario_id"], r["run_index"])] and r["calls_used"] == 1


def test_resume_repeats_nothing(finished):
    store, runner, _, eid, _ = finished
    n = len(store.list_runs(eid))
    assert runner.run(eid)["status"] == "completed" and len(store.list_runs(eid)) == n


def test_independent_budget_mode_runs_every_budget_separately(settings):
    store, runner = make_runner(settings)
    cfg = levels_cfg(difficulty_levels=["L0"], num_scenarios=1, repetitions=1, nested_budgets=False, methods=["resq_verify"], call_budgets=[2, 3])
    eid = runner.create(cfg)["experiment_id"]
    runner.run(eid)
    runs = store.list_runs(eid)
    assert {r["call_budget"] for r in runs} == {2, 3} and all(r["budget_mode"] == "independent" and r["derived_from_budget"] is None for r in runs)


def test_legacy_k_mode_still_works_and_records_the_effective_budget(settings):
    store, runner = make_runner(settings)
    cfg = ExperimentConfig(name="k", sizes=["small"], scarcity=["high"], num_scenarios=1, methods=["llm_only", "self_critique", "resq_verify", "resample_verify"])
    eid = runner.create(cfg)["experiment_id"]
    runner.run(eid)
    got = {r["method"]: (r["call_budget"], r["budget_mode"]) for r in store.list_runs(eid)}
    assert got == {"llm_only": (1, None), "self_critique": (7, "legacy_k"), "resq_verify": (4, "legacy_k"), "resample_verify": (4, "legacy_k")}


def test_unknown_tokens_stay_none_never_zero(settings):
    class NoUsage(ScriptedLLM):
        def generate(self, prompt, **kw):
            r = super().generate(prompt, **kw)
            r.input_tokens = r.output_tokens = r.total_tokens = None
            return r
    llm = NoUsage([j(broken_plan_dict()), j(valid_plan_dict())])
    c = ctx(llm, call_budget=3)
    out = run_resq_verify(c, SC, generate_initial(c, SC))
    assert out.rounds[0]["token_usage"]["total"] is None
    store, runner = make_runner(settings)
    from backend.app.experiments.records import build_run_record
    rec = build_run_record(experiment_id="E", scenario=SC, outcome=out, run_index=0, start_time="a", end_time="b", wall_time_s=0.1,
                           settings=settings, llm_meta={"provider": "x", "model": "m", "is_simulated": False}, method_cfg={}, prompt_meta=None,
                           reference=None, shared_initial_plan=True)
    assert rec["token_usage"]["total"] is None and rec["estimated_cost_usd"] is None
    df = runs_to_frame([rec])
    assert df["tokens_total"].isna().all()


def test_failure_persistence_counts_repeated_violations():
    def rnd(*keys, calls=1):
        return {"kind": "initial", "calls": calls, "validation": {"violations": [{"type": t, "entity": e} for t, e in keys]}}
    p = failure_persistence([rnd(("vehicle_capacity", "V1"), ("shelter_capacity", "S1")), rnd(("vehicle_capacity", "V1")), rnd()])
    assert p["initial_violations"] == 2 and p["persisted_in_final"] == 0 and p["by_round"] == [0.5, 0.0]
    q = failure_persistence([rnd(("a", "x")), rnd(("a", "x"))])
    assert q["final_fraction"] == 1.0 and q["by_type"]["a"] == {"initial": 1, "persisted": 1}
    assert failure_persistence([rnd(), rnd()])["final_fraction"] is None


# ------------------------------------------------------------------ analytics / exports
def test_summary_has_budget_curve_level_table_persistence_and_equal_budget_comparisons(finished):
    store, _, cfg, eid, _ = finished
    runs = store.list_runs(eid)
    s = build_summary(runs)
    assert s["call_budgets"] == [1, 2, 3, 5] and s["call_budget_selected"] == 5
    curve = s["budget_curve"]
    assert {r["call_budget"] for r in curve} == {1, 2, 3, 5}
    assert {r["method"] for r in curve} == {"llm_only", "self_critique", "resq_verify", "resample_verify", "ortools"}
    assert all(r["broadcast"] for r in curve if r["method"] == "llm_only") and all(r["reference"] for r in curve if r["method"] == "ortools")
    m3 = {r["call_budget"]: r for r in curve if r["method"] == "resq_verify"}
    assert m3[5]["feasibility"] >= m3[1]["feasibility"]
    assert {r["difficulty_level"] for r in s["by_difficulty_level"]} == {"L0", "L2"}
    assert {c["call_budget"] for c in s["comparisons"]} == {1, 2, 3, 5}
    assert any({c["method_a"], c["method_b"]} == {"resq_verify", "resample_verify"} for c in s["comparisons"])
    assert s["is_simulated"]
    # headline table uses ONE budget, so no method appears twice
    assert [m["method"] for m in s["methods"]] == ["llm_only", "self_critique", "resq_verify", "resample_verify", "ortools"]
    assert all(m["call_budget"] in (5, None) for m in s["methods"] if m["method"] != "llm_only")
    b3 = build_summary(runs, call_budget=3)
    assert b3["call_budget_selected"] == 3 and all(m["call_budget"] in (3, None) for m in b3["methods"] if m["method"] != "llm_only")


def test_markdown_and_bundle_contain_new_sections_and_a_simulated_flag(finished):
    store, _, _, eid, _ = finished
    runs, exps = store.list_runs(eid), [store.get_experiment(eid)]
    md = research_summary_md(runs, experiments=exps)
    for h in ("SIMULATED LLM", "Feasibility vs LLM-call budget", "By difficulty level", "Failure persistence", "equal call budget"):
        assert h in md, h
    z = zipfile.ZipFile(io.BytesIO(build_bundle(runs, exps, store.list_scenarios())))
    names = set(z.namelist())
    assert {"attempts.csv", "budget_curve.csv", "by_difficulty_level.csv", "failure_persistence.csv", "runs.csv"} <= names
    assert "plots/feasibility_vs_budget.png" in names and "plots/feasibility_by_difficulty_level.png" in names
    attempts = z.read("attempts.csv").decode()
    assert "calls_cumulative" in attempts and attempts.count("\n") > 10
    header = z.read("runs.csv").decode().splitlines()[0].split(",")
    assert {"call_budget", "calls_used", "difficulty_level", "revision_template_hash", "first_feasible_call"} <= set(header)


def test_no_conclusion_text_without_data():
    s = build_summary([])
    assert not s["has_results"] and s["message"] and s["budget_curve"] == []
    assert "No experimental conclusion" in research_summary_md([])


# ------------------------------------------------------------------ API
def test_api_m5_and_level_generation(settings):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from backend.app.api.app import create_app
    c = TestClient(create_app(settings))
    sc = c.post("/scenarios/generate", json={"seed": 3, "difficulty_level": "L2"}).json()
    assert sc["scenario_id"] == "L2-s3" and sc["difficulty_level"] == "L2"
    assert c.post("/scenarios/generate", json={"seed": 3, "difficulty_level": "L9"}).status_code == 400
    r = c.post("/planning/resample-verify", json={"scenario_id": "L2-s3", "call_budget": 3}).json()
    assert r["method"] == "resample_verify" and r["call_budget"] == 3 and r["calls_used"] <= 3 and r["is_simulated"]
    v = c.post("/planning/verify", json={"scenario_id": "L2-s3", "call_budget": 2}).json()
    assert v["calls_used"] <= 2
    assert [x["id"] for x in c.get("/scenarios/levels").json()["levels"]] == DIFFICULTY_LEVELS
    assert {"resample_verify"} <= {m["id"] for m in c.get("/config/methods").json()["methods"]}
