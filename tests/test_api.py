import pytest
from fastapi.testclient import TestClient

from backend.app.api.app import create_app
from tests.helpers import broken_plan_dict, valid_plan_dict


@pytest.fixture()
def api(settings):
    app = create_app(settings)
    return TestClient(app), app


def test_health_reports_simulated_llm_and_storage(api):
    c, _ = api
    h = c.get("/health").json()
    assert h["status"] == "ok" and h["llm"]["is_simulated"] and h["storage"]["active"] == "file"
    assert h["settings"]["gemini_api_key"] in ("set", "not set") and h["settings"]["mongodb_uri"] in ("set", "not set")


def test_scenario_generate_get_and_errors(api):
    c, _ = api
    sc = c.post("/scenarios/generate", json={"size": "small", "scarcity": "high", "seed": 3}).json()
    assert c.get(f"/scenarios/{sc['scenario_id']}").json()["content_hash"] == sc["content_hash"]
    assert c.get("/scenarios/nope").status_code == 404
    r = c.post("/scenarios/generate", json={"size": "huge"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "BAD_REQUEST"
    assert c.post("/scenarios/generate", json={"seed": "abc"}).status_code == 422


def test_validate_endpoint_catches_violations_and_garbage(api):
    c, _ = api
    sid = c.post("/scenarios/generate", json={"size": "small", "scarcity": "low", "seed": 1}).json()["scenario_id"]
    r = c.post("/validate", json={"scenario_id": sid, "raw_text": "garbage"}).json()
    assert not r["parsed"] and not r["report"]["valid"]
    assert c.post("/validate", json={"scenario_id": sid}).status_code == 400


def test_solve_returns_valid_plan(api):
    c, _ = api
    sid = c.post("/scenarios/generate", json={"size": "small", "scarcity": "medium", "seed": 2}).json()["scenario_id"]
    r = c.post("/solve", json={"scenario_id": sid}).json()
    assert r["report"]["valid"] and r["quality"]["quality_norm"] > 0


def test_planning_endpoints_and_adhoc_not_in_default_analytics(api):
    c, _ = api
    sid = c.post("/scenarios/generate", json={"size": "small", "scarcity": "high", "seed": 5}).json()["scenario_id"]
    for ep, method in (("generate", "llm_only"), ("self-critique", "self_critique"), ("verify", "resq_verify")):
        r = c.post(f"/planning/{ep}", json={"scenario_id": sid}).json()
        assert r["method"] == method and r["is_simulated"]
    cmp = c.post("/planning/compare", json={"scenario_id": sid}).json()
    assert {x["method"] for x in cmp["runs"]} == {"llm_only", "self_critique", "resq_verify", "resample_verify", "ortools"}
    assert c.post("/planning/verify", json={"scenario_id": "missing"}).status_code == 404
    s = c.get("/analytics/summary").json()
    assert s["has_results"] is False and s["message"] == "No experimental conclusion available yet."
    assert c.get("/analytics/summary?include_adhoc=true").json()["has_results"] is True


def test_repair_from_user_supplied_plan(api):
    c, _ = api
    sid = c.post("/scenarios/generate", json={"size": "small", "scarcity": "low", "seed": 9}).json()["scenario_id"]
    r = c.post("/planning/repair", json={"scenario_id": sid, "plan": {"vehicle_assignments": [{"vehicle_id": "ZZ", "incident_id": "I1", "trips": 1, "people": 1}]}})
    assert r.status_code == 200 and r.json()["initial_feasible"] is False


def test_experiment_flow_and_exports(api):
    c, app = api
    e = c.post("/experiments/run", json={"config": {"sizes": ["small"], "scarcity": ["high"], "num_scenarios": 2}}).json()
    eid = e["experiment_id"]
    app.state.ctx.manager.join(eid, 60)
    assert c.get(f"/experiments/{eid}").json()["status"] == "completed"
    runs = c.get(f"/experiments/{eid}/runs").json()
    assert runs["n"] == 10 and "rounds" not in runs["runs"][0]
    rid = runs["runs"][0]["run_id"]
    assert "rounds" in c.get(f"/runs/{rid}").json()
    s = c.get(f"/analytics/summary?experiment_id={eid}").json()
    assert s["has_results"] and s["is_simulated"]
    md = c.get(f"/exports/research_summary.md?experiment_id={eid}").text
    assert "SIMULATED LLM" in md and "Headline metrics" in md
    assert c.get(f"/exports/runs.csv?experiment_id={eid}").text.startswith("experiment_id")
    z = c.get(f"/exports/bundle.zip?experiment_id={eid}")
    assert z.status_code == 200 and z.content[:2] == b"PK"
    assert c.get("/exports/evil.exe").status_code == 404
    assert c.post(f"/experiments/{eid}/cancel").status_code == 409     # not running any more


def test_bad_experiment_requests(api):
    c, _ = api
    assert c.post("/experiments/run", json={}).status_code == 400
    assert c.post("/experiments/run", json={"config": {"methods": ["magic"]}}).status_code in (400, 422)
    assert c.post("/experiments/run", json={"config_file": "../../etc/passwd"}).status_code == 404
    assert c.get("/experiments/nope").status_code == 404


def test_empty_state_says_no_conclusion(api):
    c, _ = api
    md = c.get("/exports/research_summary.md").text
    assert "No experimental conclusion available yet." in md
