"""Guards the files used for the live demo: the flawed plan must really violate V1-V6, and the (simulated) repair must run."""
import json
from pathlib import Path

from backend.app.schemas.plan import Plan
from backend.app.schemas.scenario import Scenario
from backend.app.validators.validator import validate_plan

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample"


def _load():
    sc = Scenario.model_validate(json.loads((SAMPLE / "sample_scenario_small-high-s42.json").read_text(encoding="utf-8")))
    d = json.loads((SAMPLE / "demo_multi_violation_plan.json").read_text(encoding="utf-8"))
    d.pop("_label", None)
    return sc, d


def test_demo_plan_breaks_six_rules():
    sc, d = _load()
    rep = validate_plan(sc, Plan.model_validate(d))
    types = {v.type for v in rep.violations}
    assert not rep.valid
    assert {"vehicle_capacity", "team_availability", "shelter_capacity", "road_accessibility", "coverage", "compatibility"} <= types


def test_valid_sample_plan_is_valid():
    sc, _ = _load()
    d = json.loads((SAMPLE / "sample_plan_valid_cpsat.json").read_text(encoding="utf-8"))
    d.pop("_label", None)
    assert validate_plan(sc, Plan.model_validate(d)).valid


def test_demo_repair_runs_with_simulated_llm(settings):
    from fastapi.testclient import TestClient
    from backend.app.api.app import create_app
    import dataclasses
    c = TestClient(create_app(dataclasses.replace(settings)))
    sc, d = _load()
    sid = c.post("/scenarios/generate", json={"size": "small", "scarcity": "high", "seed": 42, "save": True}).json()["scenario_id"]
    r = c.post("/planning/repair", json={"scenario_id": sid, "plan": d, "max_repair_rounds": 3}).json()
    assert r["is_simulated"] is True
    assert r["n_violations_initial"] == 14 and r["n_violations_final"] < r["n_violations_initial"]
