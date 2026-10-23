"""Request-count guard for a free-tier demo: with LLM_MAX_RETRIES=0 and a call budget of 2, the REAL GeminiClient
(talking to a local fake server, never to Google) sends at most 2 HTTP requests, and a 429 stops immediately."""
import dataclasses
import json
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.api.app import create_app
from tests.test_gemini_client import Server

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample"


def _plan(name):
    d = json.loads((SAMPLE / name).read_text(encoding="utf-8"))
    d.pop("_label", None)
    return d


def _ok(plan):
    return (200, {"candidates": [{"content": {"parts": [{"text": json.dumps(plan)}]}, "finishReason": "STOP"}],
                  "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 50, "totalTokenCount": 150},
                  "modelVersion": "gemini-fake-001"}, {})


def _run(settings, script):
    srv = Server(script)
    try:
        s = dataclasses.replace(settings, llm_provider="gemini", gemini_api_key="K-TEST", gemini_model="gemini-3.5-flash-lite",
                                gemini_api_base=srv.base, llm_max_retries=0, request_delay_s=0.0)
        c = TestClient(create_app(s))
        sid = c.post("/scenarios/generate", json={"size": "small", "scarcity": "high", "seed": 42, "save": True}).json()["scenario_id"]
        r = c.post("/planning/verify", json={"scenario_id": sid, "max_repair_rounds": 1, "call_budget": 2, "compute_reference": False}).json()
        return r, srv.requests
    finally:
        srv.close()


def test_initial_valid_uses_one_request(settings):
    r, reqs = _run(settings, [_ok(_plan("sample_plan_valid_cpsat.json"))])
    assert len(reqs) == 1 and r["final_feasibility"] is True and r["provider"] == "gemini" and r["is_simulated"] is False


def test_invalid_then_repair_uses_two_requests(settings):
    r, reqs = _run(settings, [_ok(_plan("demo_multi_violation_plan.json")), _ok(_plan("sample_plan_valid_cpsat.json"))])
    assert len(reqs) == 2 and r["n_violations_initial"] > 0 and r["n_violations_final"] == 0
    assert "n_violations" in json.dumps(reqs[1]["body"])          # the validator's structured report reached the second request


def test_budget_caps_requests_even_if_plan_stays_invalid(settings):
    r, reqs = _run(settings, [_ok(_plan("demo_multi_violation_plan.json")), _ok(_plan("demo_multi_violation_plan.json")),
                              _ok(_plan("demo_multi_violation_plan.json"))])
    assert len(reqs) == 2 and r["final_feasibility"] is False


def test_quota_error_stops_after_one_request(settings):
    r, reqs = _run(settings, [(429, {"error": {"status": "RESOURCE_EXHAUSTED", "message": "quota"}}, {}),
                              _ok(_plan("sample_plan_valid_cpsat.json"))])
    assert len(reqs) == 1 and r["status"] == "error"
