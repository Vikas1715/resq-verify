"""Thin HTTP client for the FastAPI backend. The UI never touches storage or the LLM directly."""
from __future__ import annotations

import os
from typing import Any, Optional

import requests

DEFAULT_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


class ApiError(Exception):
    """Readable error for the UI: .message is safe to show to the user."""

    def __init__(self, message: str, code: str = "ERROR", status: Optional[int] = None):
        super().__init__(message)
        self.message, self.code, self.status = message, code, status


class Api:
    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 600.0, session: Optional[requests.Session] = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.s = session or requests.Session()

    def _req(self, method: str, path: str, *, raw: bool = False, timeout: Optional[float] = None, **kw) -> Any:
        try:
            r = self.s.request(method, self.base_url + path, timeout=timeout or self.timeout, **kw)
        except requests.ConnectionError as exc:
            raise ApiError(f"Cannot reach the backend at {self.base_url}. Start it with: uvicorn backend.app.api.app:app --port 8000",
                           "BACKEND_DOWN") from exc
        except requests.Timeout as exc:
            raise ApiError("The backend took too long to answer (the LLM may be slow). Try again or lower the rounds.", "TIMEOUT") from exc
        if r.status_code >= 400:
            try:
                err = r.json().get("error", {})
            except ValueError:
                err = {}
            detail = err.get("details", {}).get("errors")
            msg = err.get("message") or r.text[:300] or f"HTTP {r.status_code}"
            if detail:
                msg += " — " + "; ".join(f"{e.get('loc')}: {e.get('msg')}" for e in detail[:3])
            raise ApiError(msg, err.get("code", "ERROR"), r.status_code)
        return r.content if raw else r.json()

    # health / config
    def health(self): return self._req("GET", "/health", timeout=5)
    # scenarios
    def generate_scenario(self, size, scarcity, seed, overrides=None, difficulty_level=None):
        return self._req("POST", "/scenarios/generate", json={"size": size, "scarcity": scarcity, "seed": seed, "overrides": overrides or {},
                                                              "difficulty_level": difficulty_level})
    def scenarios(self): return self._req("GET", "/scenarios")["scenarios"]
    def scenario(self, sid): return self._req("GET", f"/scenarios/{sid}")
    # validation / solver
    def validate(self, scenario_id, *, plan=None, raw_text=None):
        return self._req("POST", "/validate", json={"scenario_id": scenario_id, "plan": plan, "raw_text": raw_text})
    def solve(self, scenario_id, time_limit_s=None): return self._req("POST", "/solve", json={"scenario_id": scenario_id, "time_limit_s": time_limit_s})
    # planning
    def plan(self, kind: str, scenario_id: str, **opts):
        return self._req("POST", f"/planning/{kind}", json={"scenario_id": scenario_id, **opts})
    def repair(self, scenario_id, plan, **opts): return self._req("POST", "/planning/repair", json={"scenario_id": scenario_id, "plan": plan, **opts})
    def compare(self, scenario_id, **opts): return self._req("POST", "/planning/compare", json={"scenario_id": scenario_id, **opts})
    # experiments
    def config_files(self): return self._req("GET", "/experiments/configs")["files"]
    def config_file(self, name): return self._req("GET", f"/experiments/configs/{name}")
    def run_experiment(self, config=None, config_file=None): return self._req("POST", "/experiments/run", json={"config": config, "config_file": config_file})
    def experiments(self, include_adhoc=True): return self._req("GET", "/experiments", params={"include_adhoc": include_adhoc})["experiments"]
    def experiment(self, eid): return self._req("GET", f"/experiments/{eid}")
    def cancel(self, eid): return self._req("POST", f"/experiments/{eid}/cancel")
    def resume(self, eid): return self._req("POST", f"/experiments/{eid}/resume")
    def runs(self, eid, **params): return self._req("GET", f"/experiments/{eid}/runs", params=params)["runs"]
    def run(self, run_id): return self._req("GET", f"/runs/{run_id}")
    # analytics / exports
    def summary(self, experiment_id=None, include_adhoc=False, call_budget=None):
        params = {"experiment_id": experiment_id, "include_adhoc": include_adhoc}
        if call_budget is not None:
            params["call_budget"] = call_budget
        return self._req("GET", "/analytics/summary", params=params)
    def analytics_runs(self, experiment_id=None, include_adhoc=False):
        return self._req("GET", "/analytics/runs", params={"experiment_id": experiment_id, "include_adhoc": include_adhoc})["rows"]
    def export(self, filename, experiment_id=None, include_adhoc=False):
        return self._req("GET", f"/exports/{filename}", raw=True, params={"experiment_id": experiment_id, "include_adhoc": include_adhoc})
