"""Storage interface. Two implementations: FileStore (default, zero setup) and MongoStore (optional)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

RunKey = tuple  # (scenario_id, method, run_index) or (scenario_id, method, run_index, call_budget)


def run_key(run: dict[str, Any]) -> RunKey:
    """Identity of a run. The call budget is part of it, so one scenario can hold one run per budget.
    Records without a budget (the solver, or files written by an older version) keep the 3-part key."""
    base = (run.get("scenario_id"), run.get("method"), run.get("run_index", 0))
    return base if run.get("call_budget") is None else base + (run["call_budget"],)


class Store(ABC):
    backend = "base"

    # scenarios
    @abstractmethod
    def save_scenario(self, scenario: dict) -> dict: ...
    @abstractmethod
    def get_scenario(self, scenario_id: str) -> Optional[dict]: ...
    @abstractmethod
    def list_scenarios(self, limit: int = 500) -> list[dict]: ...

    # experiments
    @abstractmethod
    def create_experiment(self, doc: dict) -> dict: ...
    @abstractmethod
    def update_experiment(self, experiment_id: str, patch: dict) -> Optional[dict]: ...
    @abstractmethod
    def get_experiment(self, experiment_id: str) -> Optional[dict]: ...
    @abstractmethod
    def list_experiments(self, limit: int = 200) -> list[dict]: ...

    # runs
    @abstractmethod
    def append_run(self, run: dict) -> None: ...
    @abstractmethod
    def list_runs(self, experiment_id: str) -> list[dict]: ...
    @abstractmethod
    def get_run(self, run_id: str) -> Optional[dict]: ...

    @abstractmethod
    def health(self) -> dict: ...

    # helpers shared by all backends
    def completed_run_keys(self, experiment_id: str) -> set[RunKey]:
        """Keys of runs that finished with status 'ok'. Errored runs are retried on resume."""
        return {run_key(r) for r in self.list_runs(experiment_id) if r.get("status") == "ok"}

    def list_all_runs(self, kind: Optional[str] = None) -> list[dict]:
        runs: list[dict] = []
        for e in self.list_experiments(limit=10_000):
            if kind and e.get("kind") != kind:
                continue
            runs.extend(self.list_runs(e["experiment_id"]))
        return runs


def dedupe_runs(runs: list[dict]) -> list[dict]:
    """Keep the latest record per (scenario, method, run_index): a retried run replaces its earlier failed attempt."""
    latest: dict[RunKey, dict] = {}
    for r in runs:
        latest[run_key(r)] = r
    return list(latest.values())
