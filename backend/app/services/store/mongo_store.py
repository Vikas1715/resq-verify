"""Optional MongoDB storage (pymongo).

Collections: scenarios, experiments, experiment_runs.
Every write is mirrored to the file store so research logs stay exportable without MongoDB
(set MONGO_MIRROR_TO_FILES=false in code if you really do not want this).
"""
from __future__ import annotations

from typing import Any, Optional

from backend.app.core.errors import DuplicateScenarioError, StorageError
from backend.app.services.store.base import Store, dedupe_runs
from backend.app.services.store.file_store import FileStore

_NO_ID = {"_id": 0}


class MongoStore(Store):
    backend = "mongo"

    def __init__(self, client: Any, database: str, mirror: Optional[FileStore] = None):
        self.client = client
        self.db = client[database]
        self.mirror = mirror
        self.scenarios = self.db["scenarios"]
        self.experiments = self.db["experiments"]
        self.runs = self.db["experiment_runs"]
        self._ensure_indexes()

    @classmethod
    def connect(cls, uri: str, database: str, mirror: Optional[FileStore] = None, timeout_ms: int = 3000) -> "MongoStore":
        try:
            from pymongo import MongoClient
        except ImportError as exc:
            raise StorageError("pymongo is not installed. Run: pip install pymongo") from exc
        client = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)
        try:
            client.admin.command("ping")
        except Exception as exc:  # noqa: BLE001 - any connection failure should be reported uniformly
            raise StorageError(f"Cannot reach MongoDB: {type(exc).__name__}: {exc}") from exc
        return cls(client, database, mirror)

    def _ensure_indexes(self) -> None:
        try:
            self.scenarios.create_index("scenario_id", unique=True)
            self.experiments.create_index("experiment_id", unique=True)
            self.runs.create_index([("experiment_id", 1), ("scenario_id", 1), ("method", 1), ("run_index", 1)])
            self.runs.create_index("run_id", unique=True)
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"Could not create MongoDB indexes: {exc}") from exc

    # ---- scenarios
    def save_scenario(self, scenario: dict) -> dict:
        existing = self.scenarios.find_one({"scenario_id": scenario["scenario_id"]}, _NO_ID)
        if existing is not None:
            if existing.get("content_hash") == scenario.get("content_hash"):
                return existing
            raise DuplicateScenarioError(f"Scenario id '{scenario['scenario_id']}' already exists with different content.")
        self.scenarios.insert_one(dict(scenario))
        if self.mirror:
            self.mirror.save_scenario(scenario)
        return scenario

    def get_scenario(self, scenario_id: str) -> Optional[dict]:
        return self.scenarios.find_one({"scenario_id": scenario_id}, _NO_ID)

    def list_scenarios(self, limit: int = 500) -> list[dict]:
        out = []
        for d in self.scenarios.find({}, _NO_ID).sort("created_at", -1).limit(limit):
            out.append({k: d.get(k) for k in ("scenario_id", "size", "scarcity", "seed", "content_hash", "created_at")}
                       | {"n_incidents": len(d.get("incidents", [])), "difficulty_score": (d.get("difficulty") or {}).get("difficulty_score")})
        return out

    # ---- experiments
    def create_experiment(self, doc: dict) -> dict:
        self.experiments.insert_one(dict(doc))
        if self.mirror:
            self.mirror.create_experiment(doc)
        return doc

    def get_experiment(self, experiment_id: str) -> Optional[dict]:
        return self.experiments.find_one({"experiment_id": experiment_id}, _NO_ID)

    def update_experiment(self, experiment_id: str, patch: dict) -> Optional[dict]:
        doc = self.get_experiment(experiment_id)
        if doc is None:
            return None
        for k, v in patch.items():
            if isinstance(v, dict) and isinstance(doc.get(k), dict) and k in ("progress", "references"):
                doc[k].update(v)
            else:
                doc[k] = v
        self.experiments.replace_one({"experiment_id": experiment_id}, dict(doc))
        if self.mirror:
            self.mirror.update_experiment(experiment_id, patch)
        return doc

    def list_experiments(self, limit: int = 200) -> list[dict]:
        return list(self.experiments.find({}, _NO_ID).sort("created_at", -1).limit(limit))

    # ---- runs
    def append_run(self, run: dict) -> None:
        self.runs.insert_one(dict(run))
        if self.mirror:
            self.mirror.append_run(run)

    def list_runs(self, experiment_id: str) -> list[dict]:
        return dedupe_runs(list(self.runs.find({"experiment_id": experiment_id}, _NO_ID).sort("start_time", 1)))

    def get_run(self, run_id: str) -> Optional[dict]:
        return self.runs.find_one({"run_id": run_id}, _NO_ID)

    def health(self) -> dict:
        try:
            self.client.admin.command("ping")
            return {"backend": "mongo", "ok": True, "database": self.db.name}
        except Exception as exc:  # noqa: BLE001
            return {"backend": "mongo", "ok": False, "error": str(exc)}
