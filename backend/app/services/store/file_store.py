"""Plain-file storage: JSON for scenarios/experiments, JSONL (one run per line) for runs.

Writes are atomic (temp file + rename). Corrupted JSONL lines are skipped and counted, never fatal.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Optional

from backend.app.core.errors import DuplicateScenarioError, StorageError
from backend.app.services.store.base import Store, dedupe_runs


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp, path)
    except OSError as exc:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise StorageError(f"Could not write {path}: {exc}") from exc


class FileStore(Store):
    backend = "file"

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.scn_dir = self.data_dir / "scenarios"
        self.res_dir = self.data_dir / "results"
        self.scn_dir.mkdir(parents=True, exist_ok=True)
        self.res_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.corrupted_records = 0

    # ---- scenarios
    def save_scenario(self, scenario: dict) -> dict:
        sid = scenario["scenario_id"]
        path = self.scn_dir / f"{sid}.json"
        with self._lock:
            existing = self.get_scenario(sid)
            if existing is not None:
                if existing.get("content_hash") == scenario.get("content_hash"):
                    return existing
                raise DuplicateScenarioError(
                    f"Scenario id '{sid}' already exists with different content. Use a different seed or overrides.",
                    details={"scenario_id": sid})
            _atomic_write(path, json.dumps(scenario, indent=1, ensure_ascii=False))
        return scenario

    def get_scenario(self, scenario_id: str) -> Optional[dict]:
        path = self.scn_dir / f"{scenario_id}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.corrupted_records += 1
            return None

    def list_scenarios(self, limit: int = 500) -> list[dict]:
        out = []
        for p in sorted(self.scn_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:limit]:
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self.corrupted_records += 1
                continue
            out.append({k: d.get(k) for k in ("scenario_id", "size", "scarcity", "seed", "content_hash", "created_at")}
                       | {"n_incidents": len(d.get("incidents", [])), "difficulty_score": (d.get("difficulty") or {}).get("difficulty_score")})
        return out

    # ---- experiments
    def _exp_path(self, eid: str) -> Path:
        return self.res_dir / eid / "experiment.json"

    def create_experiment(self, doc: dict) -> dict:
        with self._lock:
            if self._exp_path(doc["experiment_id"]).exists():
                raise StorageError(f"Experiment {doc['experiment_id']} already exists")
            _atomic_write(self._exp_path(doc["experiment_id"]), json.dumps(doc, indent=1, ensure_ascii=False, default=str))
        return doc

    def get_experiment(self, experiment_id: str) -> Optional[dict]:
        p = self._exp_path(experiment_id)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.corrupted_records += 1
            return None

    def update_experiment(self, experiment_id: str, patch: dict) -> Optional[dict]:
        with self._lock:
            doc = self.get_experiment(experiment_id)
            if doc is None:
                return None
            for k, v in patch.items():
                if isinstance(v, dict) and isinstance(doc.get(k), dict) and k in ("progress", "references"):
                    doc[k].update(v)
                else:
                    doc[k] = v
            _atomic_write(self._exp_path(experiment_id), json.dumps(doc, indent=1, ensure_ascii=False, default=str))
            return doc

    def list_experiments(self, limit: int = 200) -> list[dict]:
        out = []
        for p in self.res_dir.glob("*/experiment.json"):
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                self.corrupted_records += 1
        out.sort(key=lambda d: d.get("created_at", ""), reverse=True)
        return out[:limit]

    # ---- runs
    def _runs_path(self, eid: str) -> Path:
        return self.res_dir / eid / "runs.jsonl"

    def append_run(self, run: dict) -> None:
        line = json.dumps(run, ensure_ascii=False, default=str)
        p = self._runs_path(run["experiment_id"])
        with self._lock:
            p.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(p, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
                    f.flush()
                    os.fsync(f.fileno())
            except OSError as exc:
                raise StorageError(f"Could not append run to {p}: {exc}") from exc

    def list_runs(self, experiment_id: str) -> list[dict]:
        p = self._runs_path(experiment_id)
        if not p.exists():
            return []
        runs = []
        with self._lock, open(p, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    runs.append(json.loads(line))
                except json.JSONDecodeError:
                    self.corrupted_records += 1
        return dedupe_runs(runs)

    def get_run(self, run_id: str) -> Optional[dict]:
        for e in self.list_experiments(limit=10_000):
            for r in self.list_runs(e["experiment_id"]):
                if r.get("run_id") == run_id:
                    return r
        return None

    def health(self) -> dict:
        return {"backend": "file", "ok": os.access(self.data_dir, os.W_OK), "data_dir": str(self.data_dir),
                "corrupted_records_seen": self.corrupted_records}
