import json

import pytest

from backend.app.core.errors import DuplicateScenarioError
from backend.app.services.scenario_generator import generate_scenario
from backend.app.services.store.base import dedupe_runs
from backend.app.services.store.factory import make_store
from backend.app.services.store.file_store import FileStore
from backend.app.services.store.mongo_store import MongoStore
from tests.fakes import FakeMongoClient


def run(i, status="ok", method="llm_only", scenario="s1", exp="E1"):
    return {"run_id": f"r{i}", "experiment_id": exp, "scenario_id": scenario, "method": method, "run_index": 0, "status": status,
            "start_time": f"2026-01-01T00:00:0{i}"}


@pytest.fixture(params=["file", "mongo"])
def store(request, tmp_path):
    if request.param == "file":
        return FileStore(tmp_path / "d")
    return MongoStore(FakeMongoClient(), "testdb", mirror=FileStore(tmp_path / "mirror"))


def test_scenario_roundtrip_idempotent_and_conflict(store):
    sc = generate_scenario("small", "high", 1).model_dump()
    store.save_scenario(sc)
    assert store.get_scenario(sc["scenario_id"])["content_hash"] == sc["content_hash"]
    store.save_scenario(sc)                       # same content -> fine
    assert len(store.list_scenarios()) == 1
    changed = {**sc, "content_hash": "different"}
    with pytest.raises(DuplicateScenarioError):
        store.save_scenario(changed)
    assert store.get_scenario("missing") is None


def test_experiment_lifecycle_and_progress_merge(store):
    store.create_experiment({"experiment_id": "E1", "status": "queued", "created_at": "2026-01-01", "progress": {"done": 0, "total": 4, "errors": 0}})
    store.update_experiment("E1", {"status": "running", "progress": {"done": 2}})
    e = store.get_experiment("E1")
    assert e["status"] == "running" and e["progress"] == {"done": 2, "total": 4, "errors": 0}
    assert store.update_experiment("nope", {"status": "x"}) is None
    assert [x["experiment_id"] for x in store.list_experiments()] == ["E1"]


def test_runs_dedupe_latest_wins_and_errors_are_retried(store):
    store.create_experiment({"experiment_id": "E1", "status": "queued", "created_at": "2026-01-01"})
    store.append_run(run(1, status="error"))
    assert store.completed_run_keys("E1") == set()          # an errored run is not "done"
    store.append_run(run(2, status="ok"))
    runs = store.list_runs("E1")
    assert len(runs) == 1 and runs[0]["status"] == "ok"
    assert store.completed_run_keys("E1") == {("s1", "llm_only", 0)}
    assert store.get_run("r2")["run_id"] == "r2" and store.get_run("zzz") is None


def test_file_store_skips_corrupted_lines(tmp_path):
    st = FileStore(tmp_path)
    st.create_experiment({"experiment_id": "E1", "status": "queued", "created_at": "x"})
    st.append_run(run(1))
    path = st._runs_path("E1")
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"truncated": \n')
    st.append_run(run(2, method="resq_verify"))
    assert len(st.list_runs("E1")) == 2
    assert st.health()["corrupted_records_seen"] == 1


def test_dedupe_function():
    assert len(dedupe_runs([run(1, "error"), run(2, "ok")])) == 1


def test_mongo_falls_back_to_files_when_unreachable(settings):
    import dataclasses
    s = dataclasses.replace(settings, storage_backend="mongo", mongodb_uri="mongodb://127.0.0.1:1/?serverSelectionTimeoutMS=300",
                            mongo_fallback_to_file=True)
    store, info = make_store(s)
    assert store.backend == "file" and info["requested"] == "mongo" and info["active"] == "file" and info["note"]


def test_mongo_without_fallback_raises(settings):
    import dataclasses
    from backend.app.core.errors import StorageError
    s = dataclasses.replace(settings, storage_backend="mongo", mongodb_uri=None, mongo_fallback_to_file=False)
    with pytest.raises(StorageError):
        make_store(s)
