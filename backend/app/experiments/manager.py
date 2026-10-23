"""Runs experiments in background threads so the API returns immediately; tracks cancel flags."""
from __future__ import annotations

import logging
import threading
from typing import Optional

from backend.app.core.errors import ExperimentStateError, NotFoundError
from backend.app.experiments.runner import ExperimentRunner
from backend.app.schemas.experiment import ExperimentConfig
from backend.app.services.store.base import Store
from backend.app.utils.ids import utcnow_iso

log = logging.getLogger("resq.manager")


class ExperimentManager:
    def __init__(self, store: Store, runner: ExperimentRunner):
        self.store, self.runner = store, runner
        self._threads: dict[str, threading.Thread] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def mark_interrupted(self) -> int:
        """Called at startup: experiments left 'running'/'queued' by a previous process were interrupted."""
        n = 0
        for e in self.store.list_experiments(limit=10_000):
            if e.get("status") in ("running", "queued"):
                self.store.update_experiment(e["experiment_id"], {"status": "interrupted", "finished_at": utcnow_iso(),
                                                                   "message": "The server stopped while this experiment was running. Use resume to continue."})
                n += 1
        return n

    def is_running(self, eid: str) -> bool:
        t = self._threads.get(eid)
        return bool(t and t.is_alive())

    def start(self, cfg: ExperimentConfig) -> dict:
        doc = self.runner.create(cfg)
        self._launch(doc["experiment_id"])
        return self.store.get_experiment(doc["experiment_id"])

    def resume(self, eid: str) -> dict:
        doc = self.store.get_experiment(eid)
        if doc is None:
            raise NotFoundError(f"Experiment {eid} not found")
        if self.is_running(eid):
            raise ExperimentStateError(f"Experiment {eid} is already running")
        self._launch(eid)
        return self.store.get_experiment(eid)

    def cancel(self, eid: str) -> dict:
        ev = self._cancel.get(eid)
        if ev is None or not self.is_running(eid):
            raise ExperimentStateError(f"Experiment {eid} is not running")
        ev.set()
        self.store.update_experiment(eid, {"message": "Cancelling after the current run finishes..."})
        return self.store.get_experiment(eid)

    def _launch(self, eid: str) -> None:
        with self._lock:
            ev = threading.Event()
            self._cancel[eid] = ev
            t = threading.Thread(target=self._run, args=(eid, ev), name=f"exp-{eid}", daemon=True)
            self._threads[eid] = t
            self.store.update_experiment(eid, {"status": "queued"})
            t.start()

    def _run(self, eid: str, ev: threading.Event) -> None:
        try:
            self.runner.run(eid, ev)
        except Exception as exc:  # noqa: BLE001
            log.exception("experiment thread crashed")
            self.store.update_experiment(eid, {"status": "failed", "finished_at": utcnow_iso(), "message": f"{type(exc).__name__}: {exc}"})

    def join(self, eid: str, timeout: Optional[float] = None) -> None:
        t = self._threads.get(eid)
        if t:
            t.join(timeout)
