"""Interactive (single-scenario) runs for the Planning Lab. They are stored as runs of a persistent
experiment with kind="adhoc", so every click is reproducible and exportable, but never mixed into
experiment statistics unless the user asks for it."""
from __future__ import annotations

import json
import time
from typing import Any, Optional

from backend.app.core.errors import BadRequestError, NotFoundError
from backend.app.experiments.records import build_run_record
from backend.app.experiments.runner import ExperimentRunner, method_budgets
from backend.app.schemas.experiment import ExperimentConfig
from backend.app.schemas.plan import Plan
from backend.app.schemas.scenario import Scenario
from backend.app.services.methods import (
    Attempt, CallStats, LLM_METHODS, MethodContext, generate_initial, run_llm_only, run_resq_verify, run_self_critique,
)
from backend.app.services.parsing import parse_plan
from backend.app.services.store.base import Store
from backend.app.utils.ids import utcnow_iso

ADHOC_ID = "ADHOC-interactive"


class AdhocService:
    def __init__(self, store: Store, runner: ExperimentRunner):
        self.store, self.runner = store, runner

    # ------------------------------------------------------------------ helpers
    def scenario(self, scenario_id: str) -> Scenario:
        d = self.store.get_scenario(scenario_id)
        if d is None:
            raise NotFoundError(f"Scenario {scenario_id} not found. Generate it first.")
        return Scenario.model_validate(d)

    def _ensure_experiment(self, cfg: ExperimentConfig) -> None:
        if self.store.get_experiment(ADHOC_ID) is None:
            self.runner.create(cfg.model_copy(update={"name": "Interactive runs (Planning Lab)",
                                                      "description": "Single runs started from the UI/API."}),
                               kind="adhoc", experiment_id=ADHOC_ID)
            self.store.update_experiment(ADHOC_ID, {"status": "open", "message": "Container for interactive runs."})

    def _next_index(self, scenario_id: str, method: str) -> int:
        return sum(1 for r in self.store.list_runs(ADHOC_ID) if r["scenario_id"] == scenario_id and r["method"] == method)

    def _cfg(self, methods: list[str], **kw: Any) -> ExperimentConfig:
        clean = {k: v for k, v in kw.items() if v is not None}
        return ExperimentConfig(name="interactive", methods=methods, **clean)

    def _reference(self, sc: Scenario, cfg: ExperimentConfig) -> Optional[dict]:
        return self.runner._reference_only(sc, cfg) if cfg.compute_reference else None  # noqa: SLF001

    # ------------------------------------------------------------------ operations
    def run_method(self, scenario_id: str, method: str, *, max_repair_rounds: Optional[int] = None, feedback_format: Optional[str] = None,
                   temperature: Optional[float] = None, compute_reference: bool = True, call_budget: Optional[int] = None) -> dict:
        if method not in LLM_METHODS:
            raise BadRequestError(f"method must be one of {LLM_METHODS}")
        sc = self.scenario(scenario_id)
        cfg = self._cfg([method], max_repair_rounds=max_repair_rounds, feedback_format=feedback_format, temperature=temperature,
                        compute_reference=compute_reference, call_budgets=[call_budget] if call_budget else None)
        self._ensure_experiment(cfg)
        llm = self.runner.llm_factory(self.runner.settings)
        ref = self._reference(sc, cfg)
        idx = self._next_index(scenario_id, method)
        recs = self.runner.run_scenario_rep(ADHOC_ID, sc, cfg, idx, [(method, b) for b in method_budgets(cfg, method)], llm, ref)
        return recs[0]

    def compare(self, scenario_id: str, *, max_repair_rounds: Optional[int] = None, feedback_format: Optional[str] = None,
                temperature: Optional[float] = None, methods: Optional[list[str]] = None, call_budget: Optional[int] = None) -> list[dict]:
        sc = self.scenario(scenario_id)
        methods = methods or ["llm_only", "self_critique", "resq_verify", "resample_verify", "ortools"]
        cfg = self._cfg(methods, max_repair_rounds=max_repair_rounds, feedback_format=feedback_format, temperature=temperature,
                        call_budgets=[call_budget] if call_budget else None)
        self._ensure_experiment(cfg)
        out: list[dict] = []
        ref = self._reference(sc, cfg)
        if "ortools" in cfg.methods:
            rec = self.runner.run_solver(ADHOC_ID, sc, cfg, ref)
            out.append(rec)
            if rec["status"] == "ok":
                ref = self.runner.reference_from(rec)
        llm_methods = [m for m in cfg.methods if m in LLM_METHODS]
        if llm_methods:
            llm = self.runner.llm_factory(self.runner.settings)
            idx = max(self._next_index(scenario_id, m) for m in llm_methods)
            todo = [(m, b) for m in llm_methods for b in method_budgets(cfg, m)]
            out = self.runner.run_scenario_rep(ADHOC_ID, sc, cfg, idx, todo, llm, ref) + out
        return out

    def repair_given_plan(self, scenario_id: str, plan: Any, *, max_repair_rounds: Optional[int] = None,
                          feedback_format: Optional[str] = None, temperature: Optional[float] = None) -> dict:
        """Run the verify-and-repair loop starting from a plan supplied by the user (e.g. an edited or hand-broken plan)."""
        sc = self.scenario(scenario_id)
        cfg = self._cfg(["resq_verify"], max_repair_rounds=max_repair_rounds, feedback_format=feedback_format, temperature=temperature)
        self._ensure_experiment(cfg)
        raw = plan if isinstance(plan, str) else json.dumps(plan)
        res = parse_plan(raw)
        parsed, err = res.plan, res.error
        llm = self.runner.llm_factory(self.runner.settings)
        mc = self.runner.method_config(cfg, self._next_index(scenario_id, "resq_verify"))
        ctx = MethodContext(llm, self.runner.prompts, mc, sleep=self.runner.sleep)
        t0, start = time.perf_counter(), utcnow_iso()
        outcome = run_resq_verify(ctx, sc, Attempt(parsed, raw, err, CallStats()))
        rec = build_run_record(experiment_id=ADHOC_ID, scenario=sc, outcome=outcome, run_index=mc.run_index, start_time=start,
                               end_time=utcnow_iso(), wall_time_s=time.perf_counter() - t0, settings=self.runner.settings,
                               llm_meta=llm.metadata(), method_cfg=mc.__dict__,
                               prompt_meta={"version": self.runner.prompts.version, "hash": self.runner.prompts.content_hash,
                                          "template_hashes": self.runner.prompts.template_hashes},
                               reference=self._reference(sc, cfg), shared_initial_plan=False)
        rec["note"] = "repair started from a user-supplied plan"
        self.store.append_run(rec)
        return rec
