"""Experiment runner: paired design (every scenario goes through every selected method), resumable, interruptible."""
from __future__ import annotations

import dataclasses
import logging
import threading
import time
from typing import Any, Callable, Optional

from backend.app.core.config import Settings
from backend.app.core.errors import ExperimentStateError, LLMNotConfiguredError, NotFoundError, ResQError
from backend.app.experiments.records import build_run_record
from backend.app.schemas.experiment import ExperimentConfig
from backend.app.schemas.scenario import Scenario
from backend.app.services.llm.base import LLMClient
from backend.app.services.llm.factory import make_llm
from backend.app.services.methods import (
    BUDGETED_METHODS, LLM_METHODS, RUNNERS, MethodConfig, MethodContext, Outcome, effective_budget, generate_initial,
    run_ortools, truncate_outcome,
)
from backend.app.services.prompts import PromptBook
from backend.app.services.scenario_generator import generate_level_scenario, generate_scenario
from backend.app.services.store.base import Store
from backend.app.solvers.cpsat_solver import solve_scenario
from backend.app.utils.ids import new_experiment_id, utcnow_iso

log = logging.getLogger("resq.runner")


def planned_scenarios(cfg: ExperimentConfig) -> list[tuple[str, str, int, Optional[str]]]:
    """(size, scarcity, seed, level). With difficulty_levels set, the levels replace the size x scarcity grid."""
    if cfg.difficulty_levels:
        return [("", "", cfg.base_seed + k, lv) for lv in cfg.difficulty_levels for k in range(cfg.num_scenarios)]
    return [(size, sc, cfg.base_seed + k, None) for size in cfg.sizes for sc in cfg.scarcity for k in range(cfg.num_scenarios)]


def method_budgets(cfg: ExperimentConfig, method: str, mc: Optional[MethodConfig] = None) -> list[Optional[int]]:
    """Call budgets a method is run under. M1 can only ever use one call, so it is run once (analytics shows it at every budget).
    Without call_budgets the single K-based budget is used."""
    mc = mc or MethodConfig(max_repair_rounds=cfg.max_repair_rounds, self_critique_rounds=cfg.critique_rounds)
    if method == "llm_only" or not cfg.call_budgets:
        return [effective_budget(method, mc)]
    return list(cfg.call_budgets)


def planned_total(cfg: ExperimentConfig) -> int:
    per_scenario = (1 if "ortools" in cfg.methods else 0)
    per_scenario += cfg.repetitions * sum(len(method_budgets(cfg, m)) for m in cfg.methods if m in LLM_METHODS)
    return len(planned_scenarios(cfg)) * per_scenario


def _done(done_keys: set, sid: str, method: str, rep: int, budget: Optional[int]) -> bool:
    """A run is done if its budgeted key is stored (or the 3-part key an older version wrote)."""
    return (sid, method, rep, budget) in done_keys or (sid, method, rep) in done_keys


class ExperimentRunner:
    def __init__(self, store: Store, settings: Settings, *, llm_factory: Callable[[Settings], LLMClient] = make_llm,
                 sleep: Callable[[float], None] = time.sleep):
        self.store, self.settings, self.llm_factory, self.sleep = store, settings, llm_factory, sleep
        self._prompts: Optional[PromptBook] = None

    @property
    def prompts(self) -> PromptBook:
        if self._prompts is None:
            self._prompts = PromptBook(self.settings.prompt_dir)
        return self._prompts

    # ------------------------------------------------------------------ setup
    def method_config(self, cfg: ExperimentConfig, run_index: int = 0, call_budget: Optional[int] = None) -> MethodConfig:
        s = self.settings
        return MethodConfig(call_budget=call_budget,
            temperature=cfg.temperature if cfg.temperature is not None else s.temperature,
            max_output_tokens=cfg.max_output_tokens or s.max_output_tokens,
            max_repair_rounds=cfg.max_repair_rounds, self_critique_rounds=cfg.critique_rounds,
            feedback_format=cfg.feedback_format, base_seed=cfg.base_seed, run_index=run_index,
            request_delay_s=cfg.request_delay_s if cfg.request_delay_s is not None else s.request_delay_s)

    def create(self, cfg: ExperimentConfig, *, kind: str = "experiment", experiment_id: Optional[str] = None) -> dict:
        eid = experiment_id or new_experiment_id()
        uses_llm = any(m in LLM_METHODS for m in cfg.methods)
        llm_block: dict[str, Any] = {"provider": None, "model": None, "is_simulated": False}
        if uses_llm:
            llm_block = {"provider": self.settings.llm_provider,
                         "model": self.settings.gemini_model if self.settings.llm_provider == "gemini" else "simulated-llm",
                         "is_simulated": self.settings.llm_provider == "mock"}
        mc = self.method_config(cfg)
        doc = {
            "experiment_id": eid, "kind": kind, "name": cfg.name, "description": cfg.description, "status": "queued",
            "created_at": utcnow_iso(), "started_at": None, "finished_at": None,
            "config": cfg.model_dump(), "llm": {**llm_block, "temperature": mc.temperature, "max_output_tokens": mc.max_output_tokens},
            "prompt_version": self.prompts.version if uses_llm else None, "prompt_hash": self.prompts.content_hash if uses_llm else None,
            "prompt_template_hashes": self.prompts.template_hashes if uses_llm else None,
            "is_simulated": llm_block["is_simulated"], "storage_backend": self.store.backend,
            "progress": {"done": 0, "total": planned_total(cfg), "errors": 0},
            "references": {}, "message": "", "settings": self.settings.public_dict(),
        }
        return self.store.create_experiment(doc)

    # ------------------------------------------------------------------ one scenario x one repetition
    def run_scenario_rep(self, experiment_id: str, scenario: Scenario, cfg: ExperimentConfig, run_index: int,
                         todo: list[tuple[str, Optional[int]]], llm: Optional[LLMClient], reference: Optional[dict],
                         cancel: Optional[threading.Event] = None) -> list[dict]:
        """Run every (method, budget) still to do for one scenario and repetition.

        With call_budgets and nested_budgets, a budgeted method runs ONCE at the largest budget it still needs and the smaller
        budgets are cut from that trajectory (truncate_outcome); otherwise each budget is run on its own.
        """
        records: list[dict] = []
        mc = self.method_config(cfg, run_index)
        llm_meta = llm.metadata() if llm else {"provider": None, "model": None, "is_simulated": False}
        prompt_meta = {"version": self.prompts.version, "hash": self.prompts.content_hash,
                       "template_hashes": self.prompts.template_hashes}
        ctx = MethodContext(llm, self.prompts, mc, sleep=self.sleep) if llm else None
        initial = None
        if todo and cfg.shared_initial_plan:
            initial = generate_initial(ctx, scenario)
        by_method: dict[str, list[Optional[int]]] = {}
        for m, b in todo:
            by_method.setdefault(m, []).append(b)
        for m, budgets in by_method.items():
            if cancel is not None and cancel.is_set():
                break
            nested = bool(cfg.uses_budgets and cfg.nested_budgets and m in BUDGETED_METHODS and len(budgets) > 1)
            plan_runs = [(max(budgets), budgets)] if nested else [(b, [b]) for b in budgets]
            for run_budget, wanted in plan_runs:
                if cancel is not None and cancel.is_set():
                    break
                mcb = dataclasses.replace(mc, call_budget=run_budget if cfg.uses_budgets and m != "llm_only" else None)
                cctx = MethodContext(llm, self.prompts, mcb, sleep=self.sleep)
                t0, start = time.perf_counter(), utcnow_iso()
                try:
                    init = initial if initial is not None else generate_initial(cctx, scenario)
                    outcome = RUNNERS[m](cctx, scenario, init)
                except Exception as exc:  # noqa: BLE001 - one bad run must never stop the experiment
                    log.exception("run failed")
                    outcome = Outcome(method=m, status="error", stop_reason="exception", failure_reason=f"exception: {type(exc).__name__}: {exc}",
                                      call_budget=run_budget)
                wall = time.perf_counter() - t0
                mode = "legacy_k" if not cfg.uses_budgets else ("nested" if nested else "independent")
                for b in wanted:
                    out_b = truncate_outcome(outcome, scenario, b) if (nested and b != run_budget) else outcome
                    rec = build_run_record(experiment_id=experiment_id, scenario=scenario, outcome=out_b, run_index=run_index,
                                           start_time=start, end_time=utcnow_iso(), wall_time_s=wall,
                                           settings=self.settings, llm_meta=llm_meta, method_cfg=mcb.__dict__, prompt_meta=prompt_meta,
                                           reference=reference, shared_initial_plan=cfg.shared_initial_plan,
                                           budget_mode=mode, derived_from_budget=(run_budget if (nested and b != run_budget) else None))
                    self.store.append_run(rec)
                    records.append(rec)
        return records

    def run_solver(self, experiment_id: str, scenario: Scenario, cfg: ExperimentConfig, reference: Optional[dict]) -> dict:
        t0, start = time.perf_counter(), utcnow_iso()
        try:
            s = self.settings.model_copy() if hasattr(self.settings, "model_copy") else self.settings
            if cfg.solver_time_limit_s:
                import dataclasses
                s = dataclasses.replace(self.settings, solver_time_limit_s=cfg.solver_time_limit_s)
            outcome = run_ortools(scenario, s, seed=cfg.base_seed)
        except ResQError as exc:
            outcome = Outcome(method="ortools", status="error", stop_reason="solver_error", failure_reason=f"solver_error: {exc.message}")
        except Exception as exc:  # noqa: BLE001
            outcome = Outcome(method="ortools", status="error", stop_reason="exception", failure_reason=f"exception: {type(exc).__name__}: {exc}")
        rec = build_run_record(experiment_id=experiment_id, scenario=scenario, outcome=outcome, run_index=0, start_time=start,
                               end_time=utcnow_iso(), wall_time_s=time.perf_counter() - t0, settings=self.settings,
                               llm_meta={"provider": None, "model": None, "is_simulated": False}, method_cfg=None, prompt_meta=None,
                               reference=reference, shared_initial_plan=None)
        self.store.append_run(rec)
        return rec

    @staticmethod
    def reference_from(rec_or_res: dict) -> dict:
        s = rec_or_res.get("solver") or rec_or_res
        return {"objective_weighted_served": s.get("objective_weighted_served"), "objective_normalized": s.get("objective_normalized"),
                "status": s.get("status"), "optimal": s.get("optimal"), "runtime_s": s.get("runtime_s"),
                "relative_gap": s.get("relative_gap")}

    # ------------------------------------------------------------------ full experiment
    def run(self, experiment_id: str, cancel: Optional[threading.Event] = None) -> dict:
        doc = self.store.get_experiment(experiment_id)
        if doc is None:
            raise NotFoundError(f"Experiment {experiment_id} not found")
        cfg = ExperimentConfig.model_validate(doc["config"])
        done_keys = self.store.completed_run_keys(experiment_id)
        refs: dict[str, dict] = dict(doc.get("references") or {})
        total = planned_total(cfg)
        done = sum(1 for k in done_keys)
        errors = 0
        self.store.update_experiment(experiment_id, {"status": "running", "started_at": doc.get("started_at") or utcnow_iso(),
                                                      "message": "", "progress": {"done": done, "total": total, "errors": 0}})
        llm = None
        if any(m in LLM_METHODS for m in cfg.methods):
            try:
                llm = self.llm_factory(self.settings)
            except LLMNotConfiguredError as exc:
                self.store.update_experiment(experiment_id, {"status": "failed", "finished_at": utcnow_iso(), "message": exc.message})
                return self.store.get_experiment(experiment_id)
        try:
            for size, scarcity, seed, level in planned_scenarios(cfg):
                if cancel is not None and cancel.is_set():
                    break
                sc = generate_level_scenario(level, seed, cfg.overrides) if level else generate_scenario(size, scarcity, seed, cfg.overrides)
                stored = self.store.save_scenario(sc.model_dump())
                sc = Scenario.model_validate(stored)
                sid = sc.scenario_id
                ref = refs.get(sid)
                solver_key = (sid, "ortools", 0)
                if "ortools" in cfg.methods and solver_key not in done_keys:
                    rec = self.run_solver(experiment_id, sc, cfg, ref)
                    done += 1
                    errors += rec["status"] == "error"
                    if rec["status"] == "ok":
                        ref = refs[sid] = self.reference_from(rec)
                elif ref is None and cfg.compute_reference:
                    ref = refs[sid] = self._reference_only(sc, cfg)
                elif ref is None and "ortools" in cfg.methods:  # solver run already done in an earlier session
                    old = [r for r in self.store.list_runs(experiment_id) if r["scenario_id"] == sid and r["method"] == "ortools"]
                    if old:
                        ref = refs[sid] = self.reference_from(old[-1])
                for rep in range(cfg.repetitions):
                    if cancel is not None and cancel.is_set():
                        break
                    mc0 = self.method_config(cfg, rep)
                    todo = [(m, b) for m in cfg.methods if m in LLM_METHODS for b in method_budgets(cfg, m, mc0)
                            if not _done(done_keys, sid, m, rep, b)]
                    if not todo:
                        continue
                    recs = self.run_scenario_rep(experiment_id, sc, cfg, rep, todo, llm, ref, cancel)
                    done += sum(r["status"] == "ok" for r in recs)
                    errors += sum(r["status"] == "error" for r in recs)
                    self.store.update_experiment(experiment_id, {"progress": {"done": done, "total": total, "errors": errors},
                                                                  "references": refs})
                self.store.update_experiment(experiment_id, {"progress": {"done": done, "total": total, "errors": errors}, "references": refs})
        except Exception as exc:  # noqa: BLE001
            log.exception("experiment crashed")
            self.store.update_experiment(experiment_id, {"status": "failed", "finished_at": utcnow_iso(), "message": f"{type(exc).__name__}: {exc}"})
            return self.store.get_experiment(experiment_id)
        status = "cancelled" if (cancel is not None and cancel.is_set()) else "completed"
        self.store.update_experiment(experiment_id, {"status": status, "finished_at": utcnow_iso(),
                                                      "message": f"{errors} run(s) failed with infrastructure errors; resume to retry them" if errors else ""})
        return self.store.get_experiment(experiment_id)

    def _reference_only(self, sc: Scenario, cfg: ExperimentConfig) -> Optional[dict]:
        try:
            res = solve_scenario(sc, time_limit_s=cfg.solver_time_limit_s or self.settings.solver_time_limit_s,
                                 workers=self.settings.solver_workers, seed=cfg.base_seed)
            return self.reference_from(res)
        except ResQError as exc:
            log.warning("reference solve failed: %s", exc.message)
            return None
