from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from backend.app.analytics.dataframe import runs_to_frame
from backend.app.analytics.plots import all_plots
from backend.app.analytics.report import build_bundle, comparisons_csv, research_summary_md, runs_csv
from backend.app.analytics.summary import build_summary
from backend.app.api.context import AppContext
from backend.app.core.errors import BadRequestError, NotFoundError
from backend.app.experiments.configs import list_config_files, load_config_file
from backend.app.schemas.experiment import ExperimentConfig
from backend.app.schemas.plan import Plan
from backend.app.schemas.scenario import Scenario
from backend.app.services.llm.factory import llm_status
from backend.app.services.methods import METHOD_CODES, METHOD_LABELS
from backend.app.services.parsing import parse_plan
from backend.app.services.scenario_generator import DIFFICULTY_LEVELS, LEVEL_PRESETS, generate_level_scenario, generate_scenario
from backend.app.services.store.base import dedupe_runs
from backend.app.solvers.cpsat_solver import solve_scenario, solver_available
from backend.app.validators.metrics import plan_quality
from backend.app.validators.validator import report_for_unparsed, validate_plan

router = APIRouter()


def ctx(request: Request) -> AppContext:
    return request.app.state.ctx


# ------------------------------------------------------------------ request models
class GenerateScenarioReq(BaseModel):
    size: str = "small"
    scarcity: str = "high"
    seed: int = 42
    overrides: dict[str, Any] = Field(default_factory=dict)
    difficulty_level: Optional[str] = Field(None, description="L0..L4; when set, size and scarcity are taken from the level")
    save: bool = True


class PlanningReq(BaseModel):
    scenario_id: str
    max_repair_rounds: Optional[int] = Field(None, ge=0, le=10)
    feedback_format: Optional[str] = Field(None, pattern="^(structured|free_text)$")
    temperature: Optional[float] = Field(None, ge=0, le=2)
    compute_reference: bool = True
    call_budget: Optional[int] = Field(None, ge=1, le=20, description="max LLM calls for M2/M3/M5 (default: use K)")


class CompareReq(PlanningReq):
    methods: Optional[list[str]] = None


class RepairReq(PlanningReq):
    plan: Any


class ValidateReq(BaseModel):
    scenario_id: Optional[str] = None
    scenario: Optional[dict] = None
    plan: Optional[Any] = None
    raw_text: Optional[str] = None


class SolveReq(BaseModel):
    scenario_id: str
    time_limit_s: Optional[float] = Field(None, gt=0, le=600)


class RunExperimentReq(BaseModel):
    config: Optional[dict] = None
    config_file: Optional[str] = None


# ------------------------------------------------------------------ health / config
@router.get("/health")
def health(request: Request):
    c = ctx(request)
    return {"status": "ok", "version": "1.0.0", "llm": llm_status(c.settings), "storage": {**c.store.health(), **c.store_info},
            "solver": {"ortools_available": solver_available(), "time_limit_s": c.settings.solver_time_limit_s},
            "prompt_version": c.runner.prompts.version, "prompt_hash": c.runner.prompts.content_hash,
            "settings": c.settings.public_dict()}


@router.get("/config/methods")
def methods_info():
    return {"methods": [{"id": k, "code": METHOD_CODES[k], "label": v} for k, v in METHOD_LABELS.items()]}


# ------------------------------------------------------------------ scenarios
@router.get("/scenarios/levels")
def scenario_levels():
    return {"levels": [{"id": k, **{x: v[x] for x in v}} for k, v in LEVEL_PRESETS.items()]}


@router.post("/scenarios/generate")
def scenarios_generate(req: GenerateScenarioReq, request: Request):
    c = ctx(request)
    try:
        sc = (generate_level_scenario(req.difficulty_level, req.seed, req.overrides) if req.difficulty_level
              else generate_scenario(req.size, req.scarcity, req.seed, req.overrides))
    except (KeyError, ValueError) as exc:
        raise BadRequestError(f"Cannot generate scenario: {exc}") from exc
    d = c.store.save_scenario(sc.model_dump()) if req.save else sc.model_dump()
    return d


@router.get("/scenarios")
def scenarios_list(request: Request, limit: int = 500):
    return {"scenarios": ctx(request).store.list_scenarios(limit)}


@router.get("/scenarios/{scenario_id}")
def scenarios_get(scenario_id: str, request: Request):
    d = ctx(request).store.get_scenario(scenario_id)
    if d is None:
        raise NotFoundError(f"Scenario {scenario_id} not found")
    return d


# ------------------------------------------------------------------ validation / solver
@router.post("/validate")
def validate(req: ValidateReq, request: Request):
    c = ctx(request)
    if req.scenario is not None:
        sc = Scenario.model_validate(req.scenario)
    elif req.scenario_id:
        sc = c.adhoc.scenario(req.scenario_id)
    else:
        raise BadRequestError("Provide scenario_id or scenario")
    if req.raw_text is not None:
        res = parse_plan(req.raw_text)
    elif req.plan is not None:
        import json
        res = parse_plan(req.plan if isinstance(req.plan, str) else json.dumps(req.plan))
    else:
        raise BadRequestError("Provide plan (object) or raw_text")
    report = validate_plan(sc, res.plan) if res.plan is not None else report_for_unparsed(res.error or "unparseable")
    out = {"report": report.model_dump(), "parsed": res.plan is not None, "parse_error": res.error}
    if res.plan is not None:
        out["quality"] = plan_quality(sc, res.plan)
    return out


@router.post("/solve")
def solve(req: SolveReq, request: Request):
    c = ctx(request)
    sc = c.adhoc.scenario(req.scenario_id)
    res = solve_scenario(sc, time_limit_s=req.time_limit_s or c.settings.solver_time_limit_s, workers=c.settings.solver_workers, seed=42)
    plan = Plan.model_validate(res["plan"]) if res.get("plan") else None
    out = {k: v for k, v in res.items() if k != "plan"}
    out["plan"] = res.get("plan")
    if plan is not None:
        out["report"] = validate_plan(sc, plan).model_dump()
        out["quality"] = plan_quality(sc, plan)
    return out


# ------------------------------------------------------------------ planning (interactive runs)
@router.post("/planning/generate")
def planning_generate(req: PlanningReq, request: Request):
    """M1: single-shot LLM plan, validated afterwards for evaluation only."""
    return ctx(request).adhoc.run_method(req.scenario_id, "llm_only", max_repair_rounds=req.max_repair_rounds,
                                         temperature=req.temperature, compute_reference=req.compute_reference)


@router.post("/planning/self-critique")
def planning_self_critique(req: PlanningReq, request: Request):
    return ctx(request).adhoc.run_method(req.scenario_id, "self_critique", max_repair_rounds=req.max_repair_rounds,
                                         temperature=req.temperature, compute_reference=req.compute_reference,
                                         call_budget=req.call_budget)


@router.post("/planning/verify")
def planning_verify(req: PlanningReq, request: Request):
    """M3: generate, validate, feed structured violations back, repeat up to K rounds."""
    return ctx(request).adhoc.run_method(req.scenario_id, "resq_verify", max_repair_rounds=req.max_repair_rounds,
                                         feedback_format=req.feedback_format, temperature=req.temperature,
                                         compute_reference=req.compute_reference, call_budget=req.call_budget)


@router.post("/planning/resample-verify")
def planning_resample_verify(req: PlanningReq, request: Request):
    """M5: draw independent plans from the same prompt until one passes the validator or the call budget is spent (no feedback)."""
    return ctx(request).adhoc.run_method(req.scenario_id, "resample_verify", max_repair_rounds=req.max_repair_rounds,
                                         temperature=req.temperature, compute_reference=req.compute_reference,
                                         call_budget=req.call_budget)


@router.post("/planning/repair")
def planning_repair(req: RepairReq, request: Request):
    """Run the repair loop from a user-supplied (possibly hand-edited) plan."""
    return ctx(request).adhoc.repair_given_plan(req.scenario_id, req.plan, max_repair_rounds=req.max_repair_rounds,
                                                feedback_format=req.feedback_format, temperature=req.temperature)


@router.post("/planning/compare")
def planning_compare(req: CompareReq, request: Request):
    recs = ctx(request).adhoc.compare(req.scenario_id, max_repair_rounds=req.max_repair_rounds, feedback_format=req.feedback_format,
                                      temperature=req.temperature, methods=req.methods, call_budget=req.call_budget)
    return {"scenario_id": req.scenario_id, "runs": recs}


# ------------------------------------------------------------------ experiments
@router.get("/experiments/configs")
def experiment_configs():
    return {"files": list_config_files()}


@router.get("/experiments/configs/{name}")
def experiment_config_get(name: str):
    return load_config_file(name).model_dump()


@router.post("/experiments/run")
def experiments_run(req: RunExperimentReq, request: Request):
    c = ctx(request)
    if req.config_file:
        cfg = load_config_file(req.config_file)
        if req.config:
            cfg = ExperimentConfig.model_validate({**cfg.model_dump(), **req.config})
    elif req.config is not None:
        cfg = ExperimentConfig.model_validate(req.config)
    else:
        raise BadRequestError("Provide config or config_file")
    return c.manager.start(cfg)


@router.get("/experiments")
def experiments_list(request: Request, include_adhoc: bool = True):
    c = ctx(request)
    exps = c.store.list_experiments()
    if not include_adhoc:
        exps = [e for e in exps if e.get("kind") != "adhoc"]
    return {"experiments": exps}


@router.get("/experiments/{experiment_id}")
def experiments_get(experiment_id: str, request: Request):
    c = ctx(request)
    e = c.store.get_experiment(experiment_id)
    if e is None:
        raise NotFoundError(f"Experiment {experiment_id} not found")
    e["is_running"] = c.manager.is_running(experiment_id)
    return e


@router.post("/experiments/{experiment_id}/cancel")
def experiments_cancel(experiment_id: str, request: Request):
    return ctx(request).manager.cancel(experiment_id)


@router.post("/experiments/{experiment_id}/resume")
def experiments_resume(experiment_id: str, request: Request):
    return ctx(request).manager.resume(experiment_id)


@router.get("/experiments/{experiment_id}/runs")
def experiments_runs(experiment_id: str, request: Request, full: bool = False, method: Optional[str] = None,
                     status: Optional[str] = None, limit: int = 5000):
    c = ctx(request)
    if c.store.get_experiment(experiment_id) is None:
        raise NotFoundError(f"Experiment {experiment_id} not found")
    runs = dedupe_runs(c.store.list_runs(experiment_id))
    if method:
        runs = [r for r in runs if r.get("method") == method]
    if status:
        runs = [r for r in runs if r.get("status") == status]
    runs = runs[:limit]
    if not full:  # light rows for tables; use /runs/{id} for the full record
        heavy = {"rounds", "parsed_plan", "raw_llm_output", "validator_report", "solver"}
        runs = [{k: v for k, v in r.items() if k not in heavy} for r in runs]
    return {"experiment_id": experiment_id, "n": len(runs), "runs": runs}


@router.get("/runs/{run_id}")
def run_get(run_id: str, request: Request):
    r = ctx(request).store.get_run(run_id)
    if r is None:
        raise NotFoundError(f"Run {run_id} not found")
    return r


# ------------------------------------------------------------------ analytics / exports
def _select(c: AppContext, experiment_id: Optional[str], include_adhoc: bool) -> tuple[list[dict], list[dict]]:
    if experiment_id and experiment_id != "all":
        e = c.store.get_experiment(experiment_id)
        if e is None:
            raise NotFoundError(f"Experiment {experiment_id} not found")
        return dedupe_runs(c.store.list_runs(experiment_id)), [e]
    exps = [e for e in c.store.list_experiments(10_000) if include_adhoc or e.get("kind") != "adhoc"]
    runs: list[dict] = []
    for e in exps:
        runs.extend(dedupe_runs(c.store.list_runs(e["experiment_id"])))
    return runs, exps


@router.get("/analytics/summary")
def analytics_summary(request: Request, experiment_id: Optional[str] = None, include_adhoc: bool = False,
                      call_budget: Optional[int] = None):
    runs, exps = _select(ctx(request), experiment_id, include_adhoc)
    s = build_summary(runs, exps[0] if len(exps) == 1 else None, call_budget)
    s["experiments"] = [e["experiment_id"] for e in exps]
    return s


@router.get("/analytics/runs")
def analytics_runs(request: Request, experiment_id: Optional[str] = None, include_adhoc: bool = False):
    runs, _ = _select(ctx(request), experiment_id, include_adhoc)
    df = runs_to_frame(runs)
    return {"n": len(df), "rows": df.where(df.notna(), None).to_dict(orient="records")}


_EXPORTS = {"runs.csv": "text/csv", "runs.json": "application/json", "comparisons.csv": "text/csv",
            "research_summary.md": "text/markdown", "plots.zip": "application/zip", "bundle.zip": "application/zip"}


@router.get("/exports/{filename}")
def exports(filename: str, request: Request, experiment_id: Optional[str] = None, include_adhoc: bool = False):
    import io
    import json
    import zipfile
    if filename not in _EXPORTS:
        raise NotFoundError(f"Unknown export '{filename}'. Available: {sorted(_EXPORTS)}")
    c = ctx(request)
    runs, exps = _select(c, experiment_id, include_adhoc)
    if filename == "runs.csv":
        body: Any = runs_csv(runs)
    elif filename == "runs.json":
        body = json.dumps(runs, indent=1, default=str)
    elif filename == "comparisons.csv":
        body = comparisons_csv(runs)
    elif filename == "research_summary.md":
        body = research_summary_md(runs, experiments=exps)
    elif filename == "plots.zip":
        df = runs_to_frame(runs)
        sim = bool(not df.empty and df.loc[df["method"] != "ortools", "is_simulated"].astype(bool).any())
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for n, data in all_plots(df, runs, sim).items():
                z.writestr(n, data)
        body = buf.getvalue()
    else:
        scen_ids = {r["scenario_id"] for r in runs}
        scenarios = [s for s in (c.store.get_scenario(i) for i in sorted(scen_ids)) if s]
        body = build_bundle(runs, exps, scenarios)
    return Response(content=body, media_type=_EXPORTS[filename],
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})
