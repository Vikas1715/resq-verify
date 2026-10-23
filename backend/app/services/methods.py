"""The five planning methods compared in the study.

M1 llm_only         scenario -> LLM -> plan (validator used only afterwards, for evaluation)
M2 self_critique    LLM critiques its own plan, then revises it with the shared revision prompt (validator NOT used to correct)
M3 resq_verify      LLM plan -> deterministic validator -> violations -> shared revision prompt -> re-validate
M4 ortools          classical CP-SAT optimisation
M5 resample_verify  LLM plan -> validator; if infeasible, a fresh independent LLM plan from the SAME planner prompt (the model
                    gets NO feedback). It separates "more attempts" from "useful feedback".

M2 and M3 use exactly the same revision template (prompts/revision.txt); only the {{FEEDBACK}} text differs.

Call budget (cfg.call_budget = B, counts LLM calls only; the validator and the solver are not LLM calls):
    M1  always 1 call.
    M3  1 initial + up to B-1 repairs.
    M5  1 initial + up to B-1 resamples.
    M2  1 initial + rounds of (critique + revision) = 2 calls each, so floor((B-1)/2) rounds. A leftover single call is not
        spent, because a critique without a revision cannot change the plan.
If cfg.call_budget is None the older K-based behaviour is used (max_repair_rounds / self_critique_rounds).

Each function returns an `Outcome` holding every round with its plan, raw output, validator report, latency and tokens.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional

from backend.app.core.config import Settings
from backend.app.core.errors import LLMCallError
from backend.app.schemas.plan import Plan, gemini_response_schema
from backend.app.schemas.scenario import Scenario
from backend.app.schemas.validation import ValidationReport
from backend.app.services.llm.base import LLMClient, LLMResult
from backend.app.services.parsing import parse_json_object, parse_plan
from backend.app.services.prompts import PromptBook, build_feedback
from backend.app.solvers.cpsat_solver import solve_scenario
from backend.app.utils.ids import derive_seed
from backend.app.validators.validator import report_for_unparsed, validate_plan

METHODS = ["llm_only", "self_critique", "resq_verify", "resample_verify", "ortools"]
METHOD_LABELS = {"llm_only": "LLM-only", "self_critique": "Self-critique", "resq_verify": "ResQ-Verify",
                 "resample_verify": "Resample + Verify", "ortools": "OR-Tools CP-SAT"}
METHOD_CODES = {"llm_only": "M1", "self_critique": "M2", "resq_verify": "M3", "resample_verify": "M5", "ortools": "M4"}
LLM_METHODS = ["llm_only", "self_critique", "resq_verify", "resample_verify"]
BUDGETED_METHODS = ["self_critique", "resq_verify", "resample_verify"]   # methods whose result depends on the call budget


@dataclass
class MethodConfig:
    temperature: float = 0.4
    max_output_tokens: int = 8192
    max_repair_rounds: int = 3
    self_critique_rounds: int = 3
    feedback_format: str = "structured"   # "structured" | "free_text"
    base_seed: int = 0
    run_index: int = 0
    request_delay_s: float = 0.0
    call_budget: Optional[int] = None     # max LLM calls per run for M2/M3/M5 (None = K-based legacy behaviour)


@dataclass
class MethodContext:
    llm: LLMClient
    prompts: PromptBook
    cfg: MethodConfig
    sleep: Callable[[float], None] = time.sleep


def _add_opt(a: Optional[int], b: Optional[int]) -> Optional[int]:
    """None means 'the API did not report it'. It is never turned into 0 and never invented."""
    return None if (a is None and b is None) else (a or 0) + (b or 0)


@dataclass
class CallStats:
    latency_s: float = 0.0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    thoughts_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    retry_count: int = 0
    errors: list[str] = field(default_factory=list)
    estimated: bool = False
    model_version: Optional[str] = None
    finish_reason: Optional[str] = None

    @classmethod
    def from_result(cls, r: LLMResult) -> "CallStats":
        total = r.total_tokens
        if total is None and (r.input_tokens is not None or r.output_tokens is not None):
            total = (r.input_tokens or 0) + (r.output_tokens or 0) + (r.thoughts_tokens or 0)
        return cls(r.latency_s, r.input_tokens, r.output_tokens, r.thoughts_tokens, total,
                   r.retry_count, list(r.errors), r.tokens_estimated, r.model_version, r.finish_reason)

    def add(self, o: "CallStats") -> "CallStats":
        return CallStats(self.latency_s + o.latency_s, _add_opt(self.input_tokens, o.input_tokens),
                         _add_opt(self.output_tokens, o.output_tokens), _add_opt(self.thoughts_tokens, o.thoughts_tokens),
                         _add_opt(self.total_tokens, o.total_tokens),
                         self.retry_count + o.retry_count, self.errors + o.errors, self.estimated or o.estimated,
                         o.model_version or self.model_version, o.finish_reason or self.finish_reason)

    def usage(self) -> dict:
        return {"input": self.input_tokens, "output": self.output_tokens, "thoughts": self.thoughts_tokens,
                "total": self.total_tokens, "estimated": self.estimated}


@dataclass
class Attempt:
    plan: Optional[Plan]
    raw: str
    parse_error: Optional[str]
    stats: CallStats
    api_error: Optional[str] = None


@dataclass
class Outcome:
    method: str
    rounds: list[dict] = field(default_factory=list)
    final_plan: Optional[Plan] = None
    final_raw: str = ""
    final_report: Optional[ValidationReport] = None
    stop_reason: str = ""
    failure_reason: Optional[str] = None
    status: str = "ok"           # "ok" | "error" (infrastructure failure, excluded from rates)
    solver: Optional[dict] = None
    calls_used: int = 0          # LLM calls this run needed (the shared first plan counts for every method that uses it)
    call_budget: Optional[int] = None   # the budget that applied (for K-based legacy runs: the equivalent maximum)


# ---------------------------------------------------------------------------- helpers
def _evaluate(scenario: Scenario, plan: Optional[Plan], parse_error: Optional[str]) -> ValidationReport:
    return validate_plan(scenario, plan) if plan is not None else report_for_unparsed(parse_error or "unknown parse error")


def _call(ctx: MethodContext, scenario: Scenario, prompt: str, tag: str, schema: Optional[dict] = None) -> LLMResult:
    if ctx.cfg.request_delay_s > 0:
        ctx.sleep(ctx.cfg.request_delay_s)
    seed = derive_seed(ctx.cfg.base_seed, ctx.cfg.run_index, scenario.scenario_id, tag)
    return ctx.llm.generate(prompt, temperature=ctx.cfg.temperature, max_output_tokens=ctx.cfg.max_output_tokens,
                            seed=seed, expect_json=True, response_schema=schema)


def _attempt_from_call(ctx, scenario, prompt, tag) -> Attempt:
    try:
        res = _call(ctx, scenario, prompt, tag, gemini_response_schema())
    except LLMCallError as exc:
        st = CallStats(latency_s=exc.latency_s, retry_count=exc.retry_count, errors=list(exc.errors))
        return Attempt(None, "", None, st, api_error=exc.message)
    pr = parse_plan(res.text)
    return Attempt(pr.plan, res.text, pr.error, CallStats.from_result(res))


def _round(idx: int, kind: str, scenario: Scenario, att: Attempt, prev_plan: Optional[Plan], extra_stats: Optional[CallStats] = None,
           calls: int = 1, **extra: Any) -> dict:
    report = _evaluate(scenario, att.plan, att.parse_error)
    stats = att.stats if extra_stats is None else att.stats.add(extra_stats)
    ed = None
    if att.plan is not None and prev_plan is not None:
        ed = len(att.plan.record_set() ^ prev_plan.record_set())
    r = {
        "round": idx, "kind": kind, "calls": calls, "plan": att.plan.model_dump() if att.plan else None,
        "raw_llm_output": att.raw, "parse_error": att.parse_error, "validation": report.model_dump(),
        "is_valid": report.valid, "n_violations": report.n_violations, "violation_counts": report.counts_by_type,
        "latency_s": round(stats.latency_s, 4), "token_usage": stats.usage(), "retry_count": stats.retry_count,
        "api_errors": stats.errors, "model_version": stats.model_version, "finish_reason": stats.finish_reason,
        "edit_distance": ed,
    }
    r.update(extra)
    return r


def _calls_in(rounds: list[dict]) -> int:
    return sum(int(r.get("calls") or 0) for r in rounds)


def _fail(method: str, rounds: list[dict], msg: str, last_plan: Optional[Plan], last_raw: str, scenario: Scenario,
          last_parse_error: Optional[str], *, failed_calls: int = 1, budget: Optional[int] = None) -> Outcome:
    """An API failure ends the run as status='error' (excluded from rates). The failed call still counts as a call used."""
    return Outcome(method=method, rounds=rounds, final_plan=last_plan, final_raw=last_raw,
                   final_report=_evaluate(scenario, last_plan, last_parse_error), stop_reason="api_error",
                   failure_reason=f"api_error: {msg}", status="error", calls_used=_calls_in(rounds) + failed_calls,
                   call_budget=budget)


def effective_budget(method: str, cfg: MethodConfig) -> Optional[int]:
    """The call budget a method runs under (None for the solver). Used to label and key run records."""
    if method == "ortools":
        return None
    if method == "llm_only":
        return 1
    if cfg.call_budget is not None:
        return cfg.call_budget
    return 1 + (2 * max(0, cfg.self_critique_rounds) if method == "self_critique" else cfg.max_repair_rounds)


def _extra_calls(cfg: MethodConfig) -> int:
    """How many calls beyond the first plan M3 / M5 may spend."""
    return max(0, (cfg.call_budget - 1) if cfg.call_budget is not None else cfg.max_repair_rounds)


def _budget_label(cfg: MethodConfig) -> str:
    return "max_rounds" if cfg.call_budget is None else "budget_exhausted"


# ---------------------------------------------------------------------------- initial plan (shared by M1-M3 and M5)
def generate_initial(ctx: MethodContext, scenario: Scenario) -> Attempt:
    return _attempt_from_call(ctx, scenario, ctx.prompts.planner(scenario), "initial")


# ---------------------------------------------------------------------------- M1
def run_llm_only(ctx: MethodContext, scenario: Scenario, initial: Attempt) -> Outcome:
    if initial.api_error:
        return _fail("llm_only", [], initial.api_error, None, "", scenario, None, budget=1)
    r0 = _round(0, "initial", scenario, initial, None)
    return Outcome("llm_only", [r0], initial.plan, initial.raw, ValidationReport.model_validate(r0["validation"]),
                   stop_reason="single_shot", failure_reason=("parse_error" if initial.plan is None else None),
                   calls_used=1, call_budget=1)


# ---------------------------------------------------------------------------- M2
def run_self_critique(ctx: MethodContext, scenario: Scenario, initial: Attempt) -> Outcome:
    cfg = ctx.cfg
    n_rounds = max(0, (cfg.call_budget - 1) // 2) if cfg.call_budget is not None else max(0, cfg.self_critique_rounds)
    budget = cfg.call_budget if cfg.call_budget is not None else 1 + 2 * n_rounds
    if initial.api_error:
        return _fail("self_critique", [], initial.api_error, None, "", scenario, None, budget=budget)
    rounds = [_round(0, "initial", scenario, initial, None)]
    cur = initial
    final_reason = _budget_label(cfg)
    for r in range(1, n_rounds + 1):
        try:
            cres = _call(ctx, scenario, ctx.prompts.critique(scenario, cur.plan, cur.raw), f"critique{r}")
        except LLMCallError as exc:
            return _fail("self_critique", rounds, exc.message, cur.plan, cur.raw, scenario, cur.parse_error, budget=budget)
        cstats = CallStats.from_result(cres)
        crit, cerr = parse_json_object(cres.text)
        approved = bool(crit) and crit.get("plan_is_valid") is True and not crit.get("issues")
        if approved:
            same = Attempt(cur.plan, cur.raw, cur.parse_error, CallStats())
            rounds.append(_round(r, "critique_approved", scenario, same, cur.plan, cstats, calls=1, critique=crit,
                                 critique_raw=cres.text, critique_parse_error=cerr))
            final_reason = "self_approved"
            break
        crit_text = json.dumps(crit) if crit else cres.text[:2000]
        # same revision template as M3; the feedback is the model's own critique
        nxt = _attempt_from_call(ctx, scenario, ctx.prompts.revision(scenario, cur.plan, cur.raw, crit_text), f"revise{r}")
        if nxt.api_error:
            return _fail("self_critique", rounds, nxt.api_error, cur.plan, cur.raw, scenario, cur.parse_error,
                         failed_calls=2, budget=budget)
        rounds.append(_round(r, "critique_revise", scenario, nxt, cur.plan, cstats, calls=2, critique=crit,
                             critique_raw=cres.text, critique_parse_error=cerr))
        cur = nxt
    last = rounds[-1]
    rep = ValidationReport.model_validate(last["validation"])
    return Outcome("self_critique", rounds, cur.plan, cur.raw, rep, stop_reason=final_reason,
                   failure_reason=("parse_error" if cur.plan is None else None), calls_used=_calls_in(rounds), call_budget=budget)


# ---------------------------------------------------------------------------- M3 (ResQ-Verify)
def run_resq_verify(ctx: MethodContext, scenario: Scenario, initial: Attempt) -> Outcome:
    cfg = ctx.cfg
    budget = cfg.call_budget if cfg.call_budget is not None else 1 + cfg.max_repair_rounds
    if initial.api_error:
        return _fail("resq_verify", [], initial.api_error, None, "", scenario, None, budget=budget)
    cur = initial
    rounds = [_round(0, "initial", scenario, cur, None)]
    report = ValidationReport.model_validate(rounds[0]["validation"])
    for r in range(1, _extra_calls(cfg) + 1):
        if report.valid:
            break
        feedback = build_feedback(report, cfg.feedback_format)
        # same revision template as M2; the feedback is the deterministic validator's report
        nxt = _attempt_from_call(ctx, scenario, ctx.prompts.revision(scenario, cur.plan, cur.raw, feedback), f"repair{r}")
        if nxt.api_error:
            return _fail("resq_verify", rounds, nxt.api_error, cur.plan, cur.raw, scenario, cur.parse_error, budget=budget)
        rounds.append(_round(r, "repair", scenario, nxt, cur.plan, feedback_sent=feedback, feedback_format=cfg.feedback_format))
        report = ValidationReport.model_validate(rounds[-1]["validation"])
        cur = nxt  # an unparseable repair is kept as-is: the next prompt shows its raw text and the parse error
    stop = "valid" if report.valid else _budget_label(cfg)
    return Outcome("resq_verify", rounds, cur.plan, cur.raw, report, stop_reason=stop,
                   failure_reason=("parse_error" if cur.plan is None else None), calls_used=_calls_in(rounds), call_budget=budget)


# ---------------------------------------------------------------------------- M5 (blind resample + verify)
def run_resample_verify(ctx: MethodContext, scenario: Scenario, initial: Attempt) -> Outcome:
    """Draw independent plans from the unchanged planner prompt until one passes the validator or the budget is spent.

    The model never sees the validator's output (or any other feedback); the validator only decides whether to stop.
    """
    cfg = ctx.cfg
    budget = cfg.call_budget if cfg.call_budget is not None else 1 + cfg.max_repair_rounds
    if initial.api_error:
        return _fail("resample_verify", [], initial.api_error, None, "", scenario, None, budget=budget)
    cur = initial
    rounds = [_round(0, "initial", scenario, cur, None)]
    report = ValidationReport.model_validate(rounds[0]["validation"])
    planner_prompt = ctx.prompts.planner(scenario)
    for r in range(1, _extra_calls(cfg) + 1):
        if report.valid:
            break
        nxt = _attempt_from_call(ctx, scenario, planner_prompt, f"resample{r}")   # identical prompt, new seed, no feedback
        if nxt.api_error:
            return _fail("resample_verify", rounds, nxt.api_error, cur.plan, cur.raw, scenario, cur.parse_error, budget=budget)
        rounds.append(_round(r, "resample", scenario, nxt, cur.plan))
        report = ValidationReport.model_validate(rounds[-1]["validation"])
        cur = nxt
    stop = "valid" if report.valid else _budget_label(cfg)
    return Outcome("resample_verify", rounds, cur.plan, cur.raw, report, stop_reason=stop,
                   failure_reason=("parse_error" if cur.plan is None else None), calls_used=_calls_in(rounds), call_budget=budget)


# ---------------------------------------------------------------------------- M4
def run_ortools(scenario: Scenario, settings: Settings, *, seed: int = 0) -> Outcome:
    res = solve_scenario(scenario, time_limit_s=settings.solver_time_limit_s, workers=settings.solver_workers, seed=seed)
    plan = Plan.model_validate(res["plan"]) if res.get("plan") else None
    err = None if plan else f"solver returned status {res['status']}"
    att = Attempt(plan, json.dumps(res["plan"]) if plan else "", err, CallStats(latency_s=res["runtime_s"]))
    r0 = _round(0, "solver", scenario, att, None, calls=0)
    solver_info = {k: v for k, v in res.items() if k != "plan"}
    return Outcome("ortools", [r0], plan, att.raw, ValidationReport.model_validate(r0["validation"]),
                   stop_reason=res["status"].lower(), failure_reason=None if plan else f"solver_{res['status'].lower()}",
                   solver=solver_info, calls_used=0, call_budget=None)


# ---------------------------------------------------------------------------- nested call budgets
RUNNERS = {"llm_only": run_llm_only, "self_critique": run_self_critique, "resq_verify": run_resq_verify,
           "resample_verify": run_resample_verify}


def truncate_outcome(outcome: Outcome, scenario: Scenario, budget: int) -> Outcome:
    """The result a run would have had with a smaller call budget, read off the same trajectory.

    Every method spends its calls in a fixed order and stops at the first valid plan, so the run for budget B is exactly the
    first rounds of the run for a larger budget. Re-using the trajectory costs fewer API calls than re-running, and it makes
    results at different budgets nested (budget 5 can never look worse than budget 3 on the same trajectory).
    Whole rounds only: an M2 round (critique + revision) is 2 calls and is kept or dropped as a unit.
    """
    if outcome.method not in BUDGETED_METHODS:
        return outcome
    if outcome.status == "error":
        if budget >= outcome.calls_used:       # the failed call happened inside this budget -> the run is an error here too
            return replace(outcome, call_budget=budget)
    keep, cum = [], 0
    for r in outcome.rounds:
        # an M2 round is only started when both of its calls (critique + revision) fit, even if the critique then approves
        # the plan and the round ends up costing one call; this is what the run at the smaller budget does too
        need = 2 if (outcome.method == "self_critique" and r.get("round", 0) > 0) else int(r.get("calls") or 0)
        if cum + need > budget:
            break
        keep.append(r)
        cum += int(r.get("calls") or 0)
    if len(keep) == len(outcome.rounds) and outcome.status == "ok":
        return replace(outcome, call_budget=budget)
    last = keep[-1]
    plan = Plan.model_validate(last["plan"]) if last.get("plan") else None
    return Outcome(method=outcome.method, rounds=keep, final_plan=plan, final_raw=last.get("raw_llm_output") or "",
                   final_report=ValidationReport.model_validate(last["validation"]),
                   stop_reason="budget_exhausted",   # M3/M5 stop at the first valid plan, so a cut run is never valid; M2 ignores validity
                   failure_reason=("parse_error" if plan is None else None), status="ok", calls_used=cum, call_budget=budget)
