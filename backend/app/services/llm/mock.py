"""SIMULATED LLM — a test double, NOT a language model and NOT a research result.

Purpose: let the whole software stack (API, runner, UI, tests) run offline without an API key.
It builds a valid greedy plan and then injects random constraint faults, repairs some of them when asked,
and critiques with imperfect detection. Every run record made with it carries `provider="mock"` and
`is_simulated=true`, the UI shows a banner, and reports/exports flag it. Never use it for paper results.
"""
from __future__ import annotations

import json
import math
import random
from typing import Optional

from backend.app.core.config import Settings
from backend.app.schemas.plan import Plan, ShelterAllocation, TeamAssignment, UnservedDemand, VehicleAssignment
from backend.app.schemas.scenario import Scenario
from backend.app.services.llm.base import LLMClient, LLMResult
from backend.app.services.parsing import parse_json_object, parse_plan
from backend.app.utils.ids import derive_seed
from backend.app.validators.validator import validate_plan

FAULTS_PER_PLAN = {"small": 1.0, "medium": 2.0, "large": 3.5}
P_FIX, P_NEW_FAULT, P_CASCADE = 0.8, 0.10, 0.6
P_DETECT, P_FALSE_POSITIVE = 0.30, 0.15


def _between(text: str, a: str, b: str) -> str:
    i = text.find(a)
    j = text.find(b, i + len(a)) if i >= 0 else -1
    return text[i + len(a): j].strip() if i >= 0 and j >= 0 else ""


def scenario_from_prompt(prompt: str) -> Scenario:
    d = json.loads(_between(prompt, "<<<SCENARIO_JSON>>>", "<<<END_SCENARIO_JSON>>>"))
    d.update({"generator_version": "mock", "seed": 0, "size": "small", "scarcity": "low"})
    return Scenario.model_validate(d)


def greedy_plan(s: Scenario) -> Plan:
    """Feasible-by-construction greedy plan (priority first)."""
    road = s.road_map()
    trips_left = {v.id: v.max_trips for v in s.vehicles}
    room = {sh.id: sh.capacity for sh in s.shelters}
    team_left = {t.id: t.max_incidents for t in s.teams if t.available}
    plan = Plan(reasoning_summary="greedy plan by priority")
    for inc in sorted(s.incidents, key=lambda i: (-i.priority, -i.people, i.id)):
        need = min(inc.people, sum(room.values()))
        if need <= 0 or not any(n > 0 for n in team_left.values()):
            continue
        moved = 0
        for v in sorted(s.vehicles, key=lambda v: (-v.capacity, v.id)):
            if need - moved <= 0:
                break
            ok = v.type in inc.allowed_vehicle_types and (v.type == "boat" or not road[inc.road_id].blocked)
            if not ok or trips_left[v.id] <= 0 or v.capacity <= 0:
                continue
            n_trips = min(trips_left[v.id], math.ceil((need - moved) / v.capacity))
            ppl = min(need - moved, n_trips * v.capacity)
            plan.vehicle_assignments.append(VehicleAssignment(vehicle_id=v.id, incident_id=inc.id, trips=n_trips, people=ppl))
            trips_left[v.id] -= n_trips
            moved += ppl
        if moved == 0:
            continue
        tid = next(t for t, n in sorted(team_left.items()) if n > 0)
        team_left[tid] -= 1
        plan.team_assignments.append(TeamAssignment(team_id=tid, incident_id=inc.id))
        left = moved
        for sh in s.shelters:
            if left <= 0:
                break
            put = min(left, room[sh.id])
            if put > 0:
                plan.shelter_allocations.append(ShelterAllocation(incident_id=inc.id, shelter_id=sh.id, people=put))
                room[sh.id] -= put
                left -= put
        if inc.people - moved > 0:
            plan.unserved_demand.append(UnservedDemand(incident_id=inc.id, people=inc.people - moved))
    for inc in s.incidents:  # incidents never touched are fully unserved
        if not any(u.incident_id == inc.id for u in plan.unserved_demand) and not any(
                a.incident_id == inc.id for a in plan.vehicle_assignments):
            plan.unserved_demand.append(UnservedDemand(incident_id=inc.id, people=inc.people))
    return plan


def inject_fault(s: Scenario, plan: Plan, rng: random.Random) -> None:
    road = s.road_map()
    kinds = ["inflate", "trips", "blocked", "wrongtype", "team_unavail", "team_over", "shelter", "bad_id", "unserved", "dup"]
    rng.shuffle(kinds)
    for kind in kinds:
        if kind == "inflate" and plan.vehicle_assignments:
            rng.choice(plan.vehicle_assignments).people += rng.randint(3, 15)
        elif kind == "trips" and plan.vehicle_assignments:
            rng.choice(plan.vehicle_assignments).trips += 1
        elif kind == "blocked":
            blocked = [i for i in s.incidents if road[i.road_id].blocked]
            ground = [v for v in s.vehicles if v.type != "boat"]
            if blocked and ground:
                i, v = rng.choice(blocked), rng.choice(ground)
                plan.vehicle_assignments.append(VehicleAssignment(vehicle_id=v.id, incident_id=i.id, trips=1, people=min(v.capacity, i.people)))
            else:
                continue
        elif kind == "wrongtype":
            bad = [(v, i) for v in s.vehicles for i in s.incidents if v.type not in i.allowed_vehicle_types and road[i.road_id].blocked is False]
            if not bad:
                continue
            v, i = rng.choice(bad)
            plan.vehicle_assignments.append(VehicleAssignment(vehicle_id=v.id, incident_id=i.id, trips=1, people=min(v.capacity, i.people)))
        elif kind == "team_unavail":
            un = [t for t in s.teams if not t.available]
            if not un:
                continue
            plan.team_assignments.append(TeamAssignment(team_id=rng.choice(un).id, incident_id=rng.choice(s.incidents).id))
        elif kind == "team_over":
            t = rng.choice([t for t in s.teams if t.available])
            have = {a.incident_id for a in plan.team_assignments if a.team_id == t.id}
            extra = [i for i in s.incidents if i.id not in have][: t.max_incidents + 1]
            for i in extra:
                plan.team_assignments.append(TeamAssignment(team_id=t.id, incident_id=i.id))
        elif kind == "shelter" and plan.shelter_allocations:
            rng.choice(plan.shelter_allocations).people += rng.randint(5, 25)
        elif kind == "bad_id":
            plan.vehicle_assignments.append(VehicleAssignment(vehicle_id="VEH-99", incident_id=rng.choice(s.incidents).id, trips=1, people=5))
        elif kind == "unserved" and plan.unserved_demand:
            rng.choice(plan.unserved_demand).people += rng.randint(1, 9)
        elif kind == "dup" and plan.vehicle_assignments:
            plan.vehicle_assignments.append(plan.vehicle_assignments[0].model_copy())
        else:
            continue
        return


def fix_violation(s: Scenario, plan: Plan, v, rng: random.Random) -> None:
    """Narrow local fix for one violation (the simulated model sometimes forgets the knock-on updates)."""
    cascade = rng.random() < P_CASCADE
    vm = s.vehicle_map()
    if v.type == "schema_integrity":
        seen, keep = set(), []
        ids = set(vm) | {t.id for t in s.teams} | {i.id for i in s.incidents} | {x.id for x in s.shelters}
        for a in plan.vehicle_assignments:
            k = (a.vehicle_id, a.incident_id)
            if a.vehicle_id in vm and a.incident_id in ids and k not in seen and a.trips >= 0 and a.people >= 0:
                seen.add(k); keep.append(a)
        plan.vehicle_assignments = keep
        plan.team_assignments = list({(t.team_id, t.incident_id): t for t in plan.team_assignments
                                      if t.team_id in ids and t.incident_id in ids}.values())
        plan.shelter_allocations = list({(a.incident_id, a.shelter_id): a for a in plan.shelter_allocations
                                         if a.shelter_id in ids and a.incident_id in ids}.values())
    elif v.type == "vehicle_capacity":
        for a in plan.vehicle_assignments:
            if a.vehicle_id == v.entity and a.vehicle_id in vm:
                a.people = min(a.people, a.trips * vm[a.vehicle_id].capacity)
                a.trips = min(a.trips, vm[a.vehicle_id].max_trips)
    elif v.type in ("compatibility", "road_accessibility"):
        plan.vehicle_assignments = [a for a in plan.vehicle_assignments
                                    if not (a.vehicle_id == v.entity and a.incident_id == v.incident_id)]
    elif v.type == "team_availability":
        drop = [t for t in plan.team_assignments if t.team_id == v.entity]
        if drop:
            plan.team_assignments.remove(drop[-1])
    elif v.type == "shelter_capacity":
        over = (v.actual or 0) - (v.expected or 0)
        for a in plan.shelter_allocations:
            if a.shelter_id == v.entity and over > 0:
                cut = min(a.people, over); a.people -= cut; over -= cut
    elif v.type == "coverage" and v.incident_id:
        tr = sum(a.people for a in plan.vehicle_assignments if a.incident_id == v.incident_id)
        inc = s.incident_map()[v.incident_id]
        plan.unserved_demand = [u for u in plan.unserved_demand if u.incident_id != v.incident_id]
        if max(inc.people - tr, 0) > 0:
            plan.unserved_demand.append(UnservedDemand(incident_id=v.incident_id, people=max(inc.people - tr, 0)))
        if cascade and not any(t.incident_id == v.incident_id for t in plan.team_assignments) and tr > 0:
            tm = [t for t in s.teams if t.available]
            if tm:
                plan.team_assignments.append(TeamAssignment(team_id=tm[0].id, incident_id=v.incident_id))
    if cascade:  # keep bookkeeping consistent
        for inc in s.incidents:
            tr = sum(a.people for a in plan.vehicle_assignments if a.incident_id == inc.id)
            plan.unserved_demand = [u for u in plan.unserved_demand if u.incident_id != inc.id]
            if inc.people - tr > 0:
                plan.unserved_demand.append(UnservedDemand(incident_id=inc.id, people=inc.people - tr))


class MockLLM(LLMClient):
    provider = "mock"
    is_simulated = True
    model = "simulated-llm"

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings

    def generate(self, prompt, *, system=None, temperature=None, max_output_tokens=None, seed=None,
                 expect_json=True, response_schema=None) -> LLMResult:
        rng = random.Random(derive_seed("mock", seed if seed is not None else len(prompt), prompt[:80], len(prompt)))
        if "<<<FEEDBACK>>>" in prompt:
            # M2 and M3 share one revision template; the simulated model tells the feedback sources apart by their content
            fb = _between(prompt, "<<<FEEDBACK>>>", "<<<END_FEEDBACK>>>")
            if "n_violations" in fb or "problem(s) found" in fb:   # validator report (structured or free text)
                text = self._repair(prompt, rng)
            else:                                                    # the model's own critique
                text = self._revise(prompt, rng, fb)
        elif "<<<PLAN_JSON>>>" in prompt:
            text = self._critique(prompt, rng)
        else:
            text = self._plan(prompt, rng)
        n_in, n_out = max(1, len(prompt) // 4), max(1, len(text) // 4)
        return LLMResult(text=text, model=self.model, model_version="simulated-1", input_tokens=n_in, output_tokens=n_out,
                         thoughts_tokens=0, total_tokens=n_in + n_out, latency_s=0.001, finish_reason="STOP",
                         tokens_estimated=True)

    def _plan(self, prompt: str, rng: random.Random) -> str:
        s = scenario_from_prompt(prompt)
        plan = greedy_plan(s)
        lam = FAULTS_PER_PLAN[("small" if len(s.incidents) <= 5 else "medium" if len(s.incidents) <= 10 else "large")]
        for _ in range(min(8, int(rng.expovariate(1 / lam) + 0.5))):
            inject_fault(s, plan, rng)
        if rng.random() < 0.03:
            return "Here is the plan: {\"team_assignments\": [  # truncated"
        return json.dumps(plan.model_dump())

    def _current(self, prompt: str):
        s = scenario_from_prompt(prompt)
        pr = parse_plan(_between(prompt, "<<<PLAN_JSON>>>", "<<<END_PLAN_JSON>>>"))
        return s, pr.plan

    def _repair(self, prompt: str, rng: random.Random) -> str:
        s, plan = self._current(prompt)
        if plan is None:
            return json.dumps(greedy_plan(s).model_dump())
        for v in validate_plan(s, plan).violations:
            if rng.random() < P_FIX:
                fix_violation(s, plan, v, rng)
        if rng.random() < P_NEW_FAULT:
            inject_fault(s, plan, rng)
        return json.dumps(plan.model_dump())

    def _critique(self, prompt: str, rng: random.Random) -> str:
        s, plan = self._current(prompt)
        issues = []
        if plan is None:
            issues.append({"rule": "R7", "description": "plan is malformed", "entity": "plan"})
        else:
            for v in validate_plan(s, plan).violations:
                if rng.random() < P_DETECT:
                    issues.append({"rule": v.rule.replace("V", "R"), "description": v.message, "entity": v.entity})
            if rng.random() < P_FALSE_POSITIVE:
                issues.append({"rule": "R1", "description": "possible capacity problem (unverified)", "entity": "VEH-01"})
        return json.dumps({"plan_is_valid": not issues, "issues": issues})

    def _revise(self, prompt: str, rng: random.Random, feedback: str) -> str:
        s, plan = self._current(prompt)
        crit, _ = parse_json_object(feedback)
        if plan is None:
            return json.dumps(greedy_plan(s).model_dump())
        flagged = {(i.get("entity")) for i in (crit or {}).get("issues", [])}
        for v in validate_plan(s, plan).violations:
            if v.entity in flagged:
                fix_violation(s, plan, v, rng)
        return json.dumps(plan.model_dump())
