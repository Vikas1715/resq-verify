"""Classical baseline: Google OR-Tools CP-SAT.

Decision variables (all integer / boolean):
  trips[v,i]   trips of vehicle v to incident i        (0..max_trips)
  people[v,i]  people moved by v to i                  (0..capacity*max_trips), people <= capacity * trips
  assign[t,i]  team t works on incident i              (bool, available teams only)
  used[i]      incident i is served at all             (bool)
  shelter[i,s] people from incident i placed in shelter s
Constraints encode exactly the hard rules checked by the validator (V1-V7).

Objective (maximise):  1000 * sum_i priority_i * served_i  -  sum(trips)  -  sum(team assignments)
The small penalty terms only break ties (always < 1000, i.e. less than one person at priority 1), so the
primary objective is exactly "priority-weighted people served". We report that primary value separately.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from backend.app.core.errors import SolverError
from backend.app.schemas.plan import (
    Plan, ShelterAllocation, TeamAssignment, UnservedDemand, VehicleAssignment,
)
from backend.app.schemas.scenario import Scenario
from backend.app.validators.metrics import max_weighted_demand, weighted_served

try:  # pragma: no cover - import guard exercised on machines without OR-Tools
    from ortools.sat.python import cp_model
except Exception as exc:  # noqa: BLE001
    cp_model = None
    _IMPORT_ERROR = str(exc)
else:
    _IMPORT_ERROR = ""

PRIMARY_WEIGHT = 1000


def solver_available() -> bool:
    return cp_model is not None


def solve_scenario(scenario: Scenario, *, time_limit_s: float = 10.0, workers: int = 1, seed: int = 0) -> dict[str, Any]:
    """Solve one scenario. Returns a JSON-friendly dict including a `plan` in the same schema the LLM must use."""
    if cp_model is None:
        raise SolverError("OR-Tools is not installed. Run: pip install ortools", details={"import_error": _IMPORT_ERROR})

    t0 = time.perf_counter()
    m = cp_model.CpModel()
    inc, veh, shl = scenario.incident_map(), scenario.vehicle_map(), scenario.shelter_map()
    road = scenario.road_map()

    pairs = [(v.id, i.id) for v in scenario.vehicles for i in scenario.incidents
             if v.type in i.allowed_vehicle_types and (v.type == "boat" or not road[i.road_id].blocked)]
    trips, people = {}, {}
    for vid, iid in pairs:
        v = veh[vid]
        trips[vid, iid] = m.NewIntVar(0, v.max_trips, f"trips_{vid}_{iid}")
        people[vid, iid] = m.NewIntVar(0, v.capacity * v.max_trips, f"people_{vid}_{iid}")
        m.Add(people[vid, iid] <= v.capacity * trips[vid, iid])
    for v in scenario.vehicles:
        mine = [trips[k] for k in pairs if k[0] == v.id]
        if mine:
            m.Add(sum(mine) <= v.max_trips)

    served, used = {}, {}
    assign = {}
    for i in scenario.incidents:
        served[i.id] = m.NewIntVar(0, i.people, f"served_{i.id}")
        used[i.id] = m.NewBoolVar(f"used_{i.id}")
        mine = [people[k] for k in pairs if k[1] == i.id]
        m.Add(served[i.id] == (sum(mine) if mine else 0))
        m.Add(served[i.id] <= i.people * used[i.id])
    for t in scenario.teams:
        if not t.available:
            continue
        for i in scenario.incidents:
            assign[t.id, i.id] = m.NewBoolVar(f"assign_{t.id}_{i.id}")
        m.Add(sum(assign[t.id, i.id] for i in scenario.incidents) <= t.max_incidents)
    for i in scenario.incidents:
        team_vars = [assign[k] for k in assign if k[1] == i.id]
        if team_vars:
            m.Add(used[i.id] <= sum(team_vars))
        else:
            m.Add(used[i.id] == 0)

    place = {}
    for i in scenario.incidents:
        for s in scenario.shelters:
            place[i.id, s.id] = m.NewIntVar(0, min(i.people, s.capacity), f"place_{i.id}_{s.id}")
        m.Add(sum(place[i.id, s.id] for s in scenario.shelters) == served[i.id])
    for s in scenario.shelters:
        m.Add(sum(place[i.id, s.id] for i in scenario.incidents) <= s.capacity)

    weighted = sum(inc[iid].priority * served[iid] for iid in served)
    penalty = (sum(trips.values()) if trips else 0) + (sum(assign.values()) if assign else 0)
    m.Maximize(PRIMARY_WEIGHT * weighted - penalty)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.random_seed = int(seed)
    try:
        solver.parameters.num_workers = int(workers)
    except AttributeError:  # older OR-Tools versions
        solver.parameters.num_search_workers = int(workers)
    status = solver.Solve(m)
    status_name = solver.StatusName(status)
    runtime = time.perf_counter() - t0

    result: dict[str, Any] = {
        "status": status_name, "feasible": False, "optimal": status_name == "OPTIMAL",
        "runtime_s": round(runtime, 4), "wall_time_s": round(float(solver.WallTime()), 4),
        "time_limit_s": float(time_limit_s), "workers": int(workers), "seed": int(seed),
        "n_variables": len(trips) + len(people) + len(served) + len(used) + len(assign) + len(place),
        "plan": None, "objective_weighted_served": None, "objective_normalized": None,
        "internal_objective": None, "best_bound": None, "relative_gap": None,
        "objective_description": f"maximise {PRIMARY_WEIGHT}*sum(priority*people_served) - total_trips - team_assignments",
    }
    if status_name not in ("OPTIMAL", "FEASIBLE"):
        return result

    plan = Plan(
        team_assignments=[TeamAssignment(team_id=t, incident_id=i) for (t, i), var in sorted(assign.items())
                          if solver.Value(var) == 1],
        vehicle_assignments=[VehicleAssignment(vehicle_id=v, incident_id=i, trips=int(solver.Value(trips[v, i])),
                                               people=int(solver.Value(people[v, i])))
                             for (v, i) in sorted(pairs) if solver.Value(people[v, i]) > 0],
        shelter_allocations=[ShelterAllocation(incident_id=i, shelter_id=s, people=int(solver.Value(var)))
                             for (i, s), var in sorted(place.items()) if solver.Value(var) > 0],
        unserved_demand=[UnservedDemand(incident_id=i.id, people=i.people - int(solver.Value(served[i.id])))
                         for i in scenario.incidents if i.people - int(solver.Value(served[i.id])) > 0],
        reasoning_summary="CP-SAT solution",
    )
    internal = float(solver.ObjectiveValue())
    bound = float(solver.BestObjectiveBound())
    ws = weighted_served(scenario, plan)
    denom = max_weighted_demand(scenario)
    result.update({
        "feasible": True, "plan": plan.model_dump(), "objective_weighted_served": ws,
        "objective_normalized": (ws / denom) if denom else None, "internal_objective": internal,
        "best_bound": bound, "relative_gap": abs(bound - internal) / max(1.0, abs(internal)),
    })
    return result


def solve_plan(scenario: Scenario, **kw: Any) -> tuple[Optional[Plan], dict[str, Any]]:
    res = solve_scenario(scenario, **kw)
    plan = Plan.model_validate(res["plan"]) if res.get("plan") else None
    return plan, res
