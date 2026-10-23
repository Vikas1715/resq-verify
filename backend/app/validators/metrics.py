"""Plan-quality metrics. Quality is computed from the plan's actual transport and shelter records, not from
whatever the LLM claims in `unserved_demand` or `reasoning_summary`."""
from __future__ import annotations

from collections import defaultdict

from backend.app.schemas.plan import Plan
from backend.app.schemas.scenario import Scenario


def served_by_incident(scenario: Scenario, plan: Plan) -> dict[str, int]:
    """People counted as served = min(transported, sheltered, incident.people), using only valid, non-negative records."""
    inc = scenario.incident_map()
    veh = scenario.vehicle_map()
    shl = scenario.shelter_map()
    tr: dict[str, int] = defaultdict(int)
    sh: dict[str, int] = defaultdict(int)
    for a in plan.vehicle_assignments:
        if a.vehicle_id in veh and a.incident_id in inc and a.people > 0:
            tr[a.incident_id] += a.people
    for s in plan.shelter_allocations:
        if s.shelter_id in shl and s.incident_id in inc and s.people > 0:
            sh[s.incident_id] += s.people
    return {iid: min(tr[iid], sh[iid], inc[iid].people) for iid in inc}


def max_weighted_demand(scenario: Scenario) -> int:
    return sum(i.priority * i.people for i in scenario.incidents)


def weighted_served(scenario: Scenario, plan: Plan) -> int:
    inc = scenario.incident_map()
    return sum(inc[iid].priority * n for iid, n in served_by_incident(scenario, plan).items())


def plan_quality(scenario: Scenario, plan: Plan) -> dict:
    """quality_norm in [0, 1]: priority-weighted people served / priority-weighted demand (higher is better)."""
    served = served_by_incident(scenario, plan)
    ws = weighted_served(scenario, plan)
    denom = max_weighted_demand(scenario)
    return {
        "weighted_served": ws,
        "weighted_demand": denom,
        "quality_norm": (ws / denom) if denom > 0 else None,
        "people_served": sum(served.values()),
        "people_total": sum(i.people for i in scenario.incidents),
    }
