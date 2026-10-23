"""Deterministic hard-constraint validator (never calls an LLM).

Rules
-----
V1 vehicle_capacity    people <= trips * capacity per assignment; total trips per vehicle <= max_trips
V2 team_availability   only available teams; at most `max_incidents` distinct incidents per team
V3 shelter_capacity    total people allocated to a shelter <= capacity
V4 road_accessibility  non-boat vehicles cannot serve an incident whose road is blocked
V5 coverage            transported <= incident people; transported == sheltered; unserved reported exactly;
                       any incident with transported people has at least one team
V6 compatibility       vehicle type must be in the incident's allowed_vehicle_types
V7 schema_integrity    known IDs, no duplicates, non-negative numbers

Records that reference unknown IDs are reported by V7 only and are ignored by V1-V6, so one bad ID does not
create a cascade of unrelated violations.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from backend.app.schemas.plan import Plan
from backend.app.schemas.scenario import Scenario
from backend.app.schemas.validation import (
    RULE_FOR_TYPE, VIOLATION_TYPES, ValidationReport, Violation,
)

VALIDATOR_VERSION = "1.0"


def _v(vtype: str, entity: str, message: str, *, expected=None, actual=None, incident_id: Optional[str] = None) -> Violation:
    return Violation(rule=RULE_FOR_TYPE[vtype], type=vtype, entity=entity, incident_id=incident_id,
                     expected=expected, actual=actual, message=message)


def _report(violations: list[Violation], *, parse_failed: bool = False, parse_error: Optional[str] = None) -> ValidationReport:
    order = {t: i for i, t in enumerate(VIOLATION_TYPES)}
    violations = sorted(violations, key=lambda x: (order[x.type], x.entity, x.incident_id or "", x.message))
    counts = {t: 0 for t in VIOLATION_TYPES}
    for x in violations:
        counts[x.type] += 1
    return ValidationReport(valid=len(violations) == 0, n_violations=len(violations), violations=violations,
                            counts_by_type=counts, parse_failed=parse_failed, parse_error=parse_error,
                            validator_version=VALIDATOR_VERSION)


def report_for_unparsed(parse_error: str) -> ValidationReport:
    """A plan that could not be parsed is invalid (schema_integrity)."""
    v = _v("schema_integrity", "plan", f"Output could not be parsed as a valid plan: {parse_error}",
           expected="a JSON object matching the plan schema", actual="unparseable output")
    return _report([v], parse_failed=True, parse_error=parse_error)


def validate_plan(scenario: Scenario, plan: Plan) -> ValidationReport:
    inc = scenario.incident_map()
    veh = scenario.vehicle_map()
    team = scenario.team_map()
    shl = scenario.shelter_map()
    road = scenario.road_map()
    out: list[Violation] = []

    # ---------------------------------------------------------------- V7 integrity
    def seen_dups(keys: list[tuple], label: str) -> None:
        counts: dict[tuple, int] = defaultdict(int)
        for k in keys:
            counts[k] += 1
        for k, c in counts.items():
            if c > 1:
                out.append(_v("schema_integrity", str(k[0]), f"Duplicate {label} record for {k} appears {c} times",
                              expected=1, actual=c, incident_id=k[1] if len(k) > 1 else None))

    seen_dups([(t.team_id, t.incident_id) for t in plan.team_assignments], "team assignment")
    seen_dups([(v.vehicle_id, v.incident_id) for v in plan.vehicle_assignments], "vehicle assignment")
    seen_dups([(a.incident_id, a.shelter_id) for a in plan.shelter_allocations], "shelter allocation")
    seen_dups([(u.incident_id,) for u in plan.unserved_demand], "unserved-demand")

    ok_team = []
    for t in plan.team_assignments:
        bad = [x for x, m in ((t.team_id, team), (t.incident_id, inc)) if x not in m]
        if bad:
            out.append(_v("schema_integrity", t.team_id, f"Team assignment references unknown id(s): {bad}",
                          expected="existing ids", actual=bad, incident_id=t.incident_id))
        else:
            ok_team.append(t)

    ok_veh = []
    for a in plan.vehicle_assignments:
        bad = [x for x, m in ((a.vehicle_id, veh), (a.incident_id, inc)) if x not in m]
        if bad:
            out.append(_v("schema_integrity", a.vehicle_id, f"Vehicle assignment references unknown id(s): {bad}",
                          expected="existing ids", actual=bad, incident_id=a.incident_id))
        elif a.trips < 0 or a.people < 0:
            out.append(_v("schema_integrity", a.vehicle_id, "Negative trips or people in vehicle assignment",
                          expected=">= 0", actual={"trips": a.trips, "people": a.people}, incident_id=a.incident_id))
        else:
            ok_veh.append(a)

    ok_shl = []
    for s in plan.shelter_allocations:
        bad = [x for x, m in ((s.shelter_id, shl), (s.incident_id, inc)) if x not in m]
        if bad:
            out.append(_v("schema_integrity", s.shelter_id, f"Shelter allocation references unknown id(s): {bad}",
                          expected="existing ids", actual=bad, incident_id=s.incident_id))
        elif s.people < 0:
            out.append(_v("schema_integrity", s.shelter_id, "Negative people in shelter allocation",
                          expected=">= 0", actual=s.people, incident_id=s.incident_id))
        else:
            ok_shl.append(s)

    ok_uns = []
    for u in plan.unserved_demand:
        if u.incident_id not in inc:
            out.append(_v("schema_integrity", u.incident_id, "Unserved-demand record references unknown incident",
                          expected="existing incident id", actual=u.incident_id, incident_id=u.incident_id))
        elif u.people < 0:
            out.append(_v("schema_integrity", u.incident_id, "Negative unserved people",
                          expected=">= 0", actual=u.people, incident_id=u.incident_id))
        else:
            ok_uns.append(u)

    active_veh = [a for a in ok_veh if a.people > 0 or a.trips > 0]

    # ---------------------------------------------------------------- V1 vehicle capacity
    trips_by_vehicle: dict[str, int] = defaultdict(int)
    for a in active_veh:
        v = veh[a.vehicle_id]
        trips_by_vehicle[a.vehicle_id] += a.trips
        limit = a.trips * v.capacity
        if a.people > limit:
            out.append(_v("vehicle_capacity", a.vehicle_id,
                          f"{a.vehicle_id} carries {a.people} people to {a.incident_id} with {a.trips} trip(s) x capacity {v.capacity} = {limit}",
                          expected=limit, actual=a.people, incident_id=a.incident_id))
    for vid, total in trips_by_vehicle.items():
        if total > veh[vid].max_trips:
            out.append(_v("vehicle_capacity", vid, f"{vid} is assigned {total} trips in total but has only {veh[vid].max_trips}",
                          expected=veh[vid].max_trips, actual=total))

    # ---------------------------------------------------------------- V6 compatibility / V4 road
    for a in active_veh:
        v, i = veh[a.vehicle_id], inc[a.incident_id]
        if v.type not in i.allowed_vehicle_types:
            out.append(_v("compatibility", a.vehicle_id,
                          f"{a.vehicle_id} (type {v.type}) cannot serve {a.incident_id}; allowed types: {i.allowed_vehicle_types}",
                          expected=i.allowed_vehicle_types, actual=v.type, incident_id=a.incident_id))
        r = road.get(i.road_id)
        if r is not None and r.blocked and v.type != "boat":
            out.append(_v("road_accessibility", a.vehicle_id,
                          f"{a.vehicle_id} (type {v.type}) is sent to {a.incident_id} but road {r.id} is blocked (only boats can reach it by water)",
                          expected="accessible road or boat", actual=v.type, incident_id=a.incident_id))

    # ---------------------------------------------------------------- V2 team availability
    incidents_by_team: dict[str, set[str]] = defaultdict(set)
    for t in ok_team:
        tm = team[t.team_id]
        incidents_by_team[t.team_id].add(t.incident_id)
        if not tm.available:
            out.append(_v("team_availability", t.team_id, f"{t.team_id} is not available but is assigned to {t.incident_id}",
                          expected="available team", actual="unavailable", incident_id=t.incident_id))
    for tid, incs in incidents_by_team.items():
        if len(incs) > team[tid].max_incidents:
            out.append(_v("team_availability", tid, f"{tid} is assigned to {len(incs)} incidents but can handle at most {team[tid].max_incidents}",
                          expected=team[tid].max_incidents, actual=len(incs)))

    # ---------------------------------------------------------------- V3 shelter capacity
    load: dict[str, int] = defaultdict(int)
    for s in ok_shl:
        load[s.shelter_id] += s.people
    for sid, total in load.items():
        if total > shl[sid].capacity:
            out.append(_v("shelter_capacity", sid, f"{sid} receives {total} people but its capacity is {shl[sid].capacity}",
                          expected=shl[sid].capacity, actual=total))

    # ---------------------------------------------------------------- V5 coverage / bookkeeping
    transported: dict[str, int] = defaultdict(int)
    sheltered: dict[str, int] = defaultdict(int)
    reported_unserved: dict[str, int] = defaultdict(int)
    teams_at: dict[str, int] = defaultdict(int)
    for a in active_veh:
        transported[a.incident_id] += a.people
    for s in ok_shl:
        sheltered[s.incident_id] += s.people
    for u in ok_uns:
        reported_unserved[u.incident_id] += u.people
    for t in ok_team:
        teams_at[t.incident_id] += 1
    for iid, i in inc.items():
        tr, sh, un = transported[iid], sheltered[iid], reported_unserved[iid]
        if tr > i.people:
            out.append(_v("coverage", iid, f"{iid} has {i.people} people but {tr} are assigned to be transported",
                          expected=f"<= {i.people}", actual=tr, incident_id=iid))
        if sh != tr:
            out.append(_v("coverage", iid, f"At {iid}, {tr} people are transported but {sh} are allocated to shelters",
                          expected=tr, actual=sh, incident_id=iid))
        true_unserved = max(i.people - tr, 0)
        if un != true_unserved:
            out.append(_v("coverage", iid, f"At {iid}, unserved demand is reported as {un} but should be {true_unserved} ({i.people} people minus {tr} transported)",
                          expected=true_unserved, actual=un, incident_id=iid))
        if tr > 0 and teams_at[iid] == 0:
            out.append(_v("coverage", iid, f"{iid} has {tr} people transported but no team assigned",
                          expected=">= 1 team", actual=0, incident_id=iid))

    return _report(out)
