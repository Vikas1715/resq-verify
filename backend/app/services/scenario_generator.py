"""Seeded synthetic scenario generator.

Same (size, scarcity, seed, overrides, generator_version) always gives the identical scenario.

Scarcity is defined by capacity-to-demand ratios (nominal, before reachability limits):
    transport_ratio = total vehicle capacity (people per trip x trips) / total demand
    shelter_ratio   = total shelter capacity / total demand
"""
from __future__ import annotations

import random
from typing import Any, Optional

from backend.app.core.errors import BadRequestError
from backend.app.schemas.scenario import (
    Difficulty, Incident, Road, Scenario, Shelter, Team, Vehicle,
)
from backend.app.utils.ids import stable_hash, utcnow_iso

GENERATOR_VERSION = "1.0"

SIZE_PROFILES: dict[str, dict[str, int]] = {
    "small": {"n_incidents": 4, "n_teams": 3, "n_vehicles": 6, "n_shelters": 3},
    "medium": {"n_incidents": 8, "n_teams": 6, "n_vehicles": 12, "n_shelters": 5},
    "large": {"n_incidents": 14, "n_teams": 10, "n_vehicles": 20, "n_shelters": 8},
}

SCARCITY_PROFILES: dict[str, dict[str, float]] = {
    "low": {"transport_ratio": 1.6, "shelter_ratio": 1.6, "team_unavailable_prob": 0.05},
    "medium": {"transport_ratio": 1.0, "shelter_ratio": 1.0, "team_unavailable_prob": 0.15},
    "high": {"transport_ratio": 0.6, "shelter_ratio": 0.65, "team_unavailable_prob": 0.25},
}

VEHICLE_SPECS = {  # type -> (people per trip, min trips, max trips)
    "bus": (30, 1, 3), "truck": (20, 1, 3), "ambulance": (6, 2, 4), "boat": (12, 1, 3),
}
VEHICLE_WEIGHTS = [("bus", 0.25), ("truck", 0.30), ("ambulance", 0.20), ("boat", 0.25)]
HAZARDS = ["flood", "earthquake", "landslide", "building_fire"]
ZONE_NAMES = ["Riverside", "Hillcrest", "Old Market", "Lakeview", "Station Road", "Harbour", "Greenfield",
              "North Camp", "Temple Hill", "Canal Street", "Mill Quarter", "East Ward", "Orchard Lane", "Bridge End"]
OVERRIDE_KEYS = {"n_incidents", "n_teams", "n_vehicles", "n_shelters", "blocked_road_prob", "transport_ratio", "shelter_ratio"}

# ---------------------------------------------------------------------------- difficulty levels
# A level is a named bundle of the existing knobs (size profile, scarcity profile, blocked-road probability) plus four extra,
# optional knobs. It does not replace the generator: with no level the generator behaves exactly as before.
#   compat_restrict_prob  chance that each allowed vehicle type is removed from an incident (one usable type is always kept)
#   boat_cap              at most this many boats in the fleet (boats are the only way into blocked-road sites)
#   team_slot_ratio       available team slots are limited to about ratio x incidents
#   shelter_skew          share of all shelter capacity concentrated in one shelter (the rest is small and spread out)
#   interlock_prob        chance that a blocked-road site is reachable ONLY by boat, so boats, teams and shelters all interact
# Honest note: these presets are design choices, not measured difficulty. Calibrate them on a pilot (see docs/methodology.md).
LEVEL_PRESETS: dict[str, dict[str, Any]] = {
    "L0": {"size": "small", "scarcity": "low", "blocked_road_prob": 0.2,
           "desc": "easy baseline: small, ample capacity"},
    "L1": {"size": "medium", "scarcity": "medium", "blocked_road_prob": 0.2,
           "desc": "larger scenario, capacity roughly equals demand"},
    "L2": {"size": "large", "scarcity": "medium", "blocked_road_prob": 0.4,
           "desc": "large scenario with many blocked roads"},
    "L3": {"size": "large", "scarcity": "medium", "blocked_road_prob": 0.3, "compat_restrict_prob": 0.35,
           "boat_cap": 2, "team_slot_ratio": 0.7,
           "desc": "large + vehicle-type restrictions + boat and team bottlenecks"},
    "L4": {"size": "large", "scarcity": "high", "blocked_road_prob": 0.4, "compat_restrict_prob": 0.35,
           "boat_cap": 2, "team_slot_ratio": 0.6, "shelter_skew": 0.4, "interlock_prob": 0.7,
           "desc": "large + high scarcity + interacting constraints (boat-only sites, few teams, one big shelter)"},
}
DIFFICULTY_LEVELS = list(LEVEL_PRESETS)
LEVEL_GENERATOR_VERSION = "1.1"
MIN_SERVABLE_FRACTION = 0.7   # level scenarios are repaired until at least this share of incidents can be served by someone


def make_scenario_id(size: str, scarcity: str, seed: int, overrides: Optional[dict[str, Any]], level: Optional[str] = None) -> str:
    sid = f"{level}-s{seed}" if level else f"{size}-{scarcity}-s{seed}"
    if overrides:
        sid += "-o" + stable_hash(overrides, 6)
    return sid


def _weighted_choice(rng: random.Random, pairs: list[tuple[str, float]]) -> str:
    r = rng.random() * sum(w for _, w in pairs)
    acc = 0.0
    for name, w in pairs:
        acc += w
        if r <= acc:
            return name
    return pairs[-1][0]


def generate_scenario(size: str, scarcity: str, seed: int, overrides: Optional[dict[str, Any]] = None) -> Scenario:
    """The original generator (no difficulty level)."""
    return _build(size, scarcity, seed, overrides)


def generate_level_scenario(level: str, seed: int, overrides: Optional[dict[str, Any]] = None) -> Scenario:
    """Scenario for a named difficulty level (L0..L4). Deterministic in (level, seed, overrides)."""
    if level not in LEVEL_PRESETS:
        raise BadRequestError(f"Unknown difficulty level '{level}'. Use one of {DIFFICULTY_LEVELS}")
    pre = LEVEL_PRESETS[level]
    return _build(pre["size"], pre["scarcity"], seed, overrides, level=level)


def _build(size: str, scarcity: str, seed: int, overrides: Optional[dict[str, Any]] = None, level: Optional[str] = None) -> Scenario:
    if size not in SIZE_PROFILES:
        raise BadRequestError(f"Unknown size '{size}'. Use one of {list(SIZE_PROFILES)}")
    if scarcity not in SCARCITY_PROFILES:
        raise BadRequestError(f"Unknown scarcity '{scarcity}'. Use one of {list(SCARCITY_PROFILES)}")
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    unknown = set(overrides) - OVERRIDE_KEYS
    if unknown:
        raise BadRequestError(f"Unknown override(s): {sorted(unknown)}. Allowed: {sorted(OVERRIDE_KEYS)}")

    p = {**SIZE_PROFILES[size], **SCARCITY_PROFILES[scarcity], "blocked_road_prob": 0.2}
    pre = LEVEL_PRESETS[level] if level else {}
    if "blocked_road_prob" in pre:
        p["blocked_road_prob"] = pre["blocked_road_prob"]
    compat_p = float(pre.get("compat_restrict_prob", 0.0))
    boat_cap = pre.get("boat_cap")
    team_ratio = pre.get("team_slot_ratio")
    shelter_skew = float(pre.get("shelter_skew", 0.0))
    interlock_p = float(pre.get("interlock_prob", 0.0))
    p.update(overrides)
    for k in ("n_incidents", "n_teams", "n_vehicles", "n_shelters"):
        p[k] = int(p[k])
        if p[k] < 1:
            raise BadRequestError(f"{k} must be >= 1")
    if not 0.0 <= float(p["blocked_road_prob"]) <= 1.0:
        raise BadRequestError("blocked_road_prob must be between 0 and 1")
    if float(p["transport_ratio"]) <= 0 or float(p["shelter_ratio"]) <= 0:
        raise BadRequestError("transport_ratio and shelter_ratio must be > 0")

    if level:
        rng = random.Random(f"resq|{LEVEL_GENERATOR_VERSION}|{level}|{size}|{scarcity}|{seed}|{stable_hash(overrides, 8)}")
    else:  # unchanged from the original generator, so old seeds still give the same scenarios
        rng = random.Random(f"resq|{GENERATOR_VERSION}|{size}|{scarcity}|{seed}|{stable_hash(overrides, 8)}")
    n_inc, n_team, n_veh, n_shl = p["n_incidents"], p["n_teams"], p["n_vehicles"], p["n_shelters"]

    # ---- roads and incidents
    roads, incidents_raw = [], []
    for k in range(n_inc):
        blocked = rng.random() < float(p["blocked_road_prob"])
        hazard = rng.choice(HAZARDS)
        waterway = hazard == "flood" or rng.random() < 0.15
        if blocked and not waterway and rng.random() < 0.5:
            waterway = True  # some blocked sites stay reachable by water; the rest are unreachable
        narrow = hazard in ("earthquake", "building_fire") and rng.random() < 0.4
        allowed = ["truck", "ambulance"] + ([] if narrow else ["bus"]) + (["boat"] if waterway else [])
        allowed = [t for t in ("bus", "truck", "ambulance", "boat") if t in allowed]
        roads.append(Road(id=f"RD-{k + 1:02d}", blocked=blocked))
        incidents_raw.append({
            "k": k, "hazard": hazard, "waterway": waterway, "narrow": narrow, "allowed": allowed,
            "priority": rng.randint(1, 5), "weight": rng.uniform(0.5, 1.5), "road": f"RD-{k + 1:02d}",
            "zone": ZONE_NAMES[k % len(ZONE_NAMES)],
        })

    # ---- level knobs on incidents (only when a level is used; they draw from the same seeded rng, after the base draws)
    restricted = 0
    if level:
        blocked_by_road = {r.id: r.blocked for r in roads}
        for i in incidents_raw:
            blocked = blocked_by_road[i["road"]]
            if blocked and rng.random() < interlock_p:
                i["waterway"], i["allowed"] = True, ["boat"]   # reachable only by boat: boats, teams and shelters now interact
                continue
            if compat_p > 0:
                before = list(i["allowed"])
                kept = [t for t in before if rng.random() >= compat_p]
                usable = [t for t in kept if t == "boat" or not blocked]
                if not usable:  # keep one type that can actually get there
                    pool = [t for t in before if t == "boat" or not blocked]
                    if pool:
                        kept.append(pool[rng.randrange(len(pool))])
                kept = [t for t in ("bus", "truck", "ambulance", "boat") if t in kept]
                if kept and kept != before:
                    i["allowed"] = kept
                    restricted += 1

    # ---- vehicles (make sure a boat exists when a waterway site exists, and a road vehicle exists)
    types = [_weighted_choice(rng, VEHICLE_WEIGHTS) for _ in range(n_veh)]
    if n_veh >= 3:
        if any(i["waterway"] for i in incidents_raw) and "boat" not in types:
            types[rng.randrange(n_veh)] = "boat"
        if not any(t in ("bus", "truck") for t in types):
            types[rng.randrange(n_veh)] = "truck"
    if level and boat_cap is not None:  # boat bottleneck: surplus boats become trucks, but keep one boat if any site needs it
        seen = 0
        for k, t in enumerate(types):
            if t == "boat":
                seen += 1
                if seen > int(boat_cap):
                    types[k] = "truck"
        if any(i["waterway"] for i in incidents_raw) and "boat" not in types and n_veh >= 3:
            types[rng.randrange(n_veh)] = "boat"
    if level:  # solvability guard: make sure most incidents can be served by somebody (never changes non-level scenarios)
        blocked_by_road = {r.id: r.blocked for r in roads}

        def servable(i):
            return any(t in i["allowed"] and (t == "boat" or not blocked_by_road[i["road"]]) for t in set(types))

        for i in incidents_raw:
            if sum(servable(x) for x in incidents_raw) >= MIN_SERVABLE_FRACTION * len(incidents_raw):
                break
            if servable(i):
                continue
            if blocked_by_road[i["road"]]:  # a blocked site is only ever reachable by boat
                i["waterway"] = True
                i["allowed"] = [t for t in ("bus", "truck", "ambulance", "boat") if t in i["allowed"] or t == "boat"]
                if "boat" not in types:
                    types[rng.randrange(n_veh)] = "boat"
            else:
                for t in ("truck", "bus"):
                    if t in types:
                        i["allowed"] = [x for x in ("bus", "truck", "ambulance", "boat") if x in i["allowed"] or x == t]
                        break
                else:
                    types[rng.randrange(n_veh)] = "truck"
                    i["allowed"] = [x for x in ("bus", "truck", "ambulance", "boat") if x in i["allowed"] or x == "truck"]
    vehicles = []
    for k, t in enumerate(types):
        cap, lo, hi = VEHICLE_SPECS[t]
        vehicles.append(Vehicle(id=f"VEH-{k + 1:02d}", type=t, capacity=cap, max_trips=rng.randint(lo, hi)))
    total_transport = sum(v.capacity * v.max_trips for v in vehicles)

    # ---- demand sized so that nominal transport_ratio holds
    target_demand = max(n_inc * 5, round(total_transport / float(p["transport_ratio"])))
    wsum = sum(i["weight"] for i in incidents_raw)
    incidents = []
    for i in incidents_raw:
        people = max(5, round(target_demand * i["weight"] / wsum))
        incidents.append(Incident(
            id=f"INC-{i['k'] + 1:02d}", name=f"{i['zone']} ({i['hazard'].replace('_', ' ')})", hazard=i["hazard"],
            people=people, priority=i["priority"], road_id=i["road"], waterway=i["waterway"],
            narrow_access=i["narrow"], allowed_vehicle_types=i["allowed"]))
    total_demand = sum(x.people for x in incidents)

    # ---- shelters sized by shelter_ratio
    sw = [rng.uniform(0.5, 1.5) for _ in range(n_shl)]
    target_shelter = round(total_demand * float(p["shelter_ratio"]))
    shares = [w / sum(sw) for w in sw]
    if level and shelter_skew > 0 and n_shl > 1:  # bottleneck: one shelter holds a big share, the others are small
        shares = [(1 - shelter_skew) * x for x in shares]
        shares[0] += shelter_skew
    shelters = [Shelter(id=f"SHL-{k + 1:02d}", name=f"Shelter {chr(65 + k % 26)}",
                        capacity=max(10, round(target_shelter * share))) for k, share in enumerate(shares)]

    # ---- teams
    teams = []
    for k in range(n_team):
        teams.append(Team(id=f"TEAM-{k + 1:02d}", available=rng.random() >= float(p["team_unavailable_prob"]),
                          max_incidents=1 if rng.random() < 0.6 else 2))
    if level and team_ratio is not None:  # team bottleneck: only about ratio x incidents team slots are usable
        want = max(1, min(n_team, round(float(team_ratio) * n_inc)))
        order = list(range(n_team))
        rng.shuffle(order)
        for rank, idx in enumerate(order):
            teams[idx].max_incidents = 1
            teams[idx].available = rank < want
    if not any(t.available for t in teams):
        teams[0].available = True

    # ---- difficulty metadata
    roadmap = {r.id: r for r in roads}
    pairs = sum(1 for v in vehicles for i in incidents
                if v.type in i.allowed_vehicle_types and (v.type == "boat" or not roadmap[i.road_id].blocked))
    unreachable = sum(1 for i in incidents
                      if roadmap[i.road_id].blocked and not i.waterway)
    team_slots = sum(t.max_incidents for t in teams if t.available)
    blocked_frac = sum(r.blocked for r in roads) / len(roads)
    unavailable_frac = sum(not t.available for t in teams) / len(teams)
    d2t = total_demand / max(total_transport, 1)
    # heuristic: bigger, more scarce, more blocked/unavailable = harder (documented in docs/methodology.md)
    score = n_inc * (1 + max(0.0, d2t - 0.5)) * (1 + blocked_frac + unavailable_frac)
    difficulty = Difficulty(
        total_demand=total_demand, total_transport_capacity=total_transport,
        total_shelter_capacity=sum(s.capacity for s in shelters),
        demand_to_transport_ratio=round(d2t, 4),
        demand_to_shelter_ratio=round(total_demand / max(sum(s.capacity for s in shelters), 1), 4),
        blocked_road_fraction=round(blocked_frac, 4), unreachable_incidents=unreachable,
        available_team_slots=team_slots, team_load_ratio=round(n_inc / max(team_slots, 1), 4),
        feasible_vehicle_incident_pairs=pairs, difficulty_score=round(score, 3))
    if level:
        fleet = {v.type for v in vehicles}
        ok_inc = [i for i in incidents if any(t in fleet and (t == "boat" or not roadmap[i.road_id].blocked)
                                              for t in i.allowed_vehicle_types)]
        difficulty.servable_incidents = len(ok_inc)
        difficulty.servable_demand_fraction = round(sum(i.people for i in ok_inc) / max(total_demand, 1), 4)
        difficulty.boat_count = sum(1 for v in vehicles if v.type == "boat")
        difficulty.restricted_incidents = restricted

    sc = Scenario(
        scenario_id=make_scenario_id(size, scarcity, seed, overrides, level),
        generator_version=LEVEL_GENERATOR_VERSION if level else GENERATOR_VERSION,
        seed=seed, size=size, scarcity=scarcity, difficulty_level=level,
        params={**{k: p[k] for k in ("n_incidents", "n_teams", "n_vehicles", "n_shelters", "blocked_road_prob",
                                     "transport_ratio", "shelter_ratio", "team_unavailable_prob")},
                **({"level": level, "compat_restrict_prob": compat_p, "boat_cap": boat_cap, "team_slot_ratio": team_ratio,
                    "shelter_skew": shelter_skew, "interlock_prob": interlock_p} if level else {}),
                "overrides": overrides},
        incidents=incidents, roads=roads, teams=teams, vehicles=vehicles, shelters=shelters, difficulty=difficulty)
    sc.content_hash = scenario_content_hash(sc)
    sc.created_at = utcnow_iso()
    return sc


def scenario_content_hash(sc: Scenario) -> str:
    """Hash of everything that defines the problem (excludes created_at and the hash itself)."""
    d = sc.model_dump(exclude={"created_at", "content_hash"})
    # fields added with difficulty levels are left out when empty, so scenarios made without a level keep their old hash
    if d.get("difficulty_level") is None:
        d.pop("difficulty_level", None)
    diff = d.get("difficulty")
    if isinstance(diff, dict):
        for k in ("servable_incidents", "servable_demand_fraction", "boat_count", "restricted_incidents"):
            if diff.get(k) is None:
                diff.pop(k, None)
    return stable_hash(d, 16)
