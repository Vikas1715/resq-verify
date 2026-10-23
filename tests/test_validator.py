import copy

import pytest

from backend.app.schemas.plan import Plan
from backend.app.validators.metrics import plan_quality
from backend.app.validators.validator import report_for_unparsed, validate_plan
from tests.helpers import tiny_scenario, valid_plan_dict

SC = tiny_scenario()


def check(mutator):
    d = copy.deepcopy(valid_plan_dict())
    mutator(d)
    return validate_plan(SC, Plan.model_validate(d))


def test_valid_plan_has_no_violations():
    rep = validate_plan(SC, Plan.model_validate(valid_plan_dict()))
    assert rep.valid and rep.n_violations == 0 and sum(rep.counts_by_type.values()) == 0


def test_v1_capacity_per_assignment():
    rep = check(lambda d: d["vehicle_assignments"][0].update(people=45))
    assert rep.counts_by_type["vehicle_capacity"] >= 1 and not rep.valid


def test_v1_total_trips_exceed_vehicle_limit():
    def m(d):
        d["vehicle_assignments"][0].update(trips=2, people=30)
        d["vehicle_assignments"].append({"vehicle_id": "V1", "incident_id": "I2", "trips": 1, "people": 0})
        d["vehicle_assignments"][-1]["people"] = 1  # activates the record (bus is also incompatible with I2)
    rep = check(m)
    assert any(v.type == "vehicle_capacity" and "trips in total" in v.message for v in rep.violations)


def test_v2_unavailable_team():
    rep = check(lambda d: d["team_assignments"].append({"team_id": "T2", "incident_id": "I1"}))
    assert rep.counts_by_type["team_availability"] == 1


def test_v2_team_overloaded():
    def m(d):
        d["team_assignments"] = [{"team_id": "T2", "incident_id": "I1"}]  # T2 unavailable
    rep = check(m)
    assert rep.counts_by_type["team_availability"] >= 1


def test_v3_shelter_over_capacity():
    def m(d):
        d["shelter_allocations"] = [{"incident_id": "I1", "shelter_id": "S2", "people": 30},
                                    {"incident_id": "I2", "shelter_id": "S2", "people": 20}]
    rep = check(m)
    assert rep.counts_by_type["shelter_capacity"] == 1  # S2 gets 50 > 25


def test_v4_non_boat_to_blocked_road():
    def m(d):
        d["vehicle_assignments"][1] = {"vehicle_id": "V3", "incident_id": "I2", "trips": 2, "people": 20}
    rep = check(m)
    assert rep.counts_by_type["road_accessibility"] == 1
    assert rep.counts_by_type["compatibility"] == 0  # a truck is allowed at I2; only the road is the problem


def test_v5_coverage_mismatch_and_unserved_wrong():
    rep = check(lambda d: d["shelter_allocations"][0].update(people=25))
    assert rep.counts_by_type["coverage"] >= 1
    rep = check(lambda d: d["unserved_demand"].append({"incident_id": "I1", "people": 7}))
    assert any("unserved demand is reported" in v.message for v in rep.violations)


def test_v5_transport_without_team():
    rep = check(lambda d: d.update(team_assignments=[{"team_id": "T1", "incident_id": "I1"}]))
    assert any(v.type == "coverage" and v.incident_id == "I2" and "no team" in v.message for v in rep.violations)


def test_v5_more_than_people_present():
    def m(d):
        d["vehicle_assignments"][0].update(trips=2, people=40)
        d["shelter_allocations"][0].update(people=40)
    rep = check(m)
    assert any("are assigned to be transported" in v.message for v in rep.violations)


def test_v6_incompatible_vehicle_type():
    def m(d):
        d["vehicle_assignments"][0] = {"vehicle_id": "V2", "incident_id": "I1", "trips": 3, "people": 30}
        d["vehicle_assignments"][1] = {"vehicle_id": "V1", "incident_id": "I2", "trips": 1, "people": 20}
    rep = check(m)
    assert rep.counts_by_type["compatibility"] >= 1


def test_v7_unknown_ids_are_reported_once_without_cascade():
    rep = check(lambda d: d["vehicle_assignments"].append({"vehicle_id": "VX", "incident_id": "I1", "trips": 1, "people": 5}))
    assert rep.counts_by_type["schema_integrity"] == 1
    assert rep.n_violations == 1


def test_v7_duplicates_and_negatives():
    rep = check(lambda d: d["team_assignments"].append({"team_id": "T1", "incident_id": "I1"}))
    assert rep.counts_by_type["schema_integrity"] == 1
    rep = check(lambda d: d["vehicle_assignments"][0].update(people=-3))
    assert rep.counts_by_type["schema_integrity"] == 1


def test_unparseable_output_is_a_failed_report():
    rep = report_for_unparsed("invalid JSON")
    assert not rep.valid and rep.parse_failed and rep.counts_by_type["schema_integrity"] >= 1


def test_empty_plan_is_invalid_because_unserved_not_reported():
    rep = validate_plan(SC, Plan())
    assert not rep.valid
    assert all(v.type == "coverage" for v in rep.violations)


def test_quality_uses_actual_records_not_llm_claims():
    q = plan_quality(SC, Plan.model_validate(valid_plan_dict()))
    assert q["weighted_served"] == 210 and q["quality_norm"] == pytest.approx(1.0)
    d = valid_plan_dict()
    d["unserved_demand"] = []  # claims nothing unserved...
    d["vehicle_assignments"] = d["vehicle_assignments"][:1]  # ...but I2 has no transport
    q = plan_quality(SC, Plan.model_validate(d))
    assert q["people_served"] == 30
