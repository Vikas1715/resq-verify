import pytest

from backend.app.core.errors import BadRequestError

from backend.app.services.scenario_generator import generate_scenario, make_scenario_id
from backend.app.validators.metrics import max_weighted_demand


def test_same_seed_same_scenario():
    a, b = generate_scenario("medium", "high", 5), generate_scenario("medium", "high", 5)
    assert a.model_dump(exclude={"created_at"}) == b.model_dump(exclude={"created_at"})
    assert a.content_hash == b.content_hash


def test_different_seed_differs():
    assert generate_scenario("small", "high", 1).content_hash != generate_scenario("small", "high", 2).content_hash


@pytest.mark.parametrize("size,counts", [("small", (4, 3, 6, 3)), ("medium", (8, 6, 12, 5)), ("large", (14, 10, 20, 8))])
def test_size_profiles(size, counts):
    s = generate_scenario(size, "medium", 3)
    assert (len(s.incidents), len(s.teams), len(s.vehicles), len(s.shelters)) == counts


def test_scarcity_orders_difficulty():
    # same seed, scarcer resources -> less transport capacity per person
    lo, hi = generate_scenario("medium", "low", 11), generate_scenario("medium", "high", 11)
    assert lo.difficulty.demand_to_transport_ratio < hi.difficulty.demand_to_transport_ratio
    assert lo.difficulty.demand_to_shelter_ratio < hi.difficulty.demand_to_shelter_ratio


def test_scenario_is_internally_consistent():
    s = generate_scenario("large", "high", 9)
    roads = {r.id for r in s.roads}
    assert all(i.road_id in roads for i in s.incidents)
    assert all(i.allowed_vehicle_types for i in s.incidents)
    assert max_weighted_demand(s) > 0 and 0 <= s.difficulty.difficulty_score


def test_id_and_overrides():
    assert make_scenario_id("small", "high", 7, None) == "small-high-s7"
    s = generate_scenario("small", "high", 7, {"n_incidents": 5})
    assert len(s.incidents) == 5 and s.scenario_id != "small-high-s7"


def test_unknown_size_rejected():
    with pytest.raises(BadRequestError):
        generate_scenario("huge", "high", 1)
