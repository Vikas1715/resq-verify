import pytest

from backend.app.schemas.plan import Plan
from backend.app.services.scenario_generator import generate_scenario
from backend.app.solvers.cpsat_solver import solve_scenario, solver_available
from backend.app.validators.metrics import plan_quality
from backend.app.validators.validator import validate_plan
from tests.helpers import tiny_scenario

pytestmark = pytest.mark.skipif(not solver_available(), reason="ortools is not installed")


def test_known_optimum_on_tiny_scenario():
    res = solve_scenario(tiny_scenario(), time_limit_s=10, workers=1, seed=1)
    assert res["status"] in ("OPTIMAL", "FEASIBLE")
    assert res["objective_weighted_served"] == 210          # everyone can be served: 5*30 + 3*20
    assert res["objective_normalized"] == pytest.approx(1.0)


@pytest.mark.parametrize("size", ["small", "medium"])
@pytest.mark.parametrize("scarcity", ["low", "high"])
def test_solver_plans_pass_the_independent_validator(size, scarcity):
    sc = generate_scenario(size, scarcity, 42)
    res = solve_scenario(sc, time_limit_s=10, workers=1, seed=1)
    plan = Plan.model_validate(res["plan"])
    rep = validate_plan(sc, plan)
    assert rep.valid, [v.message for v in rep.violations]
    assert plan_quality(sc, plan)["weighted_served"] == res["objective_weighted_served"]


def test_scarcer_scenario_cannot_serve_everyone():
    res = solve_scenario(generate_scenario("medium", "high", 3), time_limit_s=10, workers=1, seed=1)
    assert res["objective_normalized"] < 1.0
