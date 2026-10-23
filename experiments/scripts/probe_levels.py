"""Check what a difficulty level looks like, WITHOUT any LLM: how big it is, how constrained, and how much a simple greedy
heuristic (the same one the simulated-LLM test double uses) gets compared with the CP-SAT solver.

Why: the levels are design choices, not measured difficulty. If a greedy heuristic already reaches ~100% of the solver's
objective on a level, then the level mostly tests whether the LLM can follow the rules, not whether it can optimise.
Both are legitimate questions, but you should know which one you are asking.

  python experiments/scripts/probe_levels.py --levels L0 L1 L2 L3 L4 --seeds 3000 3009 --time-limit 10
Nothing here calls an LLM and nothing is stored.
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.services.llm.mock import greedy_plan  # noqa: E402  (a greedy heuristic; not an LLM)
from backend.app.services.scenario_generator import DIFFICULTY_LEVELS, generate_level_scenario  # noqa: E402
from backend.app.solvers.cpsat_solver import solve_scenario, solver_available  # noqa: E402
from backend.app.validators.metrics import plan_quality  # noqa: E402
from backend.app.validators.validator import validate_plan  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", nargs="+", default=DIFFICULTY_LEVELS, choices=DIFFICULTY_LEVELS)
    ap.add_argument("--seeds", nargs=2, type=int, default=[3000, 3009], metavar=("FIRST", "LAST"))
    ap.add_argument("--time-limit", type=float, default=10.0)
    a = ap.parse_args()
    if not solver_available():
        print("OR-Tools is not installed (pip install ortools). Showing generator statistics only.")
    print(f"{'level':5} {'n':>3} {'incidents':>9} {'demand':>7} {'dem/transp':>10} {'blocked':>8} {'boats':>5} {'servable':>8} "
          f"{'greedy/solver':>13} {'solver s':>9} {'optimal':>8}")
    for lv in a.levels:
        scs = [generate_level_scenario(lv, s) for s in range(a.seeds[0], a.seeds[1] + 1)]
        ratios, times, opt = [], [], 0
        if solver_available():
            for sc in scs:
                r = solve_scenario(sc, time_limit_s=a.time_limit, workers=1, seed=0)
                gp = greedy_plan(sc)
                assert validate_plan(sc, gp).valid
                if r.get("objective_weighted_served"):
                    ratios.append(plan_quality(sc, gp)["weighted_served"] / r["objective_weighted_served"])
                times.append(r["runtime_s"]); opt += r["status"] == "OPTIMAL"
        d = [s.difficulty for s in scs]
        print(f"{lv:5} {len(scs):>3} {len(scs[0].incidents):>9} {st.mean(x.total_demand for x in d):>7.0f} "
              f"{st.mean(x.demand_to_transport_ratio for x in d):>10.2f} {st.mean(x.blocked_road_fraction for x in d):>8.2f} "
              f"{st.mean(x.boat_count for x in d):>5.1f} {st.mean(x.servable_demand_fraction for x in d):>8.2f} "
              f"{(st.mean(ratios) if ratios else float('nan')):>13.3f} {(st.mean(times) if times else float('nan')):>9.2f} "
              f"{opt}/{len(scs) if times else 0:>2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
