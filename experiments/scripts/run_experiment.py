"""Run an experiment from the command line (no server needed).

Examples (from the project root, virtual environment active):
  python experiments/scripts/run_experiment.py --config experiments/configs/quick_check.yaml
  python experiments/scripts/run_experiment.py --config experiments/configs/pilot.yaml --num-scenarios 3 --methods llm_only resq_verify
  python experiments/scripts/run_experiment.py --config experiments/configs/pilot.yaml --levels L0 L2 L4 --call-budgets 1 3 5
  python experiments/scripts/run_experiment.py --resume EXP-20260101-120000-abc123
Ctrl+C stops cleanly after the current run; use --resume to continue.
"""
from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.app.core.config import get_settings  # noqa: E402
from backend.app.experiments.configs import load_config_file  # noqa: E402
from backend.app.experiments.runner import ExperimentRunner  # noqa: E402
from backend.app.schemas.experiment import ExperimentConfig  # noqa: E402
from backend.app.services.store.factory import make_store  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="YAML config file")
    ap.add_argument("--resume", help="experiment id to resume")
    ap.add_argument("--name"); ap.add_argument("--size", nargs="+", dest="sizes"); ap.add_argument("--scarcity", nargs="+")
    ap.add_argument("--num-scenarios", type=int); ap.add_argument("--seed", type=int, dest="base_seed")
    ap.add_argument("--repetitions", type=int); ap.add_argument("--methods", nargs="+")
    ap.add_argument("--max-repair-rounds", type=int); ap.add_argument("--feedback-format", choices=["structured", "free_text"])
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--levels", nargs="+", dest="difficulty_levels", choices=["L0", "L1", "L2", "L3", "L4"],
                    help="difficulty levels (replaces the size x scarcity grid)")
    ap.add_argument("--call-budgets", nargs="+", type=int, dest="call_budgets", help="LLM-call budgets to compare, e.g. 1 2 3 5")
    ap.add_argument("--independent-budgets", action="store_true", help="run every budget separately instead of cutting them from one run")
    a = ap.parse_args()
    if a.independent_budgets:
        a.nested_budgets = False

    settings = get_settings()
    store, info = make_store(settings)
    if info.get("note"):
        print("NOTE:", info["note"])
    runner = ExperimentRunner(store, settings)
    if settings.llm_provider == "mock":
        print("WARNING: LLM_PROVIDER=mock -> SIMULATED LLM. Results are NOT experimental results.")

    if a.resume:
        eid = a.resume
    else:
        cfg = load_config_file(a.config) if a.config else ExperimentConfig()
        patch = {k: v for k, v in vars(a).items() if k in ExperimentConfig.model_fields and v is not None}
        cfg = ExperimentConfig.model_validate({**cfg.model_dump(), **patch})
        eid = runner.create(cfg)["experiment_id"]
        print("Created", eid)
    stop = threading.Event()
    try:
        doc = runner.run(eid, stop)
    except KeyboardInterrupt:
        stop.set()
        print("\nInterrupted. Resume with: python experiments/scripts/run_experiment.py --resume", eid)
        return 130
    p = doc["progress"]
    print(f"{eid}: {doc['status']} ({p['done']}/{p['total']} runs, {p['errors']} errors) {doc.get('message', '')}")
    print("Export: python experiments/scripts/export_results.py --experiment", eid)
    return 0 if doc["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
