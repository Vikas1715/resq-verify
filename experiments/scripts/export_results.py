"""Write the research exports to data/exports/<name>/ without starting the server.

  python experiments/scripts/export_results.py --experiment EXP-... [--all] [--include-adhoc]
"""
from __future__ import annotations

import argparse
import sys
import zipfile
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.app.analytics.dataframe import runs_to_frame  # noqa: E402
from backend.app.analytics.plots import all_plots  # noqa: E402
from backend.app.analytics.report import build_bundle, comparisons_csv, research_summary_md, runs_csv  # noqa: E402
from backend.app.core.config import get_settings  # noqa: E402
from backend.app.services.store.base import dedupe_runs  # noqa: E402
from backend.app.services.store.factory import make_store  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment"); ap.add_argument("--all", action="store_true"); ap.add_argument("--include-adhoc", action="store_true")
    a = ap.parse_args()
    s = get_settings()
    store, _ = make_store(s)
    if a.experiment:
        exps = [store.get_experiment(a.experiment)]
        if exps[0] is None:
            print("Experiment not found"); return 1
        name = a.experiment
    elif a.all:
        exps = [e for e in store.list_experiments(10_000) if a.include_adhoc or e.get("kind") != "adhoc"]
        name = "all"
    else:
        ap.error("give --experiment ID or --all")
    runs = [r for e in exps for r in dedupe_runs(store.list_runs(e["experiment_id"]))]
    out = s.data_dir / "exports" / name
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "research_summary.md").write_text(research_summary_md(runs, experiments=exps), encoding="utf-8")
    (out / "runs.csv").write_text(runs_csv(runs), encoding="utf-8")
    (out / "paired_comparisons.csv").write_text(comparisons_csv(runs), encoding="utf-8")
    df = runs_to_frame(runs)
    sim = bool(not df.empty and df.loc[df["method"] != "ortools", "is_simulated"].astype(bool).any())
    for n, data in all_plots(df, runs, sim).items():
        (out / "plots" / n).write_bytes(data)
    scen = [x for x in (store.get_scenario(i) for i in sorted({r["scenario_id"] for r in runs})) if x]
    (out / "bundle.zip").write_bytes(build_bundle(runs, exps, scen))
    print("Wrote exports to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
