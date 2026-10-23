import numpy as np
import pandas as pd
import pytest

from backend.app.analytics import stats as S
from backend.app.analytics.dataframe import runs_to_frame


def test_wilson_interval_known_values():
    lo, hi = S.wilson_interval(8, 10)
    assert lo == pytest.approx(0.490, abs=0.005) and hi == pytest.approx(0.943, abs=0.005)
    assert S.wilson_interval(0, 0) == (None, None)
    lo, hi = S.wilson_interval(10, 10)
    assert hi == pytest.approx(1.0) and lo < 1


def test_describe_and_bootstrap_are_deterministic():
    v = [1, 2, 3, 4, 10]
    a, b = S.describe(v), S.describe(v)
    assert a == b and a["mean"] == 4 and a["median"] == 3 and a["sd"] == pytest.approx(np.std(v, ddof=1))
    assert a["ci_low"] <= a["mean"] <= a["ci_high"]
    assert S.describe([])["n"] == 0 and S.describe([None, float("nan")])["n"] == 0


def test_holm_matches_hand_calculation():
    assert S.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert S.holm([None, 0.5]) == [None, 0.5]


def test_mcnemar_exact():
    a = np.array([1] * 9 + [0] * 1 + [1] * 5)
    b = np.array([0] * 9 + [1] * 1 + [1] * 5)
    r = S.mcnemar_exact(a, b)
    assert r["a_only"] == 9 and r["b_only"] == 1
    assert r["p_value"] == pytest.approx(0.02148, abs=1e-4)  # 2 * P(X<=1 | n=10, p=.5)


def test_paired_continuous_picks_wilcoxon_for_small_n_and_reports_effect():
    a = np.array([0.9, 0.8, 0.85, 0.7, 0.95, 0.6, 0.75, 0.88])
    b = a - np.array([0.1, 0.05, 0.12, 0.02, 0.2, 0.07, 0.03, 0.09])
    r = S.paired_continuous(a, b)
    assert r["test"] == "Wilcoxon signed-rank" and r["effect_size"] == pytest.approx(1.0) and r["p_value"] < 0.05


def test_paired_continuous_uses_t_test_only_when_n_large_and_normal():
    rng = np.random.default_rng(0)
    a = rng.normal(0.8, 0.05, 40)
    b = a - rng.normal(0.05, 0.03, 40)
    assert S.paired_continuous(a, b)["test"] == "paired t-test"


def test_degenerate_pairs_do_not_crash():
    assert S.paired_continuous(np.array([1.0]), np.array([1.0]))["p_value"] is None
    r = S.paired_continuous(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert r["p_value"] == 1.0 and r["effect_size"] == 0.0


def _runs(pairs):
    rows = []
    for sid, (fa, fb) in enumerate(pairs):
        for m, f in (("resq_verify", fa), ("llm_only", fb)):
            rows.append({"scenario_id": f"s{sid}", "method": m, "status": "ok", "final_feasibility": f, "plan_quality": 0.5 if f else None,
                         "is_simulated": False, "token_usage": {"total": 10}, "violation_counts": {}, "violation_counts_initial": {}})
    return runs_to_frame(rows)


def test_paired_compare_binary_uses_mcnemar_and_aligns_by_scenario():
    df = _runs([(True, False)] * 8 + [(True, True)] * 2)
    r = S.paired_compare(df, "final_feasibility", "resq_verify", "llm_only", binary=True)
    assert r["test"].startswith("McNemar") and r["n_pairs"] == 10 and r["mean_diff"] == pytest.approx(0.8) and r["p_value"] < 0.01


def test_paired_compare_with_no_common_scenarios():
    df = _runs([(True, False)])
    df = df[df["method"] == "resq_verify"]
    r = S.paired_compare(df, "final_feasibility", "resq_verify", "llm_only", binary=True)
    assert r["n_pairs"] == 0 and r["p_value"] is None


def test_errored_runs_are_excluded_from_rates():
    rows = [{"scenario_id": "a", "method": "llm_only", "status": "ok", "final_feasibility": True, "is_simulated": False},
            {"scenario_id": "b", "method": "llm_only", "status": "error", "final_feasibility": None, "is_simulated": False}]
    from backend.app.analytics.summary import method_summary
    m = method_summary(runs_to_frame(rows))[0]
    assert m["n_runs"] == 2 and m["n_ok"] == 1 and m["feasibility"]["rate"] == 1.0
