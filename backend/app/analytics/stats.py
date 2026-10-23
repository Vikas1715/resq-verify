"""Statistics used by the analysis layer.

Only tests whose assumptions are checked are used:
  * binary outcomes (0/1 per scenario)  -> exact McNemar test on discordant pairs
  * continuous outcomes                 -> Shapiro-Wilk on the paired differences, then paired t-test
                                           (only if n >= 20 and differences look normal) else Wilcoxon signed-rank
Pairing unit = scenario (repetitions of one scenario are averaged first), so repetitions are never treated
as independent samples. Multiple comparisons are adjusted with Holm. Nothing here is hard-wired to a result.
"""
from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

BOOT_N = 2000


def _clean(values: Sequence) -> np.ndarray:
    arr = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return np.array(arr, dtype=float)


def wilson_interval(k: int, n: int, z: float = 1.959964) -> tuple[Optional[float], Optional[float]]:
    if n <= 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_ci(values: Sequence, seed: int = 12345, n_boot: int = BOOT_N, stat=np.mean, level: float = 0.95):
    arr = _clean(values)
    if len(arr) < 2:
        return None, None
    if np.allclose(arr, arr[0]):
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boots = stat(arr[idx], axis=1)
    a = (1 - level) / 2
    return float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))


def describe(values: Sequence, seed: int = 12345) -> dict[str, Any]:
    arr = _clean(values)
    n = len(arr)
    if n == 0:
        return {"n": 0, "mean": None, "median": None, "sd": None, "min": None, "max": None, "ci_low": None, "ci_high": None}
    lo, hi = bootstrap_ci(arr, seed)
    return {"n": n, "mean": float(arr.mean()), "median": float(np.median(arr)),
            "sd": float(arr.std(ddof=1)) if n > 1 else None, "min": float(arr.min()), "max": float(arr.max()),
            "ci_low": lo, "ci_high": hi}


def proportion(flags: Sequence) -> dict[str, Any]:
    vals = [bool(v) for v in flags if v is not None and not (isinstance(v, float) and math.isnan(v))]
    n = len(vals)
    k = sum(vals)
    lo, hi = wilson_interval(k, n)
    return {"n": n, "k": k, "rate": (k / n if n else None), "ci_low": lo, "ci_high": hi}


def scenario_level(df: pd.DataFrame, metric: str, method: str) -> pd.Series:
    """Mean of `metric` per scenario for one method (repetitions averaged), NaN-dropped."""
    sub = df[(df["method"] == method) & (df["status"] == "ok")]
    s = pd.to_numeric(sub[metric].astype(float), errors="coerce")
    return s.groupby(sub["scenario_id"]).mean().dropna()


def holm(pvalues: list[Optional[float]]) -> list[Optional[float]]:
    idx = [i for i, p in enumerate(pvalues) if p is not None]
    order = sorted(idx, key=lambda i: pvalues[i])
    m = len(order)
    out: list[Optional[float]] = [None] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * pvalues[i])
        running = max(running, adj)
        out[i] = running
    return out


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """a, b are paired 0/1 arrays. Exact two-sided binomial test on the discordant pairs."""
    a1b0 = int(((a == 1) & (b == 0)).sum())
    a0b1 = int(((a == 0) & (b == 1)).sum())
    disc = a1b0 + a0b1
    p = 1.0 if disc == 0 else float(sps.binomtest(min(a1b0, a0b1), disc, 0.5).pvalue)
    effect = (a1b0 - a0b1) / disc if disc else 0.0
    return {"test": "McNemar (exact)", "statistic": float(min(a1b0, a0b1)), "p_value": p,
            "effect_size": effect, "effect_name": "discordant-pair difference (a-only minus b-only)/discordant",
            "a_only": a1b0, "b_only": a0b1}


def paired_continuous(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    d = a - b
    n = len(d)
    nonzero = d[d != 0]
    if n < 2 or len(nonzero) == 0:
        return {"test": "none (no non-zero differences)" if n >= 2 else "none (n<2)", "statistic": None, "p_value": None if n < 2 else 1.0,
                "effect_size": 0.0 if n >= 2 else None, "effect_name": "Cohen's dz", "normality_p": None}
    normal_p = None
    normal = False
    if 3 <= n <= 5000 and np.ptp(d) > 0:
        normal_p = float(sps.shapiro(d).pvalue)
        normal = normal_p >= 0.05
    sd = d.std(ddof=1)
    dz = float(d.mean() / sd) if sd > 0 else None
    if n >= 20 and normal:
        res = sps.ttest_rel(a, b)
        return {"test": "paired t-test", "statistic": float(res.statistic), "p_value": float(res.pvalue),
                "effect_size": dz, "effect_name": "Cohen's dz", "normality_p": normal_p}
    try:
        res = sps.wilcoxon(nonzero, zero_method="wilcox", method="auto")
        ranks = sps.rankdata(np.abs(nonzero))
        w_pos = float(ranks[nonzero > 0].sum())
        w_neg = float(ranks[nonzero < 0].sum())
        rbc = (w_pos - w_neg) / (w_pos + w_neg)
        return {"test": "Wilcoxon signed-rank", "statistic": float(res.statistic), "p_value": float(res.pvalue),
                "effect_size": float(rbc), "effect_name": "rank-biserial r", "normality_p": normal_p, "cohens_dz": dz}
    except ValueError:
        return {"test": "Wilcoxon signed-rank (failed)", "statistic": None, "p_value": None, "effect_size": dz,
                "effect_name": "Cohen's dz", "normality_p": normal_p}


def paired_compare(df: pd.DataFrame, metric: str, method_a: str, method_b: str, binary: bool = False) -> dict[str, Any]:
    """Compare method_a with method_b on `metric`, paired by scenario. Positive mean_diff = a larger than b."""
    sa = scenario_level(df, metric, method_a)
    sb = scenario_level(df, metric, method_b)
    common = sa.index.intersection(sb.index)
    base = {"metric": metric, "method_a": method_a, "method_b": method_b, "n_pairs": int(len(common))}
    if len(common) == 0:
        return {**base, "mean_a": None, "mean_b": None, "mean_diff": None, "test": "none (no paired scenarios)",
                "p_value": None, "effect_size": None, "effect_name": None, "ci_low": None, "ci_high": None, "note": "no common scenarios"}
    a, b = sa.loc[common].to_numpy(), sb.loc[common].to_numpy()
    d = a - b
    lo, hi = bootstrap_ci(d)
    out = {**base, "mean_a": float(a.mean()), "mean_b": float(b.mean()), "mean_diff": float(d.mean()), "ci_low": lo, "ci_high": hi}
    is_binary = binary and set(np.unique(np.concatenate([a, b]))) <= {0.0, 1.0}
    out.update(mcnemar_exact(a, b) if is_binary else paired_continuous(a, b))
    if len(common) < 10:
        out["note"] = "fewer than 10 paired scenarios: treat p-values as unreliable"
    return out


def apply_holm(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    adj = holm([r.get("p_value") for r in rows])
    for r, p in zip(rows, adj):
        r["p_holm"] = p
        r["significant_0_05_holm"] = (p is not None and p < 0.05)
    return rows
