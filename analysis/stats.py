"""
Paired significance tests: McNemar (binary) and Wilcoxon (continuous).

Both compare the SAME queries under two arms, which is why they are paired
tests. Per the proposal, estimation (rates with Wilson intervals, in
metrics.py) is PRIMARY; these tests are secondary — with ~100 paired
queries only large differences are reliably detectable, and a
non-significant result is not evidence of no effect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar

from src.constants import ALPHA


@dataclass(frozen=True)
class TestResult:
    name: str
    statistic: float
    p_value: float
    significant: bool
    detail: dict

    def summary(self) -> str:
        verdict = "significant" if self.significant else "not significant"
        return (f"{self.name}: statistic={self.statistic:.4f}, "
                f"p={self.p_value:.4f} ({verdict} at alpha={ALPHA})")


# ---------------------------------------------------------------------------
# McNemar — paired binary outcomes
# ---------------------------------------------------------------------------

def mcnemar_test(arm_a: Sequence[bool], arm_b: Sequence[bool],
                 alpha: float = ALPHA, exact: bool = True) -> TestResult:
    """
    arm_a / arm_b: per-query binary outcomes for the same queries, in the
    same order (use metrics.paired_safety_outcomes to produce them).

    McNemar looks ONLY at the discordant pairs — queries where the two arms
    disagree. Agreements carry no information about which arm is better,
    which is exactly why the test is appropriate here.

        b = a True,  b False   (harmed by arm B)
        c = a False, b True    (helped by arm B)

    exact=True uses the binomial test, correct at the small discordant
    counts this study will produce. The chi-square approximation is
    unreliable when b + c < ~25.
    """
    if len(arm_a) != len(arm_b):
        raise ValueError(f"unpaired lengths: {len(arm_a)} vs {len(arm_b)}")
    if not arm_a:
        raise ValueError("no paired observations")

    n00 = sum(1 for x, y in zip(arm_a, arm_b) if not x and not y)
    n01 = sum(1 for x, y in zip(arm_a, arm_b) if not x and y)
    n10 = sum(1 for x, y in zip(arm_a, arm_b) if x and not y)
    n11 = sum(1 for x, y in zip(arm_a, arm_b) if x and y)

    table = np.array([[n00, n01], [n10, n11]])
    res = mcnemar(table, exact=exact, correction=not exact)

    return TestResult(
        name="McNemar (paired binary)",
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        significant=bool(res.pvalue < alpha),
        detail={
            "table": table.tolist(),
            "discordant_a_only": n10,
            "discordant_b_only": n01,
            "discordant_total": n01 + n10,
            "n_pairs": len(arm_a),
            "exact": exact,
        },
    )


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank — paired continuous outcomes
# ---------------------------------------------------------------------------

def wilcoxon_test(scores_a: Sequence[float], scores_b: Sequence[float],
                  alpha: float = ALPHA) -> TestResult:
    """
    Paired test on continuous scores (Ragas faithfulness). Non-parametric,
    so it makes no normality assumption — appropriate for bounded scores
    that pile up near 1.0.

    Pairs where both scores are identical contribute nothing and are
    dropped by scipy's default ('wilcox'); if ALL pairs are identical the
    test is undefined and this raises rather than returning a meaningless
    p-value.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(f"unpaired lengths: {len(scores_a)} vs {len(scores_b)}")
    if not scores_a:
        raise ValueError("no paired observations")

    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    n_nonzero = sum(1 for d in diffs if d != 0)
    if n_nonzero == 0:
        raise ValueError("all pairs identical; Wilcoxon is undefined")

    res = wilcoxon(scores_a, scores_b)

    return TestResult(
        name="Wilcoxon signed-rank (paired continuous)",
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        significant=bool(res.pvalue < alpha),
        detail={
            "n_pairs": len(scores_a),
            "n_nonzero_diffs": n_nonzero,
            "median_diff": float(np.median(diffs)),
            "mean_a": float(np.mean(scores_a)),
            "mean_b": float(np.mean(scores_b)),
        },
    )