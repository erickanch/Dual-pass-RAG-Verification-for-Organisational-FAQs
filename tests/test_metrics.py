"""
Unit tests for analysis/metrics.py and analysis/stats.py.

These run on SYNTHETIC data with hand-computed expected values — no corpus,
no pipeline, no API keys. The point is not coverage but VERIFICATION: every
expected number below was computed independently (Wilson from its algebraic
definition; McNemar from the binomial; Wilcoxon from worked examples,
including the tied-score case of the reported comparisons; the nearest-rank
percentile from its rank formula) and is asserted against the code's answer.
If a library changes behaviour, these fail loudly rather than shifting a
reported result silently.

Run:  uv run pytest tests/test_metrics.py -v
"""

import math

import pytest

from analysis.metrics import (
    ConfusionMatrix,
    build_confusion_matrix,
    false_negative_rate,
    false_positive_rate,
    hallucination_reached_user,
    nearest_rank_percentile,
    paired_safety_outcomes,
    recorded_latency_s,
    strict_compliance_rate,
    total_tokens,
)
from analysis.stats import mcnemar_test, wilcoxon_test
from src.constants import (
    LABEL_CORRECT_REFUSAL,
    LABEL_EXTRINSIC,
    LABEL_FAITHFUL,
    LABEL_INTRINSIC,
    OUTCOME_FALLBACK,
    OUTCOME_FIXED_BY_REWRITE,
    OUTCOME_GENERATION_FAILED,
    OUTCOME_PASSED_FIRST,
    OUTCOME_VERIFIER_MALFORMED,
)

TOL = 1e-6


def rec(qid, label, outcome):
    return {"query_id": qid, "human_label": label, "outcome": outcome}


# ===========================================================================
# Confusion matrix construction
# ===========================================================================

def test_matrix_four_cells():
    """One record into each cell; check it lands where it should."""
    records = [
        rec("q1", LABEL_EXTRINSIC, OUTCOME_FALLBACK),      # hallucinated+blocked -> TP
        rec("q2", LABEL_FAITHFUL, OUTCOME_FALLBACK),       # faithful+blocked     -> FP
        rec("q3", LABEL_INTRINSIC, OUTCOME_PASSED_FIRST),  # hallucinated+passed  -> FN
        rec("q4", LABEL_FAITHFUL, OUTCOME_PASSED_FIRST),   # faithful+passed      -> TN
    ]
    cm = build_confusion_matrix(records)
    assert (cm.tp, cm.fp, cm.fn, cm.tn) == (1, 1, 1, 1)
    assert cm.total == 4


def test_fixed_by_rewrite_counts_as_passed():
    """The final answer WAS displayed, so a rewritten-then-verified answer
    is a pass, not a block. Getting this wrong would inflate TP/FP."""
    cm = build_confusion_matrix([
        rec("q1", LABEL_FAITHFUL, OUTCOME_FIXED_BY_REWRITE),
        rec("q2", LABEL_EXTRINSIC, OUTCOME_FIXED_BY_REWRITE),
    ])
    assert (cm.tp, cm.fp, cm.fn, cm.tn) == (0, 0, 1, 1)


def test_infrastructure_failures_excluded():
    """generation_failed / verifier_malformed leave the matrix entirely."""
    cm = build_confusion_matrix([
        rec("q1", LABEL_FAITHFUL, OUTCOME_GENERATION_FAILED),
        rec("q2", LABEL_FAITHFUL, OUTCOME_VERIFIER_MALFORMED),
        rec("q3", LABEL_FAITHFUL, OUTCOME_PASSED_FIRST),
    ])
    assert cm.total == 1
    assert cm.excluded == 2


def test_correct_refusal_treated_as_faithful():
    """Per the ratified decision in constants.CORRECT_REFUSAL_TREATED_AS:
    a passed refusal is a TN; a blocked refusal is an FP, and is also
    counted in the blocked_refusals sub-breakdown."""
    cm = build_confusion_matrix([
        rec("q1", LABEL_CORRECT_REFUSAL, OUTCOME_PASSED_FIRST),
        rec("q2", LABEL_CORRECT_REFUSAL, OUTCOME_FALLBACK),
    ])
    assert (cm.tp, cm.fp, cm.fn, cm.tn) == (0, 1, 0, 1)
    assert cm.blocked_refusals == 1


def test_unknown_label_raises():
    """A typo in a label file must fail loudly, not shift a rate silently."""
    with pytest.raises(ValueError, match="unknown human_label"):
        build_confusion_matrix([rec("q1", "faithfull", OUTCOME_PASSED_FIRST)])


def test_unknown_outcome_raises():
    with pytest.raises(ValueError, match="unknown outcome"):
        build_confusion_matrix([rec("q1", LABEL_FAITHFUL, "passed")])


# ===========================================================================
# Rates — denominators verified by hand
# ===========================================================================

def test_rate_denominators():
    """
    Matrix: TP=6, FP=2, FN=4, TN=8  (total 20)
      SCR = (6+8)/20   = 14/20 = 0.70
      FNR = 4/(4+6)    =  4/10 = 0.40
      FPR = 2/(2+8)    =  2/10 = 0.20
    """
    cm = ConfusionMatrix(tp=6, fp=2, fn=4, tn=8)
    assert strict_compliance_rate(cm).value == pytest.approx(0.70, abs=TOL)
    assert false_negative_rate(cm).value == pytest.approx(0.40, abs=TOL)
    assert false_positive_rate(cm).value == pytest.approx(0.20, abs=TOL)

    assert strict_compliance_rate(cm).denominator == 20
    assert false_negative_rate(cm).denominator == 10
    assert false_positive_rate(cm).denominator == 10


def test_baseline_fnr_is_one_by_construction():
    """The Baseline arm has no verifier: nothing is ever blocked, so TP=0
    and FNR=1.0. Expected, not a bug — documented in metrics.py."""
    cm = ConfusionMatrix(tp=0, fp=0, fn=7, tn=13)
    assert false_negative_rate(cm).value == pytest.approx(1.0, abs=TOL)


def test_empty_denominator_is_nan_not_crash():
    """A perfect arm (no hallucinations at all) has an undefined FNR.
    Return NaN rather than dividing by zero mid-run."""
    cm = ConfusionMatrix(tp=0, fp=1, fn=0, tn=9)
    assert math.isnan(false_negative_rate(cm).value)


# ===========================================================================
# Wilson intervals — reference values computed from the algebraic definition
#
#   centre = (p + z^2/2n) / (1 + z^2/n)
#   half   = z/(1+z^2/n) * sqrt( p(1-p)/n + z^2/4n^2 )        z = 1.959964
#
# Verify one by hand before trusting the rest. For 8/10:
#   p=0.8  z^2=3.841459  z^2/n=0.384146  1+z^2/n=1.384146
#   centre = (0.8 + 0.192073)/1.384146 = 0.716736
#   half   = (1.959964/1.384146) * sqrt(0.016 + 0.009604) = 1.416008*0.160012
#          = 0.226578
#   -> (0.490158, 0.943314)
# ===========================================================================

WILSON_CASES = [
    # (successes, n, expected_lower, expected_upper)
    (8, 10, 0.490162, 0.943318),
    (0, 10, 0.000000, 0.277533),   # zero events: interval must NOT be (0,0)
    (10, 10, 0.722467, 1.000000),  # all events: must NOT be (1,1)
    (3, 40, 0.025836, 0.198642),   # small proportion, realistic FPR shape
    (12, 100, 0.069994, 0.198121),
]


@pytest.mark.parametrize("k,n,lo,hi", WILSON_CASES)
def test_wilson_intervals(k, n, lo, hi):
    cm = ConfusionMatrix(tp=0, fp=k, fn=0, tn=n - k)
    r = false_positive_rate(cm)
    assert r.numerator == k and r.denominator == n
    assert r.lower == pytest.approx(lo, abs=1e-5)
    assert r.upper == pytest.approx(hi, abs=1e-5)


def test_wilson_handles_boundaries_sensibly():
    """The reason for choosing Wilson over the normal approximation: at 0
    and 1 the normal interval collapses to a point, which would let you
    report 'FNR 0% (0%-0%)' from a handful of observations."""
    zero = false_positive_rate(ConfusionMatrix(fp=0, tn=10))
    assert zero.value == 0.0 and zero.upper > 0.2

    full = false_positive_rate(ConfusionMatrix(fp=10, tn=0))
    assert full.value == 1.0 and full.lower < 0.8


# ===========================================================================
# Paired outcomes
# ===========================================================================

def test_hallucination_reached_user():
    assert hallucination_reached_user(rec("q", LABEL_EXTRINSIC, OUTCOME_PASSED_FIRST))
    assert hallucination_reached_user(rec("q", LABEL_INTRINSIC, OUTCOME_FIXED_BY_REWRITE))
    assert not hallucination_reached_user(rec("q", LABEL_EXTRINSIC, OUTCOME_FALLBACK))
    assert not hallucination_reached_user(rec("q", LABEL_FAITHFUL, OUTCOME_PASSED_FIRST))


def test_pairing_drops_queries_excluded_in_either_arm():
    """If a query failed in one arm only, keeping it in the other would
    compare the arms on different query sets."""
    arm_a = [rec("q1", LABEL_FAITHFUL, OUTCOME_PASSED_FIRST),
             rec("q2", LABEL_EXTRINSIC, OUTCOME_PASSED_FIRST),
             rec("q3", LABEL_FAITHFUL, OUTCOME_GENERATION_FAILED)]
    arm_b = [rec("q1", LABEL_FAITHFUL, OUTCOME_PASSED_FIRST),
             rec("q2", LABEL_EXTRINSIC, OUTCOME_FALLBACK),
             rec("q3", LABEL_FAITHFUL, OUTCOME_PASSED_FIRST)]

    a, b = paired_safety_outcomes(arm_a, arm_b)
    assert len(a) == len(b) == 2          # q3 dropped from BOTH
    assert a == [False, True]
    assert b == [False, False]


# ===========================================================================
# McNemar — reference p-values from the exact binomial
#
#   discordant b=10, c=2  ->  binomial(2; 12, 0.5) two-sided = 0.03857421875
#   discordant b=6,  c=1  ->  binomial(1;  7, 0.5) two-sided = 0.125
#   discordant b=3,  c=3  ->  symmetric, p = 1.0
# ===========================================================================

def _outcomes(n_both_true, n_a_only, n_b_only, n_both_false):
    a = [True]*n_both_true + [True]*n_a_only + [False]*n_b_only + [False]*n_both_false
    b = [True]*n_both_true + [False]*n_a_only + [True]*n_b_only + [False]*n_both_false
    return a, b


def test_mcnemar_exact_significant():
    """10 queries where a hallucination reached the user in the baseline
    but not the dual-pass arm; 2 the other way. p = 0.0386 -> significant."""
    a, b = _outcomes(n_both_true=5, n_a_only=10, n_b_only=2, n_both_false=50)
    res = mcnemar_test(a, b)
    assert res.p_value == pytest.approx(0.03857421875, abs=1e-9)
    assert res.significant
    assert res.detail["discordant_a_only"] == 10
    assert res.detail["discordant_b_only"] == 2
    assert res.detail["discordant_total"] == 12


def test_mcnemar_exact_not_significant():
    """Same DIRECTION, smaller counts — a good illustration for the report
    of why n matters: 6 vs 1 is not significant at alpha=0.05."""
    a, b = _outcomes(n_both_true=2, n_a_only=6, n_b_only=1, n_both_false=40)
    res = mcnemar_test(a, b)
    assert res.p_value == pytest.approx(0.125, abs=1e-9)
    assert not res.significant


def test_mcnemar_ignores_concordant_pairs():
    """Adding queries where the arms AGREE must not change the p-value —
    the defining property of the test."""
    a1, b1 = _outcomes(0, 10, 2, 0)
    a2, b2 = _outcomes(500, 10, 2, 500)
    assert mcnemar_test(a1, b1).p_value == pytest.approx(
        mcnemar_test(a2, b2).p_value, abs=1e-12)


def test_mcnemar_symmetric_is_null():
    a, b = _outcomes(n_both_true=5, n_a_only=3, n_b_only=3, n_both_false=5)
    assert mcnemar_test(a, b).p_value == pytest.approx(1.0, abs=1e-9)


def test_mcnemar_unpaired_lengths_raise():
    with pytest.raises(ValueError, match="unpaired lengths"):
        mcnemar_test([True, False], [True])


# ===========================================================================
# Wilcoxon — reference from a worked 8-pair example
#   arm B higher in all 8 pairs -> statistic 0.0, p = 2/2^8 = 0.0078125
# ===========================================================================

BASELINE_SCORES = [0.60, 0.55, 0.70, 0.62, 0.58, 0.65, 0.51, 0.68]
DUAL_SCORES     = [0.75, 0.71, 0.72, 0.80, 0.66, 0.69, 0.74, 0.70]


def test_wilcoxon_all_pairs_favour_one_arm():
    res = wilcoxon_test(BASELINE_SCORES, DUAL_SCORES)
    assert res.statistic == pytest.approx(0.0, abs=TOL)
    assert res.p_value == pytest.approx(0.0078125, abs=1e-9)
    assert res.significant
    assert res.detail["n_pairs"] == 8
    assert res.detail["median_diff"] < 0      # baseline below dual


def test_wilcoxon_all_identical_raises():
    """Undefined rather than a meaningless p-value."""
    scores = [0.8] * 6
    with pytest.raises(ValueError, match="all pairs identical"):
        wilcoxon_test(scores, scores)


def test_wilcoxon_unpaired_lengths_raise():
    with pytest.raises(ValueError, match="unpaired lengths"):
        wilcoxon_test([0.1, 0.2], [0.1])


# ===========================================================================
# Wilcoxon on tied scores — the shape of the reported comparisons
#
#   57 pairs, of which 55 are identical. Identical pairs are dropped,
#   leaving n = 2 differences. Under the null each is equally likely to be
#   positive or negative: 4 sign patterns, W+ in {0, 1, 2, 3}, each 1/4.
#   Both differences here are positive, so W = min(W+, W-) = 0 and
#     two-sided p = 2 * P(W+ <= 0) = 2 * 1/4 = 0.5
#   With n = 1: 2 sign patterns, two-sided p = 2 * 1/2 = 1.0
#   The normal approximation would give about 0.18 and 0.32 instead.
# ===========================================================================

def test_wilcoxon_exact_with_tied_pairs():
    a = [1.0] * 57
    b = [1.0] * 55 + [0.5, 0.667]
    res = wilcoxon_test(a, b)
    assert res.detail["n_nonzero_diffs"] == 2
    assert res.statistic == pytest.approx(0.0, abs=TOL)
    assert res.p_value == pytest.approx(0.5, abs=1e-9)

    one = [1.0] * 56 + [0.667]
    assert wilcoxon_test(a, one).p_value == pytest.approx(1.0, abs=1e-9)


# ===========================================================================
# Cost measures
# ===========================================================================

def test_nearest_rank_percentile():
    """
    Rank = ceil(q/100 * n), and the value is always an observed one:
      n = 20,  q = 95  -> rank 19
      n = 20,  q = 50  -> rank 10 (the lower middle value, not the median 10.5)
      n = 108, q = 95  -> rank ceil(102.6) = 103
    """
    values = list(range(20, 0, -1))          # 20..1, deliberately unsorted
    assert nearest_rank_percentile(values, 95) == 19
    assert nearest_rank_percentile(values, 50) == 10
    assert nearest_rank_percentile(values, 100) == 20
    assert nearest_rank_percentile(list(range(1, 109)), 95) == 103


def test_nearest_rank_percentile_rejects_bad_input():
    with pytest.raises(ValueError):
        nearest_rank_percentile([], 95)
    with pytest.raises(ValueError):
        nearest_rank_percentile([1.0, 2.0], 0)


def test_recorded_latency_counts_successful_calls_only():
    """Generation 2.0 + verification 1.5 + 1.0 + rewrite 3.0 = 7.5 s.
    Retrieval and wall-clock times (which include retries) are excluded."""
    record = {
        "t_retrieval_s": 0.05,
        "t_generation_s": 2.0, "t_generation_wall_s": 8.0,
        "t_verification_s": [1.5, 1.0], "t_verification_wall_s": [1.5, 6.0],
        "t_rewrite_s": [3.0], "t_rewrite_wall_s": [9.0],
        "tokens_prompt_total": 900, "tokens_completion_total": 100,
    }
    assert recorded_latency_s(record) == pytest.approx(7.5, abs=TOL)
    assert total_tokens(record) == 1000

    baseline = {"t_generation_s": 2.29, "t_verification_s": [], "t_rewrite_s": []}
    assert recorded_latency_s(baseline) == pytest.approx(2.29, abs=TOL)


# ===========================================================================
# End-to-end on a synthetic run: the numbers you would report
# ===========================================================================

def test_worked_example_end_to_end():
    """
    A synthetic 20-query dual-pass arm:
      6 hallucinations blocked          -> TP=6
      4 hallucinations passed           -> FN=4
      2 faithful answers blocked        -> FP=2
      8 faithful answers passed         -> TN=8
    Expected:  SCR 70.0%   FNR 40.0%   FPR 20.0%
    """
    records = (
        [rec(f"h_blocked_{i}", LABEL_EXTRINSIC, OUTCOME_FALLBACK) for i in range(6)]
        + [rec(f"h_passed_{i}", LABEL_INTRINSIC, OUTCOME_PASSED_FIRST) for i in range(4)]
        + [rec(f"f_blocked_{i}", LABEL_FAITHFUL, OUTCOME_FALLBACK) for i in range(2)]
        + [rec(f"f_passed_{i}", LABEL_FAITHFUL, OUTCOME_PASSED_FIRST) for i in range(8)]
    )
    cm = build_confusion_matrix(records)
    assert (cm.tp, cm.fp, cm.fn, cm.tn) == (6, 2, 4, 8)

    scr, fnr, fpr = (strict_compliance_rate(cm),
                     false_negative_rate(cm),
                     false_positive_rate(cm))

    assert scr.value == pytest.approx(0.70, abs=TOL)
    assert fnr.value == pytest.approx(0.40, abs=TOL)
    assert fpr.value == pytest.approx(0.20, abs=TOL)

    # intervals must be wide at n=20 — the honest story of a modest sample
    assert (scr.upper - scr.lower) > 0.30
    assert "70.0%" in scr.as_percent()