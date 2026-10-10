"""
Confusion matrix and the three headline rates, each with a Wilson interval,
plus the per-query cost measures (recorded latency, tokens) and the
nearest-rank percentile used to report them.

Reads records; writes nothing. All statistical TESTS live in stats.py —
this module only describes.

The labeller supplies a judgment of the text (four labels); the controller
supplies what the system did (five outcomes). This module performs the
crossing. The labeller never does — that would break blinding.

Wilson interval (score interval), reported at 95% per the proposal because
it behaves sensibly at small n and near 0 or 1, where the normal
approximation does not:

    centre = (p + z^2/2n) / (1 + z^2/n)
    half   = z/(1 + z^2/n) * sqrt( p(1-p)/n + z^2/4n^2 )
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from statsmodels.stats.proportion import proportion_confint

from src.constants import (
    ALPHA,
    BLOCKING_OUTCOMES,
    CORRECT_REFUSAL_TREATED_AS,
    EXCLUDED_OUTCOMES,
    HALLUCINATION_LABELS,
    LABEL_CORRECT_REFUSAL,
    VALID_LABELS,
    VALID_OUTCOMES,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Rate:
    """A proportion with its Wilson interval and the counts behind it."""
    value: float
    lower: float
    upper: float
    numerator: int
    denominator: int

    def as_percent(self) -> str:
        if self.denominator == 0:
            return "n/a (denominator 0)"
        return (f"{self.value * 100:.1f}% "
                f"[{self.lower * 100:.1f}, {self.upper * 100:.1f}] "
                f"({self.numerator}/{self.denominator})")


@dataclass(frozen=True)
class ConfusionMatrix:
    tp: int = 0          # hallucinated + blocked   -> correct block
    fp: int = 0          # faithful    + blocked    -> wrong block (usability)
    fn: int = 0          # hallucinated + passed    -> critical safety miss
    tn: int = 0          # faithful    + passed     -> correct pass

    # Recorded but outside the matrix
    excluded: int = 0            # verifier_malformed / generation_failed
    blocked_refusals: int = 0    # FP sub-breakdown: refusals that were blocked

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn


# ---------------------------------------------------------------------------
# Building the matrix
# ---------------------------------------------------------------------------

def _is_hallucinated(label: str) -> bool | None:
    """Ground-truth side. Returns None if the record leaves the matrix."""
    if label in HALLUCINATION_LABELS:
        return True
    if label == LABEL_CORRECT_REFUSAL:
        if CORRECT_REFUSAL_TREATED_AS == "exclude":
            return None
        return False          # "faithful": no unsupported claims
    return False              # LABEL_FAITHFUL


def build_confusion_matrix(records: Iterable[Mapping]) -> ConfusionMatrix:
    """
    records: each needs 'human_label' (one of VALID_LABELS) and 'outcome'
             (one of VALID_OUTCOMES).

    Raises on unknown labels/outcomes rather than silently miscounting —
    a typo in a label file would otherwise shift a rate with no warning.
    """
    tp = fp = fn = tn = excluded = blocked_refusals = 0

    for i, r in enumerate(records):
        label, outcome = r["human_label"], r["outcome"]

        if label not in VALID_LABELS:
            raise ValueError(f"record {i}: unknown human_label {label!r}")
        if outcome not in VALID_OUTCOMES:
            raise ValueError(f"record {i}: unknown outcome {outcome!r}")

        if outcome in EXCLUDED_OUTCOMES:
            excluded += 1
            continue

        hallucinated = _is_hallucinated(label)
        if hallucinated is None:
            excluded += 1
            continue

        blocked = outcome in BLOCKING_OUTCOMES

        if hallucinated and blocked:
            tp += 1
        elif hallucinated and not blocked:
            fn += 1
        elif not hallucinated and blocked:
            fp += 1
            if label == LABEL_CORRECT_REFUSAL:
                blocked_refusals += 1
        else:
            tn += 1

    return ConfusionMatrix(tp=tp, fp=fp, fn=fn, tn=tn,
                           excluded=excluded,
                           blocked_refusals=blocked_refusals)


# ---------------------------------------------------------------------------
# Rates
# ---------------------------------------------------------------------------

def _rate(numerator: int, denominator: int, alpha: float = ALPHA) -> Rate:
    if denominator == 0:
        return Rate(float("nan"), float("nan"), float("nan"), numerator, 0)
    lower, upper = proportion_confint(numerator, denominator,
                                      alpha=alpha, method="wilson")
    return Rate(numerator / denominator, lower, upper, numerator, denominator)


def strict_compliance_rate(cm: ConfusionMatrix, alpha: float = ALPHA) -> Rate:
    """SCR = (TP + TN) / Total — the system got it right, either way."""
    return _rate(cm.tp + cm.tn, cm.total, alpha)


def false_negative_rate(cm: ConfusionMatrix, alpha: float = ALPHA) -> Rate:
    """
    FNR = FN / (FN + TP) — of all hallucinated answers, the fraction that
    reached the user. The key safety metric.

    NOTE the Baseline arm has no verifier, so TP = 0 and FNR = 1.0 by
    construction: every hallucination reaches the user. That is correct and
    expected, not a bug — but it means FNR is not the interesting
    baseline-vs-dual comparison. Use the paired per-query test in stats.py.
    """
    return _rate(cm.fn, cm.fn + cm.tp, alpha)


def false_positive_rate(cm: ConfusionMatrix, alpha: float = ALPHA) -> Rate:
    """FPR = FP / (FP + TN) — of all faithful answers, the fraction wrongly
    blocked. Measured against the threshold set in advance."""
    return _rate(cm.fp, cm.fp + cm.tn, alpha)


# ---------------------------------------------------------------------------
# Per-query binary outcome — the paired input for McNemar (see stats.py)
# ---------------------------------------------------------------------------

def hallucination_reached_user(record: Mapping) -> bool:
    """
    True if a hallucinated answer was displayed. This is the per-query
    safety outcome that the paired McNemar comparisons use (all three pairs
    of arms).

    Excluded outcomes raise: they must be filtered before pairing, or the
    two arms would be compared on different query sets.
    """
    label, outcome = record["human_label"], record["outcome"]
    if outcome in EXCLUDED_OUTCOMES:
        raise ValueError(
            f"record {record.get('query_id')} has excluded outcome "
            f"{outcome!r}; filter these out before pairing arms")
    return (label in HALLUCINATION_LABELS) and (outcome not in BLOCKING_OUTCOMES)


def paired_safety_outcomes(
    arm_a: Sequence[Mapping], arm_b: Sequence[Mapping]
) -> tuple[list[bool], list[bool]]:
    """
    Align two arms by query_id and return their per-query safety outcomes.
    Queries excluded in EITHER arm are dropped from BOTH, so the pairing
    stays honest. Returns (outcomes_a, outcomes_b) in matching order.
    """
    by_id_a = {r["query_id"]: r for r in arm_a}
    by_id_b = {r["query_id"]: r for r in arm_b}

    usable = [
        qid for qid in by_id_a
        if qid in by_id_b
        and by_id_a[qid]["outcome"] not in EXCLUDED_OUTCOMES
        and by_id_b[qid]["outcome"] not in EXCLUDED_OUTCOMES
    ]
    usable.sort()

    return ([hallucination_reached_user(by_id_a[q]) for q in usable],
            [hallucination_reached_user(by_id_b[q]) for q in usable])


# ---------------------------------------------------------------------------
# Cost — per-query latency and tokens, and the percentile used to report them
# ---------------------------------------------------------------------------

def recorded_latency_s(record: Mapping) -> float:
    """
    Recorded latency for one query: the SUCCESSFUL generation, verification
    and rewrite calls, summed. Retrieval, failed attempts and retry waits are
    excluded (the *_wall_s fields carry those). This is the latency in the
    cost and run-to-run stability results.
    """
    return ((record.get("t_generation_s") or 0.0)
            + sum(record.get("t_verification_s") or [])
            + sum(record.get("t_rewrite_s") or []))


def total_tokens(record: Mapping) -> int:
    """Prompt plus completion tokens for one query, both nodes together."""
    return record["tokens_prompt_total"] + record["tokens_completion_total"]


def nearest_rank_percentile(values: Sequence[float], q: float) -> float:
    """
    The q-th percentile by the nearest-rank method: the ceil(q/100 * n)-th
    smallest value. Always an observed value, never an interpolation.
    """
    if not values:
        raise ValueError("no values")
    if not 0 < q <= 100:
        raise ValueError(f"q must be in (0, 100], got {q}")
    ordered = sorted(values)
    rank = math.ceil(q * len(ordered) / 100)
    return ordered[rank - 1]