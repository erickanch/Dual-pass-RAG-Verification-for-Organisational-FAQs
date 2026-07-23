"""
Shared vocabulary for the dual-pass RAG experiment.

WHY THIS FILE EXISTS (and is separate from config.py):
config.py has side effects — it loads .env and raises if API keys are
missing. Analysis code and unit tests must run WITHOUT API keys, so the
constants they share with the pipeline live here instead. This module
has no imports, no side effects, and is safe to import from anywhere.

config.py re-exports these so the "one source of truth" rule holds:
nothing defines a label, an outcome, or alpha anywhere else.
"""

# ---------------------------------------------------------------------------
# Human labels — FOUR, per the proposal (Appendix D Phase 3) and the
# Labelling Guide v3. A fifth ("wrongly withheld") was drafted and retracted:
# it required the labeller to know the answer had been blocked, which breaks
# the blinding rule. See NOTES 21.7.2026.
# ---------------------------------------------------------------------------

LABEL_FAITHFUL = "faithful"
LABEL_INTRINSIC = "intrinsic"            # contradicts the retrieved passages
LABEL_EXTRINSIC = "extrinsic"            # unsupported by the retrieved passages
LABEL_CORRECT_REFUSAL = "correct_refusal"

VALID_LABELS = (
    LABEL_FAITHFUL,
    LABEL_INTRINSIC,
    LABEL_EXTRINSIC,
    LABEL_CORRECT_REFUSAL,
)

HALLUCINATION_LABELS = (LABEL_INTRINSIC, LABEL_EXTRINSIC)


# ---------------------------------------------------------------------------
# Controller outcomes — FIVE. What the SYSTEM did (distinct from what the
# human judged). Written per query by the controller.
# ---------------------------------------------------------------------------

OUTCOME_PASSED_FIRST = "passed_first"        # verified on the first attempt
OUTCOME_FIXED_BY_REWRITE = "fixed_by_rewrite"  # rewritten, then verified
OUTCOME_FALLBACK = "fallback"                # attempts exhausted -> withheld
OUTCOME_VERIFIER_MALFORMED = "verifier_malformed"  # Node B unparseable after retry
OUTCOME_GENERATION_FAILED = "generation_failed"    # Node A API failure; no draft

VALID_OUTCOMES = (
    OUTCOME_PASSED_FIRST,
    OUTCOME_FIXED_BY_REWRITE,
    OUTCOME_FALLBACK,
    OUTCOME_VERIFIER_MALFORMED,
    OUTCOME_GENERATION_FAILED,
)

# Outcomes where the verifier BLOCKED the answer from reaching the user.
# fixed_by_rewrite counts as PASSED: the final answer was displayed.
BLOCKING_OUTCOMES = (OUTCOME_FALLBACK,)

# Outcomes excluded from the confusion matrix entirely: infrastructure
# failures, not verification decisions. Reported as counts alongside results.
EXCLUDED_OUTCOMES = (OUTCOME_VERIFIER_MALFORMED, OUTCOME_GENERATION_FAILED)


# ---------------------------------------------------------------------------
# DECISION (ratify or change here — affects every reported denominator):
# how does a Correct Refusal map onto the matrix's two ground-truth states?
#
# "faithful"  -> a refusal contains no unsupported claims, so it sits on the
#                faithful side; a BLOCKED refusal therefore counts as an FP.
#                Chosen: keeps the four labels mapping cleanly onto two
#                states, and blocked refusals should be near-zero anyway
#                (a refusal has no claims for Node B to reject). Blocked
#                refusals are reported as an FP sub-breakdown, since the
#                user sees a non-answer either way and the usability cost
#                is nominal.
# "exclude"   -> refusals leave the matrix entirely and are reported only as
#                a refusal rate. Cleaner conceptually, but shrinks the
#                denominators and loses the over-blocking signal.
# ---------------------------------------------------------------------------

CORRECT_REFUSAL_TREATED_AS = "faithful"   # "faithful" | "exclude"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

ALPHA = 0.05                      # significance level
CONFIDENCE_LEVEL = 1 - ALPHA      # 0.95 for Wilson intervals on SCR / FNR / FPR
RANDOM_SEED = 42                  # Used ONLY for: (1) seeded shuffle of the blind-labelling
                                  # presentation order, (2) the second-reviewer 15-20% sample
                                  # draw. Does NOT affect API calls — provider determinism
                                  # comes from temperature, not this seed.