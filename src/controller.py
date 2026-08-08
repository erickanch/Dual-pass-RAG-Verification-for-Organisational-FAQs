"""
The deterministic controller. Owns routing, the rewrite loop, the fallback,
and the per-query log.

CONTROL LOGIC LIVES HERE, IN ORDINARY CODE — never in the models. That is the
proposal's central design choice: letting an LLM orchestrate its own checking
would reintroduce the self-trust the architecture exists to avoid. Every
branch below is auditable and repeatable.

ROUTING (three short-circuits, then the loop):

    Node A
      ├─ raises GenerationFailed  -> outcome generation_failed, fallback shown.
      │                              No verification, no rewrite: there is no
      │                              draft to process. NOT the same as
      │                              `fallback`, which means the system judged
      │                              it could not answer. Keeping them apart is
      │                              what stops infrastructure noise polluting
      │                              the Phase 1 coverage/refusal denominators.
      │
      ├─ returns INSUFFICIENT_CONTEXT -> outcome fallback, fallback shown.
      │                              No verification (a refusal makes no claims
      │                              to check) and no rewrite (deterministic
      │                              Node A on the same passages returns the
      │                              identical refusal).
      │
      └─ returns a draft -> Node B
                             ├─ raises VerifierMalformed -> outcome
                             │    verifier_malformed, fallback shown, WITHOUT
                             │    consuming a rewrite attempt. Without verdicts
                             │    there are no flagged claims to rewrite from,
                             │    and letting a parse failure spend an attempt
                             │    would contaminate the outcome-split and
                             │    attempt-count metrics.
                             │
                             ├─ all claims supported -> outcome passed_first
                             │    (or fixed_by_rewrite if this is a later
                             │    round), answer shown.
                             │
                             └─ claims flagged -> rewrite with ONLY those
                                  claims, up to MAX_REWRITE_ATTEMPTS, then
                                  outcome fallback.

BASELINE ARM: NODE_B_ENABLED is False, so a draft is shown as soon as it
exists. The only outcomes reachable are passed_first, fallback (Node A
refused) and generation_failed.

THE DRAFT CHAIN. Every draft produced for a query is recorded with its stage
and a hash — initial, rewrite_1, rewrite_2, ... Human labelling judges drafts,
not the displayed output, and every distinct draft is labelled. The hash lets
labelling.py group identical texts ACROSS ARMS: the initial draft is produced
by the same Node A, prompt and passages in all three arms, so where the hashes
match it is read once and the label applied to every matching record. The
controller only RECORDS the hash; grouping is an analysis decision and lives
in labelling.py, because at runtime only one arm exists and the other arms'
results may not have been produced yet.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

from src import node_a, node_b, prompts
from src.config import (
    ACTIVE_ARM,
    ACTIVE_NODE_B_MODEL,
    LOG_SCHEMA_VERSION,
    MAX_REWRITE_ATTEMPTS,
    NODE_B_ENABLED,
    RETRIEVAL_K,
)
from src.constants import (
    OUTCOME_FALLBACK,
    OUTCOME_FIXED_BY_REWRITE,
    OUTCOME_GENERATION_FAILED,
    OUTCOME_PASSED_FIRST,
    OUTCOME_VERIFIER_MALFORMED,
)
from src.retrieval import retrieve


def _sha(text: str) -> str:
    """Short content hash. Truncated to 16 hex chars: collision risk across a
    few hundred short answers is negligible, and the log stays readable."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class QueryResult:
    """One query's complete record. Serialised straight into the results file;
    the field set IS the log contract in config.py."""
    query_id: str
    arm: str

    outcome: str
    answer_shown: str          # what the user saw: the answer, or the fallback
    attempts: int              # rewrite attempts consumed (0 = none)

    drafts: list[dict] = field(default_factory=list)
    verifications: list[dict] = field(default_factory=list)

    retrieved_chunk_ids: list[str] = field(default_factory=list)

    # TIMINGS — two kinds, deliberately.
    #   *_s      : the SUCCESSFUL API call's duration. This is what the
    #              latency percentiles report, because it is what a user
    #              experiences.
    #   *_wall_s : elapsed time including failed attempts and backoff sleeps.
    #              This is what the sweep costs in clock time.
    # They diverge only when retries fire, but a 429 plus a 6-second backoff
    # would otherwise be reported as 6 seconds of generation latency and
    # distort the p95.
    t_retrieval_s: float = 0.0
    t_generation_s: float = 0.0
    t_generation_wall_s: float = 0.0
    t_verification_s: list[float] = field(default_factory=list)
    t_verification_wall_s: list[float] = field(default_factory=list)
    t_rewrite_s: list[float] = field(default_factory=list)
    t_rewrite_wall_s: list[float] = field(default_factory=list)
    t_total_s: float = 0.0

    # TOKENS — split by node. Node A and Node B may be different vendors with
    # different prices, so a single total cannot be converted to a cost.
    # Rates are applied in analysis, never here: a results file with a
    # hardcoded price becomes WRONG when prices change, rather than merely
    # stale, and re-deriving beats re-running.
    tokens_a_prompt: int = 0
    tokens_a_completion: int = 0
    tokens_b_prompt: int = 0
    tokens_b_completion: int = 0

    # API attempt counts, for attributing wall/latency divergence to retries.
    generation_attempts: list[int] = field(default_factory=list)
    verification_attempts: list[int] = field(default_factory=list)

    log_schema_version: str = LOG_SCHEMA_VERSION
    error: str | None = None   # exception detail on infrastructure failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "arm": self.arm,
            "outcome": self.outcome,
            "answer_shown": self.answer_shown,
            "attempts": self.attempts,
            "drafts": self.drafts,
            "verifications": self.verifications,
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "t_retrieval_s": round(self.t_retrieval_s, 4),
            "t_generation_s": round(self.t_generation_s, 4),
            "t_generation_wall_s": round(self.t_generation_wall_s, 4),
            "t_verification_s": [round(t, 4) for t in self.t_verification_s],
            "t_verification_wall_s": [round(t, 4) for t in self.t_verification_wall_s],
            "t_rewrite_s": [round(t, 4) for t in self.t_rewrite_s],
            "t_rewrite_wall_s": [round(t, 4) for t in self.t_rewrite_wall_s],
            "t_total_s": round(self.t_total_s, 4),
            "tokens_a_prompt": self.tokens_a_prompt,
            "tokens_a_completion": self.tokens_a_completion,
            "tokens_b_prompt": self.tokens_b_prompt,
            "tokens_b_completion": self.tokens_b_completion,
            "tokens_prompt_total": self.tokens_a_prompt + self.tokens_b_prompt,
            "tokens_completion_total": (self.tokens_a_completion
                                        + self.tokens_b_completion),
            "generation_attempts": self.generation_attempts,
            "verification_attempts": self.verification_attempts,
            "log_schema_version": self.log_schema_version,
            "error": self.error,
        }


def _record_draft(result: QueryResult, stage: str, draft: node_a.Draft) -> None:
    result.drafts.append({
        "stage": stage,
        "text": draft.text,
        "sha256": _sha(draft.text),
        "refused": draft.refused,
    })
    result.tokens_a_prompt += draft.tokens_prompt
    result.tokens_a_completion += draft.tokens_completion
    result.generation_attempts.append(draft.attempts)


def _record_verification(result: QueryResult, stage: str,
                         verification: node_b.VerificationResult) -> None:
    result.verifications.append({
        "stage": stage,
        "claims": [{"text": c.text, "verdict": c.verdict.value}
                   for c in verification.claims],
        "n_claims": len(verification.claims),
        "n_flagged": len(verification.flagged),
        "all_supported": verification.all_supported,
    })
    result.tokens_b_prompt += verification.tokens_prompt
    result.tokens_b_completion += verification.tokens_completion
    result.verification_attempts.append(verification.attempts)


def answer_query(query_id: str, question: str, k: int = RETRIEVAL_K) -> QueryResult:
    """
    Run one query through the pipeline for the ACTIVE arm.

    Never raises for a pipeline failure: infrastructure faults become
    `generation_failed` or `verifier_malformed` records so a long sweep is not
    lost to one bad call. Programming errors still propagate.
    """
    t_start = time.perf_counter()
    result = QueryResult(query_id=query_id, arm=ACTIVE_ARM,
                         outcome="", answer_shown="", attempts=0)

    # --- retrieval ---------------------------------------------------------
    t0 = time.perf_counter()
    passages = retrieve(question, k=k)
    result.t_retrieval_s = time.perf_counter() - t0
    result.retrieved_chunk_ids = [p.chunk_id for p in passages]

    # --- generation --------------------------------------------------------
    t0 = time.perf_counter()
    try:
        draft = node_a.generate(question, passages)
    except node_a.GenerationFailed as exc:
        result.t_generation_wall_s = time.perf_counter() - t0
        result.outcome = OUTCOME_GENERATION_FAILED
        result.answer_shown = prompts.FALLBACK_MESSAGE
        result.error = str(exc)
        result.t_total_s = time.perf_counter() - t_start
        return result
    result.t_generation_wall_s = time.perf_counter() - t0
    result.t_generation_s = draft.api_latency_s
    _record_draft(result, "initial", draft)

    # --- short-circuit: Node A refused -------------------------------------
    if draft.refused:
        result.outcome = OUTCOME_FALLBACK
        result.answer_shown = prompts.FALLBACK_MESSAGE
        result.t_total_s = time.perf_counter() - t_start
        return result

    # --- baseline arm: no verifier -----------------------------------------
    if not NODE_B_ENABLED:
        result.outcome = OUTCOME_PASSED_FIRST
        result.answer_shown = draft.text
        result.t_total_s = time.perf_counter() - t_start
        return result

    # --- verification, then the rewrite loop -------------------------------
    # `attempt` counts REWRITES, so attempt 0 is the initial verification.
    for attempt in range(MAX_REWRITE_ATTEMPTS + 1):
        stage = "initial" if attempt == 0 else f"rewrite_{attempt}"

        t0 = time.perf_counter()
        try:
            verification = node_b.verify(question, passages, draft.text)
        except node_b.VerifierMalformed as exc:
            result.t_verification_wall_s.append(time.perf_counter() - t0)
            result.outcome = OUTCOME_VERIFIER_MALFORMED
            result.answer_shown = prompts.FALLBACK_MESSAGE
            result.error = str(exc)
            result.attempts = attempt          # rewrites consumed so far
            result.t_total_s = time.perf_counter() - t_start
            return result
        except Exception as exc:               # transport failure at Node B
            result.t_verification_wall_s.append(time.perf_counter() - t0)
            result.outcome = OUTCOME_VERIFIER_MALFORMED
            result.answer_shown = prompts.FALLBACK_MESSAGE
            result.error = f"{type(exc).__name__}: {exc}"
            result.attempts = attempt
            result.t_total_s = time.perf_counter() - t_start
            return result
        result.t_verification_wall_s.append(time.perf_counter() - t0)
        result.t_verification_s.append(verification.api_latency_s)
        _record_verification(result, stage, verification)

        if verification.all_supported:
            result.outcome = (OUTCOME_PASSED_FIRST if attempt == 0
                              else OUTCOME_FIXED_BY_REWRITE)
            result.answer_shown = draft.text
            result.attempts = attempt
            result.t_total_s = time.perf_counter() - t_start
            return result

        # Flagged. Rewrite unless the cap is reached.
        if attempt == MAX_REWRITE_ATTEMPTS:
            break

        t0 = time.perf_counter()
        try:
            draft = node_a.rewrite(question, passages, verification.flagged)
        except node_a.GenerationFailed as exc:
            result.t_rewrite_wall_s.append(time.perf_counter() - t0)
            result.outcome = OUTCOME_GENERATION_FAILED
            result.answer_shown = prompts.FALLBACK_MESSAGE
            result.error = str(exc)
            result.attempts = attempt + 1
            result.t_total_s = time.perf_counter() - t_start
            return result
        result.t_rewrite_wall_s.append(time.perf_counter() - t0)
        result.t_rewrite_s.append(draft.api_latency_s)
        _record_draft(result, f"rewrite_{attempt + 1}", draft)

        # A rewrite may itself refuse: if removing the unsupported claims
        # leaves nothing the passages support, refusing is correct. Treat it
        # as any refusal — straight to fallback, no further verification.
        if draft.refused:
            result.outcome = OUTCOME_FALLBACK
            result.answer_shown = prompts.FALLBACK_MESSAGE
            result.attempts = attempt + 1
            result.t_total_s = time.perf_counter() - t_start
            return result

    # --- attempts exhausted, still flagged ---------------------------------
    result.outcome = OUTCOME_FALLBACK
    result.answer_shown = prompts.FALLBACK_MESSAGE
    result.attempts = MAX_REWRITE_ATTEMPTS
    result.t_total_s = time.perf_counter() - t_start
    return result


# ---------------------------------------------------------------------------
# Manual check — live API calls. Not a pytest test.
#   uv run python -m src.controller
#   ACTIVE_ARM=baseline uv run python -m src.controller
# ---------------------------------------------------------------------------

def _main() -> None:
    import json

    cases = [
        ("IN-01", "What is the annual leave entitlement for full-time staff?"),
        ("IN-35", "Do I only need to give 7 days' notice during the "
                  "probationary period?"),
        ("ADV-01", "What is the paternity leave entitlement?"),
        ("ADV-31", "Who signs off on purchases above HKD 10,000?"),
        ("OOD-05", "What is the capital of France?"),
    ]

    print(f"Controller check — arm={ACTIVE_ARM} "
          f"node_b={'off' if not NODE_B_ENABLED else ACTIVE_NODE_B_MODEL}\n")

    for qid, question in cases:
        r = answer_query(qid, question)
        print(f"--- {qid}: {question}")
        print(f"  outcome={r.outcome}  attempts={r.attempts}  "
              f"drafts={len(r.drafts)}  verifications={len(r.verifications)}")
        print(f"  t_total={r.t_total_s:.2f}s  "
              f"gen={r.t_generation_s:.2f}s(api)/{r.t_generation_wall_s:.2f}s(wall)")
        print(f"  tokens A={r.tokens_a_prompt}+{r.tokens_a_completion}  "
              f"B={r.tokens_b_prompt}+{r.tokens_b_completion}")
        print(f"  shown: {r.answer_shown[:120]}")
        for v in r.verifications:
            print(f"    [{v['stage']}] {v['n_claims']} claims, "
                  f"{v['n_flagged']} flagged")
        print()

    print("Sample record (JSON as it will be logged):")
    print(json.dumps(answer_query(*cases[0]).to_dict(), indent=2)[:900])


if __name__ == "__main__":
    _main()