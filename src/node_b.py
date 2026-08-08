"""
Node B — the verifier. Decomposes a draft answer into atomic claims and judges
each against the retrieved passages.

WHICH MODEL RUNS HERE IS ARM-DEPENDENT (src/config.py):
  baseline      NODE_B_ENABLED is False — this module is never called.
  same_vendor   ACTIVE_NODE_B_MODEL is NODE_A_MODEL (Gemini): the generator
                checks its own draft under information asymmetry.
  cross_vendor  ACTIVE_NODE_B_MODEL is a different vendor (OpenAI).

TWO VENDOR PATHS, ONE SCHEMA. Both were confirmed live against the same
three-claim test (supported / unsupported / contradicted, all correct):
  OpenAI  client.responses.parse(..., text_format=Verification)
  Gemini  generate_content(..., config={"response_schema": Verification})
Structured output is SCHEMA-ENFORCED, so conformance is by construction rather
than by parsing hope. Pydantic validation still runs on the result, and one
retry is allowed, before the terminal `verifier_malformed` outcome.

WHAT THIS MODULE DOES NOT DO:
- It does not decide anything. Whether flagged claims trigger a rewrite, how
  many rewrites are allowed, when to fall back — all controller.
- It is not told the query category, the target FAQ, the arm, or whether the
  draft came from a first pass or a rewrite. It sees question, passages, draft.
- It does not judge relevance, tone, or helpfulness. Only whether each claim is
  supported by the passages.

MALFORMATION IS NOT ROUTED TO THE REWRITE LOOP (NOTES 21.7.2026): without
verdicts there are no flagged claims to rewrite from, and letting a parse
failure consume a rewrite attempt would contaminate the outcome-split and
attempt-count metrics. The Node B CALL is retried; the loop is not entered.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field

from src import prompts
from src.config import (
    ACTIVE_NODE_B_MODEL,
    GOOGLE_API_KEY,
    NODE_B_REASONING_EFFORT,
    OPENAI_API_KEY,
    REQUEST_TIMEOUT,
    RETRY_BASE_DELAY,
    _get_vendor,
)

# One corrective retry, then give up. Deliberately NOT MAX_RETRIES: that is for
# transport failures. A malformed-but-successful response is a different fault,
# and retrying it many times is unlikely to help — the contingency ladder in
# NOTES 21.7.2026 (prompt fix -> effort -> tier) is the real escalation path.
_MALFORMED_RETRIES = 1


class Verdict(str, Enum):
    """The three verdicts. An Enum, not a bare string, so the schema constrains
    the model to exactly these values — an invented fourth label ("partially
    supported") becomes impossible rather than something to detect."""
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"


class Claim(BaseModel):
    text: str = Field(description="The atomic claim, quoted or closely "
                                  "paraphrased from the draft answer.")
    verdict: Verdict = Field(description="supported | contradicted | unsupported")


class Verification(BaseModel):
    claims: list[Claim] = Field(description="Every atomic factual claim in the "
                                            "draft, one entry each.")


class VerifierMalformed(RuntimeError):
    """Node B's output failed schema validation after the retry. The controller
    records `verifier_malformed` and routes to fallback WITHOUT consuming a
    rewrite attempt."""


@dataclass(frozen=True)
class VerificationResult:
    """Node B's output for one call.

    api_latency_s is the SUCCESSFUL call's duration — the latency figure. The
    controller records wall time separately, which includes retried
    malformations and backoff. See node_a.Draft for the rationale.
    """
    claims: list[Claim]
    tokens_prompt: int
    tokens_completion: int
    attempts: int
    api_latency_s: float

    @property
    def flagged(self) -> list[Claim]:
        """Claims that are not supported — what the rewrite prompt passes back
        to Node A. Order is preserved so the rewrite sees them as the verifier
        found them."""
        return [c for c in self.claims if c.verdict is not Verdict.SUPPORTED]

    @property
    def all_supported(self) -> bool:
        """True when the draft may be shown to the user.

        NOTE an empty claims list counts as all-supported. That is a real edge
        case: a draft with no extractable factual claim (a bare refusal, say)
        has nothing to reject. The controller should not normally send such a
        draft here — Node A refusals short-circuit to fallback — but if one
        arrives, passing it is the correct behaviour: there is no unsupported
        claim to block on.
        """
        return not self.flagged


_openai_client = None
_genai_client = None


def _sleep_backoff(attempt: int) -> None:
    delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
    time.sleep(delay + random.uniform(0, 0.5 * delay))


# ---------------------------------------------------------------------------
# Vendor path: OpenAI (cross_vendor arm)
# ---------------------------------------------------------------------------

def _verify_openai(system: str, user: str) -> tuple[Verification, int, int]:
    global _openai_client
    from openai import OpenAI

    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY, timeout=REQUEST_TIMEOUT)

    response = _openai_client.responses.parse(
        model=ACTIVE_NODE_B_MODEL,
        reasoning={"effort": NODE_B_REASONING_EFFORT},
        input=[{"role": "system", "content": system},
               {"role": "user", "content": user}],
        text_format=Verification,
    )

    parsed = response.output_parsed
    if parsed is None:
        raise ValueError("OpenAI returned no parsed output "
                         f"(status={getattr(response, 'status', '?')})")

    usage = getattr(response, "usage", None)
    return (parsed,
            getattr(usage, "input_tokens", 0) or 0,
            getattr(usage, "output_tokens", 0) or 0)


# ---------------------------------------------------------------------------
# Vendor path: Gemini (same_vendor arm)
# ---------------------------------------------------------------------------

def _verify_gemini(system: str, user: str) -> tuple[Verification, int, int]:
    global _genai_client
    from google import genai
    from google.genai import types

    if _genai_client is None:
        _genai_client = genai.Client(api_key=GOOGLE_API_KEY)

    response = _genai_client.models.generate_content(
        model=ACTIVE_NODE_B_MODEL,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,          # verification is classification
            response_mime_type="application/json",
            response_schema=Verification,
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT * 1000),
        ),
    )

    # google-genai populates .parsed when response_schema is a Pydantic model;
    # fall back to validating .text if a future SDK version stops doing so.
    parsed = getattr(response, "parsed", None)
    if parsed is None:
        parsed = Verification.model_validate_json(response.text)

    usage = getattr(response, "usage_metadata", None)
    return (parsed,
            getattr(usage, "prompt_token_count", 0) or 0,
            getattr(usage, "candidates_token_count", 0) or 0)


# ---------------------------------------------------------------------------
# Public API — the controller calls this
# ---------------------------------------------------------------------------

def verify(question: str, passages, draft: str) -> VerificationResult:
    """
    Verify a draft answer against the retrieved passages.

    Routes to the vendor path implied by ACTIVE_NODE_B_MODEL, so the arm
    mechanism in config decides which model verifies without this module or
    the controller knowing which arm is live.

    Raises VerifierMalformed if the response fails schema validation after one
    retry. Transport failures propagate as-is: they are the controller's to
    handle, and conflating them with malformation would blur two distinct
    outcomes (`generation_failed`-style infrastructure faults versus
    `verifier_malformed`).
    """
    if ACTIVE_NODE_B_MODEL is None:
        raise RuntimeError(
            "Node B called with ACTIVE_NODE_B_MODEL=None — the baseline arm "
            "has no verifier. The controller should not reach this call.")

    vendor = _get_vendor(ACTIVE_NODE_B_MODEL)
    if vendor == "openai":
        call = _verify_openai
    elif vendor == "google":
        call = _verify_gemini
    else:
        raise RuntimeError(
            f"No Node B path for vendor {vendor!r} (model "
            f"{ACTIVE_NODE_B_MODEL!r}). Add one before running this arm.")

    user = prompts.NODE_B_USER.format(
        passages=prompts.format_passages(passages),
        question=question,
        draft=draft,
    )

    last_error: Exception | None = None
    for attempt in range(1, _MALFORMED_RETRIES + 2):
        try:
            _t0 = time.perf_counter()
            parsed, tp, tc = call(prompts.NODE_B_SYSTEM, user)
            return VerificationResult(claims=parsed.claims,
                                      tokens_prompt=tp,
                                      tokens_completion=tc,
                                      attempts=attempt,
                                      api_latency_s=time.perf_counter() - _t0)
        except Exception as exc:
            # Schema/validation faults are retried once. Anything else
            # (transport, auth, rate limit) is re-raised for the controller.
            if not _is_malformation(exc):
                raise
            last_error = exc
            if attempt <= _MALFORMED_RETRIES:
                _sleep_backoff(attempt)

    raise VerifierMalformed(
        f"Node B output failed validation after {_MALFORMED_RETRIES + 1} "
        f"attempts: {last_error}") from last_error


def _is_malformation(exc: Exception) -> bool:
    """
    Distinguish 'the model produced something unusable' from 'the call failed'.

    Deliberately conservative: only clear validation/parse faults count as
    malformation. A transport error misclassified as malformation would be
    logged as verifier_malformed and pollute a reported count with
    infrastructure noise.
    """
    from pydantic import ValidationError
    if isinstance(exc, (ValidationError, ValueError)):
        return True
    name = type(exc).__name__
    return "JSONDecode" in name or "Validation" in name


# ---------------------------------------------------------------------------
# Manual pre-test — live API calls. Not a pytest test.
#   uv run python -m src.node_b
#
# Answers the three questions the pre-test must settle before any sweep:
#   1. Does Node B flag DECLINING CLAUSES ("I do not have information on X")
#      as unsupported? 11 of 40 adversarial answers and all 6 compound answers
#      contain one. If flagged, every such answer becomes a False Positive
#      through the very behaviour rule 6 engineered.
#   2. How does it treat ADV-31, a real marginal over-read produced by the
#      pipeline itself ("a secondary sign-off is required above HKD 10,000",
#      where the passage says a manager may approve UP TO 10,000 WITHOUT one)?
#   3. Does it catch PLANTED contradicted and unsupported claims — the
#      instrument validation that stands in for the arm comparison if the
#      arms yield no hallucinations.
# ---------------------------------------------------------------------------

def _main() -> None:
    from src.retrieval import retrieve

    cases = [
        # (label, question, draft, what to look for)
        ("1. DECLINING CLAUSE — the critical case",
         "Is there bereavement leave for grandparents?",
         "Employees are granted 5 consecutive working days of fully paid "
         "compassionate leave for the loss of an immediate family member. I do "
         "not have information on whether grandparents count as immediate "
         "family.",
         "the declining sentence should NOT be flagged unsupported"),

        ("2. ADV-31 — real marginal over-read from the pipeline",
         "Who signs off on purchases above HKD 10,000?",
         "A secondary sign-off is required for purchase orders above HKD "
         "10,000, but I do not have information on who provides this secondary "
         "sign-off.",
         "does it accept the inference or flag it?"),

        ("3. PLANTED CONTRADICTION — must be caught",
         "What is the maternity leave entitlement?",
         "Female employees are entitled to 14 weeks of fully paid maternity "
         "leave.",
         "expect: contradicted (passage says 20 weeks)"),

        ("4. PLANTED UNSUPPORTED — must be caught",
         "What is the annual leave entitlement for full-time staff?",
         "Permanent full-time employees receive 24 calendar days of paid "
         "annual leave per year, and unused days may be carried over for up "
         "to three months.",
         "expect: 24 days supported, carry-over unsupported"),

        ("5. CLEAN FAITHFUL — must pass",
         "What is the annual leave entitlement for full-time staff?",
         "Permanent full-time employees receive 24 calendar days of paid "
         "annual leave per year.",
         "expect: all supported"),

        ("6. COURTESY FRAMING — is rule 8 necessary?",
         "How many days of compassionate leave are provided?",
         "Thank you for your question. Employees are granted 5 consecutive "
         "working days of fully paid compassionate leave for the loss of an "
         "immediate family member. Please contact HR if you need further "
         "assistance.",
         "are the greeting and sign-off flagged unsupported?"),
    ]

    print(f"Node B pre-test — model={ACTIVE_NODE_B_MODEL} "
          f"vendor={_get_vendor(ACTIVE_NODE_B_MODEL)}\n")

    for label, question, draft, expectation in cases:
        passages = retrieve(question)
        print(f"--- {label} ---")
        print(f"  look for: {expectation}")
        try:
            result = verify(question, passages, draft)
        except VerifierMalformed as exc:
            print(f"  MALFORMED: {exc}\n")
            continue

        print(f"  {len(result.claims)} claims, "
              f"tokens={result.tokens_prompt}+{result.tokens_completion}, "
              f"all_supported={result.all_supported}")
        for c in result.claims:
            mark = " " if c.verdict is Verdict.SUPPORTED else ">"
            print(f"   {mark} [{c.verdict.value}] {c.text}")
        print()


if __name__ == "__main__":
    _main()