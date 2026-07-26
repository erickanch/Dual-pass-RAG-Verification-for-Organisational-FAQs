"""
Node A — the generator. Gemini, temperature 0.

Takes a question plus retrieved passages and returns either a grounded draft
answer or the INSUFFICIENT_CONTEXT sentinel. Also serves the rewrite round:
same model, same system prompt, different user message carrying the flagged
claims.

WHAT THIS MODULE DOES NOT DO:
- It does not decide what happens next. Whether a refusal goes to fallback,
  whether a draft goes to Node B, how many rewrites are allowed — all of that
  is the controller's job. Node A answers or refuses; the controller routes.
- It does not know the query's category, target FAQ, or the experimental arm.
  Only the question text and the passage text reach the model.

FAILURE HANDLING (per NOTES 21.7.2026):
Transport failures are retried with exponential backoff. On exhaustion this
raises GenerationFailed, which the controller records as the terminal outcome
`generation_failed` — NOT `fallback`. A fallback means the system judged it
could not answer; generation_failed means the system could not run. Keeping
them separate is what stops the Phase 1 coverage/refusal denominators being
contaminated by infrastructure noise.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from google import genai
from google.genai import types

from src import prompts
from src.config import (
    GOOGLE_API_KEY,
    MAX_RETRIES,
    NODE_A_MODEL,
    NODE_A_TEMPERATURE,
    REQUEST_TIMEOUT,
    RETRY_BASE_DELAY,
)

# Shortest plausible real answer. Anything below this is treated as a
# degenerate response (empty string, stray punctuation, a truncated token)
# rather than a genuine answer, and retried.
_MIN_ANSWER_CHARS = 1


class GenerationFailed(RuntimeError):
    """Node A could not produce a response after retries. The controller
    records `generation_failed` and skips verification and the rewrite loop —
    there is no draft to process."""


@dataclass(frozen=True)
class Draft:
    """Node A's output for one call.

    refused=True means the model emitted the INSUFFICIENT_CONTEXT sentinel:
    the passages do not answer the question. The controller short-circuits
    such a draft straight to the fallback message, with no verification (no
    claims to check) and no rewrite (deterministic Node A on the same
    passages would return the identical refusal).
    """
    text: str                 # the answer, or the sentinel
    refused: bool
    tokens_prompt: int
    tokens_completion: int
    attempts: int             # API attempts consumed, including retries


_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GOOGLE_API_KEY)
    return _client


def _is_refusal(text: str) -> bool:
    """
    Detect the sentinel.

    Deliberately tolerant of surrounding whitespace and a trailing full stop,
    but NOT of a sentinel buried in prose. If the model wraps the sentinel in
    explanation ("I'm afraid INSUFFICIENT_CONTEXT because..."), that is a
    prompt defect to fix in prompts.py — not something to paper over here,
    because a tolerant matcher would hide the defect and quietly reclassify
    partial answers as refusals.
    """
    stripped = text.strip().rstrip(".").strip()
    return stripped == prompts.INSUFFICIENT_CONTEXT


def _call_gemini(system: str, user: str) -> tuple[str, int, int, int]:
    """
    One logical generation, with retry-and-backoff around transport failures.
    Returns (text, tokens_prompt, tokens_completion, attempts).

    Raises GenerationFailed when retries are exhausted, or when the model
    keeps returning degenerate output.
    """
    client = _get_client()
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=NODE_A_MODEL,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=NODE_A_TEMPERATURE,
                    http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT * 1000),
                ),
            )

            text = (response.text or "").strip()

            # Degenerate output: treat as a failure worth retrying rather than
            # passing an empty draft downstream, where Node B would verify
            # nothing and the outcome would be silently meaningless.
            if len(text) < _MIN_ANSWER_CHARS and not _is_refusal(text):
                last_error = ValueError(
                    f"degenerate response ({len(text)} chars): {text!r}")
                if attempt < MAX_RETRIES:
                    _sleep_backoff(attempt)
                    continue
                raise GenerationFailed(
                    f"Node A returned degenerate output on all {MAX_RETRIES} "
                    f"attempts; last: {text!r}")

            usage = getattr(response, "usage_metadata", None)
            tokens_prompt = getattr(usage, "prompt_token_count", 0) or 0
            tokens_completion = (
                getattr(usage, "candidates_token_count", 0) or 0)

            return text, tokens_prompt, tokens_completion, attempt

        except GenerationFailed:
            raise
        except Exception as exc:          # transport, rate limit, 5xx
            last_error = exc
            if attempt < MAX_RETRIES:
                _sleep_backoff(attempt)
                continue

    raise GenerationFailed(
        f"Node A failed after {MAX_RETRIES} attempts: {last_error}"
    ) from last_error


def _sleep_backoff(attempt: int) -> None:
    """
    Exponential backoff with jitter. The Gemini free tier is rate-limited
    (RPM and RPD) and a 108-query sweep WILL hit 429s; jitter avoids a
    thundering herd if this is ever run concurrently.
    """
    delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
    time.sleep(delay + random.uniform(0, 0.5 * delay))


# ---------------------------------------------------------------------------
# Public API — the controller calls these two
# ---------------------------------------------------------------------------

def generate(question: str, passages) -> Draft:
    """
    First-pass generation.

    passages: iterable of retrieval.Passage. Only their .text reaches the
    model — faq_id and distance are experimental bookkeeping and are
    deliberately withheld from the prompt.
    """
    user = prompts.NODE_A_USER.format(
        passages=prompts.format_passages(passages),
        question=question,
    )
    text, tp, tc, attempts = _call_gemini(prompts.NODE_A_SYSTEM, user)
    return Draft(text=text, refused=_is_refusal(text),
                 tokens_prompt=tp, tokens_completion=tc, attempts=attempts)


def rewrite(question: str, passages, flagged_claims) -> Draft:
    """
    Rewrite round: regenerate correcting ONLY the claims Node B flagged.

    flagged_claims: the Claim objects whose verdict was not "supported".
    Passing back only the flagged claims (rather than "try again") is what
    makes fixed_by_rewrite a meaningful outcome instead of a second lottery
    ticket — the supported parts of the draft are left alone.

    A rewrite may itself return the sentinel: if removing the unsupported
    claims leaves nothing the passages support, refusing is the correct
    answer. The controller treats that as it treats any refusal.
    """
    user = prompts.REWRITE_USER.format(
        passages=prompts.format_passages(passages),
        question=question,
        flagged=prompts.format_flagged_claims(flagged_claims),
    )
    text, tp, tc, attempts = _call_gemini(prompts.REWRITE_SYSTEM, user)
    return Draft(text=text, refused=_is_refusal(text),
                 tokens_prompt=tp, tokens_completion=tc, attempts=attempts)


# ---------------------------------------------------------------------------
# Manual check — costs a few Gemini calls. Not a pytest test (live API).
#   uv run python -m src.node_a
# ---------------------------------------------------------------------------

def _main() -> None:
    from src.retrieval import retrieve

    cases = [
        ("IN-01 clean in-domain",
         "What is the annual leave entitlement for full-time staff?"),
        ("IN-35 embedded wrong figure",
         "Do I only need to give 7 days' notice during the probationary period?"),
        ("ADV-01 designed gap (paternity)",
         "What is the paternity leave entitlement?"),
        ("OOD-05 out of domain",
         "What is the capital of France?"),
    ]

    for label, question in cases:
        passages = retrieve(question)
        draft = node_a_draft = generate(question, passages)
        flag = "REFUSED" if draft.refused else "ANSWERED"
        print(f"\n--- {label} ---")
        print(f"retrieved: {[p.faq_id for p in passages]}")
        print(f"{flag} (attempts={draft.attempts}, "
              f"tokens={draft.tokens_prompt}+{draft.tokens_completion})")
        print(node_a_draft.text)


if __name__ == "__main__":
    _main()