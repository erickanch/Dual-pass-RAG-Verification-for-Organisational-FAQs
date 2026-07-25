# Methodology Journal & Running Notes

Rule for this file: decisions and their reasons, dated. README says *how*,
PROJECT_STRUCTURE says *where*, this file says *why*.

## 11.7.2026

**Environment & Dependency Decisions (settled during setup)**
*   Python pinned to 3.12 (`requires-python = ">=3.12,<3.13"`): the ML ecosystem
    is not reliable on 3.14 yet.
*   `google-generativeai` SDK is dead (support ended) — migrated to `google-genai`.
*   Ragas pinned at **0.3.9** with `langchain-community<0.4`: Ragas 0.4.3 has an
    open upstream import bug (`ChatVertexAI` moved out of langchain-community),
    and even 0.3.9 breaks against langchain-community 0.4.x. Version is FROZEN
    for the study — changing the measuring instrument mid-study would make
    results non-comparable (same logic as model pinning, Appendix A).
*   GPT-5.6 requires the **Responses API** (`client.responses.create`), not the
    legacy `chat.completions` endpoint (which returns a misleading 401).
    `reasoning={'effort': 'low'}` verified working on gpt-5.6-luna.

**CHUNK_SIZE Discovery (report material — "measured, not assumed")**
*   all-MiniLM-L6-v2 has `max_seq_length = 256` (verified by direct check).
    Chunks longer than 256 word-piece tokens are SILENTLY truncated by the
    embedder — no error, no warning. Original CHUNK_SIZE=512 would have had
    half of every chunk invisible to retrieval.
*   Fixed: CHUNK_SIZE=256, CHUNK_OVERLAP=32. Cost of discovery: one command.
    Cost if undiscovered: silently degraded retrieval misattributed to the
    architecture. This is the flagship example of the verify-before-building
    discipline for the methodology chapter.

## 12.7.2026

**Retrieval Logistics & Determinism**
*   **Goal:** Ensure top-$k$ retrieval is deterministic and reproducible.
*   **Mechanism:**
    *   Exact same prompt.
    *   Same corpus with the same passage embeddings already stored.
    *   Same embedding model, same version explicitly specified in the configuration.
    *   Opted for a local open-source embedding model over a hosted one to
        prevent silent updates.
    *   Same $k$ and explicitly set similarity metric (Cosine) rather than
        relying on ChromaDB defaults.
*   **Top-$k$ Considerations:** How do we decide $k$? Driven by the corpus and
    multi-part FAQs. A human reading an FAQ prefers consolidated answers; the
    RAG should mirror this. Consider corpus size and the impact of irrelevant
    retrieved passages. If $k=3$, it pulls 3 chunks — ASSUMPTION "1 FAQ = 1
    chunk" must be VERIFIED at embedding time: multi-part FAQs may split at
    256 tokens, which changes this reasoning. Similarity threshold considered
    and set to None for baseline comparability.
*   **Open (pilot):** $k$ value to be validated against retrieval recall on the
    real corpus (does the known-supporting chunk appear in top-$k$?).

**Corpus & Experimental Design**
*   **Corpus Principle:** Designed for an experiment but simulating real-world
    organizational characteristics. Not a "gold-standard" perfect FAQ. FAQs
    standalone where possible, with deliberate exceptions for part-and-partial
    queries to test synthesis.
*   **Adversarial Alignment:** The relationship between top-$k$ and adversarial
    queries is highly coupled.
*   **Limitations to Acknowledge:**
    *   Unable to test *completeness* under this design (when relevant
        information isn't provided — same as human handling).
    *   What if there are zero hallucinations simply because the corpus is small?
    *   State explicitly: we are *not* testing retrieval optimization;
        retrieval is held constant as a control variable.

**LLM Judge Configuration (corrected 17.7 — see below for why wording matters)**
*   Ragas 0.3.9 with the default judge **explicitly overridden**. The Ragas
    default is an OpenAI model — it MUST NOT be used here, because it would
    share a vendor with Node B (the verifier under test), which is exactly the
    shared-blind-spot problem the protocol exists to avoid. The override is
    silent-by-default: omitting `llm=` in `evaluate()` falls back to OpenAI
    with no warning.
*   **Judge:** `claude-sonnet-4-5-20250929` (dated snapshot), temperature 0,
    via `LangchainLLMWrapper` — the modern `llm_factory` path is OpenAI-centric
    and cannot serve an Anthropic judge in 0.3.9. The wrapper's deprecation
    warning is accepted and documented in code. Ragas is version-frozen so the
    deprecation cannot bite.
*   Strictly independent third vendor: Google = Generator, OpenAI = Verifier
    (cross_vendor arm), Anthropic = Judge.
*   Embeddings for answer relevance: the local all-MiniLM model (keeps OpenAI
    fully out of the scoring loop). Shared-instrument note: answer relevance is
    computed in the same embedding space used for retrieval — acknowledge as a
    limitation.
*   *Note for Thesis:* Alternative or superior judge configurations left for
    "Future Research".

**Judge Validation Check (for Evaluation/Analysis chapter — zero extra cost)**
*   The Baseline arm is human-labelled in full AND Ragas-scored. The agreement
    between the Ragas faithfulness scores and the human labels on that arm is a
    direct check on the automated judge itself. A judge that scores
    human-identified hallucinations as faithful is unreliable, and the
    Same-Vendor findings (which rest on Ragas alone) must be qualified
    accordingly. Both data sets already exist under the protocol — no extra
    collection. Deliberately NOT added to the proposal (kept scope tight);
    reported in the final report as an additional check.

## 13.7.2026

**Production vs. Experiment Models**
*   In a real-world production environment, the specific models chosen for
    Node A and Node B will be organization-specific and depend on the
    complexity of the subject matter.
*   **Hypothesis (tested by RQ3, not assumed):** cross-vendor verification is
    expected to outperform same-vendor self-checking, because a different
    vendor's model is least likely to share the generator's blind spots and is
    not tied to its update stream. RQ3 tests whether vendor difference adds
    anything beyond the information asymmetry itself. A smaller or null result
    is still a valid finding (per proposal §8). If confirmed, different-vendor
    Node A/Node B is the recommended production architecture (no single point
    of failure).

**Verifier Tier (open — pilot question)**
*   NODE_B_MODEL currently `gpt-5.6-luna`. Pilot: compare Luna vs Terra as
    verifier on ~20-30 questions with planted unsupported claims. Compare
    False Negative Rate/False Positive Rate. If comparable, keep Luna and REPORT
    the check ("architecture works with a low-cost verifier" is itself a finding
    on deployability). If Luna materially worse, switch to Terra with the pilot
    as documented justification. Rationale for not simply picking the strongest:
    the verifier is the treatment under test. The JUDGE is the instrument and
    gets the stronger model (Sonnet-tier).

## 14.7.2026

**Knowledge Management (KM) Analogy**
*   Drew an analogy between traditional KM systems and KM RAG chatbots.
*   Traditional KM often uses hybrid retrieval. Why not use hybrid retrieval
    for this experiment?
    *   *Justification:* This is a controlled experiment. The results
        (verifying generated claims against retrieved context) must hold good
        irrespective of the retrieval logic.
    *   For real-world deployment, retrieval logic should be tailored to the
        organization's specific source documents and taxonomy.

## 17.7.2026

**Model Pinning & Residual Risk (Discussion Chapter Material)**
*   *Question:* Does pinning exact model IDs for production mitigate the
    "silent update" issue?
*   *Answer:* Partially. It mitigates silent updates (the provider won't swap
    `gemini-3.5-flash` to a new revision without warning; a pinned snapshot
    like `claude-sonnet-4-5-20250929` either works or 404s). However, it
    leaves three residual risks:
    1.  **Retirement:** Pinned models get deprecated (e.g., Gemini 1.5
        returning 404s). Pinning converts "silent change" to "loud death."
    2.  **Infrastructure Drift:** Providers can change serving infrastructure,
        quantisation, or safety filters behind the same ID. Unfalsifiable from
        the outside.
    3.  **Non-determinism:** Even with temperature 0, commercial APIs are not
        perfectly deterministic (batching, hardware floating-point ordering).
*   *Report Argument:* This residual risk is an argument *for* the
    architecture, not a weakness. Pinning (operational) and Cross-Vendor
    Verification (architectural) are complementary mitigations. If the
    generator's vendor breaks something, the independent verifier still
    catches unfaithful output.

**Dependency Management (`uv` vs `pip`)**
*   `uv.lock` is the true, superior manifest. It pins every transitive
    dependency, backing the reproducibility claim.
*   A hand-maintained `requirements.txt` is prone to drift.
*   *Action for Week 14 (Submission Packaging):* Generate a `requirements.txt`
    strictly as a courtesy export for non-uv readers using:
    `uv export --format requirements-txt > requirements.txt`. Do not edit it
    by hand.

## 21.7.2026

**Node A generation failure**

API failure after retries → no draft exists → skip Node B and the
rewrite loop entirely → return the fallback message. Logged as
`generation_failed` (not `fallback`) and excluded from labelling, since
there is no answer to judge.

## 22.7.2026
Five-label draft retracted: "Wrongly Withheld Answer" required the labeller
to know the answer was blocked, breaking blinding. Fixed operationally —
labeller judges the withheld DRAFT, not the fallback message. Controller
must log withheld drafts.

## 25.7.2026
One md file for each FAQ. Both question and answer would be retrieved.
Questions are confirmed not situational or contain incorrect information or
contradict the answer. To keep it simple, the question will not be separated
from the answer as metadata or be withheld. The focus of the project is
about single-pass or dual-pass (with verifier) instead of complicated retrieval.
Ultimately, the synthetic FAQs are part of the experimental design.

**Retrieval Ingestion & Token Limits**
*   **Token Budget:** The pipeline correctly identified `all-MiniLM-L6-v2`'s
    strict 256-token limit. With 2 reserved special tokens, the effective content
    budget is 254 word-piece tokens. The safety guard successfully clamped the
    `CHUNK_SIZE` to 254 to prevent silent truncation.
*   **Chunk Mapping:** Verified that the "1 FAQ = 1 chunk" across all 24 entries.
    Each retrieved chunk is therefore exactly one complete FAQ, so top-k retrieval
    returns k distinct FAQs rather than fragments of one — retrieval depth k maps
    cleanly to "number of FAQs considered." (k itself is still to be set by the
    recall pilot.)

** Retrieval recall pilot test**
Recall@k on all 47 in-domain queries (no LLM, no cost): 100% at k=3/5/8/10
after correcting one target_faq typo (IN-38 → faq_23) that the check caught.
RETRIEVAL_K set to 5: recall is perfect even at k=3, but the internal-transfer
trio (faq_21/22/23, near-identical sequential steps) fills the top-3 with
competing chunks — k=3 has zero margin on that cluster. k=5 clears every case
with headroom at negligible context cost. Numbering verified coherent: no
target_faq mismatch across corpus / CORPUS_DESIGN / query set.

---

## OPEN ITEMS (running list — move up with a date when settled)

*   [ ] ARMS_CONFIG mechanism (`ACTIVE_ARM` selection + per-arm guards) — next
        structural commit. same_vendor's node_b_model must be a REFERENCE to
        NODE_A_MODEL, not a copied string.
*   [ ] Pilot: verifier tier (Luna vs Terra) — see 13.7.2026
*   [ ] Pilot: retrieval recall at k=5/8/10 on the real corpus.
*   [ ] Verify "1 FAQ = 1 chunk" at embedding time.
*   [ ] Ragas RunConfig(max_workers=8): test whether the 81 s/query is serial
        execution; a full sweep at current speed is 6–7 hours.
*   [ ] Per-query cost: measure real $/query on ~10 corpus queries (Anthropic
        dashboard before/after) — replaces the smoke-test-derived estimate.
*   [ ] Week 14: `uv export` requirements.txt; README final pass; regenerate
        PROJECT_STRUCTURE markers.
*   [ ] Verify Luna + reasoning_effort=low supports schema-enforced
        structured output (one-liner, messy multi-claim test answer) —
        FIRST Node B session, before controller integration.
*   [ ] Verify Gemini `response_schema` structured output for the
        same-vendor arm's Node B (one-liner).

