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

**Retrieval recall pilot test**
Recall@k on all 47 in-domain queries (no LLM, no cost): 100% at k=3/5/8/10
after correcting one target_faq typo (IN-38 → faq_23) that the check caught.
RETRIEVAL_K set to 5: recall is perfect even at k=3, but the internal-transfer
trio (faq_21/22/23, near-identical sequential steps) fills the top-3 with
competing chunks — k=3 has zero margin on that cluster. k=5 clears every case
with headroom at negligible context cost. Numbering verified coherent: no
target_faq mismatch across corpus / CORPUS_DESIGN / query set.

**Structured output — both vendors confirmed**
OpenAI recovered from the outage (primaryapi_server_error was platform-wide,
confirmed via models.list also 500ing). Re-ran the Luna probe: gpt-5.6-luna
at reasoning_effort=low returns clean schema-conforming JSON on the 3-claim
test, verdicts all correct (supported/unsupported/contradicted including the
unsupported-vs-contradicted distinction) — identical to Gemini. Both Node B
vendor paths (OpenAI Responses API text_format; Gemini response_schema)
validated with the same Pydantic-schema pattern. Format-malformation risk
retired. Verifier-tier ladder's rung 1 confirmed. Tier quality comparison
(Luna vs Terra on FNR/FPR) remains for the pilot.

## 26.7.2026
Embedding model pinned to HF revision 1110a243... The config claimed a
"frozen local model", but SentenceTransformer("all-MiniLM-L6-v2") resolves
the name against the Hub on every load (visible as the unauthenticated-
request warning) — cached, not pinned, so upstream changes could have
silently altered the weights behind the index. The fingerprint hashed only
the model NAME, so the staleness guard was blind to it. Now pinned to the
revision already cached (index unaffected), added to the fingerprint and to
run_metadata. Same discipline as dated model snapshots, applied to the
component that determines the entire index.

Prompt rule 5 split to handle compound questions during initial Node A
probing: v1's single rule was silent on PARTIAL coverage, so Node A refused
outright when asked "how many days of annual leave do I get, and can I carry
unused days over?" — declining an answerable sub-question because a second
clause was uncovered. A defect (refusing clean in-domain content it could
obviously answer), not an outcome to tune. After the split, all six compound
probes answered the supported part and declined the gap without inventing
anything. PROMPT_VERSION left at v1 as no results had been recorded — the
prompt was still being drafted, not revised. v1 is the first prompt to
produce reported results.

**Generation-stage faithfulness: zero hallucinations observed**

Probes against the real corpus and index, Node A only, via throwaway scripts
(not run_experiment.py, no run_metadata recorded). Indicative, not reportable
as experimental results — to be re-measured in the pilot and the full run.

| Probe | n | Result |
|---|---|---|
| Adversarial (designed gaps) | 40 | 39 refused via sentinel; ADV-27 answered an adjacent supported fact. 0 extrinsic hallucinations. |
| Compound (supported + gap) | 6 | All answered the supported part, declined the gap. 0 inventions. |
| Embedded-wrong-figure | IN-35 | Contradicted the user's false "7 days" premise, answered 2 days. Trap resisted. |
| Out-of-domain | OOD-05 | Clean sentinel. |

If this holds in the full run, TP and FN are ~0: FNR undefined (metrics.py
returns NaN), McNemar has no discordant pairs, and the arm comparison measures
only FPR. Limitation to state (cf. 12.7.2026): with zero hallucinations it
cannot be distinguished whether the generator is genuinely faithful or the
24-FAQ corpus was too small and too clean to elicit hallucination.

**Consequence: Node B needs separate instrument validation**

If the arms yield no hallucinations, Node B's capability cannot be
demonstrated by the arm comparison. Validate it separately on hand-written
drafts containing planted contradicted and unsupported claims against real
passages. Reportable as verifier capability ("identified N of M planted
claims"), clearly labelled as instrument validation, not an experimental arm.

**ADV-27: a third behaviour — faithful but non-responsive**

Asked "Can I bypass my Line Manager for a transfer if we don't get along?",
Node A answered with the transfer procedure. Every claim is in faq_21, so
under the worst-element rule the response is FAITHFUL — but it answers an
adjacent question rather than the one asked. Neither refusal nor
hallucination.

Labelling rule fixed in advance: all claims supported => faithful.
Non-responsiveness is a relevance failure, not a faithfulness failure. This is
the case where the two Ragas metrics diverge (high faithfulness, lower answer
relevance) — an argument for reporting both rather than faithfulness alone.

**Node A as baseline**

The prompt was developed to make Node A a responsive, grounded assistant — not
to approximate a naive commercial deployment. The strict early variant was
faithful but unresponsive (refused answerable sub-questions; answered adjacent
facts rather than the question asked). Rule 2 corrected that within an explicit
no-invention boundary.

The baseline arm therefore represents a well-configured single-pass RAG system.
This is the DEMANDING comparison, not a fallback: a verifier that adds nothing
over competent prompting has no deployment case, so beating a naive baseline
would prove little. An upload-and-chat deployment configured in minutes has no
comparable specification and, being subject to undisclosed model and platform
updates (cf. 17.7.2026), is not a stable baseline to measure against.

**Bounded inference is where Node B earns its place**

Entailment — applying a stated requirement to the question asked — is a
judgement at the margin, and a generator making it in one pass MAY occasionally
over-read, stating a route or exception the passages do not support.
Verification is therefore not merely an error-catcher for a careless generator:
it is what permits Node A's helpfulness to be set higher than a single-pass
deployment could safely allow. The trade-off is measurable here — answer
relevance against FNR/FPR across arms.


## 27.7.2026

**Node B pre-test — 6 hand-written drafts, both vendor paths**

*   **Declining clauses are NOT flagged** by either vendor. "I do not have
    information on whether grandparents count as immediate family" was marked
    supported by both. Rule 6's engineered behaviour is therefore safe: the 11
    adversarial and 6 compound partial answers will not become False Positives.
    This was the critical unknown before the sweeps; it is now closed.

*   **Courtesy framing — the vendors disagree.** Luna ignored a greeting and
    "Please contact HR if you need further assistance" entirely, extracting
    only substantive claims. Gemini flagged the HR sign-off as unsupported.
    Consequence: had pleasantries been allowed, cross_vendor and same_vendor
    would differ in FPR because of how each vendor treats framing language,
    not because of the treatment under test — a direct confound in the RQ3
    comparison. **Rule 8 (no pleasantries) retained on this evidence**, which
    replaces the earlier speculative justification.

*   **ADV-31 — a real over-read produced by the pipeline itself.** The draft
    claimed "a secondary sign-off is required for purchase orders above HKD
    10,000"; the passages state only what a manager may approve *without* one,
    and are silent above that threshold. Luna flagged it (as contradicted);
    Gemini marked it supported — a false negative on an ungrounded claim, by
    the same model that generated it. Whether the correct verdict is
    contradicted or unsupported is arguable; both block, so either counts as
    the verifier succeeding. Consistent with the shared-blind-spot mechanism
    RQ3 tests. n=1 — to be characterised properly in the pilot.

*   **Decomposition granularity is inconsistent** within and across vendors:
    the same compassionate-leave sentence became 1 claim in one case and 3 in
    another. Verdicts were unaffected, but claim counts are not comparable
    across queries and should not be reported as a metric.

**Limitation — corpus structure favours the baseline**

The corpus is 24 atomic single-topic FAQ files, which yields clean chunking
(one FAQ per chunk, verified at ingestion). A real deployment more typically
ingests a monolithic staff handbook, where chunk boundaries cut across topics
and retrieval is correspondingly noisier. The corpus design therefore
minimises retrieval error by construction — as intended, so that generation
and verification effects are isolated — but the baseline's faithfulness
partly reflects that favourable structure.

**Anticipated objection — "you chose the weakest configuration to beat"**

The naive configuration is the deployment the research question is *about*,
not a competitor selected for weakness. It is also not the baseline arm: the
baseline is single-pass through the same pipeline with the full grounding
specification — the demanding comparison, since a verifier that adds nothing
over competent prompting has no deployment case. NotebookLM (purpose-built RAG
with forced citation) and AI Studio (developer GUI taking one's own system
prompt) answer different questions and are not naive deployments.

**Draft deduplication across arms — methodology, not just efficiency**

All three arms share Node A, the prompt, and retrieval, so the initial draft
is identical across arms wherever no rewrite fires. The controller therefore
hashes every draft (SHA-256) rather than assuming identity: labelling.py will
group records by hash and present each DISTINCT draft to the labeller once,
applying that label to every matching record.

This is not primarily an efficiency saving. Presenting the same text three
times in a blinded set would not produce three independent judgements anyway
— the labeller may recognise it and anchor — so deduplication makes explicit
what was already true. It also protects CONSISTENCY: the same text labelled on
separate occasions could plausibly receive different labels on a borderline
case, and that inconsistency would propagate into the confusion matrix as an
apparent arm difference. Labelling each distinct draft once removes that
source of noise by construction. Identity is verified by hash, never assumed:
commercial APIs are not perfectly deterministic even at temperature 0, so
drafts that differ in any respect are labelled separately. The divergence
rate (how often arms produced non-identical drafts from identical inputs) is
itself worth reporting as a small observation on API determinism.

Same-vendor human labelling is decided AFTER the runs, from the actual
distinct-draft count post hash-grouping — not committed in the proposal
(which excludes it on cost grounds), but potentially near-free under
deduplication if most initial drafts are shared. If drafts diverge more than
expected, the proposal's Ragas-only treatment is kept.

---

## OPEN ITEMS (running list — move up with a date when settled)

*   [x] ARMS_CONFIG mechanism (`ACTIVE_ARM` selection + per-arm guards) — next
        structural commit. same_vendor's node_b_model must be a REFERENCE to
        NODE_A_MODEL, not a copied string.
*   [ ] Pilot: verifier tier (Luna vs Terra) — see 13.7.2026
*   [x] Pilot: retrieval recall at k=5/8/10 on the real corpus.
*   [x] Verify "1 FAQ = 1 chunk" at embedding time.
*   [ ] Ragas RunConfig(max_workers=8): test whether the 81 s/query is serial
        execution; a full sweep at current speed is 6–7 hours.
*   [ ] Per-query cost: measure real $/query on ~10 corpus queries (Anthropic
        dashboard before/after) — replaces the smoke-test-derived estimate.
*   [ ] Week 14: `uv export` requirements.txt; README final pass; regenerate
        PROJECT_STRUCTURE markers.
*   [x] Verify Luna + reasoning_effort=low supports schema-enforced
        structured output (one-liner, messy multi-claim test answer) —
        FIRST Node B session, before controller integration.
*   [x] Verify Gemini `response_schema` structured output for the
        same-vendor arm's Node B (one-liner).
*   [ ] Write-up: informal Gemini app check (corpus uploaded, no system prompt,
        5-8 queries across categories, fresh chat each) — to see whether the
        naive-deployment contrast is real before asserting it in Limitations.
        Illustrative only and not a comparison.
*   [ ] Write-up: naive-baseline sanity check = consumer Gemini chat + ONE
        concatenated document + NO instruction. This alone is the configuration
        the problem statement describes. NotebookLM (purpose-built RAG, forced
        citation) and AI Studio (developer GUI, own system prompt) are NOT naive
        and must not be conflated with it — distinguish them in Limitations to
        pre-empt the objection. If condition 1 grounds well, SOFTEN the premise
        rather than defending it.
*   [ ] Write-up (Recommendations): if single-pass proves sufficient, present the
        eight-rule grounding specification as a deployment recommendation with its
        per-rule rationale. Scope honestly — validated on THIS corpus and model
        only; transfer untested. NOT a template for NotebookLM/AI Studio, neither
        of which was evaluated.
*   [ ] Write-up cluster (baseline framing):
*   [ ] Naive check: consumer Gemini + ONE concatenated doc + NO instruction.
        Only this configuration is what the problem statement describes.
*   [ ] Distinguish the four conditions (naive one-doc / naive 24-file /
        NotebookLM purpose-built RAG / AI Studio developer GUI). 2-3 sentences:
        all are quick to set up, but only the first is the deployment the
        research addresses — not a claim about which performs worst.
*   [ ] Why the pipeline and not those tools: the dual-pass design needs a
        DEFINED passage set for both nodes; long-context provides none, so the
        architecture is unimplementable there. Plus comparability across arms,
        failure attribution, reproducibility.
*   [ ] Contribution: the eight-rule specification with per-rule rationale,
        scoped to this corpus and model.
*   [ ] Limitation: the baseline is well-configured, not naive; corpus is
        atomic single-topic files, not a monolithic handbook. If the naive
        check grounds well, SOFTEN the premise rather than defend it.
*   [ ] node_a.py: distinguish 429 (quota — short backoff) from 503 (Google
        capacity — longer, more patient backoff). Current 5 attempts from 2s
        gives ~45s total, too short for a demand spike; a spike during a sweep
        would convert good queries into generation_failed. Hit 26.7.2026.
*   [ ] run_experiment.py: incremental writes + RESUME — re-run only queries
        that came back generation_failed, not the whole sweep.
*   [ ] VERIFIER NON-DETERMINISM (test-retest). ADV-31 was flagged
        `contradicted` in the Node B pre-test and `supported` in the controller
        run — same model, same passages, near-identical prompt tokens
        (711 both times), opposite decision. Run ~20 queries x3 and count
        verdict stability before the sweeps. Affects reproducibility claims;
        also a legitimate finding (a verifier that disagrees with itself is a
        deployment consideration).
*   [ ] Compound queries: UNBLOCKED — both vendors accepted declining clauses
        as supported in the pre-test, so no False-Positive risk. Decide whether
        to adopt (would need second-reviewer plausibility check, as the 108 had).
*   [ ] Same-vendor human labelling: decide AFTER the runs, from actual
        distinct-draft counts post hash-grouping. Not committed in the proposal;
        cheap if most drafts are shared, drop it if they diverge.
