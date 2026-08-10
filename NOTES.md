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

## 8.8.2026

[Contents: run_experiment.py build; baseline/cross_vendor/same_vendor
sweeps; IN-44 three-arm RQ3 exhibit; ADV-13 non-determinism; Ragas
answer_relevancy artefact + human-labelling argument]

**run_experiment.py built and first complete sweep run**

Parameterised runner: reads ACTIVE_ARM, writes one .jsonl per sweep (metadata
header + one record per query, append-only, flushed per write). Resume
retries only generation_failed/verifier_malformed outcomes. Genuine results
(passed_first, fixed_by_rewrite, fallback) are left untouched. A controller
exception not already converted to an outcome halts the sweep rather than
being logged as an infrastructure failure — that would hide a code defect
inside normal noise.

**First baseline sweep: 108/108, 0 errors, ~4.5 minutes**

    passed_first: 58   fallback: 50
    in_domain (47):     47 passed_first, 0 fallback
    adversarial (40):   11 passed_first, 29 fallback
    out_of_domain (21):  0 passed_first, 21 fallback

Zero generation failures, zero queries needed a second attempt. Matches the
throwaway-probe pattern from 26-27.7.2026 exactly, now as a logged,
metadata-stamped, reproducible run rather than an indicative script — this
is the first REPORTABLE measurement of the zero-hallucination finding.

The 11 adversarial passed_first queries are the identical set observed in
the frozen-prompt probes (ADV-03, 13, 14, 23, 24, 27, 31, 32, 36, 38, 39).
Spot-checked ADV-27 and ADV-31 in full: both faithful, both explicitly
decline the unsupported part, neither invents anything.

**ADV-27 — rule 2 confirmed working**

Earlier probe: "To apply for an internal department transfer, you must first
obtain written approval..." (recitation, no direct response to "can I
bypass?"). This run: "You cannot bypass your Line Manager. [supporting
detail] I do not have information on whether exceptions are made..." — leads
with a direct answer to the question asked, exactly the ordering rule 2 was
written to produce. Best evidence yet that the rule works. The earlier
sample simply hadn't shown it.

**Retrieval-timing artefact (record for Phase 2 analysis)**

First record of the run showed t_retrieval_s = 6.27s. Every other record was
0.01-0.05s. Not a retrieval bug — the embedding model loads on the first call
of the process ("Loading weights" warm-up), inflating only that one
measurement. EXCLUDE the first record per sweep from latency percentiles, or
warm the model with a throwaway call before the sweep's first real query.
Flag this in figures.py / the results write-up so it is not misread as a
genuine outlier.


**First complete cross_vendor sweep: 108/108, 0 errors**

    passed_first: 56   fixed_by_rewrite: 1   fallback: 51
    in_domain (47):     46 passed_first, 1 fixed_by_rewrite
    adversarial (40):   10 passed_first, 30 fallback
    out_of_domain (21):  0 passed_first, 21 fallback

57 queries reached Node B. 0 verifier_malformed. One node_a.py bug hit and
fixed mid-sweep (attempts=attempt typo in rewrite(), NameError on the first
real rewrite call — IN-25). Fixed, resumed via --resume from record 24: prior
records untouched, sweep completed cleanly from IN-25.

**IN-44 — the rewrite loop working end to end (report exhibit)**

Draft 1: "No, you cannot take marriage leave in your first year..." — Luna
correctly flagged this CONTRADICTED: the passage states a positive condition
(12 months' service), not a prohibition; asserting "cannot" overstates it.
Rewrite 1 over-corrected: dropped the direct answer, added "the passages do
not provide information on whether..." — Luna flagged THIS as contradicted
too (the question is answerable by entailment; claiming it isn't is itself
wrong). Rewrite 2 converged correctly: "No, you cannot... [because] employees
must have completed 12 months" — same conclusion as draft 1, now correctly
derived as entailment from the stated condition rather than asserted
outright. Three rounds, all-supported on the third. A clean, reproducible
demonstration of claim-level verification catching a subtle
contradiction-vs-entailment error and converging to a correct answer.

**ADV-31 — pre-test concern did not recur in the live pipeline**

Passed_first again (as in the earlier controller test). Checked why: Node
A's actual phrasing here never asserts "a secondary sign-off is required
above HKD 10,000" (the claim Luna flagged in the hand-written pre-test
draft) — it states only what a manager may approve up to, then declines the
rest. Luna correctly marked the declining clause supported (the passages
ARE silent, so saying so is accurate). Conclusion: ADV-31 was never a real
failure in live output — only in the synthetic pre-test draft, which was
worded more assertively than anything Node A actually produces. The pre-test
served its purpose (surfacing the mechanism) but should not be read as a
live-pipeline failure.

**ADV-13 — Node A non-determinism observed (limitation, not investigated further)**

Same query, IDENTICAL retrieved passages (confirmed: same 5 chunk IDs, same
order) — yet different Node A output between the baseline and cross_vendor
sweeps. Baseline: answered (SLA stated, penalty declined). Cross_vendor:
draft was INSUFFICIENT_CONTEXT before Node B was ever called — the
divergence is generation-stage, not verification-stage; Node B is not
responsible for this difference. Consistent with documented commercial API
non-determinism at temperature 0 (cf. 17.7.2026).

n=1 — not a basis for a rate estimate, and not investigated further with a
dedicated stability run. Noted as a limitation rather than generalised from
a single instance: an observed cross-arm outcome difference is not
automatically attributable to the verifier, and where this matters (e.g.
ADV-13) it should be checked against the underlying draft rather than
assumed.

**Structured output — confirmed at scale**

0 verifier_malformed across 57 real Node B calls (previously validated only
on 6 hand-written pre-test cases). Closes the structured-output risk beyond
the pre-test.

**First complete same_vendor sweep: 108/108, 0 errors**

    passed_first: 57   fallback: 51
    in_domain (47):     47 passed_first
    adversarial (40):   10 passed_first, 30 fallback
    out_of_domain (21):  0 passed_first, 21 fallback

57 queries reached Node B; 0 verifier_malformed (Gemini response_schema path
now confirmed at scale, not just the 6-case pre-test). ACTIVE_NODE_B_MODEL
correctly resolved to gemini-3.5-flash via the reference mechanism (metadata
confirms node_b_model=gpt-5.6-luna, active_node_b_model=gemini-3.5-flash —
static config vs. resolved value both visible, as intended).

Across all three arms, only TWO queries produced different outcomes:
ADV-13 (Node A non-determinism, see below) and IN-44.

**IN-44 — a controlled three-arm comparison; direct RQ3 evidence**

Node A produced the IDENTICAL initial draft in all three arms (same hash) —
the only variable across outcomes is therefore the verifier, which is
exactly the isolation the design exists to achieve:

    "No, you cannot take paid marriage leave in your first year of
    employment. Employees must have completed 12 months of continuous
    service..."

*   **baseline** (no verifier): passed_first — draft shown as-is.
*   **cross_vendor** (Luna verifies): FLAGGED "you cannot... in your first
    year" as contradicted. The passage states a SUFFICIENT condition (12
    months' service -> entitled), not explicitly a necessary one; read
    strictly, "you cannot" asserts more than the passage states. Triggered
    a rewrite; the rewrite over-corrected and Luna flagged that too;
    converged to an entailment-based answer on round 3 (fixed_by_rewrite).
*   **same_vendor** (Gemini verifies Gemini): the IDENTICAL draft, containing
    the IDENTICAL claim, marked 0 flagged, all_supported=True.

**Not a clean "one verdict right, one wrong" case — the finding is the
DISAGREEMENT, not a ruling on which vendor was correct.** Both readings are
defensible: Luna's strict reading (a sufficient condition does not licence
an unqualified "cannot") versus Gemini's colloquial reading (the condition
functions as the answer for any practical reader, absent evidence of
exceptions) — which is the same entailment move rule 2 explicitly licenses
Node A to make. There is no ground-truth fact settling which threshold is
right; the ambiguity is real, and the labelling guide does not yet have a
rule for a claim entailed from a stated SUFFICIENT (not necessary) condition.
Needs one before this class of case is labelled.

The safe, evidence-supported claim: with the confound of differing drafts
eliminated by construction, the two verifiers applied DIFFERENT STRICTNESS
THRESHOLDS to the identical claim, and same_vendor's threshold was looser
here than cross_vendor's. Whether looser is worse depends on the FPR/FNR
trade-off the evaluation protocol is designed to quantify, not on an
argument about which reading is correct. This is itself informative for
RQ3 regardless: it is direct evidence that vendor identity affects
verification behaviour beyond information asymmetry alone (same_vendor DOES
have a fresh, asymmetric call and still applied a looser threshold) — a
production deployment would need to choose a threshold, and this shows
vendors do not converge on one by default.

This is the shared-blind-spot mechanism RQ3 hypothesises, observed on a real
query with drafts held constant by construction — not a synthetic pre-test
and not inferred from a single vendor's behaviour. Gemini-as-verifier,
checking a claim shaped by Gemini-as-generator's own reasoning pattern,
applied no additional scrutiny where a different vendor's verifier did.
Same_vendor DOES have information asymmetry (a fresh call, no shared
context) and still showed no scrutiny — evidence that vendor difference
adds something beyond asymmetry alone, which is precisely RQ3's question.

Caveat for Discussion: whether the passed-through claim is meaningfully
harmful is arguable — "you cannot... in your first year" is a reasonable
colloquial gloss on "12 months' service required," even if not strictly
entailed. The finding is not that same_vendor produced something false; it
is that same_vendor's verification added measurably less scrutiny than
cross_vendor's to the identical claim. Strong candidate as a worked example
in Results/Discussion — one of very few cases where n=1 is still compelling,
because the confound (different drafts) is eliminated by construction.

**IN-44 — supplementary record detail**

Retrieved passages confirmed identical by ID across both arms:
faq_09/10/03/05/02 (target_faq=faq_09 at rank 1) — retrieval isolation is
verified from the log, not assumed.

Cost of the divergence: cross_vendor spent 23.43s / 5,720 tokens (3 rounds)
against same_vendor's 6.49s / 1,578 tokens (1 round) — a 3.6x cost for the
extra scrutiny on this query. Concrete RQ2 exhibit alongside the RQ3 finding.

Claim decomposition differs between vendors even on AGREED claims (e.g.
"entitled to marriage leave" vs "entitled to PAID marriage leave") —
consistent with the granularity inconsistency noted 27.7.2026, now
confirmed at full-sweep scale, not just the 6-case pre-test.


**ADV-13 — non-determinism note refined**

cross_vendor and same_vendor produced IDENTICAL Node A drafts (same hash,
INSUFFICIENT_CONTEXT), both differing from baseline's answered draft. So the
divergence occurred once — between the baseline run and the two later runs —
not on every run. Slightly more precise than the earlier note ("Node A
varied between runs"): consistent with an occasional deviation rather than
persistent instability on this specific query. Still n=1; still not
investigated further; still a limitation, not a rate claim.

**ADV-03 — latency outlier, not a defect**

t_generation_s = 26.20s (generation_attempts=[1], no retry) — a single slow
Gemini response, not a bug or a rate-limit retry. Flag as an outlier when
computing latency percentiles rather than investigating further.

**Ragas answer_relevancy on partial answers — a scoring artefact, and a
separate real question about value delivered**

All 12 answers containing a declining clause ("I do not have information
on X") scored answer_relevancy EXACTLY 0.000 in the baseline arm — not
merely low, uniformly zero — while 11 of the 12 scored faithfulness 1.000.
The 46 clean answers averaged relevancy 0.7993; the pooled mean (0.6339)
is a straight arithmetic consequence of the 12 zeros and is misleading if
reported alone. Full breakdown belongs in Results as two subgroup means,
not one pooled figure.

Mechanical cause: Ragas generates hypothetical questions FROM the answer
text and compares them to the real question by embedding similarity. An
answer combining a factual statement with an explicit statement of what is
NOT covered appears to produce hypothetical questions that average out to
near-zero similarity — a known brittleness on compound-content answers,
not a judgement that the underlying answer is poor. These are NOT
refusals: refusals (the fallback message) are already excluded from
scoring entirely (confirmed: 50/50 fallbacks excluded, 0 reached Ragas).
These are substantive, partially-informative answers that also decline a
sub-part — rule 6's engineered behaviour.

**This is the clearest evidence yet for why human labelling cannot be
replaced by the automated score alone.** Ragas relevancy, taken in
isolation, would wrongly flag 12 genuinely faithful answers as failures.
A human labeller applying the actual labelling-guide criteria would
correctly assess these as faithful (per the worst-element rule) and could
separately judge relevance on its merits — not zero out mechanically.

**A separate, harder question this raises — not a Ragas artefact, a real
one: does a partial answer actually deliver USABLE information, or does
the pipeline run to completion and produce nothing the enquirer can act
on?** "Faithful" only means every stated claim is grounded — it says
nothing about whether the answer contains anything responsive to what was
asked. IN-44 is the extreme case: three full rounds (retrieval once,
generation x3, verification x3) to arrive at a faithful answer — but for a
DIFFERENT declining-clause query, if the entitlement fact stated is not
what the enquirer actually needed to know, the dual-pass machinery may
have run its full, most expensive path to deliver a non-answer dressed
with an unrelated fact. This is a genuine cost-effectiveness question
distinct from faithfulness and distinct from the Ragas scoring bug above:
even a CORRECTLY human-labelled-faithful partial answer may deliver zero
practical value, at full pipeline cost. Worth checking specifically during
human labelling — not just "is this faithful" but "does this answer give
the enquirer anything they can use" — and worth a labelling-guide note or
a separate tracked flag if the guide is revised before labelling starts.

Candidates for this check, from the 12 declining-clause answers: ADV-31
(states the manager's approval limit, which the enquirer did not ask
about, then declines who signs off above it, which IS what was asked —
arguably close to zero usable information delivered despite being fully
faithful).

Pattern replicated in cross_vendor: 12 declining-clause answers, all
0.000 relevancy (identical to baseline); 45 clean answers averaged 0.8122.
Confirms this is a systematic Ragas mechanism, not a baseline-specific
artefact — strengthens the human-labelling argument above.

**IN-38 — a second Ragas artefact, this time on FAITHFULNESS**

"The interview for an internal transfer is conducted by the receiving
Department Head and an HR Business Partner." scored faithfulness=0.000 in
ALL THREE ARMS. This is near-verbatim faq_19 — the same content approved
when the corpus was written. Trivially, obviously faithful; a score of
exactly 0.000 across three independent scoring runs on a textbook-correct
answer is not a real faithfulness failure.

Likely mechanism: the answer is a single short sentence, largely lifted
from the passage. Ragas's faithfulness metric decomposes the answer into
atomic statements before checking each against the context; a decomposition
failure on a terse, single-clause answer (extracting zero or malformed
statements) may default to a 0.000 score rather than signalling "no claims
extracted". NOT currently verifiable from the saved output — the
statement-level breakdown Ragas computes internally is not saved by
ragas_eval.py, only the final score. Would need re-running with Ragas's
verbose/trace output to confirm the exact mechanism.

Consequence: min_faithfulness in every arm's summary table (all three
report min_faith=0.000) is understated for the SAME reason min relevancy
was — an isolated scoring artefact, not a genuine pipeline failure. Do not
report min/max faithfulness without inspecting the flagged record first;
this affects the faithfulness metric as well as relevancy, not just the
one already documented.

**Adversarial category: answer_relevancy is a clean, complete 0.0000 across
ALL passed_first adversarial answers, all three arms**

    cross_vendor  IN-*  (n=47): mean relevancy 0.7777
    cross_vendor  ADV-* (n=10): mean relevancy 0.0000  <- every single one

Every adversarial partial-answer scores relevancy exactly zero, with no
exceptions, in every arm. This is a stronger and cleaner pattern than the
"declining clause" framing used above: by design, every adversarial
passed_first answer targets a designed gap and therefore contains a
declining clause, so the artefact is total and complete for this entire
query category. Worth its own Results table (relevancy by category: in_domain
vs adversarial) rather than the ad-hoc "answers containing the phrase 'I do
not have information'" framing — category is a design variable already
reported elsewhere in this project; the phrase-match filter was only a
diagnostic convenience.

Reinforces the finding above: automated answer-relevancy scoring is
UNUSABLE for the adversarial category as currently computed. Any claim
about relevancy differing between arms on adversarial queries would be
measuring a Ragas artefact, not a real difference. Human labelling (and,
if adopted, genuine per-query judgement) is the only sound source of
relevance assessment for this category.

**IN-44 — Ragas relevancy scores add a small corroborating point**

    baseline (untouched draft):        relevancy 0.951
    cross_vendor (after 2 rewrites):   relevancy 1.000
    same_vendor (untouched draft):     relevancy 0.951

The rewritten answer scored marginally HIGHER relevancy than the original
phrasing used by both other arms — Ragas independently judged the corrected
answer to be at least as relevant to the question as the version it
replaced, not merely more faithful at a usability cost. Small effect (0.951
-> 1.000) but a genuine point in cross_vendor's favour for the IN-44 exhibit:
the extra scrutiny did not trade relevance for faithfulness here; it
improved (or at minimum did not harm) both.

**OPEN DECISION — tier comparisons (Luna/Terra AND same_vendor's Gemini
tier): keep or drop?**

If a tier comparison is required for cross_vendor (Luna vs Terra), the same
standard applies symmetrically to same_vendor's verifier tier — no
principled basis to hold one arm to it and not the other. Both or neither.
Adequacy evidence already exists for BOTH verifiers from full-sweep
performance (0 verifier_malformed in both arms; correct handling of IN-44,
ADV-31). Leaning toward: NEITHER comparison required — adequacy is
established by observed live performance, not a dedicated study. Revisit
with a clear head; if kept, both comparisons must be done together for
consistency, roughly doubling that scope.

**DECIDED: same_vendor included in human labelling**

Marginal cost computed by hash: adding same_vendor to baseline+cross_vendor
labelling costs only 5 additional unique labels (103/108 same_vendor
records share an initial-draft hash already covered elsewhere; 95.4%
overlap). Proposal's stated reason for exclusion ("would roughly triple
the task") does not hold at this measured cost. Benefit: same_vendor
currently has no confusion matrix / FNR / FPR, only Ragas scores — which
carry confirmed artefacts on a meaningful subset (declining-clause and
short-answer cases, see above). Human labels give a genuine three-arm RQ3
comparison on primary safety metrics, directly strengthening the IN-44
finding across the full dataset rather than one query. Total labelling
scope: 68 unique labels (was 63 for two arms), not the ~200+ the proposal
anticipated for three independently-labelled arms.

**Total human labelling requirement: 68 unique labels**

Computed by SHA-256 deduplication across all three arms (baseline,
cross_vendor, same_vendor), all 108 queries, including rewrite-chain
drafts as separate units where text genuinely differs. Down from ~200+
anticipated for two arms without deduplication in the original proposal
scope. At an estimated 2-3 minutes per careful judgement: roughly 2.5-3.5
hours of labelling time total, not 7-12+.

**Second-reviewer sample size — needs revisiting given the 68-label total**

Proposal specifies 15-20% for the reliability sample, sized against an
anticipated ~200-label total. At the actual 68, 15-20% is only 10-14 items
— too small for a stable Cohen's kappa. Consider a minimum absolute floor
(e.g., max(20%, 20 items)) rather than the percentage alone. Decide before
labelling begins; the sample must be drawn once, by the seeded shuffle, not
adjusted after seeing results.

## 9.8.2026

**Systematic audit of the two Ragas artefacts — both questions settled**

Ran a systematic pass over all scored records in all three arms (58/57/57),
rather than the curiosity-driven spot-check of 8.8.2026. Two questions were
open: (1) is the zero-relevancy artefact arm-blind, and therefore harmless to
the between-arm comparison? (2) is IN-38's zero faithfulness isolated, or the
visible tip of a class of failures on short answers?

**(1) Zero-relevancy IS arm-blind — comparison is safe**

    baseline      12 zero-relevancy / 58 scored
    cross_vendor  12 / 57
    same_vendor   11 / 57

TEN query IDs are identical across all three arms (ADV-03, 14, 23, 24, 27,
31, 32, 36, 38, 39, plus IN-42). The small count differences are fully
explained by cases already understood:
*   ADV-13 appears in baseline only — the known Node A non-determinism case
    (28.7.2026): baseline answered, the other two arms refused, so there was
    no answer to score in those arms.
*   IN-25 appears in cross_vendor only — a rewrite case where the revised
    answer contained a declining clause the other arms' answers did not.

The artefact therefore hits the same queries, for the same reason, in every
arm. This satisfies the arm-blindness condition identified on 8.8.2026 as
necessary for the comparison to remain valid. CONCLUSION: absolute relevancy
figures are not reportable as estimates of answer quality; RELATIVE
comparison between arms remains valid, because the measurement bias applies
to the same queries across all conditions.

**(2) IN-38 is ISOLATED, not a class — and the short-answer hypothesis is REFUTED**

Faithfulness by answer length, all three arms:

    <=15 words   n=13/11/12   mean faithfulness 1.000  (zeros: none)
    16-30 words  n=30/31/30   mean 0.925/0.944/0.942   (zeros: IN-38 only)
    31-50 words  n=12/13/12   mean 0.944/0.949/0.972   (zeros: none)
    >50 words    n= 3/ 2/ 3   mean 1.000               (zeros: none)

The SHORTEST answers (<=15 words) score a perfect 1.000 in every arm. IN-38
is 18 words, sitting in a bucket where everything else scores well. There is
no length gradient whatsoever. This REFUTES the hypothesis proposed on
8.8.2026 that terse single-clause answers break Ragas's statement
decomposition — a plausible mechanism, now shown by the data not to hold.

Total faithfulness zeros across all three arms: exactly ONE query (IN-38),
three times, reproducibly. Nothing else in 172 scored records resembles it.
The rest of the distribution is healthy: only 4-6 records per arm score below
1.0, and those are graded values (0.5, 0.667, 0.75) consistent with genuine
partial-faithfulness judgements rather than artefacts.

IN-38's answer ("The interview for an internal transfer is conducted by the
receiving Department Head and an HR Business Partner.") is verbatim-correct
against faq_19. The 0.000 is a scoring anomaly specific to this one
answer/context pairing; the underlying mechanism remains unverified and is
NOT explained by answer length.

**DECISION: report, do not remediate**

Both artefacts are reported as measurement limitations, not fixed:
*   No re-run, no judge-model change, no input reformatting. The evidence
    does not support a systemic problem that remediation would address, and
    a judge-strength change would not test the suspected mechanism anyway
    (8.8.2026).
*   IN-38 is LEFT IN the dataset and flagged as a known-erroneous zero rather
    than excluded. Excluding data points, even with justification, invites
    more scrutiny than a clearly-documented anomaly.
*   Absolute faithfulness/relevancy means are reported with the caveat above;
    between-arm comparisons are reported as valid.

Cost of this audit: ~30 minutes against data already held. Converted two open
worries into settled, evidenced facts — and refuted one of my own working
hypotheses in the process, which is the more valuable half of the result.

**Labelling blindness — limitation and partial mitigations**

The primary labeller (the researcher) is not truly blind: these answers have
been read and discussed during pipeline development and analysis. Three
mitigations applied: (1) a deliberate delay of several days between analysis
and labelling, to allow recall to decay; (2) seeded shuffle with the arm
identifier stripped, so that even where an answer is recognised, its arm is
not; (3) opaque row IDs, with the mapping written to a separate file not
opened during labelling. Residual limitation acknowledged: recall decay is
uneven — memorable cases (IN-44, ADV-31, IN-38) will likely remain
recognisable, and these are disproportionately the cases where judgement is
contested. The second reviewer's sample is therefore the genuinely blind
reliability check, and is weighted accordingly in interpreting kappa.

Labelling scope: 67 distinct draft texts covering 174 records across three
arms (107 labels saved by deduplication). Second-reviewer sample: 20 units
(30%), applying an absolute floor of 20 rather than the proposal's 15-20%
proportion, which at this deduplicated total would yield only 10-13 items —
too few for a stable Cohen's kappa. Documented deviation.

---

## OPEN ITEMS (running list — move up with a date when settled)

### Settled — decisions recorded, no further action

*   [x] ARMS_CONFIG mechanism (`ACTIVE_ARM` + per-arm guards). same_vendor's
        node_b_model is a REFERENCE to NODE_A_MODEL, not a copied string.
*   [x] Retrieval recall at k=3/5/8/10 on the real corpus — 100% at every k;
        RETRIEVAL_K=5 chosen for margin on the transfer cluster (26.7.2026).
*   [x] "1 FAQ = 1 chunk" verified at embedding time — holds for all 24.
*   [x] Luna + reasoning_effort=low supports schema-enforced structured output.
*   [x] Gemini `response_schema` structured output for the same_vendor arm.
        Both now confirmed AT SCALE: 0 verifier_malformed across 57+ live
        calls per arm (28.7.2026).
*   [x] run_experiment.py incremental writes + --resume — built, and proven
        under fire when the node_a rewrite bug halted the cross_vendor sweep
        at IN-25 (28.7.2026).
*   [x] Ragas artefact audit — systematic pass done 9.8.2026. Zero-relevancy
        is ARM-BLIND (same 10 query IDs in all three arms), so the between-arm
        comparison is safe. IN-38's zero faithfulness is ISOLATED (1 record,
        3 arms, reproducible) and the short-answer hypothesis is REFUTED
        (<=15-word answers score 1.000). DECISION: report as limitations, do
        NOT remediate — no re-run, no judge change, IN-38 left in and flagged.
*   [x] Same-vendor human labelling — INCLUDED. Marginal cost measured at 5
        additional labels (103/108 same_vendor drafts already covered by
        baseline/cross_vendor hashes). The proposal's cost-based reason for
        excluding it does not hold at this measured cost (8.8.2026).
*   [x] Second-reviewer sample size — 20% with an absolute floor of 20 units.
        At the deduplicated total (67), the proposal's 15-20% would yield only
        10-13 items, too few for a stable Cohen's kappa. Documented deviation;
        the fraction is unchanged, the floor is what binds.
*   [x] Ragas telemetry hang — RAGAS_DO_NOT_TRACK set in ragas_eval.py before
        import. Was 6+ min/row (~11h projected); now 35s for 116 evaluations.
        DNS failure on Ragas's usage-tracking host, not a judge/API problem.

### Open — needs a decision

*   [ ] **Tier comparisons (Luna vs Terra AND same_vendor's Gemini tier):
        keep or drop?** Both or neither — no principled basis for holding one
        arm to a comparative-adequacy standard and not the other. Adequacy
        evidence already exists for both verifiers from full-sweep
        performance. LEANING DROP; if kept, both must be done, roughly
        doubling that scope. Decide before the report's Future Work section.
*   [ ] **Compound queries: adopt or leave as probes?** UNBLOCKED — both
        vendors accepted declining clauses as supported in the pre-test, so
        no False-Positive risk. Adoption would need a second-reviewer
        plausibility check, as the 108 had.

### Open — work still to do

*   [ ] **HUMAN LABELLING — the critical path.** 67 units built and blinded;
        20-unit reviewer sample drawn. Deliberately delayed several days to
        allow recall decay (see blindness limitation, 9.8.2026).
*   [ ] **Send the reviewer their CSV + Labelling Guide v3.** Their clock runs
        independently of the delay; no reason to hold this.
*   [ ] Cohen's kappa — not yet implemented in labelling.py; pending returned
        reviewer labels.
*   [ ] Judge-validation check: Ragas faithfulness vs human labels on the
        Baseline arm (NOTES 12.7.2026). Pending labels.
*   [ ] figures.py — pending labels.
*   [ ] Per-query cost: measure real $/query. Token counts are logged per node
        (tokens_a_*, tokens_b_*); rates applied in analysis, not stored.
*   [ ] node_a.py: distinguish 429 (quota — short backoff) from 503 (capacity
        — longer, more patient backoff). Current ~45s total is too short for
        a demand spike (hit 26.7.2026). Not triggered in any completed sweep.
*   [ ] ragas_eval.py: no guard against corpus drift when reconstructing
        contexts from data/corpus/ — safe only because the corpus is closed.
*   [ ] Week 14: `uv export` requirements.txt; README final pass; regenerate
        PROJECT_STRUCTURE markers.

### Open — dropped or superseded

*   [~] VERIFIER NON-DETERMINISM (test-retest, ~20 queries x3). The original
        motivation (ADV-31 flagged `contradicted` in the pre-test, `supported`
        in the controller run) was RESOLVED on inspection: Node A's live
        phrasing never asserted the claim Luna flagged in the hand-written
        pre-test draft, so this was not verifier inconsistency (28.7.2026).
        A separate Node A non-determinism instance (ADV-13) was observed and
        quantified from existing sweep data (101/108 baseline-vs-cross_vendor
        drafts identical) — recorded as a limitation, n=1, not investigated
        further. No dedicated re-test run planned.

### Write-up cluster (baseline framing) — one task, five parts

*   [ ] Naive check: consumer Gemini + ONE concatenated doc + NO instruction.
        Only this configuration is what the problem statement describes.
        Illustrative only; not a comparison. Do at write-up, not before.
*   [ ] Distinguish the four conditions (naive one-doc / naive 24-file /
        NotebookLM purpose-built RAG / AI Studio developer GUI). 2-3
        sentences: all are quick to set up, but only the first is the
        deployment the research addresses — not a claim about which performs
        worst.
*   [ ] Why the pipeline and not those tools: the dual-pass design needs a
        DEFINED passage set for both nodes; long-context provides none, so the
        architecture is unimplementable there. Plus comparability across arms,
        failure attribution, reproducibility.
*   [ ] Anticipated objection: "you chose the weakest configuration to beat."
        Answer: the naive configuration is the deployment the RQ is ABOUT, not
        a competitor; and it is NOT the baseline arm — the baseline is
        single-pass through the same pipeline with the full specification,
        i.e. the DEMANDING comparison.
*   [ ] Limitation: the baseline is well-configured, not naive; corpus is
        atomic single-topic files, not a monolithic handbook. If the naive
        check grounds well, SOFTEN the premise rather than defend it.

### Write-up — other

*   [ ] Recommendations: present the eight-rule grounding specification with
        per-rule rationale. Scope honestly — validated on THIS corpus and
        model only; transfer untested. NOT a template for NotebookLM/AI
        Studio, neither of which was evaluated.
*   [ ] Future Work: verifier selection (vendor, tier, hosted vs open-weights)
        evaluated jointly on verification quality, cost and latency. Motivated
        by IN-44's strictness divergence and the measured cost overhead;
        held fixed here to isolate the verification step itself.
*   [ ] Limitations: Node A generation non-determinism (ADV-13, 28.7.2026).
        Cross-reference from any cross-arm outcome comparison.
*   [ ] Limitations: labelling blindness — primary labeller not truly blind;
        mitigations and residual limitation (9.8.2026).
*   [ ] Methodology: two-file results arrangement (pipeline record + scoring
        record, joined by query_id) — state explicitly so it is not inferred.