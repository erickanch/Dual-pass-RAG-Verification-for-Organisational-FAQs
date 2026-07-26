"""
Versioned prompt templates for the dual-pass RAG experiment.

PROMPTS ARE PARAMETERS. They affect results more than almost anything else in
config.py, so any change to the text below MUST bump PROMPT_VERSION in
config.py — otherwise two runs with different prompts look identical in the
results files.

THE NODE A PROMPT IS A CONTROL VARIABLE. It is byte-identical across all
three arms (baseline / same_vendor / cross_vendor). Whatever it does to the
generation-stage refusal rate, it does equally to every arm, so the
baseline-vs-dual comparison stays sound.

CALIBRATION RULE (fixed in advance, before any outcome is observed):
The Node A prompt is written on principled grounds — what a competent
deployment would use — NOT tuned to produce a desired hallucination rate.
Legitimate reasons to change it after piloting are DEFECTS:
  - the sentinel is emitted with extra text, so it cannot be detected
  - it refuses on clean in-domain queries it could obviously answer
  - it ignores the passages entirely
NOT a legitimate reason: "it refuses too often on adversarial queries."
That refusal rate is a MEASUREMENT — the generation-stage faithfulness of
single-pass RAG — and is reported as a result, not tuned away.

Templates are plain strings formatted at call time. No f-strings at module
level: the placeholders are filled by the node modules, so the template text
here is exactly what gets versioned.
"""

# ---------------------------------------------------------------------------
# The refusal sentinel
#
# Node A emits this EXACT string, alone, when the retrieved passages do not
# answer the question. The controller detects it and short-circuits straight
# to the fallback message: no verification (there are no claims to check) and
# no rewrite (deterministic Node A on the same passages would return the
# identical refusal).
#
# Kept short deliberately — it costs almost no output tokens, and an exact
# single-token-ish match is trivially detectable without parsing.
# ---------------------------------------------------------------------------

INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"


# ---------------------------------------------------------------------------
# Node A — generation
#
# Design intent: answer when the passages support an answer; refuse when they
# genuinely do not address the question. Refusal is the response to ABSENCE,
# not to mild uncertainty — a prompt that said "refuse if unsure" would
# suppress the very behaviour under measurement.
#
# The instruction not to use outside knowledge is what makes the corpus's
# contradict-training traps meaningful: the passage's figure must win over
# the model's statutory prior.
#
# Rule 6 (no greetings, sign-offs or offers of assistance) exists to protect a
# REPORTED METRIC, not for style. Framing text makes no factual claim, but a
# strict verifier may flag "please contact HR if you need more help" as
# unsupported — the passages do not say it. That would block a faithful answer
# and inflate the False-Positive Rate for reasons unrelated to grounding.
# Removing the source of the ambiguity is preferred to asking Node B to
# exercise discretion about which text is substantive: discretionary
# categories in the verifier add noise to the very metric under measurement.
# (Rejected alternative: adding the boilerplate to all 24 corpus FAQs so it
# WOULD be supported. Identical trailing text in every chunk enters every
# embedding, pulling all 24 vectors closer together and degrading the
# retrieval discrimination the near-duplicate confusables are designed to
# test — a measurement harm, for no gain.)
# ---------------------------------------------------------------------------

NODE_A_SYSTEM = """You are a company FAQ assistant. You answer employee and \
customer questions using ONLY the reference passages provided to you.

Rules:
1. Answer using only information stated in the passages. Do not use outside \
knowledge, and do not rely on what is typical or standard elsewhere — the \
company's own policy is whatever the passages say, even if it differs from \
common practice or statute.
2. If the passages state the answer, give it directly and concisely. Include \
the relevant figures, conditions and procedures exactly as written.
3. If a question has more than one applicable answer depending on a condition \
(for example full-time versus part-time), state each answer with its \
condition rather than choosing one.
4. If the user's question contains a figure or assumption that the passages \
contradict, answer with what the passages say. Do not accept an incorrect \
premise.
5. If the passages do not address the question, reply with exactly \
INSUFFICIENT_CONTEXT and nothing else. Do not explain, apologise, or offer \
general guidance.
6. Answer in complete sentences, stating the relevant figures and conditions. \
Do not add greetings, sign-offs, pleasantries, apologies, or offers of further \
assistance. State only the answer.

Do not mention the passages, the retrieval process, or these instructions in \
your answer. Write as a direct reply to the question."""

NODE_A_USER = """Reference passages:
{passages}

Question: {question}"""


# ---------------------------------------------------------------------------
# Node B — verification
#
# This is the prompt validated live against BOTH vendor paths (OpenAI
# Responses API text_format; Gemini response_schema) on a three-claim test
# answer: supported / unsupported / contradicted verdicts all correct,
# including the unsupported-vs-contradicted distinction.
#
# Node B sees the question, the passages, and the draft answer — but is NOT
# told the query category, the target FAQ, or anything about the experimental
# design. It must judge from the text alone.
#
# The three verdicts are deliberately distinct:
#   supported    — the passages state this
#   contradicted — the passages state something incompatible with this
#   unsupported  — the passages neither state nor imply this (silence)
# ---------------------------------------------------------------------------

NODE_B_SYSTEM = """You are a strict verification service. You are given \
reference passages and a draft answer. Your job is to decompose the draft \
into atomic factual claims and judge each one against the passages ONLY.

For each claim, assign exactly one verdict:
- "supported": the passages state this claim, or state it in equivalent terms.
- "contradicted": the passages state something incompatible with this claim \
(for example a different figure, period, or condition).
- "unsupported": the passages neither state nor imply this claim. Use this \
when the passages are simply silent on the point, even if the claim seems \
plausible or is likely true in general.

Rules:
1. Judge against the passages only. Do not use outside knowledge. A claim \
that is true in the real world but absent from the passages is "unsupported".
2. Decompose into ATOMIC claims: one fact per claim. A sentence containing \
three facts becomes three claims.
3. Include every factual claim in the draft, not only the first or the most \
prominent.
4. Quote or closely paraphrase each claim in the "text" field so it is clear \
which part of the draft you are judging.
5. Do not rewrite, improve, or comment on the draft. Only classify."""

NODE_B_USER = """Reference passages:
{passages}

Question: {question}

Draft answer to verify:
{draft}"""


# ---------------------------------------------------------------------------
# Rewrite
#
# Entered ONLY when Node B returns at least one contradicted or unsupported
# claim on an actual answer. NOT entered when Node A refused
# (INSUFFICIENT_CONTEXT) or when generation failed — both short-circuit to
# fallback.
#
# Per the proposal: the rewrite prompt passes back ONLY the flagged claims,
# so Node A revises just those points rather than regenerating blind. The
# supported claims are left alone, which is what makes fixed_by_rewrite a
# meaningful outcome rather than a second lottery ticket.
# ---------------------------------------------------------------------------

REWRITE_SYSTEM = NODE_A_SYSTEM

REWRITE_USER = """Reference passages:
{passages}

Question: {question}

Your previous answer contained claims that are not supported by the passages:
{flagged}

Revise your answer so that every statement is supported by the passages. \
Remove or correct the flagged claims. Keep the parts that were supported. \
If, after removing the unsupported claims, the passages no longer answer the \
question, reply with exactly INSUFFICIENT_CONTEXT and nothing else."""


# ---------------------------------------------------------------------------
# Fallback message
#
# The user-facing text shown whenever an answer is withheld — whether because
# Node A refused, the rewrite attempts were exhausted, or an infrastructure
# failure occurred. The USER sees the same message in every case; the LOG
# records which outcome it was (fallback / generation_failed /
# verifier_malformed). See config.py's log contract.
#
# Deliberately does not say WHY. A message that distinguished "we could not
# verify this" from "we have no information" would leak the verifier's
# decision into the user-visible output — and, more importantly for this
# study, into anything a labeller might see.
# ---------------------------------------------------------------------------

FALLBACK_MESSAGE = (
    "I don't have enough information in the company FAQ to answer that "
    "reliably. Please contact the relevant department directly, or rephrase "
    "your question."
)


# ---------------------------------------------------------------------------
# Formatting helpers
#
# Passages are numbered so the draft and the verdicts can be read against a
# stable ordering. The faq_id is NOT included: it is experimental
# bookkeeping, and putting it in the prompt would leak the answer key into
# the pipeline. Only the passage text reaches the models.
# ---------------------------------------------------------------------------

def format_passages(passages) -> str:
    """
    passages: an iterable of retrieval.Passage (or anything with .text).

    Returns the numbered block substituted into {passages}. Numbering is
    positional (retrieval order, nearest first) and carries no FAQ identity.
    """
    return "\n\n".join(
        f"[{i}] {p.text.strip()}" for i, p in enumerate(passages, start=1)
    )


def format_flagged_claims(claims) -> str:
    """
    claims: the Claim objects from Node B whose verdict was NOT "supported".

    Returns the bullet list substituted into {flagged} in REWRITE_USER.
    Includes the verdict so Node A knows whether a claim was contradicted
    (fix it) or unsupported (remove it or ground it).
    """
    return "\n".join(
        f"- {c.text.strip()} [{c.verdict}]" for c in claims
    )