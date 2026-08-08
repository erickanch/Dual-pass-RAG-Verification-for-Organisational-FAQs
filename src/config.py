"""
Central configuration for the dual-pass RAG experiment.

PRINCIPLE: every parameter that could affect a result lives here.
No magic numbers anywhere else in the codebase.

When writing the methodology chapter, this file should be the only
place to look to describe the experimental setup.

TERMINOLOGY MAPPING (Code vs Document):
- "same_vendor" /"cross_vendor" arms in code = "Same-Vendor /Cross-Vendor dual-pass" in the proposal.
- "rewrite_attempts" in code = "rewrite loop" in the proposal.
"""

import os
import subprocess
from dotenv import load_dotenv
from datetime import datetime, timezone
from src.constants import ALPHA, CONFIDENCE_LEVEL, RANDOM_SEED  # noqa: F401

# Config is read-once. Other modules import from here, never read .env directly.
load_dotenv()

# ---------------------------------------------------------------------------
# API keys (never hardcode; loaded from .env, which is gitignored)
# ---------------------------------------------------------------------------

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not GOOGLE_API_KEY or not OPENAI_API_KEY or not ANTHROPIC_API_KEY:
    raise RuntimeError("Missing API key(s). Check your .env file.")


# ---------------------------------------------------------------------------
# Models
#
# Pin EXACT model IDs. Never use floating aliases (e.g. "gemini-flash-latest",
# or bare "gpt-5.6" which routes to gpt-5.6-sol). Aliases get hot-swapped by
# the provider, which would silently break comparability between runs.
#
# Assume any model can be retired mid-project.
# Record the ID and the run date for every experiment.
# ---------------------------------------------------------------------------

# Node A — generation
NODE_A_MODEL = "gemini-3.5-flash"
NODE_A_TEMPERATURE = 0.0          # deterministic: not testing creativity

# Node B — verification of atomic claims
NODE_B_MODEL = "gpt-5.6-luna"     # TODO: validate against gpt-5.6-terra in pilot
NODE_B_REASONING_EFFORT = "low"   # verification is structured classification,
                                  # not deep reasoning. Set explicitly — the
                                  # default may be higher and costs output tokens.

# NOTE: GPT-5.6 requires the Responses API (client.responses.create),
# NOT the legacy chat.completions endpoint. The latter returns a 401.

# Judge — Ragas automated faithfulness/relevance scoring
# THIRD vendor, independent of Node A (Google) and Node B (OpenAI), so the
# automated metric does not share blind spots with the systems under test.
# Used only to score returned answers offline; never in the live pipeline.
JUDGE_MODEL = "claude-sonnet-4-5-20250929"   # Claude Sonnet 4.5 pinned dated snapshot; ID from Anthropic
                                             # third vendor (Anthropic);
                                             # independent on Node A (Google) and Node B (OpenAI);
                                             # with Ragas 0.3.9 via LangchainLLMWrapper
JUDGE_TEMPERATURE = 0.0                      # scoring, not creativity — same rationale as Node A

# ---------------------------------------------------------------------------
# Methodology Enforcement: Cross-Vendor Independence
# ---------------------------------------------------------------------------
def _get_vendor(model_id: str) -> str:
    """Infers vendor from model ID string for methodology checks."""
    if not model_id: return "none"
    mid = model_id.lower()
    if "gemini" in mid: return "google"
    if "gpt" in mid: return "openai"
    if "claude" in mid: return "anthropic"
    return "unknown"

_vendor_a = _get_vendor(NODE_A_MODEL)
_vendor_b = _get_vendor(NODE_B_MODEL)
_vendor_judge = _get_vendor(JUDGE_MODEL)

# Ensure the vendor model is available
if "unknown" in (_vendor_a, _vendor_b, _vendor_judge):
    raise ValueError(
        f"Unrecognised model ID (A={_vendor_a}, B={_vendor_b}, judge={_vendor_judge}). "
        "Extend _get_vendor() before running."
    )

# Ensure the vendor for the judge is different from Nodes A and B
if _vendor_judge in (_vendor_a, _vendor_b):
    raise ValueError(
        f"Methodology violation: Judge vendor ({_vendor_judge}) must be strictly "
        f"independent from Node A ({_vendor_a}) and Node B ({_vendor_b})."
    )

# Prompts are parameters which affect results more than almost anything
# else here. Update this whenever ANY prompt text changes — otherwise two runs
# with different prompts look identical in the results files.
PROMPT_VERSION = "v1"

# ---------------------------------------------------------------------------
# Embeddings
#
# Local model, deliberately. An API-hosted embedder could be updated or retired
# mid-project, which would make indexes built at different times incomparable.
# A frozen local model removes that risk entirely.
#
# Embeddings are held constant across all three experimental arms, so retrieval
# quality is not a confound in the baseline vs dual-node comparison.
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
                                  # HF commit SHA — pinned for the same reason
                                  # model IDs are pinned: the bare name resolves
                                  # against the Hub on every load, so upstream
                                  # could silently change the weights that built
                                  # the index. This is the revision already in
                                  # the local cache, so the existing embeddings
                                  # remain valid.
EMBEDDING_DIM = 384               # verified in smoke test


# ---------------------------------------------------------------------------
# Retrieval
#
# Starting values are drawn from common practice, not from theory. They are
# to be validated in a pilot and adjusted if retrieval recall is poor.
# Whatever you settle on, report BOTH the final values and the fact that they
# were tuned — and keep the pilot results.
# ---------------------------------------------------------------------------

CHUNK_SIZE = 256                  # WORD-PIECE-TOKENS (not characters) capped by all-MiniLM-L6-v2's max_seq_length (verified 256). 
                                  # Long chunks are silently truncated by the embedder — no error, no warning. 
CHUNK_OVERLAP = 32                # ~12.5% — enough to avoid splitting a claim
                                  # across a boundary
RETRIEVAL_K = 5                   # chunks retrieved per query

CHROMA_PATH = "./chroma_db"
CHROMA_COLLECTION = "msc_corpus"
SIMILARITY_METRIC = "cosine"      # Distance metric explicitly set for ChromaDB

# ---------------------------------------------------------------------------
# Experimental configurations - THREE arms as per the proposal (Appendix C)
#
#   baseline      : retrieve -> generate. No verification.
#   same_vendor   : Node B = the IDENTICAL model as Node A (proposal:
#                   same vendor, family, version and tier — NOT merely a
#                   different model from the same vendor). Tests information
#                   asymmetry alone (RQ3). Ragas-scored only.
#   cross_vendor  : Node B from a different vendor. Primary configuration
#                   (RQ1/RQ2). Ragas + full human labelling.
#
# Everything else is held identical across all three arms.
# same corpus, same index, same retrieval, same Node A model,
# same temperature. 
# Note: As Node A may equal Node B for same_vendor, the vendor guard only 
# checks the JUDGE against the nodes.
# ---------------------------------------------------------------------------

ARM_BASELINE = "baseline"
ARM_SAME_VENDOR = "same_vendor"
ARM_CROSS_VENDOR = "cross_vendor"
VALID_ARMS = (ARM_BASELINE, ARM_SAME_VENDOR, ARM_CROSS_VENDOR)

# --- Arm selection ---------------------------------------------------------
#
# ACTIVE_ARM chooses which configuration is live for a run. Set it in .env
# (ACTIVE_ARM=cross_vendor) so a run's arm is recorded outside the code, or
# leave it to default. run_experiment.py reads the DERIVED values below —
# never NODE_B_MODEL directly — so the same runner serves all three arms.

ACTIVE_ARM = os.getenv("ACTIVE_ARM", ARM_CROSS_VENDOR)

if ACTIVE_ARM not in VALID_ARMS:
    raise ValueError(
        f"ACTIVE_ARM={ACTIVE_ARM!r} is not a valid arm. "
        f"Expected one of {VALID_ARMS}."
    )

# Resolve Node B per arm. Downstream reads NODE_B_ENABLED and
# ACTIVE_NODE_B_MODEL — the single point where "which arm" becomes "which
# Node B". same_vendor assigns by REFERENCE to NODE_A_MODEL (identity
# guaranteed by construction, never a copied string).
if ACTIVE_ARM == ARM_BASELINE:
    NODE_B_ENABLED = False
    ACTIVE_NODE_B_MODEL = None
elif ACTIVE_ARM == ARM_SAME_VENDOR:
    NODE_B_ENABLED = True
    ACTIVE_NODE_B_MODEL = NODE_A_MODEL          # reference, not a copy
else:  # ARM_CROSS_VENDOR
    NODE_B_ENABLED = True
    ACTIVE_NODE_B_MODEL = NODE_B_MODEL

# cross_vendor guard — load-bearing: NODE_B_MODEL is independent and could
# share Node A's vendor if misconfigured. (same_vendor needs none — identity
# is guaranteed by the reference above. baseline has no Node B. The
# judge-independence guard above still applies in every arm.)
if ACTIVE_ARM == ARM_CROSS_VENDOR:
    if _get_vendor(NODE_A_MODEL) == _get_vendor(ACTIVE_NODE_B_MODEL):
        raise ValueError(
            f"cross_vendor arm requires DISTINCT vendors, but Node A "
            f"({NODE_A_MODEL}) and Node B ({ACTIVE_NODE_B_MODEL}) are both "
            f"{_get_vendor(NODE_A_MODEL)}."
        )


# The rewrite loop: if Node B rejects one or more atomic claims, Node A is
# asked to regenerate. This cap prevents unbounded loops (and unbounded cost).
MAX_REWRITE_ATTEMPTS = 3          # Specified in the proposal. Up to three corrective passes.
                                  # If temporarily lowered for cheap debugging, RESTORE
                                  # TO 3 before any run whose results you will report.


# ---------------------------------------------------------------------------
# Reliability
#
# The Gemini free tier is rate-limited (RPM and RPD). A long experimental run
# WILL hit 429s. Build backoff in from the start rather than discovering this
# when a 200-question run dies at question 140.
# ---------------------------------------------------------------------------

MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0            # seconds; exponential backoff
REQUEST_TIMEOUT = 60              # seconds


# ---------------------------------------------------------------------------
# Per-query log contract (populated by the controller at runtime — NOT config)
#
# Fields the controller must write for every query:
#   query_id, arm, answer_shown, attempts (REWRITES consumed, not API calls),
#   outcome — one of src.constants.VALID_OUTCOMES (five; verifier_malformed
#             and generation_failed are infrastructure failures, excluded
#             from the confusion matrix and reported as counts),
#   drafts  — the full chain: [{stage, text, sha256, refused}, ...] with
#             stage in {initial, rewrite_1, ...}. Human labelling judges
#             DRAFTS, not the displayed output, and every distinct draft is
#             labelled. The hash lets labelling.py group identical texts
#             across arms: the initial draft is produced by the same Node A,
#             prompt and passages in every arm, so where hashes match it is
#             read once and the label applied to all matching records.
#             Grouping is an analysis decision and lives in labelling.py —
#             at runtime only one arm exists.
#   verifications — [{stage, claims, n_claims, n_flagged, all_supported}, ...]
#   retrieved_chunk_ids (list)
#
#   TIMINGS, two kinds:
#     t_retrieval_s, t_generation_s, t_verification_s (list), t_rewrite_s
#       (list), t_total_s — the SUCCESSFUL call's duration. These are the
#       latency percentiles reported in Phase 2: what a user experiences.
#     t_generation_wall_s, t_verification_wall_s, t_rewrite_wall_s — elapsed
#       time INCLUDING failed attempts and backoff sleeps. These answer how
#       long a sweep took. They diverge only when retries fire, but a 429
#       plus a 6-second backoff would otherwise be reported as 6 seconds of
#       generation latency and distort the p95.
#     generation_attempts, verification_attempts (lists) — API attempts per
#       call, so wall/latency divergence is attributable to retries.
#
#   TOKENS, split by node: tokens_a_prompt, tokens_a_completion,
#     tokens_b_prompt, tokens_b_completion (plus derived totals). Node A and
#     Node B may be different vendors with different prices, so a single
#     total cannot be converted to a cost. Rates are applied in ANALYSIS,
#     never written here: a results file with a hardcoded price becomes wrong
#     when prices change, rather than merely stale.
#
# Timer: time.perf_counter() — monotonic, per the proposal's latency protocol.
# Bump this version if fields are ever added/changed mid-project, so every
# results file declares which schema it follows.
# ---------------------------------------------------------------------------
LOG_SCHEMA_VERSION = "v1"

# Statistics constants (ALPHA, CONFIDENCE_LEVEL, RANDOM_SEED) are imported
# from src.constants at the top of this file — see the import there.


# ---------------------------------------------------------------------------
# Paths
#
# Results are NEVER overwritten. Every run is timestamped and kept.
# ---------------------------------------------------------------------------

CORPUS_DIR = "./data/corpus"
QUESTIONS_FILE = "./data/questions.json"
RESULTS_DIR = "./experiments/results"


# ---------------------------------------------------------------------------
# Run metadata — call this and write it into every results file.
#
# This is what answers "which models, which settings, which day?"
# months later when checking a number in your write-up.
# ---------------------------------------------------------------------------

def run_metadata() -> dict:
    """Snapshot of every parameter affecting a run. Store with the results."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True
        ).stdout.strip(),
        "prompt_version": PROMPT_VERSION,
        "log_schema_version": LOG_SCHEMA_VERSION,
        "active_arm": ACTIVE_ARM,
        "node_b_enabled": NODE_B_ENABLED,
        "active_node_b_model": ACTIVE_NODE_B_MODEL,
        "node_a_model": NODE_A_MODEL,
        "node_a_temperature": NODE_A_TEMPERATURE,
        "node_b_model": NODE_B_MODEL,
        "node_b_reasoning_effort": NODE_B_REASONING_EFFORT,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_revision": EMBEDDING_REVISION,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "retrieval_k": RETRIEVAL_K,
        "max_rewrite_attempts": MAX_REWRITE_ATTEMPTS,
        "random_seed": RANDOM_SEED,
    }

def scoring_metadata(scored_results_file: str) -> dict:
    """Judge-run record — written by the Ragas scoring script AT SCORING TIME,
    into its own output file. Separate from run_metadata() because scoring
    happens offline, possibly days after generation, possibly re-run with a
    different judge. Links back to the results file it scored."""
    # ragas imported locally so config.py stays importable without it
    # (the live pipeline must not depend on the offline scoring library).
    import ragas
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scored_results_file": scored_results_file,
        "judge_model": JUDGE_MODEL,
        "judge_temperature": JUDGE_TEMPERATURE,
        "ragas_version": ragas.__version__,
        "embedding_model": EMBEDDING_MODEL,
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"],
                                     capture_output=True, text=True).stdout.strip(),
    }