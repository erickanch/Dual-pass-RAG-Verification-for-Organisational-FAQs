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

# Judge — Ragas automated faithfulness/relevance scoring (Phase 6 only)
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

# Ensure 
if "unknown" in (_vendor_a, _vendor_b, _vendor_judge):
    raise ValueError(
        f"Unrecognised model ID (A={_vendor_a}, B={_vendor_b}, judge={_vendor_judge}). "
        "Extend _get_vendor() before running."
    )

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
# Statistics
# ---------------------------------------------------------------------------

ALPHA = 0.05                      # significance level
CONFIDENCE_LEVEL = 1 - ALPHA      # 0.95 for Wilson intervals on SCR / FNR / FPR

RANDOM_SEED = 42                  # for any sampling; record it


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
    from datetime import datetime, timezone

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True
        ).stdout.strip(),
        "prompt_version": PROMPT_VERSION,
        "node_a_model": NODE_A_MODEL,
        "node_a_temperature": NODE_A_TEMPERATURE,
        "node_b_model": NODE_B_MODEL,
        "node_b_reasoning_effort": NODE_B_REASONING_EFFORT,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "retrieval_k": RETRIEVAL_K,
        "max_rewrite_attempts": MAX_REWRITE_ATTEMPTS,
        "random_seed": RANDOM_SEED,
    }