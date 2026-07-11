"""
Central configuration for the dual-pass RAG experiment.

PRINCIPLE: every parameter that could affect a result lives here.
No magic numbers anywhere else in the codebase.

When writing the methodology chapter, this file should be the only
place to look to describe the experimental setup.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# API keys (never hardcode; loaded from .env, which is gitignored)
# ---------------------------------------------------------------------------

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not GOOGLE_API_KEY or not OPENAI_API_KEY:
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


# ---------------------------------------------------------------------------
# Embeddings
#
# Local model, deliberately. An API-hosted embedder could be updated or retired
# mid-project, which would make indexes built at different times incomparable.
# A frozen local model removes that risk entirely.
#
# Embeddings are held constant across BOTH experimental arms, so retrieval
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

CHUNK_SIZE = 512                  # tokens per chunk
CHUNK_OVERLAP = 64                # ~12.5% — enough to avoid splitting a claim
                                  # across a boundary
RETRIEVAL_K = 5                   # chunks retrieved per query

CHROMA_PATH = "./chroma_db"
CHROMA_COLLECTION = "msc_corpus"


# ---------------------------------------------------------------------------
# Experimental configurations
#
# Exactly two arms. Everything except the verification pass is held identical
# between them — same corpus, same index, same retrieval, same Node A model,
# same temperature. The ONLY difference is whether Node B runs.
# ---------------------------------------------------------------------------

CONFIG_BASELINE = "baseline"      # retrieve -> generate. No verification.
CONFIG_DUAL_NODE = "dual_node"    # retrieve -> generate -> verify -> (rewrite)

# The rewrite loop: if Node B rejects one or more atomic claims, Node A is
# asked to regenerate. This cap prevents unbounded loops (and unbounded cost).
MAX_REWRITE_ATTEMPTS = 1          # TODO: justify in methodology. 1 = single
                                  # corrective pass. Raising this changes both
                                  # cost and the interpretation of your metrics.


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
CONFIDENCE_LEVEL = 0.95           # for Wilson intervals on SCR / FNR / FPR

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