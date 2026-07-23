# msc-rag — Dual-Pass RAG Verification

MSc project prototype: a RAG chatbot whose answers are verified
claim-by-claim by an independent second model before display, with a
deterministic controller owning the rewrite loop and fallback.

Three experimental arms: **baseline** (single-pass) / **same_vendor**
(identical model self-checks under information asymmetry) / **cross_vendor**
(different-vendor verifier — primary configuration).

- Design rationale and decisions: see `NOTES.md`
- File map and build order: see `PROJECT_STRUCTURE.md`

## Setup

Requirements: Python 3.12 (pinned), [`uv`](https://docs.astral.sh/uv/).

```bash
git clone <repo-url>
cd msc-rag
uv sync
```

Create `.env` in the project root (never committed):

```
GOOGLE_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

Verify the environment:

```bash
uv run pytest                            # unit tests — free, fast, no API keys
uv run python tests/smoke_test.py        # full-stack import/connectivity check
uv run python tests/test_ragas.py        # three-vendor Ragas end-to-end proof
```

Both should finish without errors (smoke test ends with "ALL SYSTEMS GO").

## Running

<!-- Fill each section AS THE PIECE IS BUILT. Do not document unbuilt code. -->

- **Ingestion / embedding** — TODO (added when `src/retrieval.py` is built)
- **Experiments** — TODO (added when `experiments/run_experiment.py` is built;
  arm selected via `ACTIVE_ARM`)
- **Ragas scoring** — TODO (added when `analysis/ragas_eval.py` is built)
- **Web app** — TODO (built last)

## Reproducibility

- Exact dependency graph pinned in `uv.lock` (Ragas 0.3.9,
  langchain-community <0.4 — see NOTES.md for why).
- All experimental parameters live in `src/config.py` — single source of
  truth, no magic numbers elsewhere.
- Every experimental run records `run_metadata()` (models, parameters, git
  commit, timestamp); every Ragas scoring pass records `scoring_metadata()`.
- Results are timestamped and never overwritten.
- `requirements.txt` (pip-installable export) is generated at packaging time
  via `uv export --format requirements-txt` — `uv.lock` remains canonical.
- Metrics and statistical tests are unit-tested against hand-computed
  expected values (`tests/test_metrics.py`), independent of any pipeline data.