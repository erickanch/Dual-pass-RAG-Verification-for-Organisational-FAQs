# msc-rag — Dual-Pass RAG Verification

MSc project prototype: a RAG chatbot whose answers are verified
claim-by-claim by an independent second model before display, with a
deterministic controller owning the rewrite loop and fallback.

Three experimental arms: **baseline** (single-pass) / **same_vendor**
(identical model self-checks under information asymmetry) / **cross_vendor**
(different-vendor verifier — primary configuration). Only the verifier 
assignment varies between arms; every other setting is held constant.

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
ACTIVE_ARM=cross_vendor          # baseline | same_vendor | cross_vendor
```

Verify the environment:

```bash
uv run pytest                            # unit tests — free, fast, no API keys
uv run python tests/smoke_test.py        # full-stack import/connectivity check
uv run python tests/test_ragas.py        # three-vendor Ragas end-to-end proof
```
The smoke test ends with "ALL SYSTEMS GO". pytest does not make live API calls.
smoke_test.py and test_ragas.py make live API calls with costs.

## Running the pipeline

The stages run in order. Each writes to `experiments/results/`, timestamped,
and nothing is ever overwritten.

### 1. Ingest the corpus

```bash
uv run python -m src.retrieval --ingest
```

Chunks the 24 FAQ files in `data/corpus/`, embeds them with
all-MiniLM-L6-v2, and writes the ChromaDB index. Logs chunks-per-FAQ so the
"1 FAQ = 1 chunk" assumption can be checked. The index stores a hash of the
chunking and embedding settings; a later run with different settings will
refuse to start rather than query a stale index.

Re-run only if the corpus or the retrieval settings change.

### 2. Sweep an arm

```bash
uv run python -m experiments.run_experiment                       # uses ACTIVE_ARM
ACTIVE_ARM=baseline uv run python -m experiments.run_experiment
ACTIVE_ARM=same_vendor uv run python -m experiments.run_experiment
```

Runs all 108 queries through the pipeline and writes
`{arm}_{timestamp}.jsonl` — a metadata header plus one record per query,
flushed as it goes. About five minutes per sweep under normal API latency.

If a sweep halts, `--resume` retries only the infrastructure failures
(`generation_failed`, `verifier_malformed`) and preserves genuine results:

```bash
uv run python -m experiments.run_experiment --resume experiments/results/cross_vendor_20260808_040232.jsonl
```

### 3. Score with Ragas

```bash
uv run python -m analysis.ragas_eval experiments/results/cross_vendor_20260808_040232.jsonl
```

Writes `ragas_{arm}_{timestamp}.jsonl` — the exact question, answer and
passages scored, with faithfulness and answer relevance. Joined to the
pipeline record by `query_id`. The judge is a third vendor, distinct from
both nodes; the configuration refuses to start otherwise.

Scores only answers shown to a user. Withheld answers make no factual claim
and are counted separately as a refusal rate.

### 4. Build the labelling sheets

```bash
uv run python -m analysis.labelling build     # blinded sheet + mapping
uv run python -m analysis.labelling sample    # second-reviewer subset
```

`build` deduplicates by SHA-256 of the answer text, so each distinct text is
labelled once. Writes `labelling_sheet.csv` — question, answer, passages, and
an empty `label` column — with the unit-to-arm mapping in a **separate** file
that must not be opened while labelling.

Fill the `label` column with one of: `faithful`, `intrinsic`, `extrinsic`,
`correct_refusal`. Then:

```bash
uv run python -m analysis.labelling read      # validates and expands labels
```

Unknown labels are rejected rather than silently accepted.

### 5. Analyse

```bash
uv run python -m analysis.analyse
```

Joins labels to outcomes, builds the confusion matrix per arm, and computes
SCR, FNR and FPR with Wilson intervals, plus the paired baseline-versus-
cross-vendor comparison. Reports the rewrite chains separately, since a
hallucination caught inside the loop does not appear in the matrix.

### 6. Web interface

```bash
uv run uvicorn web.main:app --reload
```

Then `http://127.0.0.1:8000`. A thin layer over `controller.answer_query()` —
it performs no routing or verification of its own, so the demonstration shows
the code path the evaluation used. The details panel exposes the claim
verdicts, retrieved passages, outcome and token counts.

The arm is fixed by `ACTIVE_ARM` before the server starts and is not
selectable in the interface, matching the evaluation setup.


## Reproducibility

- Exact dependency graph pinned in `uv.lock` (Ragas 0.3.9,
  langchain-community <0.4 — see NOTES.md for why).
- All experimental parameters live in `src/config.py` — single source of
  truth, no magic numbers elsewhere.
- Model identifiers are pinned to the most specific form each provider
  offers. Only the judge has a dated snapshot available; the generator and
  verifier are named by their published identifiers.
- The embedding model is pinned to a specific HuggingFace revision, so the
  weights cannot change between runs.
- Every experimental run records `run_metadata()` (models, parameters, git
  commit, timestamp); every Ragas scoring pass records `scoring_metadata()`
  and names the results file it scored.
- Results are timestamped and never overwritten.
- `requirements.txt` (pip-installable export) is generated at packaging time
  via `uv export --format requirements-txt` — `uv.lock` remains canonical.
- Metrics and statistical tests are unit-tested against hand-computed
  expected values (`tests/test_metrics.py`), independent of any pipeline data.

## Reproducing the reported results

The three sweeps behind the reported figures are committed in
`experiments/results/`, with their Ragas scoring files and the completed
labelling sheet. `analysis/analyse.py` regenerates every number in the
results section from them without re-running the pipeline.

Each arm was also run twice more to check output stability; those runs are
committed and reported separately.