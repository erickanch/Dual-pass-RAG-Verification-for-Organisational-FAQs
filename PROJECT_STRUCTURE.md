# Project Structure — msc-rag

The finalised design:
- THREE arms (baseline / same_vendor / cross_vendor), selected via `ACTIVE_ARM`,
  not two hard-coded run scripts.
- The three-vendor Ragas evaluation subsystem (Claude judge) has its own home.
- Node B's model is arm-dependent, not fixed to one vendor.
- FOUR human labels (proposal), FIVE controller outcomes — different things,
  different owners. Shared vocabulary lives in src/constants.py.

Marker key: DONE = built | TODO = planned | MOVE = built but not yet in this location

```
msc-rag/
│
├── .env                      # 3 API keys (Google, OpenAI, Anthropic). GITIGNORED.
├── .gitignore
├── pyproject.toml            # declared dependencies + Python 3.12 pin + pytest discovery config
├── uv.lock                   # exact resolved dependency graph — COMMIT (reproducibility)
├── requirements.txt          # DONE pip-installable export for non-uv readers.
├── README.md                 # setup guide + how to run (see README)
├── NOTES.md                  # methodology journal: decisions + reasons, dated
├── PROJECT_STRUCTURE.md      # this file
│
├── src/                      # ── THE LIVE PIPELINE ──────────────────────────
│   ├── constants.py          # DONE SHARED VOCABULARY. Four human labels, five
│   │                         #    controller outcomes, ALPHA, and the
│   │                         #    correct-refusal mapping decision. No imports,
│   │                         #    NO SIDE EFFECTS — so analysis code and unit
│   │                         #    tests can import it without API keys.
│   │                         #    config.py re-exports from here.
│   ├── config.py             # DONE Single source of truth for parameters.
│   │                         #    Includes ARM SELECTION: ACTIVE_ARM (default set
│   │                         #    here; optional override at launch or in .env)
│   │                         #    resolves NODE_B_ENABLED + ACTIVE_NODE_B_MODEL
│   │                         #    per arm. same_vendor sets Node B = NODE_A_MODEL
│   │                         #    by reference (not a copied string); cross_vendor
│   │                         #    guard requires distinct vendors.
│   ├── prompts.py            # DONE VERSIONED PROMPTS. Node A gen (8 rules), Node B
│   │                         #    verify, rewrite prompt, fallback message. Tied to
│   │                         #    PROMPT_VERSION (v1). FROZEN 27.7.2026 after
│   │                         #    defect-driven revisions; see NOTES for the
│   │                         #    calibration rule governing further changes.
│   ├── retrieval.py          # DONE INGESTION + RETRIEVAL. Chunk corpus -> embed
│   │                         #    (all-MiniLM) -> ChromaDB. Query-time top-k.
│   │                         #    Stores a config-hash with the index; refuses to
│   │                         #    run on a stale index (chunk/embed settings changed).
│   │                         #    Logs chunks-per-FAQ at ingestion (verifies the
│   │                         #    "1 FAQ = 1 chunk" assumption).
│   ├── node_a.py             # DONE GENERATOR. Gemini. (question + passages) -> draft.
│   │                         #    API failure after retries -> generation_failed;
│   │                         #    skips Node B and the rewrite loop entirely.
│   ├── node_b.py             # DONE VERIFIER. MODEL PER ACTIVE ARM (not fixed to GPT).
│   │                         #    Decomposes draft into atomic claims; labels each
│   │                         #    supported / contradicted / unsupported.
│   │                         #    SCHEMA-ENFORCED structured output: TWO vendor paths
│   │                         #    (OpenAI Responses API; Gemini response_schema).
│   │                         #    Pydantic-validated + one retry -> verifier_malformed.
│   └── controller.py         # DONE DETERMINISTIC CONTROLLER. Owns routing, the rewrite
│                             #    loop (cap = MAX_REWRITE_ATTEMPTS), and the fallback.
│                             #    Control logic lives HERE, in ordinary code — never
│                             #    in the models. Writes the per-query log
│                             #    (LOG_SCHEMA_VERSION contract in config.py), including
│                             #    the FULL DRAFT CHAIN with SHA-256 per draft —
│                             #    labelling judges drafts, not the fallback message,
│                             #    and the hashes let labelling.py dedupe across arms.
│
├── web/                      # ── WEB APP (BUILT LAST) ───────────────────────
│   └── main.py               # DONE FastAPI app. Thin layer over controller.py.
│                             #    Separated from the pipeline so evaluation never
│                             #    depends on the web server.
│                             #    calls answer_query() and does no routing or
│                             #    verification of its own, so the demonstration
│                             #    shows the code path the evaluation used.
│                             #    Details panel exposes claim verdicts, retrieved
│                             #    passages, outcome and token counts. Arm fixed by
│                             #    ACTIVE_ARM before start, not selectable in the UI.
│
├── data/                     # ── INPUTS ─────────────────────────────────────
│   ├── corpus/               # DONE 24 synthetic FAQ source documents.
│   └── questions.json        # DONE 108 queries (47 in-domain / 40 adversarial /
│                             #    21 OOD). Each: {id, text, category
│                             #    (in_domain|out_of_domain|adversarial),
│                             #    target_faq, underspecified}.
│
├── experiments/              # ── RUNNING THE SYSTEM ─────────────────────────
│   ├── run_experiment.py     # DONE ONE parameterised runner. Reads ACTIVE_ARM from
│   │                         #    config, runs the full query set through the
│   │                         #    pipeline, writes timestamped .jsonl (metadata
│   │                         #    header + one record per query, flushed per write).
│   │                         #    --resume retries ONLY infrastructure failures.
│   ├── pilot.py              # NOT BUILT AND NOT PLANNED. Pilot validation done via
│   │                         #    scratch_*.py probes (28.7.2026: adversarial,
│   │                         #    compound) and the Node B pre-test (27.7.2026),
│   │                         #    rather than a dedicated script. 
│   │                         #    DROPPED: verifier adequacy is already evidenced by 
│   │                         #    full-sweep performance (0 verifier_malformed across
│   │                         #    57+ live calls per arm). 
│   │                         #    The tier comparison (Luna vs Terra, and symmetrically 
│   │                         #    the same_vendor tier) is reported as future work.
│   │                         #    Kept it to record the decision. 
│   └── results/              # DONE Raw outputs, timestamped, NEVER overwritten:
│                             #      {arm}_{ts}.jsonl        pipeline record
│                             #      ragas_{arm}_{ts}.jsonl  scoring record
│                             #      labelling_sheet.csv     blinded, 67 units
│                             #      labelling_sheet_reviewer.csv  20-unit sample
│                             #      labelling_mapping.json  unit_id -> arm/query
│                             #    The folder also holds each arm's two repeat runs
│                             #    and two abandoned baseline runs; analyse.py names
│                             #    which runs it reads.
│                             #    Pipeline and scoring records are SEPARATE files
│                             #    joined by query_id: generation and scoring are
│                             #    distinct events, and the pipeline record is never
│                             #    mutated after the fact.
│
├── analysis/                 # ── MEASURING THE RUNS (offline) ───────────────
│   ├── metrics.py            # DONE Confusion matrix from FOUR labels x block/pass;
│   │                         #    SCR, FNR, FPR each with 95% Wilson intervals;
│   │                         #    paired_safety_outcomes() feeds stats.py. Also the
│   │                         #    cost measures: recorded latency, tokens and the
│   │                         #    nearest-rank percentile.
│   ├── stats.py              # DONE McNemar (paired binary) + Wilcoxon signed-rank
│   │                         #    (paired Ragas faithfulness), both exact, applied
│   │                         #    to all three pairs of arms.
│   ├── ragas_eval.py         # DONE RAGAS SCORING. Claude judge + local embeddings.
│   │                         #    Faithfulness + answer relevance. Writes its OWN
│   │                         #    scoring_metadata() (judge, ragas version, which
│   │                         #    results file it scored). The 3rd-vendor subsystem.
│   │                         #    Scores ONLY answers shown to a user
│   │                         #    (passed_first / fixed_by_rewrite); refusals are
│   │                         #    excluded and counted as a refusal rate.
│   │                         #    Sets RAGAS_DO_NOT_TRACK before import — the
│   │                         #    telemetry call otherwise hangs on DNS timeout
│   │                         #    (NOTES 8.8.2026). Judge-validation check vs human
│   │                         #    labels across all three arms is done.
│   ├── labelling.py          # DONE HITL SUPPORT. `build` writes a blinded CSV
│   │                         #    (query + answer + passages only; arm, outcome and
│   │                         #    verifier verdict withheld), seeded-shuffled, with
│   │                         #    the unit_id -> arm mapping in a SEPARATE file.
│   │                         #    Deduplicates by draft SHA-256: 67 distinct texts
│   │                         #    cover 174 records (107 labels saved). `sample`
│   │                         #    draws the 2nd-reviewer subset (20% or 20 minimum);
│   │                         #    `read` validates and expands labels back to all
│   │                         #    matching records. Cohen's kappa NOT implemented:
│   │                         #    with both raters assigning a single category it is  
│   │                         #    undefined, and percentage agreement is reported
│   │                         #    instead.
│   ├── analyse.py            # DONE FINAL ANALYSIS. Joins labels to outcomes via
│   │                         #    labelling_mapping.json, builds the confusion
│   │                         #    matrix per arm, computes SCR/FNR/FPR with Wilson
│   │                         #    intervals and the paired baseline-vs-cross_vendor
│   │                         #    comparison. The matrix uses the FINAL labelled
│   │                         #    draft per query; intermediate rewrite drafts are
│   │                         #    reported separately.
│   └── figures.py            # NOT BUILT. The findings are few and tabular; the
│                             #    tables in Results carry them. No figure would
│                             #    add to what the tables show.
│
├── tests/                    # ── TESTS AND DIAGNOSTICS ──────────────────────
│   ├── test_metrics.py       # DONE UNIT TESTS. Synthetic data, hand-verified
│   │                         #    expected values. Free, fast, no API keys.
│   │                         #    This is what bare `pytest` runs.
│   ├── smoke_test.py         # DONE Full-stack import/connectivity check.
│   │                         #    Uses live APIs (costs money).
│   │                         #    Run explicitly: uv run python tests/smoke_test.py
│   ├── test_ragas.py         # DONE Three-vendor Ragas end-to-end proof.
│   │                         #    Use live APIs (costs money).
│   │                         #    Run explicitly: uv run python tests/test_ragas.py
│
└── notebooks/                # Scratch exploration ONLY. Never the source of truth.
                              # Nothing that produces a reported number lives here.
```
## pytest discovery — IMPORTANT

Both diagnostics match pytest's default discovery patterns (`smoke_test.py`
matches `*_test.py`; `test_ragas.py` matches `test_*.py`), so a bare
`pytest` would collect them and make live, billable API calls. Restrict
discovery in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_metrics.py"]
pythonpath = ["."]
```

Widen `python_files` deliberately as more unit tests are added. Diagnostics
stay explicit: `uv run python tests/smoke_test.py`.

## Labels vs outcomes (a recurring confusion — keep them apart)

- **FOUR human labels** (`src/constants.py`, Labelling Guide v3): what the
  HUMAN judged the text to be — faithful / intrinsic / extrinsic /
  correct_refusal. A fifth ("wrongly withheld") was drafted and retracted:
  it required the labeller to know the answer was blocked, breaking the
  blinding rule. See NOTES 21.7.2026.
- **FIVE controller outcomes** (`src/constants.py`): what the SYSTEM did —
  passed_first / fixed_by_rewrite / fallback / verifier_malformed /
  generation_failed. The last two are infrastructure failures, excluded
  from the confusion matrix and reported as counts.

`metrics.py` performs the crossing of the two. The labeller never does.

## Layer model (how it reads)

    data/  -feeds->  src/ (the system)  -run by->  experiments/  -outputs->  results/  -measured by->  analysis/

- **src/** is the *system*: retrieve -> generate -> verify -> route.
- **experiments/** *runs* the system across the query set, once per arm.
- **analysis/** *measures* what the runs produced — Ragas, labels, stats and the final analysis.
- **web/** is a thin UI over the system, deliberately separate so evaluation
  never depends on the web server.

## Document roles (avoid duplicated facts)

- **README.md** — *how*: setup, commands, running each arm.
- **NOTES.md** — *why*: decisions, reasons, open items, dated.
- **PROJECT_STRUCTURE.md** — *where*: this map. Keep markers current.

Each fact lives in exactly ONE of the three.

## Build order (dependency-driven)

1.  DONE **corpus + questions.json**
2.  DONE **retrieval.py** + index (recall 100% at k=3/5/8/10; k=5 chosen)
3.  DONE **prompts.py, node_a.py, node_b.py, controller.py**
4.  DONE **run_experiment.py** -> all three arms swept, 108/108 each, 0 errors
5.  DONE **ragas_eval.py** -> all three arms scored
6.  DONE **labelling.py** -> 67 units generated, blinded, deduplicated
7.  DONE **human labelling** (67 units; 20 to 2nd reviewer) — the critical path
8.  DONE **analyse.py** — matrices, rates, paired comparison, judge validation
9.  DONE **web/main.py**
10. DONE the **report** + demo video
11. DONE **requirements.txt**

`analysis/metrics.py` and `stats.py` are the exception — they can be written and
unit-tested against synthetic confusion matrices at ANY point, with no pipeline
and no data required.
