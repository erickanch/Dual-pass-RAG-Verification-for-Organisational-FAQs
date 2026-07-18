# Project Structure — msc-rag

Updated to reflect the finalised design:
- THREE arms (baseline / same_vendor / cross_vendor), selected via `ACTIVE_ARM`,
  not two hard-coded run scripts.
- The three-vendor Ragas evaluation subsystem (Claude judge) has its own home.
- Node B's model is arm-dependent, not fixed to one vendor.

Marker key: DONE = built | TODO = planned | MOVE = built but not yet in this location

```
msc-rag/
│
├── .env                      # 3 API keys (Google, OpenAI, Anthropic). GITIGNORED.
├── .gitignore
├── pyproject.toml            # declared dependencies + Python 3.12 pin
├── uv.lock                   # exact resolved dependency graph — COMMIT (reproducibility)
├── requirements.txt          # TODO pip-installable export for non-uv readers (Week 14)
├── README.md                 # setup guide + how to run (see README)
├── NOTES.md                  # methodology journal: decisions + reasons, dated
├── PROJECT_STRUCTURE.md      # this file
│
├── src/                      # ── THE LIVE PIPELINE ──────────────────────────
│   ├── config.py             # DONE Single source of truth. Everything imports from here.
│   ├── arms.py               # TODO ARM SELECTION. Reads ACTIVE_ARM, derives node_b_model
│   │                         #    + node_b_enabled per arm. Per-arm guards
│   │                         #    (cross_vendor => distinct vendors;
│   │                         #     same_vendor => Node B IS the Node A model — a
│   │                         #     reference, not a copied string).
│   │                         #    (May instead live inside config.py as ARMS_CONFIG.)
│   ├── prompts.py            # TODO VERSIONED PROMPTS. Node A gen, Node B verify,
│   │                         #    rewrite prompt, fallback message. Tied to PROMPT_VERSION.
│   ├── retrieval.py          # TODO INGESTION + RETRIEVAL. Chunk corpus -> embed
│   │                         #    (all-MiniLM) -> ChromaDB. Query-time top-k.
│   │                         #    Stores a config-hash with the index; refuses to
│   │                         #    run on a stale index (chunk/embed settings changed).
│   ├── node_a.py             # TODO GENERATOR. Gemini. (question + passages) -> draft.
│   ├── node_b.py             # TODO VERIFIER. MODEL PER ACTIVE ARM (not fixed to GPT).
│   │                         #    Decomposes draft into atomic claims; labels each
│   │                         #    supported / contradicted / unsupported.
│   │                         #    Returns Pydantic-validated structured result.
│   └── controller.py         # TODO DETERMINISTIC CONTROLLER. Owns routing, the rewrite
│                             #    loop (cap = MAX_REWRITE_ATTEMPTS), and the fallback.
│                             #    Control logic lives HERE, in ordinary code — never
│                             #    in the models. Writes the per-query log
│                             #    (LOG_SCHEMA_VERSION contract in config.py).
│
├── web/                      # ── WEB APP (BUILT LAST) ───────────────────────
│   └── main.py               # TODO FastAPI app. Thin layer over controller.py.
│                             #    Separated from the pipeline so evaluation never
│                             #    depends on the web server.
│
├── data/                     # ── INPUTS ─────────────────────────────────────
│   ├── corpus/               # TODO 20 synthetic FAQ source documents.
│   └── questions.json        # TODO ~100 queries. Each: {id, text, category
│                             #    (in_domain|out_of_domain|adversarial),
│                             #    target_faq (in-domain only)}.
│
├── experiments/              # ── RUNNING THE SYSTEM ─────────────────────────
│   ├── run_experiment.py     # TODO ONE parameterised runner. Reads ACTIVE_ARM from
│   │                         #    config, runs the full query set through the
│   │                         #    pipeline, writes timestamped results + run_metadata().
│   │                         #    (Replaces the old run_baseline/run_dual_node pair.)
│   ├── pilot.py              # TODO SMALL-SCALE PRE-TEST. Validates verifier tier
│   │                         #    (Luna vs Terra), retrieval recall (k), chunk size,
│   │                         #    sweep time, and per-query cost — before the full run.
│   └── results/              # TODO Raw outputs. Every run timestamped, NEVER overwritten.
│                             #    Gitignored if large; keep a manifest.
│
├── analysis/                 # ── MEASURING THE RUNS (offline) ───────────────
│   ├── ragas_eval.py         # TODO RAGAS SCORING. Claude judge + local embeddings.
│   │                         #    Faithfulness + answer relevance. Writes its OWN
│   │                         #    scoring_metadata() (judge, ragas version, which
│   │                         #    results file it scored). The 3rd-vendor subsystem.
│   │                         #    Includes the judge-validation check: Ragas vs
│   │                         #    human labels on the Baseline arm (see NOTES 12.7.2026).
│   ├── labelling.py          # TODO HITL SUPPORT. Presents answers blind + in seeded
│   │                         #    random order for human labelling; captures labels;
│   │                         #    computes Cohen's kappa on the 2nd-reviewer sample.
│   ├── metrics.py            # TODO SCR, FNR, FPR from the confusion matrix, each with
│   │                         #    95% Wilson intervals. Unit-test on SYNTHETIC data now.
│   ├── stats.py              # TODO McNemar (paired, baseline vs cross_vendor) +
│   │                         #    Wilcoxon signed-rank (paired Ragas faithfulness).
│   └── figures.py            # TODO matplotlib figures for the report.
│
├── tests/                    # ── DIAGNOSTICS (keep, don't delete) ───────────
│   ├── smoke_test.py         # MOVE Full-stack import/connectivity check.
│   │                         #    BUILT — currently in repo root; git mv here.
│   └── test_ragas.py         # MOVE Three-vendor Ragas end-to-end proof.
│                             #    BUILT — currently in repo root; git mv here.
│
└── notebooks/                # Scratch exploration ONLY. Never the source of truth.
                              # Nothing that produces a reported number lives here.
```

## Layer model (how it reads)

    data/  -feeds->  src/ (the system)  -run by->  experiments/  -outputs->  results/  -measured by->  analysis/

- **src/** is the *system*: retrieve -> generate -> verify -> route.
- **experiments/** *runs* the system across the query set, once per arm.
- **analysis/** *measures* what the runs produced — Ragas, labels, stats, figures.
- **web/** is a thin UI over the system, deliberately separate so evaluation
  never depends on the web server.

## Document roles (avoid duplicated facts)

- **README.md** — *how*: setup, commands, running each arm.
- **NOTES.md** — *why*: decisions, reasons, open items, dated.
- **PROJECT_STRUCTURE.md** — *where*: this map. Keep markers current.

Each fact lives in exactly ONE of the three.

## Build order (dependency-driven)

1. **corpus + questions.json**  <- blocks everything (do first)
2. **retrieval.py** + provisional index
3. **arms.py, prompts.py, node_a.py, node_b.py, controller.py** (the pipeline)
4. **pilot.py** -> settle verifier tier / k / chunk size -> RE-EMBED if changed
5. **run_experiment.py** -> full runs, all three arms
6. **ragas_eval.py, labelling.py, metrics.py, stats.py** (measurement)
7. **figures.py**, then the **report**
8. **web/main.py** + demo video (last)

`analysis/metrics.py` and `stats.py` are the exception — they can be written and
unit-tested against synthetic confusion matrices at ANY point, with no pipeline
and no data. Good weekday-evening work.
