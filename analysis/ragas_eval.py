"""
Ragas scoring: faithfulness and answer relevance, judged by Claude (the
third vendor, independent of Node A/Google and Node B/OpenAI).

WHAT THIS SCRIPT DOES, per the proposal's Phase 1:
  1. Load a results file (one arm's sweep).
  2. For each record with outcome in {passed_first, fixed_by_rewrite} —
     i.e. an answer actually shown to the user — reconstruct the (question,
     answer, contexts) triple Ragas needs.
  3. Score faithfulness and answer_relevancy with the Claude judge.
  4. Write a SEPARATE scores file, plus scoring_metadata() (judge, ragas
     version, which results file this scored) — never mixed into the
     results file itself, because scoring happens offline, possibly days
     later, possibly re-run with a different judge.

WHAT IS EXCLUDED, per the proposal: "Faithfulness is computed only over the
final answers actually returned to the user... queries ending in a withheld
or fallback response make no factual claims, so they are excluded from the
faithfulness average and counted separately as a refusal rate." fallback,
verifier_malformed and generation_failed records are therefore excluded from
scoring and counted, not silently dropped.

PASSAGE TEXT RECONSTRUCTION: results files store retrieved_chunk_ids
(e.g. "faq_09::0"), not passage text — keeping the results file small and
because the corpus is the single source of truth for passage content.
This script reconstructs contexts by reading data/corpus/{faq_id}.md via
the chunk ID's faq_id prefix. If the corpus changes after a sweep, contexts
reconstructed later will NOT match what the pipeline actually saw. No
fingerprint check currently guards this — the corpus is closed and has not
changed since these sweeps ran, so it is safe today, but this is a real gap
if the corpus is ever edited again.

Run:
    uv run python -m analysis.ragas_eval experiments/results/baseline_TIMESTAMP.jsonl
"""

from __future__ import annotations

import os
 
# MUST be set before ragas is imported anywhere (including transitively via
# this module's own functions). Ragas fires a usage-tracking HTTP call on
# certain metric operations; if that host cannot be resolved (e.g. this
# machine's egress does not allow it), each call hangs until DNS times out
# and retries — observed 8.8.2026: 6+ minutes per row, ~11h projected for a
# 58-row batch, collapsing to 35s for 116 evaluations once disabled. This is
# NOT a judge/API problem; the real Claude calls were fine throughout.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

import json
import sys
from pathlib import Path

from src.config import (
    CORPUS_DIR,
    JUDGE_MODEL,
    JUDGE_TEMPERATURE,
    QUESTIONS_FILE,
    RESULTS_DIR,
    scoring_metadata,
)
from src.constants import OUTCOME_FIXED_BY_REWRITE, OUTCOME_PASSED_FIRST

_SCORABLE_OUTCOMES = (OUTCOME_PASSED_FIRST, OUTCOME_FIXED_BY_REWRITE)


# ---------------------------------------------------------------------------
# Loading and joining the three sources: results, questions, corpus
# ---------------------------------------------------------------------------

def _load_results(path: str | Path) -> tuple[dict, list[dict]]:
    """Returns (metadata_dict, records). The #METADATA line is parsed and
    stripped; everything else is a per-query record."""
    metadata = None
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#METADATA "):
                metadata = json.loads(line[len("#METADATA "):])
                continue
            records.append(json.loads(line))
    if metadata is None:
        raise ValueError(f"{path}: no #METADATA header found")
    return metadata, records


def _load_question_text(path: str | Path = QUESTIONS_FILE) -> dict[str, str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    queries = data["queries"] if isinstance(data, dict) else data
    return {q["id"]: q["text"] for q in queries}


_corpus_cache: dict[str, str] = {}


def _reconstruct_contexts(chunk_ids: list[str],
                          corpus_dir: str | Path = CORPUS_DIR) -> list[str]:
    """
    chunk_ids look like "faq_09::0" (faq_id::chunk_index). The corpus is
    small enough and chunked coarsely enough (1 FAQ = 1 chunk, verified
    25.7.2026) that reading the whole FAQ file per chunk_id is correct here.

    If the corpus is ever re-chunked (multi-chunk FAQs), this function
    would need the actual chunk boundaries, not just the file — flagged
    rather than silently wrong, since RETRIEVAL_K excludes chunk count from
    the fingerprint and this script has no independent way to detect it.
    """
    contexts = []
    for cid in chunk_ids:
        faq_id = cid.split("::")[0]
        if faq_id not in _corpus_cache:
            fpath = Path(corpus_dir) / f"{faq_id}.md"
            if not fpath.exists():
                raise FileNotFoundError(
                    f"chunk_id {cid!r} references {fpath}, which does not "
                    f"exist. Corpus may have changed since this sweep ran.")
            _corpus_cache[faq_id] = fpath.read_text(encoding="utf-8").strip()
        contexts.append(_corpus_cache[faq_id])
    return contexts


def build_scoring_rows(results_path: str | Path) -> tuple[dict, list[dict], dict]:
    """
    Returns (run_metadata, scorable_rows, exclusion_counts).

    scorable_rows: [{query_id, question, answer, contexts}, ...] — only
    records whose outcome is in _SCORABLE_OUTCOMES.
    exclusion_counts: {outcome: count} for everything NOT scored, so the
    refusal rate the proposal asks for is a byproduct of this call, not a
    separate pass.
    """
    run_meta, records = _load_results(results_path)
    questions = _load_question_text()

    rows = []
    excluded = {}
    for r in records:
        if r["outcome"] not in _SCORABLE_OUTCOMES:
            excluded[r["outcome"]] = excluded.get(r["outcome"], 0) + 1
            continue
        rows.append({
            "query_id": r["query_id"],
            "question": questions[r["query_id"]],
            "answer": r["answer_shown"],
            "contexts": _reconstruct_contexts(r["retrieved_chunk_ids"]),
        })

    return run_meta, rows, excluded


# ---------------------------------------------------------------------------
# Scoring — the validated three-vendor stack (test_ragas.py, 11.7.2026)
# ---------------------------------------------------------------------------

def _build_judge():
    """The Claude judge via LangchainLLMWrapper — the only path available in
    Ragas 0.3.9 for a non-OpenAI judge (the modern llm_factory path is
    OpenAI-only). Deprecation warning accepted; ragas is version-frozen so
    it cannot bite. See NOTES 12.7.2026."""
    from langchain_anthropic import ChatAnthropic
    from ragas.llms import LangchainLLMWrapper
    from src.config import ANTHROPIC_API_KEY, MAX_RETRIES, REQUEST_TIMEOUT

    llm = ChatAnthropic(
        model=JUDGE_MODEL,
        api_key=ANTHROPIC_API_KEY,
        temperature=JUDGE_TEMPERATURE,
        max_retries=MAX_RETRIES,
        timeout=REQUEST_TIMEOUT,
        max_tokens=4096,
    )
    return LangchainLLMWrapper(llm)


def _build_embeddings():
    """Local all-MiniLM, pinned revision — keeps OpenAI fully out of the
    scoring loop and matches the retrieval embedding space. Shared-instrument
    limitation (answer relevance computed in the same space used for
    retrieval) is acknowledged in NOTES 12.7.2026, not fixed here."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from src.config import EMBEDDING_MODEL, EMBEDDING_REVISION

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"revision": EMBEDDING_REVISION},
    )


def score(rows: list[dict]) -> "pandas.DataFrame":
    """
    Runs Ragas faithfulness + answer_relevancy over the scorable rows.

    Returns the raw Ragas result as a DataFrame — one row per query, in the
    same order as `rows`, so scores can be zipped back onto query_ids by the
    caller rather than this function assuming a particular output shape.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, faithfulness

    dataset = Dataset.from_list([
        {"question": r["question"], "answer": r["answer"],
         "contexts": r["contexts"]}
        for r in rows
    ])

    judge = _build_judge()
    embeddings = _build_embeddings()

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=judge,          # explicit override — the Ragas default is
        embeddings=embeddings,  # OpenAI and MUST NOT be used (NOTES 12.7.2026)
    )
    return result.to_pandas()


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def run(results_path: str | Path) -> Path:
    results_path = Path(results_path)
    run_meta, rows, excluded = build_scoring_rows(results_path)

    print(f"Scoring {results_path.name}")
    print(f"  arm={run_meta['active_arm']}  scorable={len(rows)}  "
          f"excluded={excluded}")

    if not rows:
        raise ValueError(
            f"No scorable rows in {results_path} — every record was "
            f"fallback/verifier_malformed/generation_failed. Nothing to "
            f"score. Excluded counts: {excluded}")

    df = score(rows)

    # The joined (question, answer, contexts) triple is saved alongside the
    # scores, not just held in memory and discarded — if a score looks
    # surprising later, this is the record of exactly what Ragas was given.
    # Kept in the SAME file as the scores (not a separate join-record file)
    # so the two can never drift apart or be matched up wrongly by query_id.
    out_records = []
    for row, (_, scored) in zip(rows, df.iterrows()):
        out_records.append({
            "query_id": row["query_id"],
            "question": row["question"],
            "answer": row["answer"],
            "contexts": row["contexts"],
            "faithfulness": float(scored["faithfulness"]),
            "answer_relevancy": float(scored["answer_relevancy"]),
        })

    stamp_path = Path(RESULTS_DIR) / f"ragas_{results_path.stem}.jsonl"
    with stamp_path.open("w", encoding="utf-8") as f:
        meta = scoring_metadata(scored_results_file=str(results_path))
        meta["scorable_count"] = len(rows)
        meta["excluded_counts"] = excluded
        meta["source_active_arm"] = run_meta["active_arm"]
        f.write("#SCORING_METADATA " + json.dumps(meta) + "\n")
        for rec in out_records:
            f.write(json.dumps(rec) + "\n")

    mean_f = sum(r["faithfulness"] for r in out_records) / len(out_records)
    mean_r = sum(r["answer_relevancy"] for r in out_records) / len(out_records)
    refusal_rate = sum(excluded.values()) / (len(out_records) + sum(excluded.values()))

    print(f"\nScored: {stamp_path}")
    print(f"  mean faithfulness:     {mean_f:.4f}")
    print(f"  mean answer_relevancy: {mean_r:.4f}")
    print(f"  refusal rate:          {refusal_rate:.4f}  "
          f"({sum(excluded.values())}/{len(out_records) + sum(excluded.values())})")

    return stamp_path


def _main() -> int:
    if len(sys.argv) != 2:
        print("Usage: uv run python -m analysis.ragas_eval "
              "experiments/results/<arm>_<timestamp>.jsonl")
        return 1
    run(sys.argv[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())