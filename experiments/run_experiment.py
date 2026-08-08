"""
The experiment runner. Reads ACTIVE_ARM from config, runs the full query set
through the controller once, and writes one timestamped results file.

ONE RUNNER SERVES ALL THREE ARMS. It never reads NODE_B_MODEL directly — only
the controller's routing, which reads NODE_B_ENABLED / ACTIVE_NODE_B_MODEL —
so the same script produces baseline, same_vendor and cross_vendor sweeps by
changing only ACTIVE_ARM (env var or .env), never the code.

INCREMENTAL WRITES. Each record is appended to the output file as soon as it
completes — never held in memory until the end. A crash or a Gemini outage at
query 80 must not lose queries 1-79. Combined with RESUME (below), a sweep
survives being interrupted.

RESUME. Re-running with --resume reads the most recent results file for the
active arm, skips any query_id that already has a result EXCEPT those whose
outcome was generation_failed or verifier_malformed — infrastructure
failures are retried, not treated as done. This is what NOTES 26.7.2026
flagged. A demand spike must not cost a full re-run.

Run:
    uv run python -m experiments.run_experiment
    uv run python -m experiments.run_experiment --resume
    ACTIVE_ARM=baseline uv run python -m experiments.run_experiment
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from src.config import ACTIVE_ARM, QUESTIONS_FILE, RESULTS_DIR, run_metadata
from src.constants import OUTCOME_GENERATION_FAILED, OUTCOME_VERIFIER_MALFORMED
from src.controller import answer_query

# Outcomes that count as "not really done" for --resume: infrastructure
# failures, not verification decisions. Everything else (passed_first,
# fixed_by_rewrite, fallback) is a genuine result and is left alone.
_RETRYABLE_OUTCOMES = (OUTCOME_GENERATION_FAILED, OUTCOME_VERIFIER_MALFORMED)


def load_queries(path: str | Path = QUESTIONS_FILE) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["queries"] if isinstance(data, dict) else data


def _latest_results_file(arm: str) -> Path | None:
    """Most recent results file for this arm, or None if none exists yet."""
    candidates = sorted(Path(RESULTS_DIR).glob(f"{arm}_*.jsonl"))
    return candidates[-1] if candidates else None


def _load_completed(path: Path) -> dict[str, dict]:
    """query_id -> last record for that query in the file (a resumed sweep may
    contain a query more than once; the latest write wins)."""
    completed = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            record = json.loads(line)
            completed[record["query_id"]] = record
    return completed


def run(resume: bool = False) -> Path:
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    queries = load_queries()

    already_ok: dict[str, dict] = {}
    out_path: Path

    if resume:
        prev = _latest_results_file(ACTIVE_ARM)
        if prev is None:
            print(f"--resume requested but no prior results file found for "
                  f"arm={ACTIVE_ARM}; starting a fresh sweep.")
            resume = False
        else:
            completed = _load_completed(prev)
            already_ok = {qid: rec for qid, rec in completed.items()
                         if rec.get("outcome") not in _RETRYABLE_OUTCOMES}
            n_retry = len(completed) - len(already_ok)
            print(f"Resuming {prev.name}: {len(already_ok)} queries already "
                  f"done, {n_retry} to retry (infrastructure failures), "
                  f"{len(queries) - len(completed)} not yet attempted.")
            out_path = prev   # append to the SAME file; already_ok are re-written as-is

    if not resume:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = Path(RESULTS_DIR) / f"{ACTIVE_ARM}_{stamp}.jsonl"

    # ---- header: run_metadata as the first line, prefixed so it is never
    # mistaken for a query record. Only written on a fresh sweep; a resumed
    # sweep keeps its original header. ----
    if not resume:
        with out_path.open("w", encoding="utf-8") as f:
            f.write("#METADATA " + json.dumps(run_metadata()) + "\n")

    n_done = n_ok = n_retryable = 0
    t_sweep_start = time.perf_counter()

    # Re-open in append mode for the sweep body, whether fresh or resumed.
    with out_path.open("a", encoding="utf-8") as f:
        for q in queries:
            qid = q["id"]

            if qid in already_ok:
                continue   # genuine prior result — leave it in the file, don't re-run

            try:
                result = answer_query(qid, q["text"])
            except Exception as exc:
                # The controller is designed not to raise for pipeline
                # failures (see controller.py). If it does, that is a
                # PROGRAMMING error, not an infrastructure one — do not
                # swallow it as generation_failed. Log what we can and stop:
                # a sweep should not silently continue past a code defect.
                print(f"\nUNEXPECTED exception on {qid} — this is a bug, not "
                      f"an API failure. Stopping the sweep.\n{type(exc).__name__}: {exc}")
                raise

            record = result.to_dict()
            record["category"] = q.get("category")
            record["target_faq"] = q.get("target_faq")
            record["underspecified"] = q.get("underspecified", False)

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()   # survive a crash between writes, not just between queries

            n_done += 1
            if record["outcome"] in _RETRYABLE_OUTCOMES:
                n_retryable += 1
            else:
                n_ok += 1

            elapsed = time.perf_counter() - t_sweep_start
            print(f"  [{n_done + len(already_ok)}/{len(queries)}] {qid}: "
                  f"{record['outcome']}  ({elapsed:.0f}s elapsed)")

    print(f"\nSweep complete: {out_path}")
    print(f"  arm={ACTIVE_ARM}  ok={n_ok + len(already_ok)}  "
          f"retryable_failures={n_retryable}  total_queries={len(queries)}")
    if n_retryable:
        print(f"  {n_retryable} quer{'y' if n_retryable == 1 else 'ies'} "
              f"had an infrastructure failure. Re-run with --resume once the "
              f"API is healthy to retry only those.")

    return out_path


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true",
                        help="Continue the most recent sweep for the active "
                             "arm, retrying only infrastructure failures.")
    args = parser.parse_args()

    print(f"Starting sweep: arm={ACTIVE_ARM}  "
          f"resume={'yes' if args.resume else 'no'}")
    run(resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())