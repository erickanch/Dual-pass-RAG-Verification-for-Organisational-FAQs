"""
Final analysis: joins human labels to pipeline outcomes, builds the confusion
matrix for each arm, and computes the reported rates and tests.

WHICH FILES. experiments/results/ holds more than the reported runs:
  - the three LABELLED runs (8.8.2026), one per arm: the main evaluation;
  - two repeat runs per arm (17.8.2026): run-to-run stability only;
  - two abandoned baseline runs (16.8.2026): kept for reference and not
    reported (NOTES 17.8.2026).
The runs are therefore NAMED below rather than picked by date. Before any
figure is computed, each labelled run is checked against the draft hashes in
labelling_mapping.json, so a label can only attach to the text it was given
for.

WHAT THIS JOINS. Three sources, all already on disk:
  labelling_sheet.csv           the human label per distinct draft text
  labelling_sheet_reviewer.csv  the second reviewer's labels
  labelling_mapping.json        which (arm, query, stage) each label covers
  {arm}_{ts}.jsonl              what the system actually did
  ragas_{arm}_{ts}.jsonl        the automated scores of each labelled run
The label attaches to the DRAFT; the outcome attaches to the QUERY. The
confusion matrix is the crossing of the two, which is performed here and
never by the labeller.

WHICH DRAFT CARRIES THE OUTCOME. A query may produce several drafts. The
outcome describes the query, not any single draft, so the matrix uses the
label of the draft that determined the outcome: the FINAL draft in the chain.
That is the answer shown to the user where one was shown, and the last
withheld draft where the answer was blocked. Intermediate rewrite drafts are
labelled and reported separately, because the rewrite chain is of interest in
its own right, but they do not each generate a matrix row.

Run:
    uv run python -m analysis.analyse
"""

from __future__ import annotations

import csv
import json
import random
import re
import statistics
from collections import Counter
from itertools import combinations
from pathlib import Path

from analysis.metrics import (
    build_confusion_matrix,
    false_negative_rate,
    false_positive_rate,
    nearest_rank_percentile,
    paired_safety_outcomes,
    recorded_latency_s,
    strict_compliance_rate,
    total_tokens,
)
from analysis.stats import mcnemar_test, wilcoxon_test
from src.config import ARM_BASELINE, ARM_CROSS_VENDOR, ARM_SAME_VENDOR, RESULTS_DIR
from src.constants import (
    BLOCKING_OUTCOMES,
    EXCLUDED_OUTCOMES,
    HALLUCINATION_LABELS,
    OUTCOME_FALLBACK,
    OUTCOME_FIXED_BY_REWRITE,
    OUTCOME_PASSED_FIRST,
    OUTCOME_VERIFIER_MALFORMED,
    RANDOM_SEED,
    VALID_LABELS,
)

# Column order of the report's tables.
ARMS = (ARM_BASELINE, ARM_CROSS_VENDOR, ARM_SAME_VENDOR)

# The runs behind the reported figures (see WHICH FILES above).
LABELLED_RUNS = {
    ARM_BASELINE: "baseline_20260808_032042.jsonl",
    ARM_CROSS_VENDOR: "cross_vendor_20260808_040232.jsonl",
    ARM_SAME_VENDOR: "same_vendor_20260808_042829.jsonl",
}
REPEAT_RUNS = {
    ARM_BASELINE: ("baseline_20260817_124708.jsonl",
                   "baseline_20260817_125235.jsonl"),
    ARM_CROSS_VENDOR: ("cross_vendor_20260817_121516.jsonl",
                       "cross_vendor_20260817_123617.jsonl"),
    ARM_SAME_VENDOR: ("same_vendor_20260817_125905.jsonl",
                      "same_vendor_20260817_130719.jsonl"),
}

DISPLAYED_OUTCOMES = (OUTCOME_PASSED_FIRST, OUTCOME_FIXED_BY_REWRITE)

# Rule 6 tells the generator to decline an uncovered point in the first
# person: "I do not have information on ...". Answers containing that
# statement are reported separately for answer relevance (Table 12).
DECLINING_STATEMENT = re.compile(r"\bI do not have information\b", re.IGNORECASE)

# Judge validation checks whether short answers explain the zero
# faithfulness score (Section 8.9).
SHORT_ANSWER_WORDS = 15

# Queries followed across all nine runs (Section 8.11), each with the claim
# whose recurrence is reported, or None where only identity is reported.
CASE_QUERIES = {"IN-44": None, "ADV-27": "cannot bypass"}


# ---------------------------------------------------------------------------
# Loading and checking
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> tuple[dict, list[dict]]:
    metadata, records = None, []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                metadata = json.loads(line.split(" ", 1)[1])
                continue
            records.append(json.loads(line))
    return metadata, records


def load_run(name: str, arm: str) -> tuple[dict, list[dict]]:
    """A named pipeline run, checked to belong to the expected arm."""
    metadata, records = _load_jsonl(Path(RESULTS_DIR) / name)
    if not metadata or metadata.get("active_arm") != arm:
        raise SystemExit(f"{name}: expected a {arm} run")
    return metadata, records


def check_against_mapping(arm: str, records: list[dict],
                          mapping: list[dict]) -> None:
    """
    Stop unless this run holds exactly the drafts labelled for this arm:
    the same (query, stage) with the same hash, none missing, none extra.
    Mirrors what labelling.py collected: refusal sentinels and
    infrastructure failures carry no labelled draft.
    """
    expected = {(m["query_id"], m["stage"]): m["sha256"]
                for m in mapping if m["arm"] == arm}
    found = {(r["query_id"], d["stage"]): d["sha256"]
             for r in records if r["outcome"] not in EXCLUDED_OUTCOMES
             for d in r.get("drafts", []) if not d.get("refused")}
    if found != expected:
        missing = sorted(expected.keys() - found.keys())
        extra = sorted(found.keys() - expected.keys())
        changed = sorted(k for k in expected.keys() & found.keys()
                         if expected[k] != found[k])
        raise SystemExit(
            f"{arm}: {LABELLED_RUNS[arm]} does not match "
            f"labelling_mapping.json (missing {missing[:3]}, "
            f"unlabelled {extra[:3]}, different text {changed[:3]})")


def load_labels() -> dict[str, str]:
    """unit_id -> label, straight from the completed sheet."""
    path = Path(RESULTS_DIR) / "labelling_sheet.csv"
    labels = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lab = (row.get("label") or "").strip()
            if not lab:
                continue
            if lab not in VALID_LABELS:
                raise ValueError(f"{row['unit_id']}: unknown label {lab!r}")
            labels[row["unit_id"]] = lab
    return labels


def load_ragas(arm: str, records: list[dict]) -> dict[str, dict]:
    """
    query_id -> Ragas record for the labelled run. Checked to be the scoring
    of that run, and of exactly the answers it displayed.
    """
    name = f"ragas_{Path(LABELLED_RUNS[arm]).stem}.jsonl"
    metadata, scored = _load_jsonl(Path(RESULTS_DIR) / name)
    scored_path = (metadata or {}).get("scored_results_file", "")
    scored_file = re.split(r"[\\/]", scored_path)[-1]
    if scored_file != LABELLED_RUNS[arm]:
        raise SystemExit(f"{name}: does not record scoring {LABELLED_RUNS[arm]}")
    shown = {r["query_id"]: r["answer_shown"] for r in records
             if r["outcome"] in DISPLAYED_OUTCOMES}
    by_query = {s["query_id"]: s for s in scored}
    if by_query.keys() != shown.keys() or any(
            by_query[q]["answer"] != shown[q] for q in shown):
        raise SystemExit(f"{name}: scored answers differ from those displayed "
                         f"in {LABELLED_RUNS[arm]}")
    return by_query


def build_arm_records(arm: str, results: list[dict], labels: dict[str, str],
                      mapping: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Returns (matrix_records, chain_records).

    matrix_records: one per query, carrying the FINAL draft's label and the
                    query's outcome — the input to the confusion matrix.
    chain_records:  every labelled draft, including intermediates, for the
                    separate rewrite-chain reporting.
    """
    # (arm, query_id, stage, hash) -> unit_id, from the mapping written at
    # build time. The hash is part of the key, so a draft whose text differs
    # from the one labelled can never pick up that label.
    unit_of = {(m["arm"], m["query_id"], m["stage"], m["sha256"]): m["unit_id"]
               for m in mapping}

    matrix_records, chain_records = [], []

    for rec in results:
        qid, outcome = rec["query_id"], rec["outcome"]

        labelled_drafts = []
        for draft in rec.get("drafts", []):
            uid = unit_of.get((arm, qid, draft["stage"], draft["sha256"]))
            if uid is None or uid not in labels:
                continue                      # refusal sentinel, or excluded
            labelled_drafts.append({
                "query_id": qid,
                "arm": arm,
                "stage": draft["stage"],
                "unit_id": uid,
                "human_label": labels[uid],
                "outcome": outcome,
                "category": rec.get("category"),
            })

        chain_records.extend(labelled_drafts)

        # The final labelled draft is the one the outcome describes.
        if labelled_drafts:
            matrix_records.append(labelled_drafts[-1])

    return matrix_records, chain_records


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    print(f"\n{'=' * 66}")
    print(title)
    print(f"{'=' * 66}")


def _row(label: str, cells, width: int = 34, cell: int = 15) -> str:
    return f"  {label:<{width}}" + "".join(f"{str(c):>{cell}}" for c in cells)


def _rewrites(n: int) -> str:
    return f"{n} rewrite{'' if n == 1 else 's'}"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _initial_hash(record: dict) -> str | None:
    drafts = record.get("drafts") or []
    return drafts[0]["sha256"] if drafts else None


# ---------------------------------------------------------------------------
# Section 8, in report order
# ---------------------------------------------------------------------------

def report_execution(runs: dict, repeats: dict) -> None:
    _banner("8.1 Execution")
    verified = [sum(1 for r in runs[a] if r["verifications"]) for a in ARMS]
    calls = [sum(len(r["verifications"]) for r in runs[a]) for a in ARMS]
    print(_row("", ARMS))
    print(_row("Queries", [len(runs[a]) for a in ARMS]))
    print(_row("Infrastructure failures",
               [sum(r["outcome"] in EXCLUDED_OUTCOMES for r in runs[a])
                for a in ARMS]))
    print(_row("Queries verified", verified))
    print(_row("Verifier calls", calls))
    print(_row("  of which re-checks after rewrite",
               [c - v for c, v in zip(calls, verified)]))

    every = [recs for a in ARMS for recs in [runs[a], *repeats[a]]]
    malformed = sum(r["outcome"] == OUTCOME_VERIFIER_MALFORMED
                    for recs in every for r in recs)
    retries = sum(n - 1 for recs in every for r in recs
                  for n in r.get("verification_attempts") or [])
    print(f"\n  Across all {len(every)} reported runs: verifier_malformed "
          f"{malformed}, verifier retries {retries}")


def report_outcomes(runs: dict) -> None:
    _banner("8.2 Outcome distribution (Table 7)")
    print(_row("Outcome", ARMS))
    for label, outcome in (
            ("Answer displayed without rewrite", OUTCOME_PASSED_FIRST),
            ("Answer displayed after rewrite", OUTCOME_FIXED_BY_REWRITE),
            ("Answer withheld", OUTCOME_FALLBACK)):
        print(_row(label, [
            "—" if arm == ARM_BASELINE and outcome == OUTCOME_FIXED_BY_REWRITE
            else sum(r["outcome"] == outcome for r in runs[arm])
            for arm in ARMS]))
    failures = [sum(r["outcome"] in EXCLUDED_OUTCOMES for r in runs[a])
                for a in ARMS]
    if any(failures):
        print(_row("Infrastructure failure", failures))
    print(_row("Total", [len(runs[a]) for a in ARMS]))

    print("\n  By query category, answered / withheld:")
    for category in ("in_domain", "adversarial", "out_of_domain"):
        cells, size = [], 0
        for arm in ARMS:
            recs = [r for r in runs[arm] if r["category"] == category]
            size = len(recs)
            shown = sum(r["outcome"] in DISPLAYED_OUTCOMES for r in recs)
            withheld = sum(r["outcome"] == OUTCOME_FALLBACK for r in recs)
            cells.append(f"{shown} / {withheld}")
        print(_row(f"  {category} ({size})", cells))

    print("\n  Why answers were withheld:")
    refused = [sum(r["outcome"] == OUTCOME_FALLBACK
                   and bool(r["drafts"]) and bool(r["drafts"][-1].get("refused"))
                   for r in runs[a]) for a in ARMS]
    withheld = [sum(r["outcome"] == OUTCOME_FALLBACK for r in runs[a])
                for a in ARMS]
    print(_row("  generator refused", refused))
    print(_row("  rewrite limit reached",
               [w - f for w, f in zip(withheld, refused)]))


def report_retrieval(runs: dict, metas: dict) -> None:
    _banner("8.3 Retrieval")
    depths = {metas[a]["retrieval_k"] for a in ARMS}
    cells = []
    for arm in ARMS:
        recs = [r for r in runs[arm] if r["category"] == "in_domain"]
        hit = sum(set(r["target_faq"]) <= {c.split("::")[0]
                                            for c in r["retrieved_chunk_ids"]}
                  for r in recs)
        cells.append(f"{hit}/{len(recs)}")
    print(f"  In-domain queries with every target FAQ retrieved, "
          f"k = {', '.join(map(str, sorted(depths)))}:")
    print(_row("", ARMS))
    print(_row("", cells))
    print("  (Depths 3 to 10 come from the retrieval pre-test, which needs the "
          "index;\n  they are not recomputed here.)")


def report_labelling(labels: dict, mapping: list[dict],
                     arm_records: dict) -> None:
    _banner("8.4 Human labelling (Table 8)")
    per_arm = Counter(m["arm"] for m in mapping)
    print(f"  Distinct texts labelled: {len(labels)}, covering {len(mapping)} "
          f"records ({', '.join(f'{a} {per_arm[a]}' for a in ARMS)})")
    counts = Counter(labels.values())
    print("  Labels by distinct text: "
          + ", ".join(f"{lab} {counts[lab]}" for lab in VALID_LABELS))

    # Each hallucinated text, where it occurred, and whether it was shown.
    final = {(r["arm"], r["query_id"]): r for a in ARMS for r in arm_records[a]}
    for uid in sorted(u for u, lab in labels.items()
                      if lab in HALLUCINATION_LABELS):
        print(f"\n  {uid} ({labels[uid]})")
        for arm in ARMS:
            occurrences = [m for m in mapping
                           if m["unit_id"] == uid and m["arm"] == arm]
            if not occurrences:
                print(f"    {arm:13s} not produced")
            for m in occurrences:
                last = final.get((arm, m["query_id"]))
                is_final = last is not None and last["unit_id"] == uid
                if not is_final:
                    status = "not shown: a later rewrite replaced it"
                elif m["outcome"] in BLOCKING_OUTCOMES:
                    status = "withheld"
                else:
                    status = "shown to the user"
                print(f"    {arm:13s} {m['query_id']} {m['stage']:10s} "
                      f"{m['outcome']:17s} {status}")

    # Second reviewer (Section 8.4, last paragraph)
    path = Path(RESULTS_DIR) / "labelling_sheet_reviewer.csv"
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    reviewer = {r["unit_id"]: r["label"].strip() for r in rows
                if (r.get("label") or "").strip()}
    agree = sum(labels.get(u) == lab for u, lab in reviewer.items())
    print(f"\n  Second reviewer: {len(reviewer)} of {len(labels)} texts "
          f"({len(reviewer) / len(labels):.1%}), agreement {agree}/"
          f"{len(reviewer)} ({agree / len(reviewer):.0%})")

    # labelling.py draws the sample with a seeded shuffle, so it can be
    # reconstructed from the main sheet.
    with (Path(RESULTS_DIR) / "labelling_sheet.csv").open(encoding="utf-8") as f:
        sheet = list(csv.DictReader(f))
    drawn = {r["unit_id"] for r in random.Random(RANDOM_SEED).sample(sheet, len(rows))}
    print(f"  Sample matches the seeded draw (seed {RANDOM_SEED}): "
          f"{'yes' if drawn == {r['unit_id'] for r in rows} else 'NO'}")

    researcher_used = {labels[u] for u in reviewer}
    reviewer_used = set(reviewer.values())
    if len(researcher_used) == 1 and researcher_used == reviewer_used:
        print(f"  Cohen's kappa: undefined. Both raters used one category "
              f"({next(iter(reviewer_used))}), so chance agreement is also "
              f"complete.")
    else:
        print(f"  Categories used: researcher {sorted(researcher_used)}, "
              f"reviewer {sorted(reviewer_used)}")


def report_arm(arm: str, records: list[dict]) -> None:
    cm = build_confusion_matrix(records)
    scr = strict_compliance_rate(cm)
    fnr = false_negative_rate(cm)
    fpr = false_positive_rate(cm)

    print(f"\n  {arm}")
    print(f"  {'-' * 40}")
    print(f"  labelled queries in matrix: {cm.total}   excluded: {cm.excluded}")
    print("\n  confusion matrix")
    print("                        hallucinated    faithful")
    print(f"    verifier blocks     TP = {cm.tp:<10d}  FP = {cm.fp}")
    print(f"    verifier passes     FN = {cm.fn:<10d}  TN = {cm.tn}")
    if cm.blocked_refusals:
        print(f"    (of which {cm.blocked_refusals} blocked correct refusals)")

    print(f"\n  Strict-Compliance Rate  {scr.as_percent()}")
    print(f"  False-Negative Rate     {fnr.as_percent()}")
    print(f"  False-Positive Rate     {fpr.as_percent()}")

    labels = Counter(r["human_label"] for r in records)
    print(f"\n  label distribution: {dict(labels)}")

    by_cat = {}
    for r in records:
        by_cat.setdefault(r["category"], Counter())[r["human_label"]] += 1
    print("  by category:")
    for cat, c in sorted(by_cat.items()):
        print(f"    {cat}: {dict(c)}")


def report_paired(arm_records: dict) -> None:
    _banner("8.6 Paired comparisons of delivered hallucinations (Table 10)")
    common = set.intersection(*({r["query_id"] for r in arm_records[a]}
                                for a in ARMS))
    print(f"  Queries answered in all three arms: {len(common)}")
    for a, b in combinations(ARMS, 2):
        xa, xb = paired_safety_outcomes(arm_records[a], arm_records[b])
        a_only = sum(x and not y for x, y in zip(xa, xb))
        b_only = sum(y and not x for x, y in zip(xa, xb))
        line = (f"  {a} vs {b}: {len(xa)} paired queries, "
                f"{a_only + b_only} discordant")
        if a_only + b_only == 0:
            line += " — McNemar not reported"
        else:
            who = ", ".join(f"{n} {arm} only"
                            for n, arm in ((a_only, a), (b_only, b)) if n)
            line += f" ({who}), exact p = {mcnemar_test(xa, xb).p_value:.3f}"
        print(line)


def report_rewrite_chains(runs: dict, arm_chains: dict) -> None:
    _banner("8.7 The rewrite chain (Table 11)")
    for arm in ARMS:
        for rec in runs[arm]:
            if not rec.get("attempts"):
                continue
            qid = rec["query_id"]
            labelled = {c["stage"]: c for c in arm_chains[arm]
                        if c["query_id"] == qid}
            verdicts = {v["stage"]: v for v in rec["verifications"]}
            print(f"  {arm} {qid}: {rec['outcome']} after "
                  f"{_rewrites(rec['attempts'])}")
            for d in rec["drafts"]:
                lab = labelled.get(d["stage"])
                v = verdicts.get(d["stage"])
                if v is None:
                    verdict = "not verified"
                elif v["all_supported"]:
                    verdict = "passed"
                else:
                    verdict = "flagged as " + ", ".join(sorted(
                        {c["verdict"] for c in v["claims"]
                         if c["verdict"] != "supported"}))
                text = "INSUFFICIENT_CONTEXT" if d.get("refused") else d["text"]
                print(f"    {d['stage']:10s} {lab['unit_id'] if lab else '-':5s} "
                      f"{lab['human_label'] if lab else 'unlabelled':10s} "
                      f"{verdict:24s} \"{text[:48]}...\"")

            print(f"\n    {qid} in each arm:")
            for a in ARMS:
                r = next(x for x in runs[a] if x["query_id"] == qid)
                same_draft = _initial_hash(r) == _initial_hash(rec)
                same_passages = r["retrieved_chunk_ids"] == rec["retrieved_chunk_ids"]
                print(f"      {a:13s} {r['outcome']:17s} initial draft "
                      f"{'identical' if same_draft else 'differs'}, passages "
                      f"{'identical' if same_passages else 'differ'}, "
                      f"{total_tokens(r):,} tokens, "
                      f"{recorded_latency_s(r):.2f} s")


def report_automated_scores(runs: dict, ragas: dict) -> None:
    _banner("8.8 Automated scoring (Tables 12 and 13)")
    declining = {a: [s for s in ragas[a].values()
                     if DECLINING_STATEMENT.search(s["answer"])] for a in ARMS}
    plain = {a: [s for s in ragas[a].values()
                 if not DECLINING_STATEMENT.search(s["answer"])] for a in ARMS}

    def relevance(scored):
        return f"{_mean([s['answer_relevancy'] for s in scored]):.3f}"

    print(_row("", ARMS))
    print(_row("Answers scored", [len(ragas[a]) for a in ARMS]))
    print(_row("Mean faithfulness",
               [f"{_mean([s['faithfulness'] for s in ragas[a].values()]):.3f}"
                for a in ARMS]))
    print(_row("Mean answer relevance",
               [relevance(ragas[a].values()) for a in ARMS]))
    print(_row("  without a declining statement",
               [relevance(plain[a]) for a in ARMS]))
    print(_row("  with a declining statement",
               [relevance(declining[a]) for a in ARMS]))
    print(_row("  number with a declining statement",
               [len(declining[a]) for a in ARMS]))
    withheld = {a: sum(r["outcome"] == OUTCOME_FALLBACK for r in runs[a])
                for a in ARMS}
    print(_row("Refusal rate (of all queries)",
               [f"{withheld[a] / len(runs[a]):.1%}" for a in ARMS]))

    common = sorted(set.intersection(*(set(ragas[a]) for a in ARMS)))
    print(f"\n  Table 13: paired faithfulness, exact Wilcoxon signed-rank, "
          f"on the {len(common)} queries scored in all three arms")
    for a, b in combinations(ARMS, 2):
        xa = [ragas[a][q]["faithfulness"] for q in common]
        xb = [ragas[b][q]["faithfulness"] for q in common]
        differing = sum(x != y for x, y in zip(xa, xb))
        if differing == 0:
            print(f"  {a} vs {b}: no pair differs; the test is undefined")
            continue
        res = wilcoxon_test(xa, xb)
        print(f"  {a} vs {b}: {differing} pair{'' if differing == 1 else 's'} "
              f"with different scores, W = {res.statistic:g}, "
              f"exact p = {res.p_value:.3f}")


def report_judge_validation(ragas: dict, arm_records: dict) -> None:
    _banner("8.9 Judge validation (Table 14)")
    label_of = {a: {r["query_id"]: r["human_label"] for r in arm_records[a]}
                for a in ARMS}
    cells = {a: Counter((f"{s['faithfulness']:.3f}", label_of[a][q])
                        for q, s in ragas[a].items()) for a in ARMS}
    print(_row("Faithfulness  Human label", ARMS))
    for score, lab in sorted(set().union(*cells.values())):
        print(_row(f"{score}         {lab}", [cells[a][(score, lab)] for a in ARMS]))
    print(_row("Total answers", [len(ragas[a]) for a in ARMS]))
    print(_row("Scored below 1.000",
               [sum(s["faithfulness"] < 1.0 for s in ragas[a].values())
                for a in ARMS]))

    print(f"\n  Length check: answers of {SHORT_ANSWER_WORDS} words or fewer, "
          f"and answers scoring zero")
    for a in ARMS:
        short = [s["faithfulness"] for s in ragas[a].values()
                 if len(s["answer"].split()) <= SHORT_ANSWER_WORDS]
        zero = [f"{q} ({len(s['answer'].split())} words, {label_of[a][q]})"
                for q, s in sorted(ragas[a].items()) if s["faithfulness"] == 0]
        print(f"    {a:13s} {len(short)} short, lowest score "
              f"{min(short):.3f}; scored zero: {', '.join(zero) or 'none'}")


def report_cost(runs: dict) -> None:
    _banner("8.10 Cost (Table 15)")
    s = {}
    for a in ARMS:
        recs = runs[a]
        tokens = [total_tokens(r) for r in recs]
        latency = [recorded_latency_s(r) for r in recs]
        first = [r["t_verification_s"][0] for r in recs if r["t_verification_s"]]
        s[a] = {
            "total": sum(tokens),
            "mean": sum(tokens) / len(tokens),
            "median": statistics.median(latency),
            "generation": statistics.median([r["t_generation_s"] for r in recs]),
            "first": statistics.median(first) if first else None,
            "p95": nearest_rank_percentile(latency, 95),
            "max": max(latency),
        }
    base = s[ARM_BASELINE]

    print(_row("", ARMS))
    print(_row("Total tokens", [f"{s[a]['total']:,}" for a in ARMS]))
    print(_row("Mean tokens per query", [f"{s[a]['mean']:,.0f}" for a in ARMS]))
    print(_row("Token use relative to baseline",
               ["—" if a == ARM_BASELINE else f"{s[a]['total'] / base['total']:.2f}×"
                for a in ARMS]))
    print(_row("Median latency", [f"{s[a]['median']:.2f} s" for a in ARMS]))
    print(_row("Median generation", [f"{s[a]['generation']:.2f} s" for a in ARMS]))
    print(_row("Median first-verification-call",
               ["—" if s[a]["first"] is None else f"{s[a]['first']:.2f} s"
                for a in ARMS]))
    print(_row("95th percentile (nearest rank)",
               [f"{s[a]['p95']:.2f} s" for a in ARMS]))
    print(_row("Maximum", [f"{s[a]['max']:.2f} s" for a in ARMS]))

    print()
    for a in ARMS:
        if a == ARM_BASELINE:
            continue
        print(f"  {a} vs baseline: {s[a]['mean'] - base['mean']:+.0f} tokens per "
              f"query ({s[a]['total'] / base['total'] - 1:+.0%}), median "
              f"latency {s[a]['median'] / base['median'] - 1:+.0%}")
    medians = [s[a]["generation"] for a in ARMS]
    print(f"  Generation medians span {max(medians) - min(medians):.2f} s "
          f"across arms")


def report_stability(runs: dict, repeats: dict) -> None:
    _banner("8.11 Run-to-run stability (Table 16)")
    print("  Runs compared, in order (labelled run first):")
    for a in ARMS:
        print(f"    {a:13s} {', '.join([LABELLED_RUNS[a], *REPEAT_RUNS[a]])}")

    outcome_row, draft_row, latency_row, labelled_row, changed = [], [], [], [], []
    for a in ARMS:
        by_query = [{r["query_id"]: r for r in recs}
                    for recs in [runs[a], *repeats[a]]]
        queries = sorted(set.intersection(*(set(b) for b in by_query)))
        n = len(queries)
        same_outcome = [q for q in queries
                        if len({b[q]["outcome"] for b in by_query}) == 1]
        same_draft = [q for q in queries
                      if len({_initial_hash(b[q]) for b in by_query}) == 1]
        medians = [statistics.median([recorded_latency_s(r) for r in b.values()])
                   for b in by_query]
        outcome_row.append(f"{len(same_outcome)}/{n} ({len(same_outcome) / n:.1%})")
        draft_row.append(f"{len(same_draft)}/{n} ({len(same_draft) / n:.1%})")
        latency_row.append(f"{min(medians):.2f}–{max(medians):.2f} s")
        labelled_row.append(f"{medians[0]:.2f} s")
        changed += [(a, q, [b[q]["outcome"] for b in by_query])
                    for q in queries if q not in same_outcome]

    print()
    print(_row("", ARMS, width=40, cell=17))
    print(_row("Outcome identical across 3 runs", outcome_row, width=40, cell=17))
    print(_row("Generator draft identical across 3 runs", draft_row, width=40, cell=17))
    print(_row("Median latency, range across 3 runs", latency_row, width=40, cell=17))
    print(_row("  median of the labelled run", labelled_row, width=40, cell=17))

    print("\n  Queries whose outcome changed (labelled / repeat 1 / repeat 2):")
    for a, q, outcomes in changed:
        print(f"    {a:13s} {q:7s} {' / '.join(outcomes)}")

    print("\n  Case queries across all nine runs:")
    for qid, claim in CASE_QUERIES.items():
        per_arm = {a: [next(r for r in recs if r["query_id"] == qid)
                       for recs in [runs[a], *repeats[a]]] for a in ARMS}
        nine = [r for a in ARMS for r in per_arm[a]]
        same_draft = len({_initial_hash(r) for r in nine}) == 1
        same_passages = len({tuple(r["retrieved_chunk_ids"]) for r in nine}) == 1
        print(f"    {qid}: initial draft identical in all nine runs: "
              f"{'yes' if same_draft else 'no'}; retrieved passages identical: "
              f"{'yes' if same_passages else 'no'}")
        for a in ARMS:
            print(f"      {a:13s} " + " / ".join(
                r["outcome"]
                + (f" ({_rewrites(r['attempts'])})" if r["attempts"] else "")
                for r in per_arm[a]))
        if claim:
            print(f"      initial draft contains '{claim}':")
            for a in ARMS:
                print(f"      {a:13s} " + " / ".join(
                    "yes" if r["drafts"]
                    and claim.lower() in r["drafts"][0]["text"].lower()
                    else "no" for r in per_arm[a]))


# ---------------------------------------------------------------------------

def main() -> None:
    labels = load_labels()
    mapping = json.loads((Path(RESULTS_DIR) / "labelling_mapping.json")
                         .read_text(encoding="utf-8"))["mapping"]

    metas, runs, repeats = {}, {}, {}
    for arm in ARMS:
        metas[arm], runs[arm] = load_run(LABELLED_RUNS[arm], arm)
        check_against_mapping(arm, runs[arm], mapping)
        repeats[arm] = [load_run(name, arm)[1] for name in REPEAT_RUNS[arm]]
    ragas = {arm: load_ragas(arm, runs[arm]) for arm in ARMS}

    print(f"Loaded {len(labels)} labels covering {len(mapping)} records")
    print("Labelled runs, checked against labelling_mapping.json:")
    for arm in ARMS:
        print(f"  {arm:13s} {LABELLED_RUNS[arm]}")

    arm_records, arm_chains = {}, {}
    for arm in ARMS:
        arm_records[arm], arm_chains[arm] = build_arm_records(
            arm, runs[arm], labels, mapping)

    report_execution(runs, repeats)
    report_outcomes(runs)
    report_retrieval(runs, metas)
    report_labelling(labels, mapping, arm_records)
    _banner("8.5 Confusion matrices and rates (Table 9)")
    for arm in ARMS:
        report_arm(arm, arm_records[arm])
    report_paired(arm_records)
    report_rewrite_chains(runs, arm_chains)
    report_automated_scores(runs, ragas)
    report_judge_validation(ragas, arm_records)
    report_cost(runs)
    report_stability(runs, repeats)


if __name__ == "__main__":
    main()
