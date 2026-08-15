"""
Final analysis: joins human labels to pipeline outcomes, builds the confusion
matrix for each arm, and computes the reported rates and tests.

WHAT THIS JOINS. Three sources, all already on disk:
  labelling_sheet.csv        the human label per distinct draft text
  labelling_mapping.json     which (arm, query, stage) each label covers
  {arm}_{ts}.jsonl           what the system actually did
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

import json
from collections import Counter
from pathlib import Path

from analysis.metrics import (
    build_confusion_matrix,
    false_negative_rate,
    false_positive_rate,
    paired_safety_outcomes,
    strict_compliance_rate,
)
from analysis.stats import mcnemar_test
from src.config import RESULTS_DIR
from src.constants import (
    EXCLUDED_OUTCOMES,
    HALLUCINATION_LABELS,
    VALID_LABELS,
)


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


def _results_files() -> dict[str, Path]:
    """arm -> most recent pipeline results file for that arm."""
    out = {}
    for p in sorted(Path(RESULTS_DIR).glob("*.jsonl")):
        if p.name.startswith(("ragas_", "labelling")):
            continue
        meta, _ = _load_jsonl(p)
        out[meta["active_arm"]] = p          # later timestamps overwrite earlier
    return out


def load_labels() -> dict[str, str]:
    """unit_id -> label, straight from the completed sheet."""
    import csv
    path = Path(RESULTS_DIR) / "labelling_sheet.csv"
    labels = {}
    for row in csv.DictReader(path.open(encoding="utf-8")):
        lab = (row.get("label") or "").strip()
        if not lab:
            continue
        if lab not in VALID_LABELS:
            raise ValueError(f"{row['unit_id']}: unknown label {lab!r}")
        labels[row["unit_id"]] = lab
    return labels


def build_arm_records(arm: str, path: Path, labels: dict[str, str],
                      mapping: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Returns (matrix_records, chain_records).

    matrix_records: one per query, carrying the FINAL draft's label and the
                    query's outcome — the input to the confusion matrix.
    chain_records:  every labelled draft, including intermediates, for the
                    separate rewrite-chain reporting.
    """
    _, results = _load_jsonl(path)

    # (arm, query_id, stage) -> unit_id, from the mapping written at build time
    unit_of = {(m["arm"], m["query_id"], m["stage"]): m["unit_id"]
               for m in mapping}

    matrix_records, chain_records = [], []

    for rec in results:
        qid, outcome = rec["query_id"], rec["outcome"]

        labelled_drafts = []
        for draft in rec.get("drafts", []):
            uid = unit_of.get((arm, qid, draft["stage"]))
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


def report_arm(arm: str, records: list[dict]) -> None:
    cm = build_confusion_matrix(records)
    scr = strict_compliance_rate(cm)
    fnr = false_negative_rate(cm)
    fpr = false_positive_rate(cm)

    print(f"\n{'='*66}")
    print(f"{arm}")
    print(f"{'='*66}")
    print(f"  labelled queries in matrix: {cm.total}   excluded: {cm.excluded}")
    print(f"\n  confusion matrix")
    print(f"                        hallucinated    faithful")
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
    print(f"  by category:")
    for cat, c in sorted(by_cat.items()):
        print(f"    {cat}: {dict(c)}")


def main() -> None:
    labels = load_labels()
    mapping = json.loads(
        (Path(RESULTS_DIR) / "labelling_mapping.json").read_text())["mapping"]
    files = _results_files()

    print(f"Loaded {len(labels)} labels covering {len(mapping)} records")
    print(f"Arms found: {', '.join(sorted(files))}")

    arm_records, arm_chains = {}, {}
    for arm, path in sorted(files.items()):
        m, c = build_arm_records(arm, path, labels, mapping)
        arm_records[arm], arm_chains[arm] = m, c
        report_arm(arm, m)

    # ---- rewrite chains, reported separately -----------------------------
    print(f"\n{'='*66}")
    print("REWRITE CHAINS (labels on intermediate drafts)")
    print(f"{'='*66}")
    for arm, chain in sorted(arm_chains.items()):
        multi = {}
        for r in chain:
            multi.setdefault(r["query_id"], []).append(r)
        multi = {q: rs for q, rs in multi.items() if len(rs) > 1}
        if not multi:
            print(f"  {arm}: no query produced more than one labelled draft")
            continue
        for q, rs in sorted(multi.items()):
            print(f"  {arm} {q} (outcome={rs[0]['outcome']}):")
            for r in rs:
                print(f"      {r['stage']:12s} {r['unit_id']}  {r['human_label']}")

    # ---- paired comparison ------------------------------------------------
    if "baseline" in arm_records and "cross_vendor" in arm_records:
        print(f"\n{'='*66}")
        print("PAIRED COMPARISON: baseline vs cross_vendor")
        print(f"{'='*66}")
        a, b = paired_safety_outcomes(arm_records["baseline"],
                                      arm_records["cross_vendor"])
        print(f"  paired queries: {len(a)}")
        print(f"  hallucination reached user — baseline: {sum(a)}, "
              f"cross_vendor: {sum(b)}")
        discordant = sum(1 for x, y in zip(a, b) if x != y)
        if discordant == 0:
            print("  no discordant pairs: McNemar is undefined and is not "
                  "reported.")
        else:
            print(f"  {mcnemar_test(a, b).summary()}")

    # ---- overall label counts --------------------------------------------
    print(f"\n{'='*66}")
    print("ALL LABELS (by distinct draft, not by record)")
    print(f"{'='*66}")
    print(f"  {dict(Counter(labels.values()))}")
    halluc = [u for u, l in labels.items() if l in HALLUCINATION_LABELS]
    print(f"  hallucinated drafts: {len(halluc)} — {sorted(halluc)}")


if __name__ == "__main__":
    main()