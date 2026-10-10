"""
Human labelling support: builds the deduplicated, blinded CSV a labeller
fills in, and reads completed labels back for analysis.

WHY A CSV RATHER THAN A CLI OR WEB UI: the second reviewer needs to label a
sample without running any of this code. A spreadsheet can be emailed,
filled offline, and returned — no environment setup, no coordination
overhead. It also doubles as the archived labelling record.

DEDUPLICATION (verified 8.8.2026): the three arms share a generator, prompt
and retrieval configuration, so the initial draft is byte-identical across
arms for most queries (96/108 identical in all three; 101/108 baseline vs
cross_vendor). Each DISTINCT draft text is therefore labelled ONCE and the
label applied to every record whose draft hash matches. This is not merely
an efficiency saving: presenting the same text three times would not produce
three independent judgements (the labeller may recognise it and anchor), and
labelling the same text on separate occasions risks inconsistent labels on
borderline cases, which would propagate into the confusion matrix as an
apparent arm difference. Identity is verified by SHA-256, never assumed —
commercial APIs are not perfectly deterministic even at temperature 0
(ADV-13, 28.7.2026).

BLINDING: the labeller must not see the arm, the verifier's verdict, or the
outcome (Labelling Guide v3). The CSV therefore carries an opaque unit_id,
the query, the answer text and the retrieved passages — nothing else. Rows
are shuffled with RANDOM_SEED so the order is reproducible but carries no
information. The unit_id -> (arm, query_id, stage, hash) mapping is written
to a SEPARATE file which must not be opened during labelling.

The primary labeller is not truly blind (these answers were read during
development). See NOTES 9.8.2026 for the limitation and mitigations; the
second reviewer's sample is the genuinely blind reliability check.

WHAT GETS LABELLED: every distinct draft text, including intermediate
rewrite drafts (they are genuinely different text and the rewrite chain is
itself of interest). Records whose outcome is fallback, generation_failed or
verifier_malformed have no answer to judge and are excluded — but a
WITHHELD draft (one the verifier blocked) IS labelled, because the FP cell
of the confusion matrix depends on knowing whether a blocked answer was
actually faithful.

Run:
    uv run python -m analysis.labelling build      # write the CSV + mapping
    uv run python -m analysis.labelling sample     # draw 2nd-reviewer sample
    uv run python -m analysis.labelling read       # parse completed labels
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

from src.config import CORPUS_DIR, RESULTS_DIR
from src.constants import (
    RANDOM_SEED,
    VALID_LABELS,
)

# Outcomes with no answer to judge. Note `fallback` is excluded here but its
# WITHHELD DRAFT is still labelled — see _collect_units.
_NO_ANSWER_OUTCOMES = ("generation_failed", "verifier_malformed")

# Second-reviewer sample: the proposal specifies 15-20%. At the actual
# deduplicated total (~68) that yields only 10-14 items, too few for a stable
# Cohen's kappa, so an absolute floor is applied. Documented deviation — see
# NOTES 9.8.2026.
_REVIEWER_FRACTION = 0.20
_REVIEWER_MINIMUM = 20


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


_corpus_cache: dict[str, str] = {}


def _passages_for(chunk_ids: list[str]) -> str:
    """Reconstruct the passage block the labeller reads. Numbered as the
    model saw them; faq_id is NOT shown, since it is the answer key."""
    out = []
    for i, cid in enumerate(chunk_ids, start=1):
        faq_id = cid.split("::")[0]
        if faq_id not in _corpus_cache:
            _corpus_cache[faq_id] = (
                Path(CORPUS_DIR) / f"{faq_id}.md").read_text(encoding="utf-8").strip()
        out.append(f"[{i}] {_corpus_cache[faq_id]}")
    return "\n\n".join(out)


def _find_results_files() -> list[Path]:
    """All pipeline result files (not ragas_* scoring files). The folder must
    hold only the runs to be labelled when build() runs: it now also holds
    the repeat runs, which were never labelled."""
    return sorted(p for p in Path(RESULTS_DIR).glob("*.jsonl")
                  if not p.name.startswith("ragas_"))


def _refuse_if_labelled(path: Path) -> None:
    """Never overwrite a sheet that already holds labels: they are the
    archived labelling record, and rebuilding cannot recreate them."""
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        if any((row.get("label") or "").strip() for row in csv.DictReader(f)):
            raise SystemExit(f"{path} already holds labels and would be "
                             f"overwritten. Move it aside first.")


def _collect_units() -> tuple[list[dict], list[dict]]:
    """
    Returns (units, mapping).

    units: one per DISTINCT draft text — what the labeller sees.
    mapping: every (arm, query_id, stage) that shares each unit's hash, so
             labels can be re-attached to all matching records afterwards.
    """
    by_hash: dict[str, dict] = {}
    mapping: list[dict] = []

    for path in _find_results_files():
        meta, records = _load_jsonl(path)
        arm = meta["active_arm"]

        for rec in records:
            if rec["outcome"] in _NO_ANSWER_OUTCOMES:
                continue
            for draft in rec.get("drafts", []):
                # A refusal sentinel makes no factual claim; the fallback
                # message shown in its place is fixed text, not generated.
                if draft.get("refused"):
                    continue

                h = draft["sha256"]
                if h not in by_hash:
                    by_hash[h] = {
                        "sha256": h,
                        "question": None,     # filled below from questions.json
                        "query_id": rec["query_id"],
                        "answer": draft["text"],
                        "passages": _passages_for(rec["retrieved_chunk_ids"]),
                    }
                mapping.append({
                    "sha256": h,
                    "arm": arm,
                    "query_id": rec["query_id"],
                    "stage": draft["stage"],
                    "outcome": rec["outcome"],
                })

    return list(by_hash.values()), mapping


def build() -> tuple[Path, Path]:
    """Write the blinded labelling CSV and the separate mapping file."""
    from src.config import QUESTIONS_FILE

    _refuse_if_labelled(Path(RESULTS_DIR) / "labelling_sheet.csv")

    questions = {q["id"]: q["text"] for q in
                 json.loads(Path(QUESTIONS_FILE).read_text())["queries"]}

    units, mapping = _collect_units()
    for u in units:
        u["question"] = questions[u["query_id"]]

    # Seeded shuffle: reproducible, but the order carries no information.
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(units)

    # Opaque ids assigned AFTER shuffling, so the id ordering reveals nothing.
    for i, u in enumerate(units, start=1):
        u["unit_id"] = f"U{i:03d}"

    out_dir = Path(RESULTS_DIR)
    csv_path = out_dir / "labelling_sheet.csv"
    map_path = out_dir / "labelling_mapping.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["unit_id", "question", "answer", "retrieved_passages",
                    "label", "notes"])
        for u in units:
            w.writerow([u["unit_id"], u["question"], u["answer"],
                        u["passages"], "", ""])

    # unit_id must be attached to the mapping too, via the shared hash.
    hash_to_unit = {u["sha256"]: u["unit_id"] for u in units}
    for m in mapping:
        m["unit_id"] = hash_to_unit.get(m["sha256"])

    map_path.write_text(json.dumps(
        {"seed": RANDOM_SEED,
         "valid_labels": list(VALID_LABELS),
         "n_units": len(units),
         "n_records_covered": len(mapping),
         "mapping": mapping},
        indent=2), encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"  {len(units)} distinct draft texts to label")
    print(f"  covering {len(mapping)} records across all arms")
    print(f"  ({len(mapping) - len(units)} labels saved by deduplication)")
    print(f"Wrote {map_path}  — DO NOT OPEN THIS DURING LABELLING")
    print(f"\nValid labels: {', '.join(VALID_LABELS)}")
    return csv_path, map_path


def sample() -> Path:
    """Draw the second reviewer's sample from the built sheet."""
    csv_path = Path(RESULTS_DIR) / "labelling_sheet.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))

    n = max(_REVIEWER_MINIMUM, round(len(rows) * _REVIEWER_FRACTION))
    n = min(n, len(rows))

    rng = random.Random(RANDOM_SEED)   # seeded: the draw is reconstructible
    chosen = rng.sample(rows, n)

    out = Path(RESULTS_DIR) / "labelling_sheet_reviewer.csv"
    _refuse_if_labelled(out)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        # Blank label and notes, so the reviewer never sees the researcher's
        # judgement, even if the sample is drawn after labelling.
        w.writerows({**r, **{k: "" for k in ("label", "notes") if k in r}}
                    for r in chosen)

    print(f"Wrote {out}")
    print(f"  {n}/{len(rows)} units ({n/len(rows):.0%}) — "
          f"{_REVIEWER_FRACTION:.0%} or {_REVIEWER_MINIMUM} minimum, "
          f"whichever is larger")
    return out


def read(sheet: str | Path | None = None) -> list[dict]:
    """
    Parse a completed sheet and expand labels back onto every record the
    unit covers. Validates labels against VALID_LABELS and refuses to
    silently accept typos.
    """
    sheet = Path(sheet) if sheet else Path(RESULTS_DIR) / "labelling_sheet.csv"
    mapping = json.loads(
        (Path(RESULTS_DIR) / "labelling_mapping.json").read_text())

    labels: dict[str, str] = {}
    blank = 0
    for row in csv.DictReader(sheet.open(encoding="utf-8")):
        lab = (row.get("label") or "").strip()
        if not lab:
            blank += 1
            continue
        if lab not in VALID_LABELS:
            raise ValueError(
                f"{row['unit_id']}: unknown label {lab!r}. "
                f"Expected one of {VALID_LABELS}")
        labels[row["unit_id"]] = lab

    expanded = []
    for m in mapping["mapping"]:
        if m["unit_id"] in labels:
            expanded.append({**m, "human_label": labels[m["unit_id"]]})

    print(f"Read {len(labels)} labels ({blank} blank) from {sheet.name}")
    print(f"  expanded to {len(expanded)} labelled records across arms")
    return expanded


def _main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "build":
        build()
    elif cmd == "sample":
        sample()
    elif cmd == "read":
        read(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())