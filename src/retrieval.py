"""
Ingestion and retrieval: corpus -> chunks -> embeddings -> ChromaDB, and
top-k retrieval at query time.

THREE THINGS THIS MODULE GUARDS AGAINST (each cost real debugging elsewhere):

1. SILENT TRUNCATION. Chunk size is measured in the EMBEDDER'S OWN word-piece
   tokens, not characters or whitespace words. all-MiniLM-L6-v2 truncates at
   max_seq_length with no error. Every chunk is re-encoded WITH special tokens
   and asserted to fit before it is stored — if this assertion ever fires, the
   chunker is wrong, not the model.

2. A SECOND EMBEDDING MODEL. ChromaDB ships its own ONNX copy of
   all-MiniLM-L6-v2 and uses it by default when you add documents without
   embeddings. That is NOT the pinned SentenceTransformer instance. Embeddings
   are therefore computed here and passed explicitly, so exactly one model
   touches the corpus and the queries.

3. A STALE INDEX. The index is only valid for the settings that built it.
   A fingerprint of (embedding model, chunk size, overlap, corpus content) is
   stored with the collection and checked on every retrieval. Change any of
   them and retrieval REFUSES to run rather than quietly querying an index
   built under different rules.

Chunking is PER FAQ: each FAQ is chunked independently, so a chunk never
spans two FAQs and every chunk has an unambiguous faq_id. This matches how
ground truth is defined (target_faq per query) and keeps the retrieval-recall
check meaningful.

Run as a script:
    uv run python -m src.retrieval ingest          # build the index
    uv run python -m src.retrieval ingest --force  # rebuild from scratch
    uv run python -m src.retrieval recall          # recall@k on in-domain queries
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import (
    CHROMA_COLLECTION,
    CHROMA_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CORPUS_DIR,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    QUESTIONS_FILE,
    RETRIEVAL_K,
    SIMILARITY_METRIC,
)

# Key that carries the fingerprint in the collection's metadata.
_FINGERPRINT_KEY = "config_fingerprint"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Passage:
    """One retrieved chunk. `chunk_id` is what the controller logs as part of
    retrieved_chunk_ids; `faq_id` is what the recall check and the
    retrieval-miss attribution use."""
    chunk_id: str
    faq_id: str
    text: str
    distance: float


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_corpus(corpus_dir: str | Path = CORPUS_DIR) -> list[tuple[str, str]]:
    """
    Returns [(faq_id, text), ...] sorted by faq_id.

    faq_id comes from the FILENAME (faq_01.md -> "faq_01"), so the id is
    unambiguous and matches the target_faq references in the query set.
    """
    path = Path(corpus_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {path}")

    files = sorted(p for p in path.iterdir()
                   if p.suffix in {".md", ".txt"} and not p.name.startswith("."))
    if not files:
        raise FileNotFoundError(f"no .md/.txt files in {path}")

    corpus = []
    for f in files:
        text = f.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"{f.name} is empty")
        corpus.append((f.stem, text))
    return corpus


# ---------------------------------------------------------------------------
# Chunking — in the embedder's own tokens
# ---------------------------------------------------------------------------

def _effective_chunk_size(model: SentenceTransformer) -> int:
    """
    The true content budget per chunk.

    CHUNK_SIZE is the configured target, but the model's max_seq_length
    INCLUDES the special tokens it adds ([CLS], [SEP]). A 256-token chunk
    would become 258 after encoding and lose its tail. Reserve them.
    """
    max_len = model.max_seq_length
    reserved = model.tokenizer.num_special_tokens_to_add(pair=False)
    budget = max_len - reserved

    if CHUNK_SIZE > budget:
        print(f"  NOTE: CHUNK_SIZE={CHUNK_SIZE} exceeds the model's content "
              f"budget ({max_len} max_seq_length - {reserved} special tokens "
              f"= {budget}). Using {budget}.")
        return budget
    return CHUNK_SIZE


def chunk_text(text: str, model: SentenceTransformer) -> list[str]:
    """
    Token-window chunking with overlap, measured in the embedder's word-piece
    tokens. Returns a list of chunk strings.

    Decoding tokens back to text is not perfectly lossless (spacing around
    punctuation can shift). That is acceptable — what matters is that the
    stored text and the embedded text are the SAME string, which they are,
    because the embedding is computed from this returned text.
    """
    tok = model.tokenizer
    size = _effective_chunk_size(model)
    stride = size - CHUNK_OVERLAP
    if stride <= 0:
        raise ValueError(
            f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) must be smaller than the "
            f"effective chunk size ({size}); otherwise the window never advances")

    ids = tok.encode(text, add_special_tokens=False)

    if len(ids) <= size:
        return [text]                      # fits whole; keep the original string

    chunks = []
    start = 0
    while start < len(ids):
        window = ids[start:start + size]
        chunks.append(tok.decode(window, skip_special_tokens=True).strip())
        if start + size >= len(ids):
            break
        start += stride
    return chunks


def _assert_no_truncation(chunks: list[str], model: SentenceTransformer,
                          faq_id: str) -> None:
    """
    Belt and braces: re-encode each chunk exactly as the model will (special
    tokens included) and confirm it fits. If this fires, the chunker is wrong
    — which is precisely the failure that would otherwise be silent.
    """
    tok = model.tokenizer
    limit = model.max_seq_length
    for i, c in enumerate(chunks):
        n = len(tok.encode(c, add_special_tokens=True))
        if n > limit:
            raise AssertionError(
                f"{faq_id} chunk {i}: {n} tokens with special tokens exceeds "
                f"max_seq_length {limit} — it would be SILENTLY TRUNCATED. "
                f"Chunker is wrong.")


# ---------------------------------------------------------------------------
# Fingerprint — what makes an index valid
# ---------------------------------------------------------------------------

def config_fingerprint(corpus: list[tuple[str, str]]) -> str:
    """
    Hash of everything that determines the index's contents. Change any of it
    and the stored vectors no longer correspond to the current settings.

    NOTE: RETRIEVAL_K is deliberately EXCLUDED. k is a query-time parameter;
    changing it does not invalidate the index, and including it would force a
    pointless re-embed during the k pilot.
    """
    h = hashlib.sha256()
    h.update(EMBEDDING_MODEL.encode())
    h.update(str(CHUNK_SIZE).encode())
    h.update(str(CHUNK_OVERLAP).encode())
    h.update(SIMILARITY_METRIC.encode())
    for faq_id, text in corpus:                 # corpus is sorted by load_corpus
        h.update(faq_id.encode())
        h.update(text.encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest(force: bool = False) -> dict:
    """
    Chunk -> embed -> store. Returns a summary dict (also printed).

    Logs chunks per FAQ, which is the standing verification of the
    "1 FAQ = 1 chunk" assumption that the k reasoning rests on.
    """
    corpus = load_corpus()
    fingerprint = config_fingerprint(corpus)

    print(f"Corpus: {len(corpus)} FAQs from {CORPUS_DIR}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Fingerprint: {fingerprint}")

    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"max_seq_length: {model.max_seq_length}  "
          f"(CHUNK_SIZE={CHUNK_SIZE}, CHUNK_OVERLAP={CHUNK_OVERLAP})")

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    existing = [c.name for c in client.list_collections()]
    if CHROMA_COLLECTION in existing:
        if not force:
            coll = client.get_collection(CHROMA_COLLECTION)
            stored = (coll.metadata or {}).get(_FINGERPRINT_KEY)
            if stored == fingerprint:
                print(f"\nIndex already current ({coll.count()} chunks). "
                      f"Nothing to do. Use --force to rebuild.")
                return {"status": "current", "fingerprint": fingerprint,
                        "chunks": coll.count()}
            print(f"\nFingerprint changed ({stored} -> {fingerprint}); "
                  f"rebuilding.")
        client.delete_collection(CHROMA_COLLECTION)

    # hnsw:space MUST be set at creation and cannot be changed afterwards.
    # Chroma's default is l2 — set cosine explicitly (NOTES 12.7.2026).
    coll = client.create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": SIMILARITY_METRIC, _FINGERPRINT_KEY: fingerprint},
    )

    ids, docs, metas, per_faq = [], [], [], {}

    for faq_id, text in corpus:
        chunks = chunk_text(text, model)
        _assert_no_truncation(chunks, model, faq_id)
        per_faq[faq_id] = len(chunks)
        for i, c in enumerate(chunks):
            ids.append(f"{faq_id}::{i}")
            docs.append(c)
            metas.append({"faq_id": faq_id, "chunk_index": i,
                          "source_file": f"{faq_id}.md"})

    # Embeddings computed HERE and passed explicitly, so Chroma's own ONNX
    # copy of the model is never used. See the module docstring.
    embeddings = model.encode(docs, show_progress_bar=False).tolist()

    if embeddings and len(embeddings[0]) != EMBEDDING_DIM:
        raise AssertionError(
            f"embedding dim {len(embeddings[0])} != EMBEDDING_DIM "
            f"{EMBEDDING_DIM} in config")

    coll.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)

    # --- chunks per FAQ: the standing "1 FAQ = 1 chunk" check ---------------
    print(f"\nChunks per FAQ ({len(ids)} chunks total):")
    multi = {f: n for f, n in per_faq.items() if n > 1}
    for faq_id, n in per_faq.items():
        flag = "  <-- SPLIT" if n > 1 else ""
        print(f"  {faq_id}: {n}{flag}")

    if multi:
        print(f"\n  {len(multi)} FAQ(s) split into multiple chunks: "
              f"{', '.join(multi)}")
        print("  The '1 FAQ = 1 chunk' assumption does NOT hold. Revisit the "
              "k reasoning in NOTES 12.7.2026 — a split FAQ needs more than "
              "one retrieved chunk to be fully represented.")
    else:
        print("\n  1 FAQ = 1 chunk holds for all FAQs. "
              "(OPEN ITEM verified — record in NOTES.)")

    return {"status": "built", "fingerprint": fingerprint,
            "faqs": len(corpus), "chunks": len(ids), "chunks_per_faq": per_faq}


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

_model: SentenceTransformer | None = None
_collection = None


def _get_collection():
    """Open the collection and refuse to proceed on a stale index."""
    global _collection
    if _collection is not None:
        return _collection

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        coll = client.get_collection(CHROMA_COLLECTION)
    except Exception as exc:
        raise RuntimeError(
            f"collection {CHROMA_COLLECTION!r} not found at {CHROMA_PATH}. "
            f"Run: uv run python -m src.retrieval ingest") from exc

    stored = (coll.metadata or {}).get(_FINGERPRINT_KEY)
    current = config_fingerprint(load_corpus())
    if stored != current:
        raise RuntimeError(
            f"STALE INDEX. Built under fingerprint {stored}, current config "
            f"is {current}. The embedding model, chunk settings, similarity "
            f"metric, or corpus has changed. Re-ingest before retrieving:\n"
            f"    uv run python -m src.retrieval ingest --force")

    _collection = coll
    return coll


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def retrieve(query: str, k: int = RETRIEVAL_K) -> list[Passage]:
    """
    Top-k passages for a query, nearest first.

    The query is embedded with the SAME model instance used at ingestion —
    not Chroma's internal copy — so query and document vectors are
    comparable by construction.
    """
    coll = _get_collection()
    q_emb = _get_model().encode([query]).tolist()

    res = coll.query(
        query_embeddings=q_emb,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    return [
        Passage(chunk_id=cid,
                faq_id=meta["faq_id"],
                text=doc,
                distance=dist)
        for cid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0],
            res["metadatas"][0], res["distances"][0])
    ]


# ---------------------------------------------------------------------------
# Retrieval recall check — no LLM calls, no cost
# ---------------------------------------------------------------------------

def recall_at_k(queries: list[dict], k_values: tuple[int, ...] = (3, 5, 8, 10)
                ) -> dict[int, float]:
    """
    For each in-domain query, is at least one target FAQ present in the top-k?

    queries: [{"id": ..., "text": ..., "target_faqs": ["faq_01", ...]}, ...]

    This validates RETRIEVAL only — it needs no LLM and costs nothing, so it
    can be run against draft queries before the query set is finalised.
    Misses are printed individually: they are the interesting rows.
    """
    results = {}
    for k in k_values:
        hits, misses = 0, []
        for q in queries:
            passages = retrieve(q["text"], k=k)
            found = {p.faq_id for p in passages}
            if found & set(q["target_faqs"]):
                hits += 1
            else:
                misses.append((q["id"], q["text"], sorted(found)))

        results[k] = hits / len(queries) if queries else float("nan")
        print(f"\nrecall@{k}: {hits}/{len(queries)} = {results[k]:.1%}")
        for qid, text, found in misses:
            print(f"  MISS {qid}: {text[:60]}...")
            print(f"       retrieved {found}")

    return results


def load_indomain_queries(path: str | Path = QUESTIONS_FILE) -> list[dict]:
    """
    Read in-domain queries from questions.json.

    Expected shape per query: {"id", "text", "category", "target_faq"}.
    target_faq may be a string or a list; both are normalised to a list.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Until the query set is finalised, pass a list of "
            f"dicts to recall_at_k() directly.")

    data = json.loads(p.read_text(encoding="utf-8"))
    queries = data["queries"] if isinstance(data, dict) else data

    out = []
    for q in queries:
        if q.get("category") != "in_domain":
            continue
        target = q.get("target_faq")
        if target is None:
            raise ValueError(f"in-domain query {q['id']} has no target_faq")
        out.append({"id": q["id"], "text": q["text"],
                    "target_faqs": [target] if isinstance(target, str) else list(target)})
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1

    cmd = args[0]

    if cmd == "ingest":
        ingest(force="--force" in args)
        return 0

    if cmd == "recall":
        queries = load_indomain_queries()
        print(f"Recall check on {len(queries)} in-domain queries "
              f"(no LLM calls, no cost)")
        recall_at_k(queries)
        return 0

    print(f"unknown command {cmd!r}; expected 'ingest' or 'recall'")
    return 1


if __name__ == "__main__":
    raise SystemExit(_main())