"""
Thin wrapper around ChromaDB (embedded, on-disk, free) using a local
sentence-transformers model for embeddings (also free, runs on CPU).

Multi-tenancy is done with one Chroma *collection* per org, named
`org_<org_id>`, so one org's documents never leak into another org's
retrieval results.
"""
from __future__ import annotations
import chromadb
from chromadb.utils import embedding_functions

from config import CHROMA_DIR, EMBEDDING_MODEL
from ingestion import Chunk

_client = chromadb.PersistentClient(path=CHROMA_DIR)
_embed_fn = None  # lazily created on first use, so importing this module never triggers a model download


def _get_embed_fn():
    global _embed_fn
    if _embed_fn is None:
        _embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    return _embed_fn


def _collection_name(org_id: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in org_id.lower())
    return f"org_{safe}"


def get_collection(org_id: str):
    return _client.get_or_create_collection(
        name=_collection_name(org_id),
        embedding_function=_get_embed_fn(),
    )


def add_chunks(org_id: str, chunks: list[Chunk]) -> int:
    """Embed and store chunks for an org. Returns number of chunks added."""
    if not chunks:
        return 0
    collection = get_collection(org_id)
    ids = [f"{c.source}::{c.chunk_index}" for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [{"source": c.source, "doc_type": c.doc_type} for c in chunks]
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(chunks)


def query(org_id: str, question: str, n_results: int = 5, doc_type: str | None = None) -> list[dict]:
    """Retrieve the most relevant chunks for a query, optionally filtered by doc_type."""
    collection = get_collection(org_id)
    if collection.count() == 0:
        return []

    where = {"doc_type": doc_type} if doc_type else None
    results = collection.query(
        query_texts=[question],
        n_results=min(n_results, collection.count()),
        where=where,
    )

    hits = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        hits.append({"text": doc, "source": meta.get("source"), "doc_type": meta.get("doc_type"), "distance": dist})
    return hits


def list_sources(org_id: str) -> list[str]:
    collection = get_collection(org_id)
    if collection.count() == 0:
        return []
    metas = collection.get()["metadatas"]
    return sorted({m["source"] for m in metas})


def delete_org(org_id: str):
    _client.delete_collection(_collection_name(org_id))
