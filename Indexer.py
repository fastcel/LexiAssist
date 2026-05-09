"""
indexer.py — Build FAISS + BM25 + chunk store
"""

import os
import pickle
import numpy as np
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from chunker import chunk_document, Chunk

# ── CONFIG ─────────────────────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"

NUM_DOCS = 200
CHUNK_SIZE = 250
OVERLAP_SENTS = 2

INDEX_PATH = "legal.faiss"
CHUNKS_PATH = "chunks.pkl"
BM25_PATH = "bm25.pkl"
# ───────────────────────────────────────────────────────────────────────


def load_corpus(n):
    print(f"Loading {n} documents...")
    ds = load_dataset(
        "isaacus/open-australian-legal-corpus",
        split="corpus",
        streaming=True,
    )

    docs = []
    for i, d in enumerate(ds):
        docs.append(d)
        if i + 1 >= n:
            break

    return docs


def build_chunks(docs):
    print("Chunking...")
    chunks = []

    for i, doc in enumerate(docs):
        text = doc.get("text", "").strip()
        if not text:
            continue

        chunks.extend(
            chunk_document(
                text=text,
                doc_id=doc.get("id", str(i)),
                citation=doc.get("citation", ""),
                url=doc.get("url", ""),
                jurisdiction=doc.get("jurisdiction", ""),
                chunk_size=CHUNK_SIZE,
                overlap_sentences=OVERLAP_SENTS,
            )
        )

    print("Total chunks:", len(chunks))
    return chunks


def embed(chunks):
    print("Embedding...")
    model = SentenceTransformer(EMBED_MODEL)

    texts = [c.text for c in chunks]
    emb = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return np.array(emb, dtype="float32")


def build_faiss(emb):
    print("Building FAISS...")

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    print("Vectors:", index.ntotal)
    return index


def build_bm25(chunks):
    print("Building BM25...")
    tokenized = [c.text.lower().split() for c in chunks]
    return BM25Okapi(tokenized)


def save(index, chunks, bm25):
    faiss.write_index(index, INDEX_PATH)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)

    print("Saved all artifacts.")


def main():
    docs = load_corpus(NUM_DOCS)
    chunks = build_chunks(docs)

    emb = embed(chunks)
    index = build_faiss(emb)
    bm25 = build_bm25(chunks)

    save(index, chunks, bm25)
    print("DONE ✔")


if __name__ == "__main__":
    main()