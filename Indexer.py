"""
indexer.py — Build TF-IDF + BM25 + chunk store
No pretrained embedding models used.
"""

import os
import pickle
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from scipy.sparse import save_npz

from chunker import chunk_document, Chunk

# ── CONFIG ─────────────────────────────────────────────────────────────
NUM_DOCS        = 1000
CHUNK_SIZE      = 250
OVERLAP_SENTS   = 2

CHUNKS_PATH     = "chunks.pkl"
BM25_PATH       = "bm25.pkl"
TFIDF_PATH      = "tfidf_vectorizer.pkl"
TFIDF_MAT_PATH  = "tfidf_matrix.npz"
# ───────────────────────────────────────────────────────────────────────


def load_corpus(n):
    print(f"Loading {n} documents from HuggingFace...")
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

    print(f"Loaded {len(docs)} documents.")
    return docs


def build_chunks(docs):
    print("Chunking documents...")
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

    print(f"Total chunks: {len(chunks)}")
    return chunks


def build_tfidf(chunks):
    """
    Build TF-IDF vectors from chunk texts.
    TF-IDF is a classical algorithm — no pretrained weights.
    It learns term importance purely from the legal corpus.
    """
    print("Building TF-IDF vectorizer from corpus...")

    texts = [c.text for c in chunks]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),       # unigrams + bigrams for better legal term matching
        max_df=0.85,              # ignore terms that appear in >85% of chunks (too common)
        min_df=2,                 # ignore terms that appear in <2 chunks (too rare)
        sublinear_tf=True,        # apply log normalization to term frequency
    )

    tfidf_matrix = vectorizer.fit_transform(texts)

    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    return vectorizer, tfidf_matrix


def build_bm25(chunks):
    print("Building BM25 index...")
    tokenized = [c.text.lower().split() for c in chunks]
    return BM25Okapi(tokenized)


def save_artifacts(chunks, vectorizer, tfidf_matrix, bm25):
    print("Saving artifacts to disk...")

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    with open(TFIDF_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    # sparse matrix — use scipy's save_npz for efficiency
    save_npz(TFIDF_MAT_PATH, tfidf_matrix)

    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)

    print("Saved:")
    print(f"  {CHUNKS_PATH}      — {len(chunks)} chunks")
    print(f"  {TFIDF_PATH}  — TF-IDF vectorizer")
    print(f"  {TFIDF_MAT_PATH}  — TF-IDF matrix")
    print(f"  {BM25_PATH}        — BM25 index")


def main():
    docs        = load_corpus(NUM_DOCS)
    chunks      = build_chunks(docs)
    vectorizer, tfidf_matrix = build_tfidf(chunks)
    bm25        = build_bm25(chunks)
    save_artifacts(chunks, vectorizer, tfidf_matrix, bm25)
    print("\nDONE ✔  Run python server.py to start the app.")


if __name__ == "__main__":
    main()