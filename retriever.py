"""
retriever.py — Hybrid retrieval using TF-IDF + BM25.
No pretrained embedding or reranking models used.

Retrieval pipeline:
  1. TF-IDF cosine similarity  (dense-equivalent, corpus-learned)
  2. BM25 keyword search       (sparse)
  3. Reciprocal Rank Fusion    (merge both ranked lists)
  4. BM25 reranking            (score top candidates more precisely)
  5. Deduplication             (Jaccard similarity)
"""

import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz
from rank_bm25 import BM25Okapi

CHUNKS_PATH    = "chunks.pkl"
BM25_PATH      = "bm25.pkl"
TFIDF_PATH     = "tfidf_vectorizer.pkl"
TFIDF_MAT_PATH = "tfidf_matrix.npz"

DENSE_K       = 20
SPARSE_K      = 20
RRF_K         = 60
TOP_RERANK    = 10
FINAL_TOP     = 8
DUP_THRESHOLD = 0.75


class Retriever:
    def __init__(self):
        print("Loading retriever...")

        # ── TF-IDF (replaces sentence-transformers + FAISS) ──
        if not os.path.exists(TFIDF_PATH):
            raise FileNotFoundError("Run indexer.py first (missing tfidf_vectorizer.pkl)")
        if not os.path.exists(TFIDF_MAT_PATH):
            raise FileNotFoundError("Run indexer.py first (missing tfidf_matrix.npz)")

        with open(TFIDF_PATH, "rb") as f:
            self.vectorizer = pickle.load(f)

        # sparse matrix — memory efficient
        self.tfidf_matrix = load_npz(TFIDF_MAT_PATH)

        # ── Chunks ──
        if not os.path.exists(CHUNKS_PATH):
            raise FileNotFoundError("Run indexer.py first (missing chunks.pkl)")

        with open(CHUNKS_PATH, "rb") as f:
            self.chunks = pickle.load(f)

        # ── BM25 (sparse retrieval + reranker) ──
        if os.path.exists(BM25_PATH):
            with open(BM25_PATH, "rb") as f:
                self.bm25 = pickle.load(f)
        else:
            print("⚠ BM25 missing — building from chunks now...")
            tokenized = [c.text.lower().split() for c in self.chunks]
            self.bm25 = BM25Okapi(tokenized)

        print(f"✔ Loaded {len(self.chunks)} chunks")

    # ─────────────────────────────────────────────────────────────────

    def retrieve(self, query, sub_queries=None):
        queries = [query] + (sub_queries or [])

        dense  = self._tfidf_search(queries)
        sparse = self._bm25_search(queries)

        fused = self._rrf(dense, sparse)
        ids   = [i for i, _ in fused[:TOP_RERANK]]

        reranked = self._rerank(query, ids)
        return self._dedupe(reranked)[:FINAL_TOP]

    # ─────────────────────────────────────────────────────────────────

    def _tfidf_search(self, queries):
        """
        Convert queries to TF-IDF vectors and find top-K
        chunks by cosine similarity. Replaces FAISS dense search.
        """
        query_vecs = self.vectorizer.transform(queries)
        sims       = cosine_similarity(query_vecs, self.tfidf_matrix)

        results = []
        for i in range(len(queries)):
            # argsort ascending → reverse → take top DENSE_K
            top_ids = sims[i].argsort()[::-1][:DENSE_K].tolist()
            results.append(top_ids)

        return results

    def _bm25_search(self, queries):
        """BM25 keyword search across all queries."""
        out = []
        for q in queries:
            scores  = self.bm25.get_scores(q.lower().split())
            top_ids = np.argsort(scores)[::-1][:SPARSE_K].tolist()
            out.append(top_ids)
        return out

    def _rrf(self, dense, sparse):
        """
        Reciprocal Rank Fusion — merges TF-IDF and BM25 ranked lists.
        Score = sum of 1 / (RRF_K + rank) across all lists.
        """
        scores = {}
        for lst in dense + sparse:
            for rank, i in enumerate(lst):
                scores[i] = scores.get(i, 0) + 1 / (RRF_K + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def _rerank(self, query, ids):
        """
        Rerank top candidates using BM25 scores.
        Replaces the cross-encoder reranker.
        BM25 scores the query against each candidate chunk more precisely.
        """
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)

        results = []
        for i, cid in enumerate(ids):
            results.append({
                "chunk": self.chunks[cid],
                "score": float(scores[cid]),
                "rank":  i,
            })

        return sorted(results, key=lambda x: x["score"], reverse=True)

    def _dedupe(self, results):
        """Remove near-duplicate chunks using Jaccard similarity."""
        kept = []

        def jaccard(a, b):
            sa, sb = set(a.lower().split()), set(b.lower().split())
            if not sa | sb:
                return 0.0
            return len(sa & sb) / len(sa | sb)

        for r in results:
            if any(
                jaccard(r["chunk"].text, k["chunk"].text) > DUP_THRESHOLD
                for k in kept
            ):
                continue
            kept.append(r)

        return kept