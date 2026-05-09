"""
retriever.py — Clean hybrid retrieval system
"""

import os
import pickle
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

INDEX_PATH = "legal.faiss"
CHUNKS_PATH = "chunks.pkl"
BM25_PATH = "bm25.pkl"

DENSE_K = 20
SPARSE_K = 20
RRF_K = 60
TOP_RERANK = 10
FINAL_TOP = 5
DUP_THRESHOLD = 0.75


class Retriever:
    def __init__(self):
        print("Loading retriever...")

        self.embed = SentenceTransformer(EMBED_MODEL)
        self.reranker = CrossEncoder(RERANK_MODEL)

        self.index = faiss.read_index(INDEX_PATH)

        if not os.path.exists(CHUNKS_PATH):
            raise FileNotFoundError("Run indexer.py first (missing chunks.pkl)")

        with open(CHUNKS_PATH, "rb") as f:
            self.chunks = pickle.load(f)

        if os.path.exists(BM25_PATH):
            with open(BM25_PATH, "rb") as f:
                self.bm25 = pickle.load(f)
        else:
            print("⚠ BM25 missing → sparse disabled")
            self.bm25 = None

        print(f"✔ Loaded {len(self.chunks)} chunks")

    def retrieve(self, query, sub_queries=None):
        queries = [query] + (sub_queries or [])

        dense = self._dense(queries)
        sparse = self._sparse(queries)

        fused = self._rrf(dense, sparse)
        ids = [i for i, _ in fused[:TOP_RERANK]]

        reranked = self._rerank(query, ids)
        return self._dedupe(reranked)[:FINAL_TOP]

    def _dense(self, queries):
        emb = self.embed.encode(
            queries,
            normalize_embeddings=True
        ).astype("float32")

        _, idx = self.index.search(emb, DENSE_K)
        return [[i for i in row if i >= 0] for row in idx.tolist()]

    def _sparse(self, queries):
        if not self.bm25:
            return []

        out = []
        for q in queries:
            scores = self.bm25.get_scores(q.lower().split())
            top = np.argsort(scores)[::-1][:SPARSE_K]
            out.append(top.tolist())

        return out

    def _rrf(self, dense, sparse):  # fixed: indented as class method
        scores = {}
        for lst in dense + sparse:
            for rank, i in enumerate(lst):
                scores[i] = scores.get(i, 0) + 1 / (RRF_K + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def _rerank(self, query, ids):
        pairs = [(query, self.chunks[i].text) for i in ids]
        scores = self.reranker.predict(pairs)

        results = []
        for i, (cid, score) in enumerate(zip(ids, scores)):
            results.append({
                "chunk": self.chunks[cid],
                "score": float(score),
                "rank": i
            })

        return sorted(results, key=lambda x: x["score"], reverse=True)

    def _dedupe(self, results):
        kept = []

        def jacc(a, b):
            sa, sb = set(a.lower().split()), set(b.lower().split())
            return len(sa & sb) / len(sa | sb)

        for r in results:
            if any(jacc(r["chunk"].text, k["chunk"].text) > DUP_THRESHOLD for k in kept):
                continue
            kept.append(r)

        return kept