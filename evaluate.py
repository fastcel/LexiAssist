"""
evaluate.py — Measure retrieval quality with standard IR metrics.

Metrics computed:
  - Hit Rate @K   : did any retrieved chunk come from the right document?
  - Precision @K  : what fraction of the top-K chunks are relevant?
  - MRR           : Mean Reciprocal Rank (how high is the first correct hit?)

A small hand-labelled test set is included. Each test case has:
  - query         : the user question
  - relevant_docs : list of doc_id substrings or citation keywords that count
                    as a correct retrieval (partial match is fine)

You can extend TEST_CASES with more examples for a stronger evaluation.

Usage:
    python evaluate.py
    python evaluate.py --k 3    # evaluate at a different K
"""

import argparse
import sys
from retriever import Retriever
from query_expander import expand_query

# ── Test set ──────────────────────────────────────────────────────────────────
# relevant_docs: list of strings — a retrieved chunk counts as relevant if
# its citation, jurisdiction, or text contains ANY of these substrings
# (case-insensitive). Add more rows as you manually inspect your corpus.

TEST_CASES = [
    {
        "query": "What is a defacto relationship under Australian law?",
        "relevant_docs": ["family law", "de facto", "4aa", "domestic"],
    },
    {
        "query": "What are a tenant's rights when a landlord doesn't make repairs?",
        "relevant_docs": ["residential tenancies", "tenant", "repair", "maintenance", "lessor"],
    },
    {
        "query": "How does negligence work in tort law?",
        "relevant_docs": ["negligence", "duty of care", "tort", "damages"],
    },
    {
        "query": "What constitutes a valid contract in Australia?",
        "relevant_docs": ["contract", "agreement", "offer", "consideration", "acceptance"],
    },
    {
        "query": "What are the rights of a director in a company?",
        "relevant_docs": ["corporations act", "director", "fiduciary", "officer"],
    },
    {
        "query": "What happens if someone breaches a lease agreement?",
        "relevant_docs": ["lease", "tenancy", "breach", "termination", "vacate"],
    },
    {
        "query": "What is the process for appealing a court decision?",
        "relevant_docs": ["appeal", "appellate", "review", "tribunal"],
    },
    {
        "query": "What are parental responsibilities after separation?",
        "relevant_docs": ["parental responsibility", "family law", "custody", "children"],
    },
    {
        "query": "What penalties apply for criminal offences in Australia?",
        "relevant_docs": ["criminal code", "penalty", "sentence", "offence", "conviction"],
    },
    {
        "query": "Can an employer terminate employment without notice?",
        "relevant_docs": ["employment", "termination", "notice", "unfair dismissal", "workplace"],
    },
]

# ── Relevance check ───────────────────────────────────────────────────────────

def _is_relevant(chunk, relevant_keywords: list[str]) -> bool:
    """Return True if any keyword appears in the chunk's text or metadata."""
    haystack = " ".join([
        chunk.text,
        chunk.citation or "",
        chunk.jurisdiction or "",
        chunk.url or "",
    ]).lower()
    return any(kw.lower() in haystack for kw in relevant_keywords)


# ── Metrics ───────────────────────────────────────────────────────────────────

def hit_rate(results: list[dict], relevant_keywords: list[str]) -> float:
    """1 if any retrieved chunk is relevant, else 0."""
    return float(any(_is_relevant(r["chunk"], relevant_keywords) for r in results))


def precision_at_k(results: list[dict], relevant_keywords: list[str]) -> float:
    """Fraction of retrieved chunks that are relevant."""
    if not results:
        return 0.0
    hits = sum(_is_relevant(r["chunk"], relevant_keywords) for r in results)
    return hits / len(results)


def mean_reciprocal_rank(results: list[dict], relevant_keywords: list[str]) -> float:
    """1 / rank of the first relevant result. 0 if none found."""
    for i, r in enumerate(results, start=1):
        if _is_relevant(r["chunk"], relevant_keywords):
            return 1.0 / i
    return 0.0


# ── Runner ────────────────────────────────────────────────────────────────────

def run_evaluation(k: int = 5):
    print(f"\n{'═' * 65}")
    print(f"  LEXIASSIST RAG EVALUATION  —  K={k}")
    print(f"{'═' * 65}\n")

    retriever = Retriever()

    all_hit      = []
    all_precision = []
    all_mrr      = []

    for i, case in enumerate(TEST_CASES, start=1):
        query    = case["query"]
        relevant = case["relevant_docs"]

        sub_queries = expand_query(query, n=3)
        results     = retriever.retrieve(query, sub_queries=sub_queries)
        results_k   = results[:k]

        hr  = hit_rate(results_k, relevant)
        p   = precision_at_k(results_k, relevant)
        mrr = mean_reciprocal_rank(results_k, relevant)

        all_hit.append(hr)
        all_precision.append(p)
        all_mrr.append(mrr)

        status = "✓" if hr else "✗"
        print(f"  [{status}] Q{i:02d}: {query[:55]}{'...' if len(query) > 55 else ''}")
        print(f"        Hit@{k}={hr:.0f}  P@{k}={p:.2f}  MRR={mrr:.2f}")
        if sub_queries:
            print(f"        Sub-queries: {sub_queries[0][:50]}...")
        print()

    n = len(TEST_CASES)
    avg_hit  = sum(all_hit) / n
    avg_p    = sum(all_precision) / n
    avg_mrr  = sum(all_mrr) / n

    print(f"{'─' * 65}")
    print(f"  RESULTS OVER {n} TEST QUERIES")
    print(f"{'─' * 65}")
    print(f"  Hit Rate  @{k}  : {avg_hit:.3f}  ({sum(all_hit):.0f}/{n} queries had a relevant result)")
    print(f"  Precision @{k}  : {avg_p:.3f}  (avg fraction of top-{k} that were relevant)")
    print(f"  MRR            : {avg_mrr:.3f}  (avg reciprocal rank of first relevant hit)")
    print(f"{'═' * 65}\n")

    interpret(avg_hit, avg_p, avg_mrr, k)


def interpret(hit: float, prec: float, mrr: float, k: int):
    print("  INTERPRETATION")
    print(f"  {'─' * 45}")
    thresholds = [
        ("Hit Rate",  hit,  0.8, 0.6),
        ("Precision", prec, 0.5, 0.3),
        ("MRR",       mrr,  0.6, 0.4),
    ]
    for name, val, good, ok in thresholds:
        if val >= good:
            grade = "Good"
        elif val >= ok:
            grade = "Acceptable"
        else:
            grade = "Needs improvement"
        print(f"  {name:<12}: {val:.3f}  → {grade}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LexiAssist RAG pipeline.")
    parser.add_argument("--k", type=int, default=5, help="Evaluate at top-K (default: 5)")
    args = parser.parse_args()
    run_evaluation(k=args.k)