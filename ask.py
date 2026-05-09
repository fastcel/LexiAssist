"""
ask.py — LexiAssist query interface. The main script you run to ask questions.

Usage:
    python ask.py "What is a defacto relationship under Australian law?"

Or run interactively:
    python ask.py

Pipeline:
    User query
      → Query expansion (GPT-4o-mini generates 3 sub-queries)
      → Hybrid retrieval (FAISS dense + BM25 sparse per sub-query)
      → Reciprocal Rank Fusion (merge all ranked lists)
      → Cross-encoder reranking (precise scoring of top candidates)
      → Deduplication (remove near-duplicate chunks)
      → GPT-4 generation (grounded answer with citations)
      → Structured output (answer + citation list)
"""

import sys
import os
import json
import textwrap

# ensure OPENAI_API_KEY is set
if not os.environ.get("OPENAI_API_KEY"):
    print("ERROR: Set the OPENAI_API_KEY environment variable before running.")
    print("  export OPENAI_API_KEY=sk-...")
    sys.exit(1)

from retriever import Retriever
from query_expander import expand_query
from generator import generate_answer

# ── Pretty printer ────────────────────────────────────────────────────────────

def print_result(result: dict):
    width = 80
    print("\n" + "═" * width)
    print(f"  LEXIASSIST ANSWER")
    print("═" * width)

    print(f"\nQuery: {result['query']}\n")

    # wrap answer text
    for para in result["answer"].split("\n"):
        if para.strip():
            print(textwrap.fill(para.strip(), width=width))
        else:
            print()

    print("\n" + "─" * width)
    print("  RETRIEVED SOURCES")
    print("─" * width)
    for c in result["citations"]:
        print(f"\n  [{c['index']}] {c['citation'] or 'Unknown citation'}")
        print(f"       Jurisdiction : {c['jurisdiction'] or 'N/A'}")
        print(f"       Rerank score : {c['rerank_score']}")
        if c['url'] and c['url'] != 'N/A':
            print(f"       URL          : {c['url']}")
        print(f"       Excerpt      : {c['excerpt'][:120]}...")

    print("\n" + "═" * width + "\n")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def ask(query: str, retriever: Retriever, verbose: bool = True) -> dict:
    """
    Run the full RAG pipeline for a single query.

    Args:
        query:     User question string.
        retriever: Loaded Retriever instance (reuse across queries).
        verbose:   Print intermediate steps.

    Returns:
        Structured result dict (answer + citations).
    """
    if verbose:
        print(f"\n[1/4] Query expansion...")
    sub_queries = expand_query(query, n=3)
    if verbose and sub_queries:
        for i, sq in enumerate(sub_queries, 1):
            print(f"      Sub-query {i}: {sq}")

    if verbose:
        print(f"[2/4] Hybrid retrieval (FAISS + BM25)...")
    retrieved = retriever.retrieve(query, sub_queries=sub_queries)
    if verbose:
        print(f"      Retrieved {len(retrieved)} chunks after reranking + deduplication.")

    if verbose:
        print(f"[3/4] Generating answer with GPT-4o...")
    result = generate_answer(query, retrieved)

    if verbose:
        print(f"[4/4] Done.")

    return result


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    # Load once; reuse for all queries in this session
    retriever = Retriever()

    # Accept query from CLI arg or run interactive loop
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        result = ask(query, retriever)
        print_result(result)
    else:
        print("\nLexiAssist — Australian Legal RAG")
        print("Type your legal question, or 'quit' to exit.\n")
        while True:
            try:
                query = input("Question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break
            if query.lower() in ("quit", "exit", "q"):
                print("Goodbye.")
                break
            if not query:
                continue
            result = ask(query, retriever)
            print_result(result)

            # optionally save to JSON
            save = input("Save result to JSON? [y/N]: ").strip().lower()
            if save == "y":
                fname = f"result_{len(query[:20].split())}.json"
                with open(fname, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"Saved to {fname}")


if __name__ == "__main__":
    main()