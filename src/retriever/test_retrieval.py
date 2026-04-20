"""
test_retrieval.py
-----------------
End-to-end smoke test for the retrieval pipeline.

Generates synthetic chunks, builds a FAISS index, and runs a query.

Usage (from project root, venv active):
    python src/retriever/test_retrieval.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.retriever.retriever import Retriever


# ---------------------------------------------------------------------------
# Synthetic test data (no real PDF needed)
# ---------------------------------------------------------------------------
TEST_CHUNKS = [
    {
        "chunk_id"  : 0,
        "content"   : (
            "Retrieval-Augmented Generation (RAG) is a technique that combines "
            "a retrieval system with a language model. Instead of relying solely "
            "on parametric knowledge, RAG fetches relevant documents at inference "
            "time and includes them in the context window."
        ),
        "type" : "text",
        "page" : 1,
        "char_count": 280,
    },
    {
        "chunk_id"  : 1,
        "content"   : (
            "FAISS (Facebook AI Similarity Search) is an open-source library for "
            "efficient similarity search and clustering of dense vectors. It supports "
            "exact and approximate nearest-neighbor search at scale."
        ),
        "type" : "text",
        "page" : 2,
        "char_count": 215,
    },
    {
        "chunk_id"  : 2,
        "content"   : (
            "Sentence-transformers is a Python library for generating state-of-the-art "
            "sentence and text embeddings. The all-MiniLM-L6-v2 model produces 384-dimensional "
            "embeddings and is optimized for semantic similarity tasks."
        ),
        "type" : "text",
        "page" : 3,
        "char_count": 240,
    },
    {
        "chunk_id"  : 3,
        "content"   : (
            "Groq is an AI inference company that provides ultra-fast LLM serving via "
            "its Language Processing Unit (LPU). The Groq API is compatible with the "
            "OpenAI SDK interface."
        ),
        "type" : "text",
        "page" : 4,
        "char_count": 185,
    },
    {
        "chunk_id"  : 4,
        "content"   : (
            "Streamlit is an open-source Python framework for building interactive "
            "web applications for machine learning and data science projects."
        ),
        "type" : "text",
        "page" : 5,
        "char_count": 155,
    },
]

TEST_QUERY   = "What is RAG and how does it work?"
TEST_INDEX   = "test_retrieval_smoke"
TOP_K        = 3


def run_test():
    print("=" * 60)
    print("  AgenticRag — Retrieval Smoke Test")
    print("=" * 60)

    # ── Build ────────────────────────────────────────────────────
    print("\n[Step 1/2] Building index from synthetic chunks ...")
    retriever = Retriever(index_name=TEST_INDEX)
    retriever.build_from_chunks(TEST_CHUNKS)

    # ── Query ────────────────────────────────────────────────────
    print(f"\n[Step 2/2] Querying: \"{TEST_QUERY}\"")
    results = retriever.retrieve(TEST_QUERY, top_k=TOP_K)

    print(f"\n{'─'*60}")
    print(f"  Top {TOP_K} Results")
    print(f"{'─'*60}")
    for r in results:
        print(f"\n  Rank  : #{r['_rank']}")
        print(f"  Score : {r['_score']:.4f}")
        print(f"  Page  : {r['page']}")
        print(f"  Type  : {r['type']}")
        print(f"  Text  : {r['content'][:120]}...")

    print(f"\n{'─'*60}")
    print("\n[format_context] LLM-ready context string:\n")
    context = retriever.format_context(results)
    print(context)

    print("\n" + "=" * 60)
    print("  ✓ Retrieval test PASSED")
    print("=" * 60)


if __name__ == "__main__":
    try:
        run_test()
    except Exception as e:
        print(f"\n[✗] Test FAILED: {type(e).__name__}: {e}")
        sys.exit(1)
