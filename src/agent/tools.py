"""
tools.py
--------
Tool functions available to the agent.
Each tool is a plain callable so the agent loop can invoke them explicitly.
"""

from typing import List, Dict, Any

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.retriever.retriever import Retriever


# ---------------------------------------------------------------------------
# Retrieval tool
# ---------------------------------------------------------------------------

def retrieval_tool(
    query: str,
    retriever: Retriever,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Retrieve the most semantically relevant document chunks for a query.

    Args:
        query     : Natural-language search query.
        retriever : Loaded Retriever instance backed by the FAISS index.
        top_k     : Number of chunks to return.

    Returns:
        List of chunk dicts with _score and _rank fields injected.

    Raises:
        RuntimeError: If the retriever/index is unavailable.
    """
    print(f"[tool:retrieval] Query   = \"{query[:80]}\"")
    print(f"[tool:retrieval] Top-K   = {top_k}")

    try:
        results = retriever.retrieve(query, top_k=top_k)
    except Exception as exc:
        raise RuntimeError(f"[tool:retrieval] Retrieval failed: {exc}") from exc

    print(f"[tool:retrieval] [OK] Retrieved {len(results)} chunks")
    return results


# ---------------------------------------------------------------------------
# Tool registry  (extensible -- add more tools here later)
# ---------------------------------------------------------------------------

TOOL_REGISTRY = {
    "retrieval": retrieval_tool,
}
