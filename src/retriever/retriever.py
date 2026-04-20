"""
retriever.py
------------
High-level retriever that wraps VectorDB + Embedder into a single
clean interface for the RAG pipeline.

Usage:
    from src.retriever.retriever import Retriever

    r = Retriever(index_name="my_index")
    r.build_from_chunks(chunks)      # embed + index + save
    # -- later --
    results = r.retrieve("What is RAG?", top_k=5)
"""

import numpy as np
from typing import List, Dict, Any

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.embeddings.embedder  import embed, embed_single
from src.vectorstore.vectordb import VectorDB


class Retriever:
    """
    Orchestrates embedding -> indexing -> retrieval.

    Args:
        index_name: Name used for FAISS index files in vector_db/.
    """

    def __init__(self, index_name: str = "default"):
        self.index_name = index_name
        self._db = VectorDB(index_name)
        self._loaded = False

    # ------------------------------------------------------------------
    # Build path  (ingest -> embed -> store)
    # ------------------------------------------------------------------

    def build_from_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Embed all chunks and persist the FAISS index.

        Args:
            chunks: Output of chunking.chunk_elements() -- list of dicts,
                    each with at least a "content" key.
        """
        print(f"[retriever] Building index '{self.index_name}' from {len(chunks)} chunks ...")

        texts = [c["content"] for c in chunks]
        embeddings = embed(texts)                   # (N, 384) float32

        self._db.add_documents(chunks, embeddings)
        self._db.save_index()
        self._loaded = True

        print(f"[retriever] [OK] Index ready -- {self._db.total_vectors} vectors stored")

    # ------------------------------------------------------------------
    # Query path  (embed -> search)
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load a previously saved index from disk."""
        if not self._loaded:
            self._db.load_index()
            self._loaded = True

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant chunks for a natural-language query.

        Args:
            query : Plain-text query string.
            top_k : Number of results to return.

        Returns:
            List of chunk dicts sorted by similarity (best first), each
            augmented with "_score" and "_rank" fields.
        """
        if not self._loaded:
            self.load()

        print(f"[retriever] Searching for: \"{query[:80]}\" (top_k={top_k})")

        query_vec = embed_single(query)             # (384,) float32
        results   = self._db.similarity_search(query_vec, k=top_k)

        print(f"[retriever] [OK] Retrieved {len(results)} results")
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into a single context string for the LLM prompt.

        Args:
            results: Output of retrieve().

        Returns:
            Formatted multi-chunk context string.
        """
        parts = []
        for r in results:
            header = (
                f"[Source: page {r.get('page', '?')} | "
                f"type: {r.get('type', '?')} | "
                f"score: {r.get('_score', 0):.4f}]"
            )
            parts.append(f"{header}\n{r['content']}")

        return "\n\n---\n\n".join(parts)
