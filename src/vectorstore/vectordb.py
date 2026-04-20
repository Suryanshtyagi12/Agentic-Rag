"""
vectordb.py
-----------
FAISS-backed vector store with metadata side-car.

Persists two files to vector_db/:
    <name>.index     — FAISS flat index (Inner Product on L2-normalised vecs)
    <name>.meta.json — list of metadata dicts, one per vector

Design notes:
  • IVFFlat is used when n_vectors > IVFFLAT_THRESHOLD for faster search.
  • Below that threshold a simple FlatIP index is used (exact, no training).
  • Metadata is kept in a plain JSON file alongside the index for portability.
"""

import os
import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VECTOR_DIM         = 384            # all-MiniLM-L6-v2 output dim
IVFFLAT_THRESHOLD  = 1000           # switch to IVFFlat above this count
IVFFLAT_NLIST      = 100            # number of Voronoi cells
TOP_K_DEFAULT      = 5

# Project-root vector_db/ directory
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
VECTOR_DB_DIR = _PROJECT_ROOT / "vector_db"


# ---------------------------------------------------------------------------
# VectorDB class
# ---------------------------------------------------------------------------

class VectorDB:
    """
    Thin wrapper around a FAISS index with JSON metadata storage.

    Usage:
        db = VectorDB("my_index")
        db.add_documents(chunks, embeddings)
        db.save_index()

        db2 = VectorDB("my_index")
        db2.load_index()
        results = db2.similarity_search("my query embedding", k=5)
    """

    def __init__(self, index_name: str = "default"):
        self.index_name  = index_name
        self.index_path  = VECTOR_DB_DIR / f"{index_name}.index"
        self.meta_path   = VECTOR_DB_DIR / f"{index_name}.meta.json"

        self._index: Optional[faiss.Index] = None
        self._metadata: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_index(self, n_vectors: int) -> faiss.Index:
        """Return an appropriate FAISS index for the given vector count."""
        if n_vectors >= IVFFLAT_THRESHOLD:
            quantizer = faiss.IndexFlatIP(VECTOR_DIM)
            index = faiss.IndexIVFFlat(quantizer, VECTOR_DIM, IVFFLAT_NLIST, faiss.METRIC_INNER_PRODUCT)
            print(f"[vectordb] Using IVFFlat index (nlist={IVFFLAT_NLIST})")
        else:
            index = faiss.IndexFlatIP(VECTOR_DIM)
            print(f"[vectordb] Using FlatIP index (exact search)")
        return index

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_documents(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray,
    ) -> None:
        """
        Add document chunks and their embeddings to the index.

        Args:
            chunks     : List of chunk dicts (from chunking.py).
                         Each dict is stored as metadata verbatim.
            embeddings : np.ndarray shape (len(chunks), VECTOR_DIM), float32.

        Raises:
            ValueError: If lengths don't match.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"[vectordb] Chunks ({len(chunks)}) and embeddings "
                f"({len(embeddings)}) length mismatch."
            )

        embeddings = np.asarray(embeddings, dtype=np.float32)
        n = len(chunks)

        if self._index is None:
            self._index = self._build_index(n)
            if hasattr(self._index, "train"):
                try:
                    self._index.train(embeddings)   # IVFFlat needs training
                except Exception:
                    pass                            # FlatIP has no-op train

        self._index.add(embeddings)
        self._metadata.extend(chunks)

        print(f"[vectordb] ✓ Added {n} vectors  (total: {self._index.ntotal})")

    def save_index(self) -> None:
        """Persist the FAISS index and metadata JSON to disk."""
        VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

        if self._index is None or self._index.ntotal == 0:
            raise RuntimeError("[vectordb] Nothing to save — index is empty.")

        faiss.write_index(self._index, str(self.index_path))

        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False, indent=2)

        idx_kb = self.index_path.stat().st_size / 1024
        print(f"[vectordb] ✓ Index  saved → {self.index_path} ({idx_kb:.1f} KB)")
        print(f"[vectordb] ✓ Metadata saved → {self.meta_path}")

    def load_index(self) -> None:
        """Load FAISS index and metadata from disk."""
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"[vectordb] Index not found: {self.index_path}\n"
                "Run the ingestion pipeline first."
            )

        self._index = faiss.read_index(str(self.index_path))

        with open(self.meta_path, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)

        print(
            f"[vectordb] ✓ Loaded index '{self.index_name}' — "
            f"{self._index.ntotal} vectors, {len(self._metadata)} metadata records"
        )

    def similarity_search(
        self,
        query_vector: np.ndarray,
        k: int = TOP_K_DEFAULT,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most similar documents for a query vector.

        Args:
            query_vector : 1-D np.ndarray of shape (VECTOR_DIM,), float32.
            k            : Number of results to return.

        Returns:
            List of dicts, each containing original chunk metadata plus:
                "_score"   : float  — inner product similarity (higher = better)
                "_rank"    : int    — 1-indexed rank
        """
        if self._index is None:
            raise RuntimeError("[vectordb] Index not loaded. Call load_index() first.")

        k = min(k, self._index.ntotal)
        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)

        scores, indices = self._index.search(query, k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx == -1:
                continue        # FAISS returns -1 for empty slots
            meta = dict(self._metadata[idx])
            meta["_score"] = float(score)
            meta["_rank"]  = rank
            results.append(meta)

        return results

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal if self._index else 0
