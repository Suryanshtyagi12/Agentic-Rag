"""
embedder.py
-----------
Generates dense vector embeddings using sentence-transformers.

Model : all-MiniLM-L6-v2  (384-dimensional, fast & accurate)
Device: auto-detected (CPU / CUDA)
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE  = 64          # tune down if you hit memory limits
NORMALIZE   = True        # L2-normalise -> cosine sim == dot product


# ---------------------------------------------------------------------------
# Singleton loader (avoids re-downloading on every call)
# ---------------------------------------------------------------------------
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[embedder] Loading model: {MODEL_NAME} ...")
        _model = SentenceTransformer(MODEL_NAME)
        print(f"[embedder] [OK] Model loaded  (dim={_model.get_sentence_embedding_dimension()})")
    return _model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed(texts: List[str]) -> np.ndarray:
    """
    Embed a list of strings into dense float32 vectors.

    Args:
        texts: List of strings to embed.

    Returns:
        np.ndarray of shape (len(texts), 384), dtype float32.

    Raises:
        ValueError: If texts is empty.
    """
    if not texts:
        raise ValueError("[embedder] Cannot embed an empty list of texts.")

    model = _get_model()

    print(f"[embedder] Embedding {len(texts)} texts (batch_size={BATCH_SIZE}) ...")
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=NORMALIZE,
        show_progress_bar=len(texts) > 100,   # only show bar for large batches
        convert_to_numpy=True,
    )

    print(f"[embedder] [OK] Embeddings shape: {vectors.shape}")
    return vectors.astype(np.float32)


def embed_single(text: str) -> np.ndarray:
    """
    Convenience wrapper to embed one string.

    Returns:
        np.ndarray of shape (384,), dtype float32.
    """
    return embed([text])[0]
