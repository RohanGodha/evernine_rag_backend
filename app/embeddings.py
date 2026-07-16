"""Local text embeddings for vector RAG.

Groq has no embedding model, so we embed locally with sentence-transformers
(all-MiniLM-L6-v2, 384-dim). The heavy import + model load are lazy and behind
a guard: if the package/model is unavailable the rest of the app keeps running,
and callers degrade to SQL keyword search.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional, List

from .db import EMBEDDING_DIM

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"

_model = None
_load_attempted = False
_lock = threading.Lock()


def _load_model():
    """Lazily load the embedding model once. Returns the model or None if it
    can't be loaded (missing dep, no network for first download, etc.)."""
    global _model, _load_attempted
    if _model is not None or _load_attempted:
        return _model
    with _lock:
        if _model is not None or _load_attempted:
            return _model
        _load_attempted = True
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model %s ...", MODEL_NAME)
            _model = SentenceTransformer(MODEL_NAME)
            # Sanity check dimensionality matches the DB column.
            dim = _model.get_sentence_embedding_dimension()
            if dim != EMBEDDING_DIM:
                logger.error(
                    "Embedding dim %d != expected %d; disabling embeddings.",
                    dim, EMBEDDING_DIM,
                )
                _model = None
        except Exception as e:  # pragma: no cover - environment dependent
            logger.warning("Embeddings unavailable (%s); using SQL fallback.", e)
            _model = None
    return _model


def is_available() -> bool:
    """True if embeddings can be produced (model loads successfully)."""
    return _load_model() is not None


def encode(text: str) -> Optional[List[float]]:
    """Return a 384-dim embedding for `text`, or None if embeddings are
    unavailable (caller should degrade to SQL)."""
    model = _load_model()
    if model is None or not text:
        return None
    try:
        vec = model.encode(text, normalize_embeddings=True)
        return vec.tolist()
    except Exception as e:  # pragma: no cover
        logger.warning("Embedding encode failed: %s", e)
        return None


def build_embedding_text(
    name: Optional[str],
    description: Optional[str],
    normalized_channel: str,
    inferred_objective: str,
) -> str:
    """Canonical text representation of a campaign used for embedding."""
    parts = [
        name or "",
        description or "",
        f"channel={normalized_channel}",
        f"objective={inferred_objective}",
    ]
    return ". ".join(p for p in parts if p)
