"""
Embedding service for WALL-E memory system.

Lazy-loads sentence-transformers on first use. After a failure (missing library
or model load error), a sentinel prevents repeated retries.
"""

import logging
import threading
from typing import Optional

import numpy as np

from .config import conf

_log = logging.getLogger("walle.embeddings")

_embedding_model = None
_embedding_device = None
_embedding_lock = threading.Lock()
_embedding_failed = False


def get_embedding_model():
    """Lazy load sentence-transformer model (thread-safe, fail-once)."""
    global _embedding_model, _embedding_device, _embedding_failed

    if _embedding_failed:
        return None
    if _embedding_model is not None:
        return _embedding_model

    with _embedding_lock:
        if _embedding_failed:
            return None
        if _embedding_model is not None:
            return _embedding_model

        try:
            import torch
            from sentence_transformers import SentenceTransformer

            _embedding_device = (
                conf.EMBEDDING_DEVICE if torch.cuda.is_available() else "cpu"
            )

            _log.info(
                "Loading embedding model: %s on %s...",
                conf.EMBEDDING_MODEL,
                _embedding_device,
            )
            _embedding_model = SentenceTransformer(
                conf.EMBEDDING_MODEL, device=_embedding_device
            )

            if _embedding_device == "cuda":
                _embedding_model.half()

            _log.info("Embedding model loaded")

        except ImportError:
            _embedding_failed = True
            _log.warning("sentence-transformers not installed. Embeddings disabled.")
            return None
        except Exception as e:
            _embedding_failed = True
            _log.warning("Failed to load embedding model: %s", e)
            return None

    return _embedding_model


def get_embedding(text: str) -> Optional[bytes]:
    """Generate a normalized embedding as bytes, or None on failure."""
    model = get_embedding_model()
    if model is None:
        return None

    try:
        embedding = model.encode(
            text,
            convert_to_tensor=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        if _embedding_device == "cuda":
            embedding = embedding.cpu()

        return embedding.numpy().astype(np.float32).tobytes()

    except Exception as e:
        _log.warning("Embedding error: %s", e)
        return None
