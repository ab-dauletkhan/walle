"""
FAISS vector index manager for WALL-E memory system.

Provides O(log n) similarity search using IndexFlatIP (inner product on
normalized vectors = cosine similarity).
"""

import json
import logging
import threading
from typing import List

import numpy as np

_log = logging.getLogger("walle.faiss")

_faiss_available = False
try:
    import torch  # noqa: F401 — must import before faiss to avoid OpenMP segfault on macOS
    import faiss
    _faiss_available = True
except ImportError:
    faiss = None


def _connect_db(db_path: str):
    import sqlite3
    conn = sqlite3.connect(db_path, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


class FAISSManager:
    """Thread-safe FAISS index with disk persistence."""

    def __init__(self, dimension: int = 384, index_path: str = None):
        self.dimension = dimension
        self.index_path = index_path
        self.index = None
        self.id_map: list[int] = []
        self._insertions_since_save = 0
        self._save_lock = threading.Lock()
        self._index_lock = threading.RLock()
        self._save_in_progress = False

        if _faiss_available:
            self._load_or_create_index()
        else:
            _log.warning("FAISS not available. Install with: pip install faiss-cpu")

    def _load_or_create_index(self):
        from pathlib import Path

        if self.index_path and Path(self.index_path).exists():
            try:
                self.index = faiss.read_index(self.index_path)
                id_map_path = self.index_path + ".ids"
                if Path(id_map_path).exists():
                    with open(id_map_path, 'r') as f:
                        self.id_map = json.load(f)
                _log.info("FAISS index loaded: %s vectors", self.index.ntotal)
            except Exception as e:
                _log.warning("Failed to load FAISS index: %s", e)
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self):
        if not _faiss_available:
            return
        self.index = faiss.IndexFlatIP(self.dimension)
        self.id_map = []

    def add(self, embedding_bytes: bytes, row_id: int):
        if not _faiss_available or self.index is None:
            return
        vec = np.frombuffer(embedding_bytes, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)
        with self._index_lock:
            self.index.add(vec)
            self.id_map.append(row_id)
            self._insertions_since_save += 1

    def search(self, query_embedding_bytes: bytes, k: int = 10) -> List[tuple]:
        if not _faiss_available or self.index is None or self.index.ntotal == 0:
            return []
        q_vec = np.frombuffer(query_embedding_bytes, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(q_vec)
        with self._index_lock:
            k = min(k, self.index.ntotal)
            distances, indices = self.index.search(q_vec, k)
            id_map_snapshot = list(self.id_map)
        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(id_map_snapshot):
                results.append((id_map_snapshot[idx], float(distances[0][i])))
        return results

    def get_indexed_ids(self) -> set:
        with self._index_lock:
            return set(self.id_map)

    def save(self):
        if not _faiss_available or self.index is None or not self.index_path:
            return
        with self._save_lock:
            try:
                faiss.write_index(self.index, self.index_path)
                with open(self.index_path + ".ids", 'w') as f:
                    json.dump(self.id_map, f)
                self._insertions_since_save = 0
            except Exception as e:
                _log.error("Failed to save FAISS index: %s", e)

    def save_async(self):
        if not _faiss_available or self.index is None or not self.index_path:
            return
        if self._save_in_progress:
            return
        self._save_in_progress = True
        threading.Thread(target=self._save_async_impl, daemon=True).start()

    def _save_async_impl(self):
        try:
            self.save()
        finally:
            self._save_in_progress = False

    def rebuild_from_db(self, db_path: str, table: str = "recall_memory"):
        if not _faiss_available:
            return
        allowed_tables = {"recall_memory", "archival_memory"}
        if table not in allowed_tables:
            raise ValueError(f"Invalid table name: {table}")
        with _connect_db(db_path) as conn:
            rows = conn.execute(f"SELECT id, embedding FROM {table} WHERE embedding IS NOT NULL").fetchall()
        if not rows:
            with self._index_lock:
                self._create_new_index()
            return
        embeddings = []
        new_id_map = []
        for row_id, emb_bytes in rows:
            embeddings.append(np.frombuffer(emb_bytes, dtype=np.float32))
            new_id_map.append(row_id)
        embeddings_array = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_array)
        with self._index_lock:
            self._create_new_index()
            self.index.add(embeddings_array)
            self.id_map = new_id_map
            self._insertions_since_save = 0
        _log.info("FAISS index rebuilt: %s vectors from %s", len(rows), table)

    def needs_save(self, threshold: int = 10) -> bool:
        return self._insertions_since_save >= threshold

    @property
    def is_available(self) -> bool:
        return _faiss_available and self.index is not None


def is_faiss_available() -> bool:
    return _faiss_available
