"""
MemGPT-inspired memory system for WALL-E robot.

This module provides three memory tiers:
  - Memory (core)   — always in LLM context window
  - RecallMemory     — recent conversation history with semantic search
  - ArchivalMemory   — long-term facts with importance decay

Each tier is a Facade that composes:
  - Repository (CRUD) from repository.py
  - Embeddings from embeddings.py
  - Vector index from vector_index.py
  - Search chain from search_chain.py
"""

import json
import logging
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field

from .config import conf, MEMORY_DIR
from .embeddings import get_embedding
from .vector_index import FAISSManager, is_faiss_available
from .repository import RecallRepository, ArchivalRepository, _connect_db
from .search_chain import (
    FAISSSearchHandler, FTS5SearchHandler, RecentSearchHandler,
    ensure_fts5_table, insert_fts5, delete_fts5_before,
)

_log = logging.getLogger("walle.memory")


# ---------------------------------------------------------------------------
# Block & Memory (core memory — always in context window)
# ---------------------------------------------------------------------------
@dataclass
class Block:
    """A reserved section of the LLM's context window."""
    label: str
    value: str
    limit: int
    description: str = ""
    read_only: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.value) > self.limit:
            self.value = self.value[:self.limit]
        if not self.metadata:
            self.metadata = {
                "created_at": datetime.now().isoformat(),
                "last_modified": datetime.now().isoformat(),
            }

    @property
    def chars_current(self) -> int:
        return len(self.value)

    @property
    def chars_remaining(self) -> int:
        return self.limit - self.chars_current

    def append(self, content: str) -> tuple[bool, str]:
        if self.read_only:
            return False, f"Block '{self.label}' is read-only."
        new_value = self.value + content
        if len(new_value) <= self.limit:
            self.value = new_value
            self.metadata["last_modified"] = datetime.now().isoformat()
            return True, "Success"
        return False, f"Not enough space ({self.chars_remaining} chars remaining)"

    def replace(self, old_content: str, new_content: str) -> tuple[bool, str]:
        if self.read_only:
            return False, f"Block '{self.label}' is read-only."
        if old_content not in self.value:
            return False, "Old content not found in block"
        new_value = self.value.replace(old_content, new_content, 1)
        if len(new_value) <= self.limit:
            self.value = new_value
            self.metadata["last_modified"] = datetime.now().isoformat()
            return True, "Success"
        return False, f"New content too large ({len(new_value)} > {self.limit})"

    def compile(self) -> str:
        readonly_tag = " [READ-ONLY]" if self.read_only else ""
        return (
            f"<{self.label}{readonly_tag}>\n"
            f"<description>{self.description}</description>\n"
            f"<metadata>chars={self.chars_current}/{self.limit}, "
            f"modified={self.metadata.get('last_modified')}</metadata>\n"
            f"<value>\n{self.value}\n</value>\n"
            f"</{self.label}>"
        )


@dataclass
class Memory:
    """Core Memory — always in context window."""
    blocks: List[Block] = field(default_factory=list)
    db_path: str = field(default_factory=lambda: os.path.join(MEMORY_DIR, "walle_core_memory.db"))

    def __post_init__(self):
        if not self.blocks:
            if not self._load_from_db():
                self.blocks = [
                    Block("persona", "I am WALL-E, a robot companion.", 2000, "My identity and capabilities."),
                    Block("human", "The human is my operator.", 2000, "User profile and preferences."),
                    Block("system", "System initialized.", 1000, "System status.", read_only=True),
                ]
                self.save()

    def _init_db(self):
        with _connect_db(self.db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS core_memory
                          (label TEXT PRIMARY KEY, value TEXT, limit_chars INTEGER,
                           description TEXT, read_only INTEGER, metadata TEXT)""")

    def _load_from_db(self) -> bool:
        self._init_db()
        with _connect_db(self.db_path) as conn:
            rows = conn.execute(
                "SELECT label, value, limit_chars, description, read_only, metadata FROM core_memory"
            ).fetchall()
        if not rows:
            return False
        self.blocks = [
            Block(label=r[0], value=r[1], limit=r[2], description=r[3],
                  read_only=bool(r[4]), metadata=json.loads(r[5] or "{}"))
            for r in rows
        ]
        return True

    def save(self):
        self._init_db()
        with _connect_db(self.db_path) as conn:
            for b in self.blocks:
                conn.execute(
                    "INSERT OR REPLACE INTO core_memory VALUES (?, ?, ?, ?, ?, ?)",
                    (b.label, b.value, b.limit, b.description, int(b.read_only), json.dumps(b.metadata)),
                )

    def get_block(self, label: str) -> Optional[Block]:
        return next((b for b in self.blocks if b.label == label), None)

    def compile(self) -> str:
        return "<memory_blocks>\n" + "\n".join(b.compile() for b in self.blocks) + "\n</memory_blocks>"


# ---------------------------------------------------------------------------
# RecallMemory — Facade over repository + embeddings + FAISS + search chain
# ---------------------------------------------------------------------------
class RecallMemory:
    """Recent conversation history with FAISS-accelerated semantic search."""

    _MAX_PENDING_EMBEDDINGS = 50

    def __init__(self, db_path: str = None, use_semantic: bool = False):
        self.use_semantic = use_semantic
        self._repo = RecallRepository(db_path=db_path)
        self.db_path = self._repo.db_path
        self._embedding_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="emb")
        self._pending_futures: set = set()
        self._pending_lock = threading.Lock()

        # FTS5
        ensure_fts5_table(self.db_path, "recall_memory")

        # FAISS
        self.faiss_manager = None
        if use_semantic and conf.USE_FAISS and is_faiss_available():
            faiss_path = os.path.join(MEMORY_DIR, conf.FAISS_INDEX_PATH.replace(".index", "_recall.index"))
            self.faiss_manager = FAISSManager(dimension=conf.FAISS_DIMENSION, index_path=faiss_path)
            self._ensure_faiss_index()

        # Search chain: FAISS → FTS5 → Recent
        self._search_chain = FAISSSearchHandler(
            faiss_manager=self.faiss_manager,
            get_embedding_fn=get_embedding,
            use_semantic=use_semantic,
            next_handler=FTS5SearchHandler(next_handler=RecentSearchHandler()),
        )

    def _ensure_faiss_index(self):
        if self.faiss_manager and self.faiss_manager.is_available:
            if self.faiss_manager.index.ntotal == 0 and self._repo.get_count() > 0:
                self.faiss_manager.rebuild_from_db(self.db_path, "recall_memory")

    def _sync_faiss_if_needed(self):
        if not (self.faiss_manager and self.faiss_manager.is_available):
            return
        faiss_ids = self.faiss_manager.get_indexed_ids()
        with _connect_db(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, embedding FROM recall_memory WHERE embedding IS NOT NULL"
            ).fetchall()
        missing = [(rid, emb) for rid, emb in rows if rid not in faiss_ids and emb]
        if not missing:
            return
        for rid, emb in missing:
            self.faiss_manager.add(emb, rid)
        self.faiss_manager.save_async()

    def insert(self, role: str, content: str, tools_used: list = None,
               metadata: dict = None, defer_embedding: bool = False):
        embedding = get_embedding(content) if (self.use_semantic and not defer_embedding) else None
        row_id = self._repo.insert(role, content, embedding, tools_used, metadata)

        insert_fts5(self.db_path, "recall_memory", row_id, content)

        if embedding and self.faiss_manager and self.faiss_manager.is_available:
            self.faiss_manager.add(embedding, row_id)
            if self.faiss_manager.needs_save(threshold=10):
                self.faiss_manager.save_async()

        if self.use_semantic and defer_embedding:
            with self._pending_lock:
                if len(self._pending_futures) >= self._MAX_PENDING_EMBEDDINGS:
                    _log.debug("Embedding queue full, skipping deferred for row %d", row_id)
                else:
                    future = self._embedding_pool.submit(self._generate_embedding_async, row_id, content)
                    self._pending_futures.add(future)
                    future.add_done_callback(self._on_embedding_done)

    def _on_embedding_done(self, future):
        with self._pending_lock:
            self._pending_futures.discard(future)

    def _generate_embedding_async(self, row_id: int, content: str):
        try:
            embedding = get_embedding(content)
            if not embedding:
                return
            self._repo.update_embedding(row_id, embedding)
            if self.faiss_manager and self.faiss_manager.is_available:
                self.faiss_manager.add(embedding, row_id)
                if self.faiss_manager.needs_save(threshold=10):
                    self.faiss_manager.save()
        except Exception as e:
            _log.warning("Background embedding failed: %s", e)

    def search(self, query: str = None, limit: int = 10) -> List[Dict]:
        self._sync_faiss_if_needed()
        return self._search_chain.search(query, limit, self.db_path, table="recall_memory") or []

    def get_count(self) -> int:
        return self._repo.get_count()

    def compress_old_memories(self, summarizer_func: Callable[[str], str],
                              archival_memory: 'ArchivalMemory', keep_recent: int = 50):
        count = self.get_count()
        if count <= keep_recent:
            return 0

        rows = self._repo.get_old_memories(keep_recent)
        if not rows:
            return 0

        chronological = rows[::-1]
        text = "\n".join(f"[{r[2]}] {r[0]}: {r[1]}" for r in chronological)

        _log.info("Summarizing old memories...")
        summary = summarizer_func(text)
        archival_memory.insert("conversation_summary", summary, importance=3)

        newest_of_old = rows[0][2]
        self._repo.delete_before(newest_of_old)
        delete_fts5_before(self.db_path, "recall_memory")

        if self.faiss_manager and self.faiss_manager.is_available:
            self.faiss_manager.rebuild_from_db(self.db_path, "recall_memory")
            self.faiss_manager.save()

        return len(rows)

    def shutdown(self, timeout: float = 30):
        self._embedding_pool.shutdown(wait=False, cancel_futures=True)
        import time
        deadline = time.monotonic() + timeout
        with self._pending_lock:
            pending = list(self._pending_futures)
        for future in pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _log.warning("Embedding workers did not finish within timeout")
                break
            try:
                future.result(timeout=max(remaining, 0.1))
            except Exception:
                pass


# ---------------------------------------------------------------------------
# ArchivalMemory — Facade over repository + embeddings + FAISS + search chain
# ---------------------------------------------------------------------------
class ArchivalMemory:
    """Long-term storage with FAISS search and importance decay."""

    def __init__(self, db_path: str = None, use_semantic: bool = False):
        self.use_semantic = use_semantic
        self._repo = ArchivalRepository(db_path=db_path)
        self.db_path = self._repo.db_path

        ensure_fts5_table(self.db_path, "archival_memory")

        self.faiss_manager = None
        if use_semantic and conf.USE_FAISS and is_faiss_available():
            faiss_path = os.path.join(MEMORY_DIR, conf.FAISS_INDEX_PATH.replace(".index", "_archival.index"))
            self.faiss_manager = FAISSManager(dimension=conf.FAISS_DIMENSION, index_path=faiss_path)
            self._ensure_faiss_index()

        self._search_chain = FAISSSearchHandler(
            faiss_manager=self.faiss_manager,
            get_embedding_fn=get_embedding,
            use_semantic=use_semantic,
            next_handler=FTS5SearchHandler(next_handler=RecentSearchHandler()),
        )

    def _ensure_faiss_index(self):
        if self.faiss_manager and self.faiss_manager.is_available:
            if self.faiss_manager.index.ntotal == 0 and self._repo.get_count() > 0:
                self.faiss_manager.rebuild_from_db(self.db_path, "archival_memory")

    def _sync_faiss_if_needed(self):
        if not (self.faiss_manager and self.faiss_manager.is_available):
            return
        faiss_ids = self.faiss_manager.get_indexed_ids()
        with _connect_db(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, embedding FROM archival_memory WHERE embedding IS NOT NULL"
            ).fetchall()
        missing = [(rid, emb) for rid, emb in rows if rid not in faiss_ids and emb]
        if not missing:
            return
        for rid, emb in missing:
            self.faiss_manager.add(emb, rid)
        self.faiss_manager.save_async()

    def insert(self, category: str, content: str, importance: int = 5, metadata: dict = None):
        embedding = get_embedding(content) if self.use_semantic else None
        row_id = self._repo.insert(category, content, importance, embedding, metadata)

        insert_fts5(self.db_path, "archival_memory", row_id, content)

        if embedding and self.faiss_manager and self.faiss_manager.is_available:
            self.faiss_manager.add(embedding, row_id)
            if self.faiss_manager.needs_save(threshold=10):
                self.faiss_manager.save_async()

    def search(self, query: str, limit: int = 5) -> List[Dict]:
        fetch_limit = limit * 3
        self._sync_faiss_if_needed()
        results = self._search_chain.search(
            query, fetch_limit, self.db_path, table="archival_memory",
            format_archival=self._apply_importance_decay,
        )
        if results:
            return results[:limit]
        return []

    def _apply_importance_decay(self, rows: List[tuple]) -> List[Dict]:
        current_time = datetime.now()
        results = []
        for row in rows:
            _, category, content, importance, timestamp = row
            age_days = 0
            if timestamp:
                try:
                    if isinstance(timestamp, str):
                        clean_ts = timestamp.replace('Z', '+00:00')
                        ts = datetime.fromisoformat(clean_ts)
                        if ts.tzinfo is not None:
                            ts = ts.replace(tzinfo=None)
                    else:
                        ts = timestamp
                    age_days = max(0, (current_time - ts).total_seconds() / 86400)
                except Exception:
                    age_days = 0

            recency_score = math.exp(-age_days / conf.IMPORTANCE_DECAY_HALF_LIFE)
            effective_importance = (
                importance * conf.IMPORTANCE_STATIC_WEIGHT
                + recency_score * 10 * conf.IMPORTANCE_RECENCY_WEIGHT
            )
            results.append({
                "category": category,
                "content": content,
                "importance": importance,
                "effective_importance": round(effective_importance, 2),
                "age_days": round(age_days, 1),
                "timestamp": str(timestamp) if timestamp else None,
            })
        results.sort(key=lambda x: x["effective_importance"], reverse=True)
        return results

    def get_count(self) -> int:
        return self._repo.get_count()
