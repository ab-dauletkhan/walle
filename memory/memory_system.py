"""
MemGPT-inspired memory system for WALL-E robot
Refactored for robustness, sentence-transformers embeddings, and proper resource management.
Optimized for NVIDIA Jetson Orin Nano.
"""

import json
import logging
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field

_log = logging.getLogger("walle.memory")

# Configuration for Embeddings (Optimized for Jetson)
from .config import conf, MEMORY_DIR
from .search_fallback import get_default_fallback_search_strategy


def _connect_db(db_path: str) -> sqlite3.Connection:
    """Open a SQLite connection with WAL mode and a reasonable busy timeout."""
    conn = sqlite3.connect(db_path, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

# Lazy load embedding model to save memory
_embedding_model = None
_embedding_device = None
_embedding_lock = threading.Lock()

def get_embedding_model():
    """Lazy load sentence-transformer model (thread-safe)"""
    global _embedding_model, _embedding_device

    if _embedding_model is not None:
        return _embedding_model

    with _embedding_lock:
        # Double-check after acquiring lock
        if _embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                import torch

                # Determine device
                _embedding_device = conf.EMBEDDING_DEVICE if torch.cuda.is_available() else "cpu"

                _log.info("🔧 Loading embedding model: %s on %s...", conf.EMBEDDING_MODEL, _embedding_device)
                _embedding_model = SentenceTransformer(
                    conf.EMBEDDING_MODEL,
                    device=_embedding_device
                )

                # Optimize for inference
                if _embedding_device == "cuda":
                    _embedding_model.half()  # Use FP16 for GPU efficiency

                _log.info("✅ Embedding model loaded")

            except ImportError:
                _log.warning("⚠️ sentence-transformers not installed. Embeddings disabled.")
                _log.warning("   Install with: pip install sentence-transformers")
                return None
            except Exception as e:
                _log.warning("⚠️ Failed to load embedding model: %s", e)
                return None

    return _embedding_model

def get_embedding(text: str) -> Optional[bytes]:
    """
    Generate embedding using sentence-transformers.

    Args:
        text: Text to embed

    Returns:
        Embedding as bytes, or None if failed
    """
    model = get_embedding_model()
    if model is None:
        return None

    try:
        # Generate embedding
        embedding = model.encode(
            text,
            convert_to_tensor=True,
            show_progress_bar=False,
            normalize_embeddings=True  # Normalize for cosine similarity
        )

        # Convert to numpy and bytes
        if _embedding_device == "cuda":
            embedding = embedding.cpu()

        embedding_np = embedding.numpy().astype(np.float32)
        return embedding_np.tobytes()

    except Exception as e:
        _log.warning("⚠️ Embedding error: %s", e)
        return None


# =============================================================================
# FAISS Integration for Fast Vector Search
# =============================================================================

_faiss_available = False
try:
    # IMPORTANT: Import torch BEFORE faiss to avoid OpenMP segfault on macOS
    # This is a known compatibility issue between PyTorch and FAISS
    import torch
    import faiss
    _faiss_available = True
except ImportError:
    pass  # FAISS not installed, will use fallback


class FAISSManager:
    """
    Manages FAISS index for fast vector similarity search.
    Uses IndexFlatIP (Inner Product) for normalized vectors (equivalent to cosine similarity).
    Provides O(log n) search instead of O(n) naive cosine similarity.
    """

    def __init__(self, dimension: int = 384, index_path: str = None):
        self.dimension = dimension
        self.index_path = index_path
        self.index = None
        self.id_map = []  # Maps FAISS index position to database row ID
        self._insertions_since_save = 0
        self._save_lock = threading.Lock()
        self._index_lock = threading.RLock()  # RLock for index operations (allows nested calls)
        self._save_in_progress = False

        if _faiss_available:
            self._load_or_create_index()
        else:
            _log.warning("⚠️ FAISS not available. Install with: pip install faiss-cpu")

    def _load_or_create_index(self):
        """Load existing index from disk or create new one"""
        from pathlib import Path

        if self.index_path and Path(self.index_path).exists():
            try:
                self.index = faiss.read_index(self.index_path)
                # Load id_map from companion file
                id_map_path = self.index_path + ".ids"
                if Path(id_map_path).exists():
                    with open(id_map_path, 'r') as f:
                        self.id_map = json.load(f)
                _log.info("✅ FAISS index loaded: %s vectors", self.index.ntotal)
            except Exception as e:
                _log.warning("⚠️ Failed to load FAISS index: %s", e)
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self):
        """Create new FAISS index"""
        if not _faiss_available:
            return
        # IndexFlatIP: Inner product (equals cosine similarity for normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.id_map = []

    def add(self, embedding_bytes: bytes, row_id: int):
        """Add single embedding to index (thread-safe)"""
        if not _faiss_available or self.index is None:
            return

        vec = np.frombuffer(embedding_bytes, dtype=np.float32).reshape(1, -1)
        # Normalize (embeddings should already be normalized, but ensure it)
        faiss.normalize_L2(vec)

        with self._index_lock:
            self.index.add(vec)
            self.id_map.append(row_id)
            self._insertions_since_save += 1

    def search(self, query_embedding_bytes: bytes, k: int = 10) -> List[tuple]:
        """
        Search for k nearest neighbors (thread-safe).
        Returns: List of (row_id, score) tuples, sorted by score descending
        """
        if not _faiss_available or self.index is None or self.index.ntotal == 0:
            return []

        q_vec = np.frombuffer(query_embedding_bytes, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(q_vec)

        with self._index_lock:
            # Limit k to available vectors
            k = min(k, self.index.ntotal)
            distances, indices = self.index.search(q_vec, k)
            # Copy id_map reference while holding lock
            id_map_snapshot = list(self.id_map)

        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(id_map_snapshot):
                results.append((id_map_snapshot[idx], float(distances[0][i])))

        return results

    def get_indexed_ids(self) -> List[int]:
        """Return a stable snapshot of indexed database row IDs."""
        with self._index_lock:
            return list(self.id_map)

    def save(self):
        """Persist index to disk (synchronous)"""
        if not _faiss_available or self.index is None or not self.index_path:
            return

        with self._save_lock:
            try:
                faiss.write_index(self.index, self.index_path)
                # Save id_map to companion file
                with open(self.index_path + ".ids", 'w') as f:
                    json.dump(self.id_map, f)
                self._insertions_since_save = 0
            except Exception as e:
                _log.error("⚠️ Failed to save FAISS index: %s", e)

    def save_async(self):
        """Persist index to disk in background thread (non-blocking)"""
        if not _faiss_available or self.index is None or not self.index_path:
            return

        # Skip if save already in progress
        if self._save_in_progress:
            return

        self._save_in_progress = True
        threading.Thread(target=self._save_async_impl, daemon=True).start()

    def _save_async_impl(self):
        """Background save implementation"""
        try:
            self.save()
        finally:
            self._save_in_progress = False

    def rebuild_from_db(self, db_path: str, table: str = "recall_memory"):
        """Rebuild index from database embeddings (thread-safe)"""
        if not _faiss_available:
            return

        # Validate table name to prevent SQL injection
        allowed_tables = {"recall_memory", "archival_memory"}
        if table not in allowed_tables:
            raise ValueError(f"Invalid table name: {table}. Allowed: {allowed_tables}")

        with _connect_db(db_path) as conn:
            cursor = conn.execute(f"SELECT id, embedding FROM {table} WHERE embedding IS NOT NULL")
            rows = cursor.fetchall()

        if not rows:
            with self._index_lock:
                self._create_new_index()
                self._insertions_since_save = 0
            return

        embeddings = []
        new_id_map = []
        for row_id, emb_bytes in rows:
            vec = np.frombuffer(emb_bytes, dtype=np.float32)
            embeddings.append(vec)
            new_id_map.append(row_id)

        embeddings_array = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_array)

        with self._index_lock:
            self._create_new_index()
            self.index.add(embeddings_array)
            self.id_map = new_id_map
            self._insertions_since_save = 0

        _log.info("✅ FAISS index rebuilt: %s vectors from %s", len(rows), table)

    def needs_save(self, threshold: int = 10) -> bool:
        """Check if index should be saved"""
        return self._insertions_since_save >= threshold

    @property
    def is_available(self) -> bool:
        return _faiss_available and self.index is not None


@dataclass
class Block:
    """A Block represents a reserved section of the LLM's context window."""
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
                "last_modified": datetime.now().isoformat()
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
        return f"""<{self.label}{readonly_tag}>
<description>{self.description}</description>
<metadata>chars={self.chars_current}/{self.limit}, modified={self.metadata.get('last_modified')}</metadata>
<value>
{self.value}
</value>
</{self.label}>"""

@dataclass
class Memory:
    """Core Memory - Always in context window"""
    blocks: List[Block] = field(default_factory=list)
    db_path: str = field(default_factory=lambda: os.path.join(MEMORY_DIR, "walle_core_memory.db"))
    
    def __post_init__(self):
        if not self.blocks:
            if not self._load_from_db():
                self.blocks = [
                    Block("persona", "I am WALL-E, a robot companion.", 2000, "My identity and capabilities."),
                    Block("human", "The human is my operator.", 2000, "User profile and preferences."),
                    Block("system", "System initialized.", 1000, "System status.", read_only=True)
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
            cursor = conn.execute("SELECT label, value, limit_chars, description, read_only, metadata FROM core_memory")
            rows = cursor.fetchall()
        
        if not rows: return False
        
        self.blocks = []
        for row in rows:
            self.blocks.append(Block(
                label=row[0], value=row[1], limit=row[2], description=row[3],
                read_only=bool(row[4]), metadata=json.loads(row[5] or "{}")
            ))
        return True
    
    def save(self):
        self._init_db()
        with _connect_db(self.db_path) as conn:
            for b in self.blocks:
                conn.execute("""INSERT OR REPLACE INTO core_memory VALUES (?, ?, ?, ?, ?, ?)""",
                             (b.label, b.value, b.limit, b.description, int(b.read_only), json.dumps(b.metadata)))

    def get_block(self, label: str) -> Optional[Block]:
        return next((b for b in self.blocks if b.label == label), None)
    
    def compile(self) -> str:
        return "<memory_blocks>\n" + "\n".join(b.compile() for b in self.blocks) + "\n</memory_blocks>"

class FAISSIndexMixin:
    """Mixin for classes that use FAISS indexing."""
    faiss_manager: FAISSManager = None
    db_path: str = ""

    def _ensure_faiss_index(self, table_name: str):
        """Rebuild FAISS index from database if index is empty but DB has data."""
        if self.faiss_manager and self.faiss_manager.is_available:
            if self.faiss_manager.index.ntotal == 0 and self.get_count() > 0:
                self.faiss_manager.rebuild_from_db(self.db_path, table_name)

    _ALLOWED_TABLES = {"recall_memory", "archival_memory"}

    def _sync_faiss_if_needed(self, table_name: str):
        """Incrementally sync FAISS with SQLite when deferred embeddings have landed.

        This is cheaper than a full rebuild: it only adds rows whose IDs are in
        SQLite but missing from the FAISS id_map.
        """
        if not (self.faiss_manager and self.faiss_manager.is_available):
            return
        if table_name not in self._ALLOWED_TABLES:
            raise ValueError(f"Invalid table name: {table_name}")

        faiss_ids = set(self.faiss_manager.get_indexed_ids())

        with _connect_db(self.db_path) as conn:
            rows = conn.execute(
                f"SELECT id, embedding FROM {table_name} WHERE embedding IS NOT NULL"
            ).fetchall()

        missing = [(rid, emb) for rid, emb in rows if rid not in faiss_ids and emb]
        if not missing:
            return

        for rid, emb in missing:
            self.faiss_manager.add(emb, rid)

        if missing:
            self.faiss_manager.save_async()

    def _faiss_search(self, query: str, limit: int, sql_template: str) -> Optional[List[tuple]]:
        """
        Common FAISS search: embed query → FAISS lookup → fetch rows from SQLite.
        Returns raw rows or None if FAISS unavailable/no results (caller should fallback).

        sql_template must contain {placeholders} for IN clause, e.g.:
            "SELECT id, content FROM table WHERE id IN ({placeholders})"
        """
        if not (self.use_semantic and query and self.faiss_manager and self.faiss_manager.is_available):
            return None

        q_emb = get_embedding(query)
        if not q_emb:
            return None

        faiss_results = self.faiss_manager.search(q_emb, limit)
        if not faiss_results:
            return None

        ids = [r[0] for r in faiss_results]
        placeholders = ','.join('?' * len(ids))
        with _connect_db(self.db_path) as conn:
            rows = conn.execute(
                sql_template.format(placeholders=placeholders), ids
            ).fetchall()

        if not rows:
            return None

        # Return (rows, faiss_results) so caller can preserve ranking
        return rows, faiss_results


class RecallMemory(FAISSIndexMixin):
    """Recall Memory - Recent history with FAISS-accelerated vector search"""

    _MAX_PENDING_EMBEDDINGS = 50  # Backpressure: skip deferred embedding if queue is full

    def __init__(self, db_path: str = None, use_semantic: bool = False):
        self.db_path = db_path or os.path.join(MEMORY_DIR, "walle_recall_memory.db")
        self.use_semantic = use_semantic
        self._fallback_search = get_default_fallback_search_strategy()
        self._embedding_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="emb")
        self._pending_futures: set = set()
        self._pending_lock = threading.Lock()
        self._init_db()

        # Initialize FAISS manager for fast vector search
        self.faiss_manager = None
        if use_semantic and conf.USE_FAISS and _faiss_available:
            faiss_path = os.path.join(MEMORY_DIR, conf.FAISS_INDEX_PATH.replace(".index", "_recall.index"))
            self.faiss_manager = FAISSManager(
                dimension=conf.FAISS_DIMENSION,
                index_path=faiss_path
            )
            self._ensure_faiss_index("recall_memory")

    def _init_db(self):
        with _connect_db(self.db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS recall_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                role TEXT, content TEXT, tools_used TEXT, metadata TEXT, embedding BLOB)""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON recall_memory(timestamp DESC)")

    def insert(self, role: str, content: str, tools_used: List[str] = None, metadata: Dict = None, defer_embedding: bool = False):
        """
        Insert a message into recall memory.

        Args:
            defer_embedding: If True, generates embedding in background thread (faster TTFT)
        """
        # Use Python timestamp for microsecond precision
        timestamp = datetime.now().isoformat()

        # Generate embedding synchronously or defer to background
        if self.use_semantic and not defer_embedding:
            embedding = get_embedding(content)
        else:
            embedding = None

        with _connect_db(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO recall_memory (timestamp, role, content, tools_used, metadata, embedding) VALUES (?, ?, ?, ?, ?, ?)",
                (timestamp, role, content, json.dumps(tools_used) if tools_used else None,
                 json.dumps(metadata) if metadata else None, embedding)
            )
            row_id = cursor.lastrowid

        # Add to FAISS index (sync path)
        if embedding and self.faiss_manager and self.faiss_manager.is_available:
            self.faiss_manager.add(embedding, row_id)
            if self.faiss_manager.needs_save(threshold=10):
                self.faiss_manager.save_async()  # Non-blocking save

        # Deferred embedding - generate in thread pool (with backpressure)
        if self.use_semantic and defer_embedding:
            with self._pending_lock:
                if len(self._pending_futures) >= self._MAX_PENDING_EMBEDDINGS:
                    _log.debug("Embedding queue full (%d), skipping deferred embedding for row %d",
                               len(self._pending_futures), row_id)
                else:
                    future = self._embedding_pool.submit(self._generate_embedding_async, row_id, content)
                    self._pending_futures.add(future)
                    future.add_done_callback(self._on_embedding_done)

    def _on_embedding_done(self, future):
        """Callback for completed embedding futures — lock-protected set removal."""
        with self._pending_lock:
            self._pending_futures.discard(future)

    def _generate_embedding_async(self, row_id: int, content: str):
        """Generate embedding in background and update DB + FAISS."""
        try:
            embedding = get_embedding(content)
            if not embedding:
                return

            # Update SQLite
            with _connect_db(self.db_path) as conn:
                conn.execute(
                    "UPDATE recall_memory SET embedding = ? WHERE id = ?",
                    (embedding, row_id)
                )

            # Update FAISS index
            if self.faiss_manager and self.faiss_manager.is_available:
                self.faiss_manager.add(embedding, row_id)
                if self.faiss_manager.needs_save(threshold=10):
                    self.faiss_manager.save()  # Already in background, sync is fine
        except Exception as e:
            _log.warning("⚠️ Background embedding failed: %s", e)

    def search(self, query: str = None, limit: int = 10) -> List[Dict]:
        """Search recall memory with FAISS-accelerated semantic search"""

        # Sync any deferred embeddings that landed in SQLite but not yet in FAISS
        self._sync_faiss_if_needed("recall_memory")

        # FAISS-accelerated semantic search (O(log n))
        result = self._faiss_search(
            query, limit,
            "SELECT id, timestamp, role, content FROM recall_memory WHERE id IN ({placeholders})"
        )
        if result:
            rows, faiss_results = result
            row_map = {r[0]: r for r in rows}
            results = []
            for row_id, score in faiss_results:
                if row_id in row_map:
                    r = row_map[row_id]
                    results.append({"role": r[2], "content": r[3], "timestamp": r[1], "score": round(score, 4)})
            return results

        # Fallback: Naive semantic search (if FAISS unavailable)
        with _connect_db(self.db_path) as conn:
            if self.use_semantic and query:
                q_emb = get_embedding(query)
                if q_emb:
                    rows = conn.execute(
                        "SELECT id, timestamp, role, content, tools_used, embedding FROM recall_memory WHERE embedding IS NOT NULL"
                    ).fetchall()
                    q_vec = np.frombuffer(q_emb, dtype=np.float32)
                    results = []
                    for r in rows:
                        vec = np.frombuffer(r[5], dtype=np.float32)
                        score = np.dot(q_vec, vec) / (np.linalg.norm(q_vec) * np.linalg.norm(vec) + 1e-8)
                        results.append((score, r))
                    results.sort(key=lambda x: x[0], reverse=True)
                    return [{"role": r[1][2], "content": r[1][3], "timestamp": r[1][1]} for r in results[:limit]]

                rows = conn.execute(
                    "SELECT role, content, timestamp FROM recall_memory ORDER BY timestamp DESC"
                ).fetchall()
                ranked = self._fallback_search.rank_rows(query, rows, lambda row: row[1])
                if ranked:
                    ranked.sort(key=lambda item: (item.score, item.row[2]), reverse=True)
                    return [
                        {"role": row[0], "content": row[1], "timestamp": row[2]}
                        for row in (item.row for item in ranked[:limit])
                    ]

            # Text search fallback
            if query:
                rows = conn.execute(
                    "SELECT role, content, timestamp FROM recall_memory WHERE content LIKE ? ORDER BY timestamp DESC LIMIT ?",
                    (f"%{query}%", limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT role, content, timestamp FROM recall_memory ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                ).fetchall()

            return [{"role": r[0], "content": r[1], "timestamp": r[2]} for r in rows]

    def get_count(self) -> int:
        with _connect_db(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM recall_memory").fetchone()[0]

    def compress_old_memories(self, summarizer_func: Callable[[str], str], archival_memory: 'ArchivalMemory', keep_recent: int = 50):
        """Summarizes old memories into archival storage before deletion."""
        count = self.get_count()
        if count <= keep_recent:
            return 0

        with _connect_db(self.db_path) as conn:
            # Fetch old memories
            rows = conn.execute("""SELECT role, content, timestamp FROM recall_memory 
                                 ORDER BY timestamp DESC LIMIT -1 OFFSET ?""", (keep_recent,)).fetchall()

            if not rows:
                return 0

            # Format for summarization
            # Reverse to get chronological order for the summary
            chronological_rows = rows[::-1]
            text_to_summarize = "\n".join([f"[{r[2]}] {r[0]}: {r[1]}" for r in chronological_rows])

            _log.info("⏳ Summarizing old memories...")
            summary = summarizer_func(text_to_summarize)

            # Store in Archival
            archival_memory.insert("conversation_summary", summary, importance=3)

            # Delete from Recall
            # We strictly rely on timestamps to delete what we just fetched
            newest_of_old = rows[0][2]  # The timestamp of the newest item in the old batch
            conn.execute("DELETE FROM recall_memory WHERE timestamp <= ?", (newest_of_old,))
            deleted_count = len(rows)

        if self.faiss_manager and self.faiss_manager.is_available:
            self.faiss_manager.rebuild_from_db(self.db_path, "recall_memory")
            self.faiss_manager.save()

        return deleted_count

    def shutdown(self, timeout: float = 30):
        """Wait for pending background embeddings (with timeout to avoid hangs)."""
        self._embedding_pool.shutdown(wait=False, cancel_futures=True)
        # Wait for in-flight futures to complete (cancel_futures only
        # prevents queued futures from starting, not in-flight ones).
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
                pass  # Already logged in _generate_embedding_async

class ArchivalMemory(FAISSIndexMixin):
    """Archival Memory - Long term storage with FAISS search and importance decay"""

    def __init__(self, db_path: str = None, use_semantic: bool = False):
        self.db_path = db_path or os.path.join(MEMORY_DIR, "walle_archival_memory.db")
        self.use_semantic = use_semantic
        self._fallback_search = get_default_fallback_search_strategy()
        self._init_db()

        # Initialize FAISS manager for archival memory
        self.faiss_manager = None
        if use_semantic and conf.USE_FAISS and _faiss_available:
            faiss_path = os.path.join(MEMORY_DIR, conf.FAISS_INDEX_PATH.replace(".index", "_archival.index"))
            self.faiss_manager = FAISSManager(
                dimension=conf.FAISS_DIMENSION,
                index_path=faiss_path
            )
            self._ensure_faiss_index("archival_memory")

    def _init_db(self):
        with _connect_db(self.db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS archival_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                category TEXT, content TEXT, importance INTEGER, metadata TEXT, embedding BLOB)""")

    def insert(self, category: str, content: str, importance: int = 5, metadata: Dict = None):
        embedding = get_embedding(content) if self.use_semantic else None

        with _connect_db(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO archival_memory (category, content, importance, metadata, embedding) VALUES (?, ?, ?, ?, ?)",
                (category, content, importance, json.dumps(metadata) if metadata else None, embedding)
            )
            row_id = cursor.lastrowid

        # Add to FAISS index (use threshold-based save like RecallMemory)
        if embedding and self.faiss_manager and self.faiss_manager.is_available:
            self.faiss_manager.add(embedding, row_id)
            if self.faiss_manager.needs_save(threshold=10):
                self.faiss_manager.save_async()  # Non-blocking save

    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search archival memory with FAISS and importance decay"""
        fetch_limit = limit * 3  # Fetch more for decay filtering

        # Sync any embeddings that landed in SQLite but not yet in FAISS
        self._sync_faiss_if_needed("archival_memory")

        # FAISS-accelerated semantic search
        result = self._faiss_search(
            query, fetch_limit,
            "SELECT id, category, content, importance, timestamp FROM archival_memory WHERE id IN ({placeholders})"
        )
        if result:
            rows, _ = result
            return self._apply_importance_decay(rows)[:limit]

        # Fallback: Text search with decay
        with _connect_db(self.db_path) as conn:
            if self.use_semantic:
                q_emb = get_embedding(query)
                if not q_emb:
                    rows = conn.execute(
                        "SELECT id, category, content, importance, timestamp FROM archival_memory"
                    ).fetchall()
                    ranked = self._fallback_search.rank_rows(query, rows, lambda row: row[2])
                    if ranked:
                        ranked.sort(key=lambda item: item.score, reverse=True)
                        return self._apply_importance_decay(
                            [item.row for item in ranked[:fetch_limit]]
                        )[:limit]

            rows = conn.execute(
                "SELECT id, category, content, importance, timestamp FROM archival_memory WHERE content LIKE ? ORDER BY importance DESC LIMIT ?",
                (f"%{query}%", fetch_limit)
            ).fetchall()

        if not rows:
            return []

        return self._apply_importance_decay(rows)[:limit]

    def _apply_importance_decay(self, rows: List[tuple]) -> List[Dict]:
        """
        Apply temporal importance decay to search results.
        Formula: effective = static_importance * 0.7 + recency_score * 0.3
        Recency score uses exponential decay with configurable half-life.
        """
        import math

        current_time = datetime.now()
        results = []

        for row in rows:
            row_id, category, content, importance, timestamp = row

            # Calculate age in days
            age_days = 0
            if timestamp:
                try:
                    if isinstance(timestamp, str):
                        # Parse ISO format timestamp, normalizing to naive local time
                        # Handle: "2024-01-01T12:00:00", "...Z", "...+00:00", "...+05:00"
                        clean_ts = timestamp.replace('Z', '+00:00')
                        ts = datetime.fromisoformat(clean_ts)
                        # Convert timezone-aware to naive local time for consistent comparison
                        if ts.tzinfo is not None:
                            ts = ts.replace(tzinfo=None)
                    else:
                        ts = timestamp
                    age_days = max(0, (current_time - ts).total_seconds() / 86400)
                except Exception:
                    age_days = 0

            # Calculate recency score: exp(-age_days / half_life)
            # At half_life days, recency_score = 0.5
            recency_score = math.exp(-age_days / conf.IMPORTANCE_DECAY_HALF_LIFE)

            # Effective importance formula
            # Scale recency to 0-10 range to match importance scale
            effective_importance = (
                importance * conf.IMPORTANCE_STATIC_WEIGHT +
                recency_score * 10 * conf.IMPORTANCE_RECENCY_WEIGHT
            )

            results.append({
                "category": category,
                "content": content,
                "importance": importance,
                "effective_importance": round(effective_importance, 2),
                "age_days": round(age_days, 1),
                "timestamp": str(timestamp) if timestamp else None
            })

        # Sort by effective importance (descending)
        results.sort(key=lambda x: x["effective_importance"], reverse=True)
        return results

    def get_count(self) -> int:
        with _connect_db(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM archival_memory").fetchone()[0]
