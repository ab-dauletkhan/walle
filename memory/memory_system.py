"""
MemGPT-inspired memory system for WALL-E robot
Refactored for robustness, sentence-transformers embeddings, and proper resource management.
Optimized for NVIDIA Jetson Orin Nano.
"""

import json
import sqlite3
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field

# Configuration for Embeddings (Optimized for Jetson)
from config import conf

# Lazy load embedding model to save memory
_embedding_model = None
_embedding_device = None

def get_embedding_model():
    """Lazy load sentence-transformer model"""
    global _embedding_model, _embedding_device

    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            import torch

            # Determine device
            _embedding_device = conf.EMBEDDING_DEVICE if torch.cuda.is_available() else "cpu"

            print(f"🔧 Loading embedding model: {conf.EMBEDDING_MODEL} on {_embedding_device}...")
            _embedding_model = SentenceTransformer(
                conf.EMBEDDING_MODEL,
                device=_embedding_device
            )

            # Optimize for inference
            if _embedding_device == "cuda":
                _embedding_model.half()  # Use FP16 for GPU efficiency

            print(f"✅ Embedding model loaded")

        except ImportError:
            print("⚠️ sentence-transformers not installed. Embeddings disabled.")
            print("   Install with: pip install sentence-transformers")
            return None
        except Exception as e:
            print(f"⚠️ Failed to load embedding model: {e}")
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
        print(f"⚠️ Embedding error: {e}")
        return None


# =============================================================================
# FAISS Integration for Fast Vector Search
# =============================================================================

_faiss_available = False
try:
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

        if _faiss_available:
            self._load_or_create_index()
        else:
            print("⚠️ FAISS not available. Install with: pip install faiss-cpu")

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
                print(f"✅ FAISS index loaded: {self.index.ntotal} vectors")
            except Exception as e:
                print(f"⚠️ Failed to load FAISS index: {e}")
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
        """Add single embedding to index"""
        if not _faiss_available or self.index is None:
            return

        vec = np.frombuffer(embedding_bytes, dtype=np.float32).reshape(1, -1)
        # Normalize (embeddings should already be normalized, but ensure it)
        faiss.normalize_L2(vec)

        self.index.add(vec)
        self.id_map.append(row_id)
        self._insertions_since_save += 1

    def search(self, query_embedding_bytes: bytes, k: int = 10) -> List[tuple]:
        """
        Search for k nearest neighbors.
        Returns: List of (row_id, score) tuples, sorted by score descending
        """
        if not _faiss_available or self.index is None or self.index.ntotal == 0:
            return []

        q_vec = np.frombuffer(query_embedding_bytes, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(q_vec)

        # Limit k to available vectors
        k = min(k, self.index.ntotal)

        distances, indices = self.index.search(q_vec, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.id_map):
                results.append((self.id_map[idx], float(distances[0][i])))

        return results

    def save(self):
        """Persist index to disk"""
        if not _faiss_available or self.index is None or not self.index_path:
            return

        try:
            faiss.write_index(self.index, self.index_path)
            # Save id_map to companion file
            with open(self.index_path + ".ids", 'w') as f:
                json.dump(self.id_map, f)
            self._insertions_since_save = 0
        except Exception as e:
            print(f"⚠️ Failed to save FAISS index: {e}")

    def rebuild_from_db(self, db_path: str, table: str = "recall_memory"):
        """Rebuild index from database embeddings"""
        if not _faiss_available:
            return

        self._create_new_index()

        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(f"SELECT id, embedding FROM {table} WHERE embedding IS NOT NULL")
            rows = cursor.fetchall()

        if not rows:
            return

        embeddings = []
        for row_id, emb_bytes in rows:
            vec = np.frombuffer(emb_bytes, dtype=np.float32)
            embeddings.append(vec)
            self.id_map.append(row_id)

        embeddings_array = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_array)
        self.index.add(embeddings_array)

        self._insertions_since_save = 0
        print(f"✅ FAISS index rebuilt: {len(rows)} vectors from {table}")

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
    db_path: str = "walle_core_memory.db"
    
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
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS core_memory 
                          (label TEXT PRIMARY KEY, value TEXT, limit_chars INTEGER, 
                           description TEXT, read_only INTEGER, metadata TEXT)""")

    def _load_from_db(self) -> bool:
        self._init_db()
        with sqlite3.connect(self.db_path) as conn:
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
        with sqlite3.connect(self.db_path) as conn:
            for b in self.blocks:
                conn.execute("""INSERT OR REPLACE INTO core_memory VALUES (?, ?, ?, ?, ?, ?)""",
                             (b.label, b.value, b.limit, b.description, int(b.read_only), json.dumps(b.metadata)))

    def get_block(self, label: str) -> Optional[Block]:
        return next((b for b in self.blocks if b.label == label), None)
    
    def compile(self) -> str:
        return "<memory_blocks>\n" + "\n".join(b.compile() for b in self.blocks) + "\n</memory_blocks>"

class RecallMemory:
    """Recall Memory - Recent history with FAISS-accelerated vector search"""

    def __init__(self, db_path: str = "walle_recall_memory.db", use_semantic: bool = False):
        self.db_path = db_path
        self.use_semantic = use_semantic
        self._init_db()

        # Initialize FAISS manager for fast vector search
        self.faiss_manager = None
        if use_semantic and conf.USE_FAISS and _faiss_available:
            faiss_path = conf.FAISS_INDEX_PATH.replace(".index", "_recall.index")
            self.faiss_manager = FAISSManager(
                dimension=conf.FAISS_DIMENSION,
                index_path=faiss_path
            )
            # Rebuild index if empty but DB has data
            if self.faiss_manager.is_available and self.faiss_manager.index.ntotal == 0:
                if self.get_count() > 0:
                    self.faiss_manager.rebuild_from_db(self.db_path, "recall_memory")
                    self.faiss_manager.save()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS recall_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                role TEXT, content TEXT, tools_used TEXT, metadata TEXT, embedding BLOB)""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON recall_memory(timestamp DESC)")

    def insert(self, role: str, content: str, tools_used: List[str] = None, metadata: Dict = None):
        embedding = get_embedding(content) if self.use_semantic else None
        # Use Python timestamp for microsecond precision (SQLite CURRENT_TIMESTAMP is second-precision)
        timestamp = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO recall_memory (timestamp, role, content, tools_used, metadata, embedding) VALUES (?, ?, ?, ?, ?, ?)",
                (timestamp, role, content, json.dumps(tools_used) if tools_used else None,
                 json.dumps(metadata) if metadata else None, embedding)
            )
            row_id = cursor.lastrowid

        # Add to FAISS index
        if embedding and self.faiss_manager and self.faiss_manager.is_available:
            self.faiss_manager.add(embedding, row_id)
            # Save periodically
            if self.faiss_manager.needs_save(threshold=10):
                self.faiss_manager.save()

    def search(self, query: str = None, limit: int = 10) -> List[Dict]:
        """Search recall memory with FAISS-accelerated semantic search"""

        # FAISS-accelerated semantic search (O(log n) instead of O(n))
        if self.use_semantic and query and self.faiss_manager and self.faiss_manager.is_available:
            q_emb = get_embedding(query)
            if q_emb:
                faiss_results = self.faiss_manager.search(q_emb, limit)

                if faiss_results:
                    # Fetch full records by IDs from SQLite
                    ids = [r[0] for r in faiss_results]
                    with sqlite3.connect(self.db_path) as conn:
                        placeholders = ','.join('?' * len(ids))
                        rows = conn.execute(
                            f"SELECT id, timestamp, role, content FROM recall_memory WHERE id IN ({placeholders})",
                            ids
                        ).fetchall()

                    # Create lookup and preserve FAISS ranking order
                    row_map = {r[0]: r for r in rows}
                    results = []
                    for row_id, score in faiss_results:
                        if row_id in row_map:
                            r = row_map[row_id]
                            results.append({
                                "role": r[2],
                                "content": r[3],
                                "timestamp": r[1],
                                "score": round(score, 4)
                            })
                    return results

        # Fallback: Naive semantic search (if FAISS unavailable)
        with sqlite3.connect(self.db_path) as conn:
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
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM recall_memory").fetchone()[0]

    def compress_old_memories(self, summarizer_func: Callable[[str], str], archival_memory: 'ArchivalMemory', keep_recent: int = 50):
        """Summarizes old memories into archival storage before deletion."""
        count = self.get_count()
        if count <= keep_recent: return 0

        with sqlite3.connect(self.db_path) as conn:
            # Fetch old memories
            rows = conn.execute("""SELECT role, content, timestamp FROM recall_memory 
                                 ORDER BY timestamp DESC LIMIT -1 OFFSET ?""", (keep_recent,)).fetchall()
            
            if not rows: return 0
            
            # Format for summarization
            # Reverse to get chronological order for the summary
            chronological_rows = rows[::-1] 
            text_to_summarize = "\n".join([f"[{r[2]}] {r[0]}: {r[1]}" for r in chronological_rows])
            
            print("⏳ Summarizing old memories...")
            summary = summarizer_func(text_to_summarize)
            
            # Store in Archival
            archival_memory.insert("conversation_summary", summary, importance=3)
            
            # Delete from Recall
            # We strictly rely on timestamps to delete what we just fetched
            newest_of_old = rows[0][2] # The timestamp of the 'newest' item in the 'old' batch
            conn.execute("DELETE FROM recall_memory WHERE timestamp <= ?", (newest_of_old,))
            
            return len(rows)

class ArchivalMemory:
    """Archival Memory - Long term storage with FAISS search and importance decay"""

    def __init__(self, db_path: str = "walle_archival_memory.db", use_semantic: bool = False):
        self.db_path = db_path
        self.use_semantic = use_semantic
        self._init_db()

        # Initialize FAISS manager for archival memory
        self.faiss_manager = None
        if use_semantic and conf.USE_FAISS and _faiss_available:
            faiss_path = conf.FAISS_INDEX_PATH.replace(".index", "_archival.index")
            self.faiss_manager = FAISSManager(
                dimension=conf.FAISS_DIMENSION,
                index_path=faiss_path
            )
            # Rebuild index if empty but DB has data
            if self.faiss_manager.is_available and self.faiss_manager.index.ntotal == 0:
                if self.get_count() > 0:
                    self.faiss_manager.rebuild_from_db(self.db_path, "archival_memory")
                    self.faiss_manager.save()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS archival_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                category TEXT, content TEXT, importance INTEGER, metadata TEXT, embedding BLOB)""")

    def insert(self, category: str, content: str, importance: int = 5, metadata: Dict = None):
        embedding = get_embedding(content) if self.use_semantic else None

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO archival_memory (category, content, importance, metadata, embedding) VALUES (?, ?, ?, ?, ?)",
                (category, content, importance, json.dumps(metadata) if metadata else None, embedding)
            )
            row_id = cursor.lastrowid

        # Add to FAISS index
        if embedding and self.faiss_manager and self.faiss_manager.is_available:
            self.faiss_manager.add(embedding, row_id)
            self.faiss_manager.save()

    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search archival memory with FAISS and importance decay"""
        fetch_limit = limit * 3  # Fetch more for decay filtering

        # FAISS-accelerated semantic search
        if self.use_semantic and self.faiss_manager and self.faiss_manager.is_available:
            q_emb = get_embedding(query)
            if q_emb:
                faiss_results = self.faiss_manager.search(q_emb, fetch_limit)

                if faiss_results:
                    ids = [r[0] for r in faiss_results]
                    with sqlite3.connect(self.db_path) as conn:
                        placeholders = ','.join('?' * len(ids))
                        rows = conn.execute(
                            f"SELECT id, category, content, importance, timestamp FROM archival_memory WHERE id IN ({placeholders})",
                            ids
                        ).fetchall()

                    # Only return if we found matching rows, otherwise fall back to text search
                    if rows:
                        results = self._apply_importance_decay(rows)
                        return results[:limit]

        # Fallback: Text search with decay
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, category, content, importance, timestamp FROM archival_memory WHERE content LIKE ? ORDER BY importance DESC LIMIT ?",
                (f"%{query}%", fetch_limit)
            ).fetchall()

        if not rows:
            return []

        # Apply importance decay
        results = self._apply_importance_decay(rows)
        return results[:limit]

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
                        # Parse ISO format timestamp
                        ts = datetime.fromisoformat(timestamp.replace('Z', '+00:00').replace('+00:00', ''))
                    else:
                        ts = timestamp
                    age_days = (current_time - ts).total_seconds() / 86400
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
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM archival_memory").fetchone()[0]