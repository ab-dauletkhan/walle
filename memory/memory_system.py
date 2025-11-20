"""
MemGPT-inspired memory system for WALL-E robot
Refactored for robustness, Ollama embeddings, and proper resource management.
"""

import json
import sqlite3
import requests
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field

# Configuration for Ollama Embeddings
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"  # Ensure you run: ollama pull nomic-embed-text

def get_ollama_embedding(text: str) -> bytes:
    """Fetch embedding from Ollama API to save RAM"""
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBEDDING_MODEL, "prompt": text}
        )
        if resp.status_code == 200:
            vector = resp.json().get('embedding')
            return np.array(vector, dtype=np.float32).tobytes()
    except Exception as e:
        print(f"⚠️ Embedding error: {e}")
    return None

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
    """Recall Memory - Recent history with vector search"""
    def __init__(self, db_path: str = "walle_recall_memory.db", use_semantic: bool = False):
        self.db_path = db_path
        self.use_semantic = use_semantic
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS recall_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                role TEXT, content TEXT, tools_used TEXT, metadata TEXT, embedding BLOB)""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON recall_memory(timestamp DESC)")

    def insert(self, role: str, content: str, tools_used: List[str] = None, metadata: Dict = None):
        embedding = get_ollama_embedding(content) if self.use_semantic else None
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO recall_memory (role, content, tools_used, metadata, embedding) VALUES (?, ?, ?, ?, ?)",
                         (role, content, json.dumps(tools_used) if tools_used else None, 
                          json.dumps(metadata) if metadata else None, embedding))

    def search(self, query: str = None, limit: int = 10) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            if self.use_semantic and query:
                q_emb = get_ollama_embedding(query)
                if q_emb:
                    # Note: This is a naive Python-side cosine similarity for SQLite. 
                    # Production would use pgvector or sqlite-vss.
                    rows = conn.execute("SELECT id, timestamp, role, content, tools_used, embedding FROM recall_memory WHERE embedding IS NOT NULL").fetchall()
                    q_vec = np.frombuffer(q_emb, dtype=np.float32)
                    results = []
                    for r in rows:
                        vec = np.frombuffer(r[5], dtype=np.float32)
                        score = np.dot(q_vec, vec) / (np.linalg.norm(q_vec) * np.linalg.norm(vec))
                        results.append((score, r))
                    results.sort(key=lambda x: x[0], reverse=True)
                    return [{"role": r[1][2], "content": r[1][3], "timestamp": r[1][1]} for r in results[:limit]]
            
            # Fallback Text Search
            if query:
                rows = conn.execute("SELECT role, content, timestamp FROM recall_memory WHERE content LIKE ? ORDER BY timestamp DESC LIMIT ?", (f"%{query}%", limit)).fetchall()
            else:
                rows = conn.execute("SELECT role, content, timestamp FROM recall_memory ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
            
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
    """Archival Memory - Long term storage"""
    def __init__(self, db_path: str = "walle_archival_memory.db", use_semantic: bool = False):
        self.db_path = db_path
        self.use_semantic = use_semantic
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS archival_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                category TEXT, content TEXT, importance INTEGER, metadata TEXT, embedding BLOB)""")

    def insert(self, category: str, content: str, importance: int = 5, metadata: Dict = None):
        embedding = get_ollama_embedding(content) if self.use_semantic else None
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO archival_memory (category, content, importance, metadata, embedding) VALUES (?, ?, ?, ?, ?)",
                         (category, content, importance, json.dumps(metadata) if metadata else None, embedding))

    def search(self, query: str, limit: int = 5) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            # (Simplified: Only implementing Text Search for brevity, add vector logic if needed similar to Recall)
            rows = conn.execute("SELECT category, content, importance FROM archival_memory WHERE content LIKE ? ORDER BY importance DESC LIMIT ?",
                                (f"%{query}%", limit)).fetchall()
            return [{"category": r[0], "content": r[1], "importance": r[2]} for r in rows]

    def get_count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM archival_memory").fetchone()[0]