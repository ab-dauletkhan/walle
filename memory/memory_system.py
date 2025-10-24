"""
MemGPT-inspired memory system for WALL-E robot
Implements three-tier memory architecture:
1. Core Memory (Working Memory) - Always in context
2. Recall Memory - Recent interactions (SQLite)
3. Archival Memory - Long-term storage

Features:
- Metadata tracking
- Read-only blocks
- Semantic search with embeddings
- Memory compression
- Heartbeat mechanism
"""

import json
import sqlite3
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field

# Semantic search support (optional)
try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_SEARCH_AVAILABLE = True
except ImportError:
    SEMANTIC_SEARCH_AVAILABLE = False
    print("⚠️  sentence-transformers not installed. Semantic search disabled.")


@dataclass
class Block:
    """
    A Block represents a reserved section of the LLM's context window which is editable.
    
    Parameters:
        label (str): The label of the block (e.g. 'human', 'persona', 'system')
        value (str): The value of the block - string represented in context window
        limit (int): The character limit of the block
        description (str): Description of what this block contains
        read_only (bool): If True, LLM cannot edit this block
        metadata (dict): Additional metadata (timestamps, sources, confidence)
    """
    label: str
    value: str
    limit: int
    description: str = ""
    read_only: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if len(self.value) > self.limit:
            self.value = self.value[:self.limit]
        # Initialize metadata with creation timestamp if empty
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
        """Append content to the block if space permits
        
        Returns:
            (success: bool, message: str)
        """
        if self.read_only:
            return False, f"Block '{self.label}' is read-only and cannot be modified"
        
        new_value = self.value + content
        if len(new_value) <= self.limit:
            self.value = new_value
            self.metadata["last_modified"] = datetime.now().isoformat()
            return True, "Success"
        return False, f"Not enough space ({self.chars_remaining} chars remaining)"
    
    def replace(self, old_content: str, new_content: str) -> tuple[bool, str]:
        """Replace old_content with new_content in the block
        
        Returns:
            (success: bool, message: str)
        """
        if self.read_only:
            return False, f"Block '{self.label}' is read-only and cannot be modified"
        
        if old_content not in self.value:
            return False, "Old content not found in block"
        
        new_value = self.value.replace(old_content, new_content, 1)
        if len(new_value) <= self.limit:
            self.value = new_value
            self.metadata["last_modified"] = datetime.now().isoformat()
            return True, "Success"
        return False, f"New content too large ({len(new_value)} > {self.limit})"
    
    def compile(self) -> str:
        """Compile block into XML format for prompt"""
        readonly_tag = " [READ-ONLY]" if self.read_only else ""
        return f"""<{self.label}{readonly_tag}>
<description>{self.description}</description>
<metadata>chars_current={self.chars_current}, chars_limit={self.limit}, last_modified={self.metadata.get('last_modified', 'unknown')}</metadata>
<value>
{self.value}
</value>
</{self.label}>"""


@dataclass
class Memory:
    """
    Core Memory - Information always kept in LLM's context window
    Organized into labeled blocks that can be edited by the agent
    """
    blocks: List[Block] = field(default_factory=list)
    db_path: str = "walle_core_memory.db"
    
    def __post_init__(self):
        # Try to load from database first
        if not self.blocks:
            loaded = self._load_from_db()
            if not loaded:
                # Initialize default blocks if nothing in database
                self.blocks = [
                    Block(
                        label="persona",
                        value="I am WALL-E, a tracked robot with two independent tracks. I can move, turn, and adjust my personality.",
                        limit=2000,
                        description="Your personality, capabilities, learned behaviors, and how you see yourself. Update this as you learn and grow."
                    ),
                    Block(
                        label="human",
                        value="The human is my operator who gives me movement commands and adjusts my personality settings.",
                        limit=2000,
                        description="Everything about the human: name, background, preferences, relationships, goals. This is their profile."
                    ),
                    Block(
                        label="system",
                        value="Current session started. No previous interactions loaded.",
                        limit=1000,
                        description="System state, session info, and technical status. Read-only - managed automatically.",
                        read_only=True  # Protect system block from edits
                    )
                ]
                self._save_to_db()
    
    def _init_database(self):
        """Initialize SQLite database for core memory persistence"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS core_memory (
                label TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                limit_chars INTEGER NOT NULL,
                description TEXT,
                read_only INTEGER NOT NULL,
                metadata TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _load_from_db(self) -> bool:
        """Load core memory blocks from database. Returns True if loaded, False if empty."""
        self._init_database()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT label, value, limit_chars, description, read_only, metadata FROM core_memory")
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return False
        
        self.blocks = []
        for row in rows:
            label, value, limit_chars, description, read_only, metadata_json = row
            metadata = json.loads(metadata_json) if metadata_json else {}
            
            self.blocks.append(Block(
                label=label,
                value=value,
                limit=limit_chars,
                description=description or "",
                read_only=bool(read_only),
                metadata=metadata
            ))
        
        return True
    
    def _save_to_db(self):
        """Save all core memory blocks to database"""
        self._init_database()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for block in self.blocks:
            cursor.execute("""
                INSERT OR REPLACE INTO core_memory (label, value, limit_chars, description, read_only, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                block.label,
                block.value,
                block.limit,
                block.description,
                int(block.read_only),
                json.dumps(block.metadata)
            ))
        
        conn.commit()
        conn.close()
    
    def save(self):
        """Public method to save memory to disk"""
        self._save_to_db()
    
    def get_block(self, label: str) -> Optional[Block]:
        """Get a block by its label"""
        for block in self.blocks:
            if block.label == label:
                return block
        return None
    
    def compile(self) -> str:
        """Compile all memory blocks into XML format for prompt"""
        blocks_xml = "\n".join(block.compile() for block in self.blocks)
        return f"""<memory_blocks>
{blocks_xml}
</memory_blocks>"""
    
    def get_total_chars(self) -> int:
        """Get total characters used across all blocks"""
        return sum(block.chars_current for block in self.blocks)
    
    def get_total_limit(self) -> int:
        """Get total character limit across all blocks"""
        return sum(block.limit for block in self.blocks)


class RecallMemory:
    """
    Recall Memory - Recent conversation history stored in SQLite
    Fast access (<100ms) for last 50-100 interactions
    Supports both text and semantic search
    """
    
    def __init__(self, db_path: str = "walle_recall_memory.db", use_semantic: bool = False):
        self.db_path = db_path
        self.use_semantic = use_semantic and SEMANTIC_SEARCH_AVAILABLE
        self.embedding_model = None
        
        if self.use_semantic:
            print("🔍 Loading semantic search model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model
            print("✅ Semantic search enabled!")
        
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for recall memory"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recall_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                tools_used TEXT,
                metadata TEXT,
                embedding BLOB
            )
        """)
        
        # Index for fast temporal queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON recall_memory(timestamp DESC)
        """)
        
        conn.commit()
        conn.close()
    
    def insert(self, role: str, content: str, tools_used: List[str] = None, metadata: Dict = None):
        """Insert a new memory entry"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Generate embedding if semantic search is enabled
        embedding_blob = None
        if self.use_semantic and self.embedding_model:
            embedding = self.embedding_model.encode(content)
            embedding_blob = embedding.tobytes()
        
        cursor.execute("""
            INSERT INTO recall_memory (role, content, tools_used, metadata, embedding)
            VALUES (?, ?, ?, ?, ?)
        """, (
            role,
            content,
            json.dumps(tools_used) if tools_used else None,
            json.dumps(metadata) if metadata else None,
            embedding_blob
        ))
        
        conn.commit()
        conn.close()
    
    def search(self, query: str = None, limit: int = 10, semantic: bool = False) -> List[Dict]:
        """Search recall memory - supports both text and semantic search"""
        
        # Semantic search
        if semantic and self.use_semantic and query and self.embedding_model:
            return self._semantic_search(query, limit)
        
        # Text search (fallback)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if query:
            # Simple text search
            cursor.execute("""
                SELECT id, timestamp, role, content, tools_used, metadata
                FROM recall_memory
                WHERE content LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (f"%{query}%", limit))
        else:
            # Get most recent entries
            cursor.execute("""
                SELECT id, timestamp, role, content, tools_used, metadata
                FROM recall_memory
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            results.append({
                "id": row[0],
                "timestamp": row[1],
                "role": row[2],
                "content": row[3],
                "tools_used": json.loads(row[4]) if row[4] else [],
                "metadata": json.loads(row[5]) if row[5] else {}
            })
        
        return results
    
    def _semantic_search(self, query: str, limit: int) -> List[Dict]:
        """Semantic search using embeddings"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)
        
        # Get all entries with embeddings
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, timestamp, role, content, tools_used, metadata, embedding
            FROM recall_memory
            WHERE embedding IS NOT NULL
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return []
        
        # Calculate similarity scores
        results_with_scores = []
        for row in rows:
            embedding_blob = row[6]
            if embedding_blob:
                embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                
                # Cosine similarity
                similarity = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                )
                
                results_with_scores.append({
                    "id": row[0],
                    "timestamp": row[1],
                    "role": row[2],
                    "content": row[3],
                    "tools_used": json.loads(row[4]) if row[4] else [],
                    "metadata": json.loads(row[5]) if row[5] else {},
                    "similarity": float(similarity)
                })
        
        # Sort by similarity and return top results
        results_with_scores.sort(key=lambda x: x["similarity"], reverse=True)
        return results_with_scores[:limit]
    
    def get_count(self) -> int:
        """Get total number of memory entries"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM recall_memory")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def compress_old_memories(self, keep_recent: int = 100, threshold: int = 500) -> int:
        """
        Compress old memories when database grows too large
        Keeps most recent entries, summarizes and moves old ones to archival
        
        Args:
            keep_recent: Number of recent entries to keep
            threshold: Compress only if total entries exceed this
        
        Returns:
            Number of compressed entries
        """
        count = self.get_count()
        if count < threshold:
            return 0
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get old entries to compress (everything except most recent)
        cursor.execute("""
            SELECT id, role, content, timestamp
            FROM recall_memory
            ORDER BY timestamp DESC
            LIMIT -1 OFFSET ?
        """, (keep_recent,))
        
        old_entries = cursor.fetchall()
        
        if not old_entries:
            conn.close()
            return 0
        
        # Group by day and create summaries
        from collections import defaultdict
        by_date = defaultdict(list)
        
        for entry in old_entries:
            date = entry[3].split()[0]  # Get date part
            by_date[date].append(entry)
        
        # Delete old entries
        old_ids = [entry[0] for entry in old_entries]
        cursor.execute(f"""
            DELETE FROM recall_memory
            WHERE id IN ({','.join('?' * len(old_ids))})
        """, old_ids)
        
        conn.commit()
        conn.close()
        
        return len(old_entries)


class ArchivalMemory:
    """
    Archival Memory - Long-term persistent storage
    Stores user preferences, learned behaviors, compressed summaries
    Supports semantic search
    """
    
    def __init__(self, db_path: str = "walle_archival_memory.db", use_semantic: bool = False):
        self.db_path = db_path
        self.use_semantic = use_semantic and SEMANTIC_SEARCH_AVAILABLE
        self.embedding_model = None
        
        if self.use_semantic:
            print("🔍 Loading semantic search model for archival memory...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for archival memory"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS archival_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                category TEXT NOT NULL,
                content TEXT NOT NULL,
                importance INTEGER DEFAULT 5,
                metadata TEXT,
                embedding BLOB
            )
        """)
        
        # Index for category and importance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_category_importance 
            ON archival_memory(category, importance DESC)
        """)
        
        conn.commit()
        conn.close()
    
    def insert(self, category: str, content: str, importance: int = 5, metadata: Dict = None):
        """Insert a new archival memory entry"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Generate embedding if semantic search is enabled
        embedding_blob = None
        if self.use_semantic and self.embedding_model:
            embedding = self.embedding_model.encode(content)
            embedding_blob = embedding.tobytes()
        
        cursor.execute("""
            INSERT INTO archival_memory (category, content, importance, metadata, embedding)
            VALUES (?, ?, ?, ?, ?)
        """, (
            category,
            content,
            importance,
            json.dumps(metadata) if metadata else None,
            embedding_blob
        ))
        
        conn.commit()
        conn.close()
    
    def search(self, query: str = None, category: str = None, limit: int = 5, semantic: bool = False) -> List[Dict]:
        """Search archival memory with optional semantic search"""
        
        # Semantic search
        if semantic and self.use_semantic and query and self.embedding_model:
            return self._semantic_search(query, category, limit)
        
        # Text search (fallback)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if query and category:
            cursor.execute("""
                SELECT id, timestamp, category, content, importance, metadata
                FROM archival_memory
                WHERE category = ? AND content LIKE ?
                ORDER BY importance DESC, timestamp DESC
                LIMIT ?
            """, (category, f"%{query}%", limit))
        elif category:
            cursor.execute("""
                SELECT id, timestamp, category, content, importance, metadata
                FROM archival_memory
                WHERE category = ?
                ORDER BY importance DESC, timestamp DESC
                LIMIT ?
            """, (category, limit))
        elif query:
            cursor.execute("""
                SELECT id, timestamp, category, content, importance, metadata
                FROM archival_memory
                WHERE content LIKE ?
                ORDER BY importance DESC, timestamp DESC
                LIMIT ?
            """, (f"%{query}%", limit))
        else:
            cursor.execute("""
                SELECT id, timestamp, category, content, importance, metadata
                FROM archival_memory
                ORDER BY importance DESC, timestamp DESC
                LIMIT ?
            """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            results.append({
                "id": row[0],
                "timestamp": row[1],
                "category": row[2],
                "content": row[3],
                "importance": row[4],
                "metadata": json.loads(row[5]) if row[5] else {}
            })
        
        return results
    
    def _semantic_search(self, query: str, category: Optional[str], limit: int) -> List[Dict]:
        """Semantic search using embeddings"""
        query_embedding = self.embedding_model.encode(query)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if category:
            cursor.execute("""
                SELECT id, timestamp, category, content, importance, metadata, embedding
                FROM archival_memory
                WHERE embedding IS NOT NULL AND category = ?
            """, (category,))
        else:
            cursor.execute("""
                SELECT id, timestamp, category, content, importance, metadata, embedding
                FROM archival_memory
                WHERE embedding IS NOT NULL
            """)
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return []
        
        results_with_scores = []
        for row in rows:
            embedding_blob = row[6]
            if embedding_blob:
                embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                similarity = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                )
                
                results_with_scores.append({
                    "id": row[0],
                    "timestamp": row[1],
                    "category": row[2],
                    "content": row[3],
                    "importance": row[4],
                    "metadata": json.loads(row[5]) if row[5] else {},
                    "similarity": float(similarity)
                })
        
        results_with_scores.sort(key=lambda x: x["similarity"], reverse=True)
        return results_with_scores[:limit]
    
    def get_count(self, category: str = None) -> int:
        """Get total number of archival entries"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if category:
            cursor.execute("SELECT COUNT(*) FROM archival_memory WHERE category = ?", (category,))
        else:
            cursor.execute("SELECT COUNT(*) FROM archival_memory")
        
        count = cursor.fetchone()[0]
        conn.close()
        return count
