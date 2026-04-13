"""
Repository layer for WALL-E memory system.

Pure CRUD — no search logic, no FAISS, no embeddings generation.
Each repository owns its SQLite table schema and provides data access.
"""

import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import List, Optional

from .config import MEMORY_DIR

_log = logging.getLogger("walle.repository")


def _connect_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


class RecallRepository:
    """CRUD for the recall_memory table."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.path.join(MEMORY_DIR, "walle_recall_memory.db")
        self._init_db()

    def _init_db(self):
        with _connect_db(self.db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS recall_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                role TEXT, content TEXT, tools_used TEXT, metadata TEXT, embedding BLOB)""")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_timestamp ON recall_memory(timestamp DESC)"
            )

    def insert(
        self,
        role: str,
        content: str,
        embedding: Optional[bytes] = None,
        tools_used: list = None,
        metadata: dict = None,
    ) -> int:
        timestamp = datetime.now().isoformat()
        with _connect_db(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO recall_memory (timestamp, role, content, tools_used, metadata, embedding) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    timestamp,
                    role,
                    content,
                    json.dumps(tools_used) if tools_used else None,
                    json.dumps(metadata) if metadata else None,
                    embedding,
                ),
            )
            return cursor.lastrowid

    def update_embedding(self, row_id: int, embedding: bytes) -> None:
        with _connect_db(self.db_path) as conn:
            conn.execute(
                "UPDATE recall_memory SET embedding = ? WHERE id = ?",
                (embedding, row_id),
            )

    def get_count(self) -> int:
        with _connect_db(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM recall_memory").fetchone()[0]

    def get_old_memories(self, keep_recent: int) -> List[tuple]:
        """Fetch memories older than the most recent `keep_recent`."""
        with _connect_db(self.db_path) as conn:
            return conn.execute(
                "SELECT role, content, timestamp FROM recall_memory ORDER BY timestamp DESC LIMIT -1 OFFSET ?",
                (keep_recent,),
            ).fetchall()

    def delete_before(self, timestamp: str) -> int:
        with _connect_db(self.db_path) as conn:
            conn.execute("DELETE FROM recall_memory WHERE timestamp <= ?", (timestamp,))
            return conn.total_changes


class ArchivalRepository:
    """CRUD for the archival_memory table."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.path.join(MEMORY_DIR, "walle_archival_memory.db")
        self._init_db()

    def _init_db(self):
        with _connect_db(self.db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS archival_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                category TEXT, content TEXT, importance INTEGER, metadata TEXT, embedding BLOB)""")

    def insert(
        self,
        category: str,
        content: str,
        importance: int = 5,
        embedding: Optional[bytes] = None,
        metadata: dict = None,
    ) -> int:
        with _connect_db(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO archival_memory (category, content, importance, metadata, embedding) VALUES (?, ?, ?, ?, ?)",
                (
                    category,
                    content,
                    importance,
                    json.dumps(metadata) if metadata else None,
                    embedding,
                ),
            )
            return cursor.lastrowid

    def get_count(self) -> int:
        with _connect_db(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM archival_memory").fetchone()[0]
