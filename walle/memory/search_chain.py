"""
Search Chain of Responsibility for WALL-E memory system.

Each handler tries its search strategy; if it returns no results,
the request passes to the next handler in the chain.

Chain order: FAISS (vector) → FTS5 (BM25) → Recent (timestamp)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

_log = logging.getLogger("walle.search")


def _connect_db(db_path: str):
    """Import-free re-use of the project's WAL-mode SQLite opener."""
    import sqlite3

    conn = sqlite3.connect(db_path, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ---------------------------------------------------------------------------
# Base handler
# ---------------------------------------------------------------------------
class SearchHandler(ABC):
    """Base class for Chain of Responsibility search handlers."""

    def __init__(self, next_handler: Optional[SearchHandler] = None):
        self._next = next_handler

    @abstractmethod
    def search(
        self,
        query: Optional[str],
        limit: int,
        db_path: str,
        table: str = "recall_memory",
        **ctx,
    ) -> Optional[List[Dict]]: ...

    def _try_next(self, query, limit, db_path, table, **ctx):
        if self._next is not None:
            return self._next.search(query, limit, db_path, table, **ctx)
        return None


# ---------------------------------------------------------------------------
# FAISS vector search (O(log n))
# ---------------------------------------------------------------------------
class FAISSSearchHandler(SearchHandler):
    """Semantic search via FAISS index. Falls through if FAISS unavailable or no results."""

    def __init__(
        self,
        faiss_manager,
        get_embedding_fn,
        use_semantic: bool = False,
        next_handler: Optional[SearchHandler] = None,
    ):
        super().__init__(next_handler)
        self._faiss = faiss_manager
        self._get_embedding = get_embedding_fn
        self._use_semantic = use_semantic

    def search(self, query, limit, db_path, table="recall_memory", **ctx):
        if not (
            self._use_semantic and query and self._faiss and self._faiss.is_available
        ):
            return self._try_next(query, limit, db_path, table, **ctx)

        q_emb = self._get_embedding(query)
        if not q_emb:
            return self._try_next(query, limit, db_path, table, **ctx)

        faiss_results = self._faiss.search(q_emb, limit)
        if not faiss_results:
            return self._try_next(query, limit, db_path, table, **ctx)

        ids = [r[0] for r in faiss_results]

        if table == "recall_memory":
            sql = f"SELECT id, timestamp, role, content FROM {table} WHERE id IN ({','.join('?' * len(ids))})"
        else:
            sql = f"SELECT id, category, content, importance, timestamp FROM {table} WHERE id IN ({','.join('?' * len(ids))})"

        with _connect_db(db_path) as conn:
            rows = conn.execute(sql, ids).fetchall()

        if not rows:
            return self._try_next(query, limit, db_path, table, **ctx)

        if table == "recall_memory":
            row_map = {r[0]: r for r in rows}
            results = []
            for row_id, score in faiss_results:
                if row_id in row_map:
                    r = row_map[row_id]
                    results.append(
                        {
                            "role": r[2],
                            "content": r[3],
                            "timestamp": r[1],
                            "score": round(score, 4),
                        }
                    )
            return results
        else:
            # Archival — return raw tuples for importance decay processing
            return ctx.get("format_archival", lambda rows: rows)(rows)


# ---------------------------------------------------------------------------
# FTS5 full-text search (BM25)
# ---------------------------------------------------------------------------
class FTS5SearchHandler(SearchHandler):
    """SQLite FTS5 BM25 ranking. Falls through if query is empty or no matches."""

    def search(self, query, limit, db_path, table="recall_memory", **ctx):
        if not query:
            return self._try_next(query, limit, db_path, table, **ctx)

        fts_table = f"{table}_fts"

        try:
            with _connect_db(db_path) as conn:
                # Check FTS table exists
                exists = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                    (fts_table,),
                ).fetchone()
                if not exists:
                    return self._try_next(query, limit, db_path, table, **ctx)

                # FTS5 MATCH with BM25 ranking
                # Escape special FTS5 characters in query
                safe_query = self._sanitize_fts_query(query)
                if not safe_query:
                    return self._try_next(query, limit, db_path, table, **ctx)

                if table == "recall_memory":
                    rows = conn.execute(
                        f"""SELECT rm.role, rm.content, rm.timestamp
                            FROM {fts_table} fts
                            JOIN {table} rm ON rm.id = fts.rowid
                            WHERE {fts_table} MATCH ?
                            ORDER BY fts.rank
                            LIMIT ?""",
                        (safe_query, limit),
                    ).fetchall()
                    if rows:
                        return [
                            {"role": r[0], "content": r[1], "timestamp": r[2]}
                            for r in rows
                        ]
                else:
                    rows = conn.execute(
                        f"""SELECT am.id, am.category, am.content, am.importance, am.timestamp
                            FROM {fts_table} fts
                            JOIN {table} am ON am.id = fts.rowid
                            WHERE {fts_table} MATCH ?
                            ORDER BY fts.rank
                            LIMIT ?""",
                        (safe_query, limit),
                    ).fetchall()
                    if rows:
                        return ctx.get("format_archival", lambda r: r)(rows)

        except Exception as e:
            _log.debug("FTS5 search failed: %s", e)

        return self._try_next(query, limit, db_path, table, **ctx)

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Convert a natural language query into a safe FTS5 query.

        Wraps each token in quotes to avoid FTS5 syntax errors from
        special characters like colons, hyphens, or parentheses.
        """
        tokens = query.split()
        if not tokens:
            return ""
        # Quote each token and join with implicit AND
        return " ".join(f'"{t}"' for t in tokens)


# ---------------------------------------------------------------------------
# Recent messages fallback (ORDER BY timestamp DESC)
# ---------------------------------------------------------------------------
class RecentSearchHandler(SearchHandler):
    """Last resort: return most recent messages regardless of query."""

    def search(self, query, limit, db_path, table="recall_memory", **ctx):
        with _connect_db(db_path) as conn:
            if table == "recall_memory":
                if query:
                    rows = conn.execute(
                        f"SELECT role, content, timestamp FROM {table} WHERE content LIKE ? ORDER BY timestamp DESC LIMIT ?",
                        (f"%{query}%", limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        f"SELECT role, content, timestamp FROM {table} ORDER BY timestamp DESC LIMIT ?",
                        (limit,),
                    ).fetchall()
                return [
                    {"role": r[0], "content": r[1], "timestamp": r[2]} for r in rows
                ]
            else:
                fetch_limit = limit * 3
                if query:
                    rows = conn.execute(
                        f"SELECT id, category, content, importance, timestamp FROM {table} WHERE content LIKE ? ORDER BY importance DESC LIMIT ?",
                        (f"%{query}%", fetch_limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        f"SELECT id, category, content, importance, timestamp FROM {table} ORDER BY importance DESC LIMIT ?",
                        (fetch_limit,),
                    ).fetchall()
                if not rows:
                    return []
                return ctx.get("format_archival", lambda r: r)(rows)


# ---------------------------------------------------------------------------
# FTS5 table management
# ---------------------------------------------------------------------------
def ensure_fts5_table(db_path: str, table: str) -> None:
    """Create FTS5 virtual table and backfill from existing data if needed."""
    fts_table = f"{table}_fts"
    with _connect_db(db_path) as conn:
        # Create FTS5 table if it doesn't exist
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {fts_table} "
            f"USING fts5(content, content_rowid='id', tokenize='porter unicode61')"
        )
        # Backfill: insert any rows that exist in main table but not in FTS
        conn.execute(
            f"INSERT OR IGNORE INTO {fts_table}(rowid, content) "
            f"SELECT id, content FROM {table} WHERE id NOT IN (SELECT rowid FROM {fts_table})"
        )


def insert_fts5(db_path: str, table: str, row_id: int, content: str) -> None:
    """Insert a single row into the FTS5 index."""
    fts_table = f"{table}_fts"
    try:
        with _connect_db(db_path) as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO {fts_table}(rowid, content) VALUES (?, ?)",
                (row_id, content),
            )
    except Exception:
        pass  # FTS table may not exist yet; gracefully skip


def delete_fts5_before(db_path: str, table: str, _timestamp: str = "") -> None:
    """Remove FTS5 entries for rows that no longer exist in the main table."""
    fts_table = f"{table}_fts"
    try:
        with _connect_db(db_path) as conn:
            conn.execute(
                f"DELETE FROM {fts_table} WHERE rowid NOT IN (SELECT id FROM {table})"
            )
    except Exception:
        pass
