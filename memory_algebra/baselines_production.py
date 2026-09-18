"""Production baselines addressing critic.md protocol (lines 52-62).

Implements:
1. ProductionVectorDB: Vector store with cosine similarity, configurable with:
   - Config A (naive kNN): time=False, delete=False
   - Config B (time-only): time=True, delete=False
   - Config C (delete-only): time=False, delete=True
   - Config D (full production vector DB): time=True, delete=True
2. SQLiteTripleStore: In-memory SQLite triple store with:
   - Triples table (subj, rel, obj, t, is_valid)
   - Recursive CTE queries for multi-hop reaching depths 2, 3, 4
   - Soft and hard deletion, plus cascading retraction (dependency tracking)
3. BM25TemporalBaseline: Lexical BM25 token matching with temporal recency decay.
"""
from __future__ import annotations

import math
import re
import sqlite3
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class ProductionVectorDB:
    """Production Vector DB baseline supporting time filtering and deletion toggles."""

    def __init__(
        self,
        use_time_filter: bool = False,
        supports_deletion: bool = False,
        time_scale: float = 0.5,
    ) -> None:
        self.use_time_filter = use_time_filter
        self.supports_deletion = supports_deletion
        self.time_scale = time_scale
        self._items: dict[str, tuple[np.ndarray, float, dict[str, Any]]] = {}
        self._deleted: set[str] = set()

    @classmethod
    def config_a(cls, time_scale: float = 0.5) -> ProductionVectorDB:
        """Config A (naive kNN): time=False, delete=False."""
        return cls(use_time_filter=False, supports_deletion=False, time_scale=time_scale)

    @classmethod
    def config_b(cls, time_scale: float = 0.5) -> ProductionVectorDB:
        """Config B (time-only): time=True, delete=False."""
        return cls(use_time_filter=True, supports_deletion=False, time_scale=time_scale)

    @classmethod
    def config_c(cls, time_scale: float = 0.5) -> ProductionVectorDB:
        """Config C (delete-only): time=False, delete=True."""
        return cls(use_time_filter=False, supports_deletion=True, time_scale=time_scale)

    @classmethod
    def config_d(cls, time_scale: float = 0.5) -> ProductionVectorDB:
        """Config D (full production vector DB): time=True, delete=True."""
        return cls(use_time_filter=True, supports_deletion=True, time_scale=time_scale)

    def insert(
        self,
        key: str,
        vector: np.ndarray,
        timestamp: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._items[key] = (
            np.asarray(vector, dtype=float).copy(),
            float(timestamp),
            metadata or {},
        )
        self._deleted.discard(key)

    def delete(self, key: str, hard: bool = False) -> None:
        """Attempt to delete key. Ignored if supports_deletion is False."""
        if not self.supports_deletion:
            return
        if hard:
            self._items.pop(key, None)
            self._deleted.discard(key)
        else:
            if key in self._items:
                self._deleted.add(key)

    def search_with_scores(
        self,
        q_vec: np.ndarray,
        k: int = 4,
        time: float | None = None,
        time_scale: float | None = None,
    ) -> list[tuple[str, float]]:
        q = np.asarray(q_vec, dtype=float)
        t_scale = time_scale if time_scale is not None else self.time_scale
        scored: list[tuple[str, float]] = []

        for key, (vec, ts, _) in self._items.items():
            if self.supports_deletion and key in self._deleted:
                continue

            sim = _cosine(q, vec)
            if self.use_time_filter and time is not None:
                dt = time - ts
                recency = float(np.exp(-(dt * dt) / (2.0 * t_scale * t_scale)))
                score = sim * 0.7 + recency * 0.3
            else:
                score = sim
            scored.append((key, score))

        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    def search(
        self,
        q_vec: np.ndarray,
        k: int = 4,
        time: float | None = None,
        time_scale: float | None = None,
    ) -> list[str]:
        scored = self.search_with_scores(q_vec, k=k, time=time, time_scale=time_scale)
        return [key for key, _ in scored]


class SQLiteTripleStore:
    """In-memory SQLite triple store with recursive CTE multi-hop and dependency tracking."""

    def __init__(self, memory_db: bool = True) -> None:
        self.conn = sqlite3.connect(":memory:" if memory_db else "triple_store.db")
        self.embeddings: dict[str, np.ndarray] = {}
        self._setup_schema()

    def _setup_schema(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS triples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subj TEXT NOT NULL,
                    rel TEXT NOT NULL,
                    obj TEXT NOT NULL,
                    t REAL DEFAULT 0.0,
                    is_valid INTEGER DEFAULT 1
                );
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dependencies (
                    parent_id INTEGER NOT NULL,
                    child_id INTEGER NOT NULL,
                    FOREIGN KEY(parent_id) REFERENCES triples(id),
                    FOREIGN KEY(child_id) REFERENCES triples(id)
                );
                """
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_subj_valid ON triples(subj, is_valid);"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_obj_valid ON triples(obj, is_valid);"
            )

    def add_entity(self, name: str, vector: np.ndarray) -> None:
        self.embeddings[name] = np.asarray(vector, dtype=float).copy()

    def add_triple(
        self,
        subj: str,
        rel: str,
        obj: str,
        t: float = 0.0,
        is_valid: int = 1,
    ) -> int:
        with self.conn:
            cursor = self.conn.execute(
                """
                INSERT INTO triples (subj, rel, obj, t, is_valid)
                VALUES (?, ?, ?, ?, ?)
                """,
                (subj, rel, obj, float(t), int(is_valid)),
            )
            return int(cursor.lastrowid)

    def add_dependency(self, parent_id: int, child_id: int) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO dependencies (parent_id, child_id) VALUES (?, ?)",
                (parent_id, child_id),
            )

    def delete_triple(self, subj: str, rel: str, obj: str, hard: bool = False) -> None:
        with self.conn:
            if hard:
                self.conn.execute(
                    "DELETE FROM triples WHERE subj = ? AND rel = ? AND obj = ?",
                    (subj, rel, obj),
                )
            else:
                self.conn.execute(
                    "UPDATE triples SET is_valid = 0 WHERE subj = ? AND rel = ? AND obj = ?",
                    (subj, rel, obj),
                )

    def retract_edge(
        self,
        subj: str,
        rel: str,
        obj: str,
        cascade: bool = False,
        hard: bool = False,
    ) -> list[int]:
        """Retract edge. If cascade=True, retracts dependent triples via dependencies table

        or downstream paths that lose support. Returns list of affected triple IDs.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id FROM triples WHERE subj = ? AND rel = ? AND obj = ? AND is_valid = 1",
            (subj, rel, obj),
        )
        rows = cursor.fetchall()
        if not rows:
            return []
        root_ids = [r[0] for r in rows]

        if not cascade:
            affected_ids = root_ids
        else:
            # Check explicit dependencies first
            placeholders = ",".join("?" for _ in root_ids)
            query = f"""
            WITH RECURSIVE dep(id) AS (
                SELECT id FROM triples WHERE id IN ({placeholders})
                UNION
                SELECT d.child_id
                FROM dependencies d
                JOIN dep ON d.parent_id = dep.id
            )
            SELECT DISTINCT id FROM dep;
            """
            cursor.execute(query, root_ids)
            explicit_deps = [r[0] for r in cursor.fetchall()]

            # Also cascade downstream triples originating from obj if they form a chain
            downstream_query = """
            WITH RECURSIVE downstream(curr_node, triple_id) AS (
                SELECT obj, id FROM triples WHERE subj = ? AND rel = ? AND obj = ? AND is_valid = 1
                UNION
                SELECT t.obj, t.id
                FROM triples t
                JOIN downstream d ON t.subj = d.curr_node
                WHERE t.is_valid = 1
            )
            SELECT DISTINCT triple_id FROM downstream;
            """
            cursor.execute(downstream_query, (subj, rel, obj))
            downstream_ids = [r[0] for r in cursor.fetchall()]
            affected_ids = list(set(explicit_deps) | set(downstream_ids))

        with self.conn:
            aff_placeholders = ",".join("?" for _ in affected_ids)
            if hard:
                self.conn.execute(
                    f"DELETE FROM triples WHERE id IN ({aff_placeholders})",
                    affected_ids,
                )
            else:
                self.conn.execute(
                    f"UPDATE triples SET is_valid = 0 WHERE id IN ({aff_placeholders})",
                    affected_ids,
                )
        return affected_ids

    def query_multihop(
        self,
        start_entity: str,
        max_depth: int = 3,
        directed: bool = True,
    ) -> list[str]:
        """Execute recursive CTE query for multi-hop graph traversal."""
        cursor = self.conn.cursor()
        if directed:
            query = """
            WITH RECURSIVE path(node, depth, path_str) AS (
                SELECT obj, 1, subj || '->' || obj
                FROM triples
                WHERE is_valid = 1 AND subj = ?
                UNION ALL
                SELECT t.obj, p.depth + 1, p.path_str || '->' || t.obj
                FROM path p
                JOIN triples t ON p.node = t.subj
                WHERE t.is_valid = 1
                  AND p.depth < ?
                  AND instr(p.path_str, t.obj) = 0
            )
            SELECT DISTINCT node FROM path WHERE depth <= ?;
            """
            cursor.execute(query, (start_entity, max_depth, max_depth))
        else:
            query = """
            WITH RECURSIVE undirected_edges(u, v) AS (
                SELECT subj, obj FROM triples WHERE is_valid = 1
                UNION
                SELECT obj, subj FROM triples WHERE is_valid = 1
            ),
            path(node, depth, path_str) AS (
                SELECT v, 1, u || '--' || v
                FROM undirected_edges
                WHERE u = ?
                UNION ALL
                SELECT e.v, p.depth + 1, p.path_str || '--' || e.v
                FROM path p
                JOIN undirected_edges e ON p.node = e.u
                WHERE p.depth < ?
                  AND instr(p.path_str, e.v) = 0
            )
            SELECT DISTINCT node FROM path WHERE depth <= ?;
            """
            cursor.execute(query, (start_entity, max_depth, max_depth))

        return [r[0] for r in cursor.fetchall()]

    def search_with_hops(
        self,
        q_vec: np.ndarray,
        k: int = 4,
        hops: int = 1,
        directed: bool = True,
    ) -> list[str]:
        """Multi-hop semantic search: seeds nearest entities, then traverses via recursive CTE."""
        if not self.embeddings:
            return []
        q = np.asarray(q_vec, dtype=float)
        scored = [(name, _cosine(q, vec)) for name, vec in self.embeddings.items()]
        scored.sort(key=lambda x: -x[1])
        top_seeds = [name for name, _ in scored[:k]]

        reached: set[str] = set(top_seeds)
        for seed in top_seeds:
            for node in self.query_multihop(seed, max_depth=hops, directed=directed):
                reached.add(node)

        # Rank all reached entities by cosine similarity if embeddings are available
        all_candidates = list(reached)
        zero = np.zeros_like(q)
        all_candidates.sort(
            key=lambda name: -_cosine(q, self.embeddings.get(name, zero))
        )
        return all_candidates

    def search_temporal(
        self,
        subj: str,
        rel: str | None = None,
        time: float = 0.0,
        k: int = 4,
    ) -> list[tuple[str, float]]:
        """Query valid triples for subj ranked by temporal proximity to time."""
        cursor = self.conn.cursor()
        query = """
        SELECT obj, t, ABS(t - ?) as dt
        FROM triples
        WHERE is_valid = 1 AND subj = ? AND (? IS NULL OR rel = ?)
        ORDER BY dt ASC, t DESC
        LIMIT ?;
        """
        cursor.execute(query, (time, subj, rel, rel, k))
        return [(r[0], r[1]) for r in cursor.fetchall()]


class BM25TemporalBaseline:
    """Okapi BM25 baseline with temporal recency decay as lower bound."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        time_scale: float = 0.5,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.time_scale = time_scale
        self.docs: dict[str, tuple[list[str], float]] = {}
        self.doc_freq: dict[str, int] = {}
        self.avg_dl: float = 0.0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def add_document(self, doc_id: str, text: str, timestamp: float = 0.0) -> None:
        tokens = self._tokenize(text)
        self.docs[doc_id] = (tokens, float(timestamp))
        self._recompute_stats()

    def delete_document(self, doc_id: str) -> None:
        if doc_id in self.docs:
            del self.docs[doc_id]
            self._recompute_stats()

    def _recompute_stats(self) -> None:
        self.doc_freq.clear()
        total_len = 0
        for tokens, _ in self.docs.values():
            total_len += len(tokens)
            unique = set(tokens)
            for t in unique:
                self.doc_freq[t] = self.doc_freq.get(t, 0) + 1
        n_docs = len(self.docs)
        self.avg_dl = (total_len / n_docs) if n_docs > 0 else 0.0

    def search(
        self,
        query: str,
        k: int = 4,
        time: float | None = None,
        time_scale: float | None = None,
    ) -> list[tuple[str, float]]:
        q_tokens = self._tokenize(query)
        if not q_tokens or not self.docs:
            return []

        n_docs = len(self.docs)
        t_scale = time_scale if time_scale is not None else self.time_scale
        scored: list[tuple[str, float]] = []

        for doc_id, (tokens, ts) in self.docs.items():
            doc_len = len(tokens)
            if doc_len == 0:
                continue

            tf_map: dict[str, int] = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1

            bm25_score = 0.0
            for qt in q_tokens:
                if qt not in tf_map:
                    continue
                df = self.doc_freq.get(qt, 0)
                # Robertson-Spärck Jones IDF
                idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
                tf = tf_map[qt]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(self.avg_dl, 1e-9)))
                bm25_score += idf * (tf * (self.k1 + 1.0)) / max(denom, 1e-9)

            if bm25_score <= 0.0:
                continue

            if time is not None:
                dt = time - ts
                recency = float(np.exp(-(dt * dt) / (2.0 * t_scale * t_scale)))
                final_score = bm25_score * (0.5 + 0.5 * recency)
            else:
                final_score = bm25_score

            scored.append((doc_id, final_score))

        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    def search_keys(
        self,
        query: str,
        k: int = 4,
        time: float | None = None,
        time_scale: float | None = None,
    ) -> list[str]:
        scored = self.search(query, k=k, time=time, time_scale=time_scale)
        return [doc_id for doc_id, _ in scored]
