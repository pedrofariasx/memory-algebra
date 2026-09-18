from __future__ import annotations

import numpy as np


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class TemporalVectorStore:
    def __init__(self):
        self._items: dict[str, tuple[np.ndarray, float]] = {}

    def insert(self, key: str, vector: np.ndarray, timestamp: float) -> None:
        self._items[key] = (np.asarray(vector, dtype=float).copy(), float(timestamp))

    def delete(self, key: str) -> None:
        self._items.pop(key, None)

    def search(self, q_vec: np.ndarray, k: int = 4, time: float | None = None, time_scale: float = 0.5) -> list[str]:
        scored = []
        q = np.asarray(q_vec, dtype=float)
        for key, (vec, ts) in self._items.items():
            sim = _cosine(q, vec)
            if time is not None:
                recency = np.exp(-((time - ts) ** 2) / (2 * time_scale**2))
                score = sim * 0.7 + recency * 0.3
            else:
                score = sim
            scored.append((key, score))
        scored.sort(key=lambda x: -x[1])
        return [key for key, _ in scored[:k]]


class SimpleKG:
    def __init__(self):
        self._triples: list[tuple[str, str, str]] = []
        self._vectors: dict[str, np.ndarray] = {}

    def add_node(self, key: str, vector: np.ndarray) -> None:
        self._vectors[key] = np.asarray(vector, dtype=float).copy()

    def add_edge(self, src: str, rel: str, dst: str) -> None:
        self._triples.append((src, rel, dst))

    def delete_node(self, key: str) -> None:
        self._vectors.pop(key, None)
        self._triples = [(s, r, d) for s, r, d in self._triples if s != key and d != key]

    def neighbors(self, key: str) -> list[str]:
        result = []
        for s, _, d in self._triples:
            if s == key and d in self._vectors:
                result.append(d)
            elif d == key and s in self._vectors:
                result.append(s)
        return result

    def search_with_hops(self, q_vec: np.ndarray, k: int = 4, hops: int = 1) -> list[str]:
        q = np.asarray(q_vec, dtype=float)
        scored = [(key, _cosine(q, vec)) for key, vec in self._vectors.items()]
        scored.sort(key=lambda x: -x[1])
        top = [key for key, _ in scored[:k]]
        visited = set(top)
        frontier = list(top)
        for _ in range(hops):
            next_frontier = []
            for node in frontier:
                for nb in self.neighbors(node):
                    if nb not in visited:
                        visited.add(nb)
                        next_frontier.append(nb)
            frontier = next_frontier
        all_candidates = list(visited)
        all_candidates.sort(key=lambda key: -_cosine(q, self._vectors.get(key, np.zeros_like(q))))
        return all_candidates[:k * 2]
