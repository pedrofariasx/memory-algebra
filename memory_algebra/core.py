from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

Edge = tuple[str, str, str]
Cluster = tuple[str, ...]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return v
    return v / n


class _UnionFind:
    __slots__ = ("parent",)

    def __init__(self, items: Sequence[str]) -> None:
        self.parent: dict[str, str] = {x: x for x in items}

    def find(self, x: str) -> str:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if ra > rb:
            ra, rb = rb, ra
        self.parent[rb] = ra


@dataclass(frozen=True, eq=False)
class MemoryObject:
    nodes: frozenset[str] = frozenset()
    vectors: Mapping[str, np.ndarray] = field(default_factory=dict)
    edges: frozenset[Edge] = frozenset()
    times: Mapping[str, float] = field(default_factory=dict)
    weights: Mapping[str, float] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.nodes)

    def __add__(self, other: MemoryObject) -> MemoryObject:
        return compose(self, other)

    def norm(self) -> float:
        return float(len(self.nodes) + len(self.edges))


EMPTY = MemoryObject()


def compose(a: MemoryObject, b: MemoryObject) -> MemoryObject:
    if a.nodes & b.nodes:
        raise ValueError("colisão de identificadores; prefixe as memórias antes de compor")
    return MemoryObject(
        nodes=a.nodes | b.nodes,
        vectors={**a.vectors, **b.vectors},
        edges=a.edges | b.edges,
        times={**a.times, **b.times},
        weights={**a.weights, **b.weights},
    )


@dataclass(frozen=True)
class Quotient:
    clusters: tuple[Cluster, ...]
    member_cluster: Mapping[str, Cluster]
    embeddings: Mapping[Cluster, np.ndarray]
    times: Mapping[Cluster, float]
    weights: Mapping[Cluster, float]
    edges: frozenset[tuple[Cluster, Cluster, str]]
    adjacency: Mapping[Cluster, frozenset[Cluster]]


def _similar_pairs_exact(unit: np.ndarray, theta: float, block: int = 2048):
    n = unit.shape[0]
    chunks_i = []
    chunks_j = []
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        for j0 in range(i0, n, block):
            j1 = min(j0 + block, n)
            sub = unit[i0:i1] @ unit[j0:j1].T
            if i0 == j0:
                mask = np.triu(sub >= theta, k=1)
            else:
                mask = sub >= theta
            ii, jj = np.nonzero(mask)
            if ii.size:
                chunks_i.append(ii + i0)
                chunks_j.append(jj + j0)
    if not chunks_i:
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
    return np.concatenate(chunks_i), np.concatenate(chunks_j)


def _similar_pairs_lsh(
    unit: np.ndarray,
    theta: float,
    bands: int = 12,
    rows: int = 3,
    seed: int = 12345,
):
    n, d = unit.shape
    rng = np.random.default_rng(seed)
    planes = rng.standard_normal((bands * rows, d))
    signs = (unit @ planes.T) >= 0.0
    candidate_i = []
    candidate_j = []
    for b in range(bands):
        band = signs[:, b * rows : (b + 1) * rows]
        keys = np.packbits(band, axis=1)
        order = np.lexsort(keys.T[::-1])
        sorted_keys = keys[order]
        same = np.all(sorted_keys[1:] == sorted_keys[:-1], axis=1)
        boundaries = np.flatnonzero(~same)
        starts = np.concatenate([[0], boundaries + 1])
        ends = np.concatenate([boundaries + 1, [n]])
        for s, e in zip(starts, ends):
            bucket = order[s:e]
            size = len(bucket)
            if size < 2:
                continue
            for i in range(size):
                for j in range(i + 1, size):
                    candidate_i.append(bucket[i])
                    candidate_j.append(bucket[j])
    if not candidate_i:
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
    ci = np.array(candidate_i, dtype=np.intp)
    cj = np.array(candidate_j, dtype=np.intp)
    keep = ci < cj
    ci, cj = ci[keep], cj[keep]
    ci, cj = np.minimum(ci, cj), np.maximum(ci, cj)
    pairs = np.unique(np.stack([ci, cj], axis=1), axis=0)
    ci, cj = pairs[:, 0], pairs[:, 1]
    sims = np.einsum("ij,ij->i", unit[ci], unit[cj])
    keep = sims >= theta
    return ci[keep], cj[keep]


def _similar_pairs(unit: np.ndarray, theta: float, block: int = 2048):
    n = unit.shape[0]
    if n >= 4096:
        return _similar_pairs_lsh(unit, theta)
    return _similar_pairs_exact(unit, theta, block)


def quotient(m: MemoryObject, theta: float) -> Quotient:
    if not m.nodes:
        return Quotient((), {}, {}, {}, {}, frozenset(), {})
    ids = sorted(m.nodes)
    matrix = np.stack([np.asarray(m.vectors[n], dtype=float) for n in ids])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    unit = matrix / norms
    pi, pj = _similar_pairs(unit, theta)
    uf = _UnionFind(ids)
    for k in range(pi.size):
        uf.union(ids[pi[k]], ids[pj[k]])
    groups: dict[str, list[str]] = {}
    for n in ids:
        groups.setdefault(uf.find(n), []).append(n)
    clusters = tuple(tuple(sorted(g)) for g in sorted(groups.values()))
    member_cluster: dict[str, Cluster] = {}
    embeddings: dict[Cluster, np.ndarray] = {}
    times: dict[Cluster, float] = {}
    weights: dict[Cluster, float] = {}
    for c in clusters:
        for n in c:
            member_cluster[n] = c
        embeddings[c] = np.mean([np.asarray(m.vectors[n], dtype=float) for n in c], axis=0)
        times[c] = max(m.times.get(n, 0.0) for n in c)
        weights[c] = max(m.weights.get(n, 0.0) for n in c)
    q_edges: set[tuple[Cluster, Cluster, str]] = set()
    adjacency: dict[Cluster, set[Cluster]] = {c: set() for c in clusters}
    for u, v, label in m.edges:
        cu, cv = member_cluster[u], member_cluster[v]
        if cu == cv:
            continue
        q_edges.add((cu, cv, label))
        adjacency[cu].add(cv)
        adjacency[cv].add(cu)
    return Quotient(
        clusters=clusters,
        member_cluster=member_cluster,
        embeddings=embeddings,
        times=times,
        weights=weights,
        edges=frozenset(q_edges),
        adjacency={c: frozenset(nb) for c, nb in adjacency.items()},
    )


@dataclass(frozen=True)
class Query:
    vector: np.ndarray
    time: float | None = None
    time_scale: float = 1.0
    top_k: int = 1
    hops: int = 1


@dataclass(frozen=True)
class Retrieval:
    memory: MemoryObject
    focus: tuple[Cluster, ...]
    scores: Mapping[Cluster, float]


def retrieve(m: MemoryObject, q: Query, theta: float = 0.9) -> Retrieval:
    if not m.nodes:
        return Retrieval(EMPTY, (), {})
    quot = quotient(m, theta)
    qv = _unit(np.asarray(q.vector, dtype=float))
    scores: dict[Cluster, float] = {}
    for c, emb in quot.embeddings.items():
        s = max(cosine(qv, emb), 0.0)
        if q.time is not None:
            dt = (quot.times[c] - q.time) / max(q.time_scale, 1e-9)
            s *= float(np.exp(-(dt * dt)))
        s *= 0.5 + 0.5 * quot.weights[c]
        scores[c] = s
    focus = tuple(sorted(scores, key=lambda c: (-scores[c], c))[: q.top_k])
    selected = set(focus)
    frontier = set(focus)
    for _ in range(max(q.hops, 0)):
        nxt: set[Cluster] = set()
        for c in frontier:
            for nb in quot.adjacency.get(c, ()):
                if nb not in selected:
                    selected.add(nb)
                    nxt.add(nb)
        frontier = nxt
    nodes = frozenset(n for c in selected for n in c)
    vectors = {n: m.vectors[n] for n in nodes}
    edges = frozenset(e for e in m.edges if e[0] in nodes and e[1] in nodes)
    times = {n: m.times.get(n, 0.0) for n in nodes}
    weights = {n: m.weights.get(n, 0.0) for n in nodes}
    return Retrieval(
        memory=MemoryObject(nodes=nodes, vectors=vectors, edges=edges, times=times, weights=weights),
        focus=focus,
        scores=scores,
    )


def retract(m: MemoryObject, sub: MemoryObject, project: bool = False) -> MemoryObject:
    gone = m.nodes & sub.nodes
    if not gone:
        return m
    keep = m.nodes - gone
    vectors = {n: m.vectors[n] for n in keep}
    if project:
        removed = [np.asarray(sub.vectors[n], dtype=float) for n in gone if n in sub.vectors]
        removed = [v for v in removed if float(np.linalg.norm(v)) > 0.0]
        if removed:
            basis, _ = np.linalg.qr(np.stack([_unit(v) for v in removed], axis=1))
            for n in keep:
                v = np.asarray(vectors[n], dtype=float)
                vectors[n] = v - basis @ (basis.T @ v)
    edges = frozenset(e for e in m.edges if e[0] in keep and e[1] in keep)
    times = {n: m.times[n] for n in keep if n in m.times}
    weights = {n: m.weights[n] for n in keep if n in m.weights}
    return MemoryObject(nodes=keep, vectors=vectors, edges=edges, times=times, weights=weights)


def transform(
    m: MemoryObject,
    matrix: np.ndarray,
    label_rules: Mapping[str, str] | None = None,
) -> MemoryObject:
    matrix = np.asarray(matrix, dtype=float)
    vectors = {n: matrix @ np.asarray(v, dtype=float) for n, v in m.vectors.items()}
    if label_rules:
        edges = frozenset((u, v, label_rules.get(l, l)) for (u, v, l) in m.edges)
    else:
        edges = m.edges
    return MemoryObject(
        nodes=m.nodes,
        vectors=vectors,
        edges=edges,
        times=dict(m.times),
        weights=dict(m.weights),
    )


def signature(m: MemoryObject) -> tuple:
    vec_sig = tuple(
        sorted(
            (n, tuple(np.round(np.asarray(v, dtype=float), 9).tolist()))
            for n, v in m.vectors.items()
        )
    )
    return (
        tuple(sorted(m.nodes)),
        vec_sig,
        tuple(sorted(m.edges)),
        tuple(sorted((n, round(t, 9)) for n, t in m.times.items())),
        tuple(sorted((n, round(w, 9)) for n, w in m.weights.items())),
    )
