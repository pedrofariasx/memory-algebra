from __future__ import annotations

from memory_algebra.core import MemoryObject


def prune(
    m: MemoryObject,
    weight_floor: float = 0.05,
    capacity: int | None = None,
    shortcut: bool = True,
) -> MemoryObject:
    low = {n for n in m.nodes if m.weights.get(n, 1.0) < weight_floor}
    vectors = {n: m.vectors[n] for n in m.nodes if n not in low}
    times = {n: m.times[n] for n in m.nodes if n not in low and n in m.times}
    weights = {n: m.weights[n] for n in m.nodes if n not in low and n in m.weights}
    keep_nodes = set(vectors.keys())
    edges = set()
    shortcuts: set[tuple[str, str, str]] = set()
    for u, v, label in m.edges:
        if u in low or v in low:
            if shortcut:
                incoming = [(a, l1) for a, b, l1 in m.edges if b == u and a not in low]
                outgoing = [(b, l2) for a, b, l2 in m.edges if a == v and b not in low]
                if u in low and v in low:
                    continue
                if u in low:
                    for a, l1 in incoming:
                        shortcuts.add((a, v, f"{l1}>{label}"))
                else:
                    for b, l2 in outgoing:
                        shortcuts.add((u, b, f"{label}>{l2}"))
            continue
        edges.add((u, v, label))
    for u, v, label in shortcuts:
        if u in keep_nodes and v in keep_nodes and u != v:
            edges.add((u, v, label))
    result = MemoryObject(
        nodes=frozenset(keep_nodes),
        vectors=vectors,
        edges=frozenset(edges),
        times=times,
        weights=weights,
    )
    if capacity is not None and len(result.nodes) > capacity:
        ranked = sorted(result.nodes, key=lambda n: (result.weights.get(n, 0.0), n))
        drop = set(ranked[: len(result.nodes) - capacity])
        vectors = {n: result.vectors[n] for n in result.nodes if n not in drop}
        edges = frozenset(
            e for e in result.edges if e[0] not in drop and e[1] not in drop
        )
        times = {n: result.times[n] for n in result.nodes if n not in drop and n in result.times}
        weights = {n: result.weights[n] for n in result.nodes if n not in drop and n in result.weights}
        result = MemoryObject(
            nodes=frozenset(vectors.keys()),
            vectors=vectors,
            edges=edges,
            times=times,
            weights=weights,
        )
    return result
