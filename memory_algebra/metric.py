from __future__ import annotations

import numpy as np

from memory_algebra.core import MemoryObject, cosine


def _jaccard_distance(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a.symmetric_difference(b)) / len(union)


def _pooled(a: MemoryObject) -> np.ndarray | None:
    if not a.nodes:
        return None
    stack = np.stack([np.asarray(a.vectors[n], dtype=float) for n in sorted(a.nodes)])
    return stack.mean(axis=0)


def _mean_scalar(d, default: float = 0.0) -> float:
    if not d:
        return default
    return float(np.mean(list(d.values())))


def context_weights(
    base: tuple[float, float, float, float] = (0.4, 0.3, 0.15, 0.15),
    time_anchored: bool = False,
    structural: bool = False,
) -> tuple[float, float, float, float]:
    alpha, beta, gamma, delta = base
    if time_anchored:
        shift = min(0.15, alpha)
        alpha -= shift
        gamma += shift
    if structural:
        shift = min(0.15, alpha)
        alpha -= shift
        beta += shift
    return (alpha, beta, gamma, delta)


def semantic_distance(
    m1: MemoryObject,
    m2: MemoryObject,
    alpha: float = 0.4,
    beta: float = 0.3,
    gamma: float = 0.15,
    delta: float = 0.15,
) -> float:
    if not m1.nodes and not m2.nodes:
        return 0.0
    p1, p2 = _pooled(m1), _pooled(m2)
    if p1 is None or p2 is None:
        d_v = 1.0
    else:
        d_v = 1.0 - cosine(p1, p2)
    d_g = 0.5 * _jaccard_distance(m1.nodes, m2.nodes) + 0.5 * _jaccard_distance(m1.edges, m2.edges)
    dt = abs(_mean_scalar(m1.times) - _mean_scalar(m2.times))
    d_t = dt / (1.0 + dt)
    d_p = abs(_mean_scalar(m1.weights) - _mean_scalar(m2.weights))
    total = alpha + beta + gamma + delta
    return (alpha * d_v + beta * d_g + gamma * d_t + delta * d_p) / total
