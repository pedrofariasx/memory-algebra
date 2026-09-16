from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from memory_algebra.core import MemoryObject


def make_memory(
    prefix: str,
    nodes: Mapping[str, tuple[np.ndarray, float, float]],
    edges: Iterable[tuple[str, str, str]] = (),
) -> MemoryObject:
    pnodes = frozenset(f"{prefix}:{k}" for k in nodes)
    vectors = {
        f"{prefix}:{k}": np.asarray(spec[0], dtype=float) for k, spec in nodes.items()
    }
    times = {f"{prefix}:{k}": float(spec[1]) for k, spec in nodes.items()}
    weights = {f"{prefix}:{k}": float(spec[2]) for k, spec in nodes.items()}
    pedges = frozenset((f"{prefix}:{u}", f"{prefix}:{v}", l) for u, v, l in edges)
    return MemoryObject(nodes=pnodes, vectors=vectors, edges=pedges, times=times, weights=weights)


def orthogonal_concepts(n: int, dim: int, rng: np.random.Generator | None = None) -> list[np.ndarray]:
    if rng is None:
        rng = np.random.default_rng(42)
    if n > dim:
        raise ValueError("n deve ser menor ou igual a dim")
    raw = rng.standard_normal((dim, n))
    q, _ = np.linalg.qr(raw)
    return [q[:, i].copy() for i in range(n)]


def perturb(v: np.ndarray, rng: np.random.Generator, scale: float = 0.05) -> np.ndarray:
    noisy = np.asarray(v, dtype=float) + scale * rng.standard_normal(len(v))
    return noisy / np.linalg.norm(noisy)
