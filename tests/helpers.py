from __future__ import annotations

import numpy as np

from memory_algebra import compose, make_memory, orthogonal_concepts
from memory_algebra.core import EMPTY

DIM = 16
THETA = 0.85
SEED = 7


def noisy(v: np.ndarray, rng: np.random.Generator, scale: float = 0.02) -> np.ndarray:
    w = np.asarray(v, dtype=float) + scale * rng.standard_normal(len(v))
    return w / np.linalg.norm(w)


def fold_left(mems):
    M = EMPTY
    for m in mems:
        M = compose(M, m)
    return M


def fold_right(mems):
    M = EMPTY
    for m in reversed(mems):
        M = compose(m, M)
    return M


def fold_parallel(mems, block: int = 10):
    blocks = [mems[i : i + block] for i in range(0, len(mems), block)]
    return fold_left([fold_left(b) for b in blocks])


def gen_memories(rng: np.random.Generator, concepts: list[np.ndarray], n: int):
    mems = []
    for i in range(n):
        k = int(rng.integers(1, 4))
        idx = rng.integers(0, len(concepts), size=k)
        nodes = {}
        for j, ci in enumerate(idx):
            nodes[f"n{j}"] = (
                noisy(concepts[ci], rng),
                float(i) + float(rng.uniform(0.0, 0.5)),
                float(rng.uniform(0.2, 1.0)),
            )
        edges = [(f"n{j}", f"n{j + 1}", "rel") for j in range(k - 1)]
        mems.append(make_memory(f"m{i}", nodes, edges))
    return mems
