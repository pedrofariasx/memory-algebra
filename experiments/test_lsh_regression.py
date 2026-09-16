"""LSH regression test: verify zero false negatives vs exact path at N>=4096."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory_algebra.core import _similar_pairs_exact, _similar_pairs_lsh

DIM = 384
THETA = 0.85
N = 5000
N_CLUSTERS = 50
CLUSTER_SIZE = 20


def main():
    rng = np.random.default_rng(42)
    raw = rng.standard_normal((N, DIM))
    for c in range(N_CLUSTERS):
        base = rng.standard_normal(DIM)
        base /= np.linalg.norm(base)
        start = c * CLUSTER_SIZE
        for i in range(CLUSTER_SIZE):
            noise = rng.standard_normal(DIM) * 0.02
            raw[start + i] = base + noise
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    unit = raw / norms

    t0 = time.time()
    exact_i, exact_j = _similar_pairs_exact(unit, THETA)
    t_exact = time.time() - t0

    t0 = time.time()
    lsh_i, lsh_j = _similar_pairs_lsh(unit, THETA)
    t_lsh = time.time() - t0

    exact_set = set(zip(exact_i.tolist(), exact_j.tolist()))
    lsh_set = set(zip(lsh_i.tolist(), lsh_j.tolist()))

    false_negatives = exact_set - lsh_set
    false_positives = lsh_set - exact_set

    print(f"N={N}, DIM={DIM}, theta={THETA}")
    print(f"Exact pairs: {len(exact_set)} ({t_exact:.3f}s)")
    print(f"LSH pairs:   {len(lsh_set)} ({t_lsh:.3f}s)")
    print(f"False negatives (exact missed by LSH): {len(false_negatives)}")
    print(f"False positives (LSH pairs below theta): {len(false_positives)}")

    if false_negatives:
        print(f"FAIL: {len(false_negatives)} true pairs missed by LSH")
        for pair in list(false_negatives)[:5]:
            sim = float(np.dot(unit[pair[0]], unit[pair[1]]))
            print(f"  pair {pair}: cosine={sim:.4f}")
        return 1
    else:
        print("PASS: zero false negatives")
        return 0


if __name__ == "__main__":
    sys.exit(main())
