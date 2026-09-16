from __future__ import annotations

import numpy as np

from memory_algebra import context_weights, make_memory, semantic_distance


def test_semantic_distance_identity():
    m = make_memory("m", {"a": (np.array([1.0, 0.0]), 0.0, 1.0)})
    assert semantic_distance(m, m) == 0.0


def test_semantic_distance_positive():
    a = make_memory("a", {"x": (np.array([1.0, 0.0]), 0.0, 1.0)})
    b = make_memory("b", {"y": (np.array([0.0, 1.0]), 5.0, 0.2)})
    assert semantic_distance(a, b) > 0.0


def test_context_weights_partition():
    base = (0.4, 0.3, 0.15, 0.15)
    wt = context_weights(time_anchored=True)
    ws = context_weights(structural=True)
    wb = context_weights(time_anchored=True, structural=True)
    for w in (base, wt, ws, wb):
        assert abs(sum(w) - 1.0) < 1e-12
    assert wt[2] > base[2]
    assert ws[1] > base[1]
    assert wb[0] < base[0]


def test_semantic_distance_with_context_weights():
    a = make_memory("a", {"x": (np.array([1.0, 0.0]), 0.0, 1.0)})
    b = make_memory("b", {"x": (np.array([1.0, 0.0]), 5.0, 1.0)})
    alpha, beta, gamma, delta = context_weights(time_anchored=True)
    d_ctx = semantic_distance(a, b, alpha=alpha, beta=beta, gamma=gamma, delta=delta)
    d_def = semantic_distance(a, b)
    assert d_ctx > d_def
