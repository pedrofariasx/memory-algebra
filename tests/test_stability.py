from __future__ import annotations

import numpy as np

from memory_algebra import Query, compose, cosine, make_memory, prune, retrieve, semantic_distance
from memory_algebra.core import EMPTY

from helpers import THETA, fold_left, noisy


def test_protocol_3_2_stability(rng, concepts):
    n = 50
    v1 = concepts[0]
    m1 = make_memory("m0", {"a": (v1, 0.0, 1.0)})
    memories = [m1]
    pool = concepts[1:]
    for i in range(1, n):
        c1 = pool[i % len(pool)]
        c2 = pool[(i + 3) % len(pool)]
        memories.append(
            make_memory(
                f"m{i}",
                {
                    "x": (noisy(c1, rng), float(i), float(rng.uniform(0.3, 0.8))),
                    "y": (noisy(c2, rng), float(i), float(rng.uniform(0.3, 0.8))),
                },
                [("x", "y", "rel")],
            )
        )
    q1 = Query(v1, time=0.0, time_scale=100.0, top_k=1, hops=0)
    M = EMPTY
    vec_sum = np.zeros_like(v1)
    d_algebra, d_sum, d_fifo = [], [], []
    fifo_capacity = 10
    buffer: list = []
    for t, m in enumerate(memories):
        M = compose(M, m)
        buffer.append(m)
        if len(buffer) > fifo_capacity:
            buffer.pop(0)
        first = sorted(m.nodes)[0]
        vec_sum = vec_sum + np.asarray(m.vectors[first], dtype=float) / np.linalg.norm(m.vectors[first])
        if t % 5 != 0 and t != n - 1:
            continue
        d_algebra.append(semantic_distance(m1, retrieve(M, q1, THETA).memory))
        d_sum.append(1.0 - cosine(vec_sum, v1))
        fifo_nodes = set().union(*(b.nodes for b in buffer)) if buffer else set()
        d_fifo.append(0.0 if m1.nodes <= fifo_nodes else 1.0)
    assert max(d_algebra) < 1e-9
    assert d_sum[-1] > 0.5
    assert d_fifo[-1] == 1.0
    assert d_algebra[-1] < d_sum[-1]


def test_prune_bounded(rng, concepts):
    anchors = [
        make_memory(f"anchor{i}", {"a": (concepts[i], float(i), 1.0)}) for i in range(5)
    ]
    filler_pool = concepts[5:]
    fillers = [
        make_memory(f"filler{i}", {"a": (filler_pool[i % len(filler_pool)], float(i), float(rng.uniform(0.0, 1.0)))})
        for i in range(100)
    ]
    M = fold_left(anchors + fillers)
    P = prune(M, weight_floor=0.5)
    assert len(P.nodes) < len(M.nodes)
    for i, a in enumerate(anchors):
        node = sorted(a.nodes)[0]
        q = Query(a.vectors[node], time=float(i), time_scale=100.0, top_k=1, hops=0)
        assert semantic_distance(a, retrieve(P, q, THETA).memory) < 1e-9


def test_prune_shortcut(rng, concepts):
    m = make_memory(
        "chain",
        {
            "a": (concepts[0], 0.0, 1.0),
            "b": (concepts[1], 0.0, 0.01),
            "c": (concepts[2], 0.0, 1.0),
        },
        [("a", "b", "r1"), ("b", "c", "r2")],
    )
    P = prune(m, weight_floor=0.05, shortcut=True)
    assert "chain:b" not in P.nodes
    assert ("chain:a", "chain:c", "r1>r2") in P.edges


def test_prune_capacity(concepts):
    mems = [
        make_memory(f"m{i}", {"a": (concepts[i % 8], float(i), float(i) / 20.0)})
        for i in range(20)
    ]
    M = fold_left(mems)
    P = prune(M, weight_floor=0.0, capacity=5)
    assert len(P.nodes) == 5


def test_protocol_3_2_lstm_baseline(rng, concepts):
    from memory_algebra.baselines import LSTMCell

    n = 50
    v1 = concepts[0]
    m1 = make_memory("m0", {"a": (v1, 0.0, 1.0)})
    memories = [m1]
    pool = concepts[1:]
    for i in range(1, n):
        c1 = pool[i % len(pool)]
        c2 = pool[(i + 3) % len(pool)]
        memories.append(
            make_memory(
                f"m{i}",
                {
                    "x": (noisy(c1, rng), float(i), float(rng.uniform(0.3, 0.8))),
                    "y": (noisy(c2, rng), float(i), float(rng.uniform(0.3, 0.8))),
                },
                [("x", "y", "rel")],
            )
        )
    cell = LSTMCell(dim=len(v1), hidden=len(v1), seed=11)
    M = EMPTY
    q1 = Query(v1, time=0.0, time_scale=100.0, top_k=1, hops=0)
    d_lstm_final = 0.0
    for m in memories:
        M = compose(M, m)
        first = sorted(m.nodes)[0]
        x = np.asarray(m.vectors[first], dtype=float)
        h = cell.step(x / np.linalg.norm(x))
        d_lstm_final = 1.0 - cosine(h, v1)
    assert semantic_distance(m1, retrieve(M, q1, THETA).memory) < 1e-9
    assert d_lstm_final > 0.2
