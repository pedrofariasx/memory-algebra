from __future__ import annotations

import numpy as np

from memory_algebra import (
    Query,
    compose,
    cosine,
    lens_get,
    lens_put,
    make_memory,
    retract,
    retrieve,
    semantic_distance,
    signature,
    transform,
)

from helpers import DIM, THETA, fold_left


def test_put_get(concepts):
    mA = make_memory(
        "mA",
        {"x": (concepts[0], 0.0, 1.0), "y": (concepts[1], 0.0, 1.0)},
        [("x", "y", "e")],
    )
    mB = make_memory("mB", {"z": (concepts[2], 1.0, 1.0)})
    M = compose(mA, mB)
    delta = make_memory("delta", {"w": (concepts[3], 5.0, 1.0)})
    M2 = lens_put(M, Query(concepts[0], top_k=1, hops=1), delta, THETA)
    r = lens_get(M2, Query(concepts[3], time=5.0, time_scale=0.5, top_k=1, hops=0), THETA)
    assert semantic_distance(delta, r.memory) < 1e-9
    assert not (M2.nodes & mA.nodes)


def test_get_put_isolated_view(concepts):
    mA = make_memory(
        "mA",
        {"x": (concepts[0], 0.0, 1.0), "y": (concepts[1], 0.0, 1.0)},
        [("x", "y", "e")],
    )
    mB = make_memory("mB", {"z": (concepts[2], 1.0, 1.0)})
    M = compose(mA, mB)
    q = Query(concepts[2], top_k=1, hops=0)
    view = lens_get(M, q, THETA).memory
    M2 = lens_put(M, q, view, THETA)
    assert signature(M2) == signature(M)


def test_transform_functorial(rng, concepts):
    mems = [
        make_memory(
            f"m{i}",
            {f"n{j}": (concepts[(i + j) % 4], float(i), 1.0) for j in range(2)},
            [("n0", "n1", "e")],
        )
        for i in range(5)
    ]
    M = fold_left(mems)
    W, _ = np.linalg.qr(rng.standard_normal((DIM, DIM)))
    q = Query(concepts[0], top_k=1, hops=1)
    r1 = retrieve(M, q, THETA).memory
    r2 = retrieve(transform(M, W), Query(W @ q.vector, top_k=1, hops=1), THETA).memory
    assert semantic_distance(r2, transform(r1, W)) < 1e-6


def test_transform_label_rules(concepts):
    m = make_memory(
        "m",
        {"x": (concepts[0], 0.0, 1.0), "y": (concepts[1], 0.0, 1.0)},
        [("x", "y", "antes")],
    )
    T = transform(m, np.eye(DIM), label_rules={"antes": "depois"})
    assert ("m:x", "m:y", "depois") in T.edges


def test_retract_projection(concepts):
    mA = make_memory("mA", {"x": (concepts[0], 0.0, 1.0)})
    mB = make_memory("mB", {"y": ((concepts[0] + concepts[1]) / np.sqrt(2), 0.0, 1.0)})
    M = compose(mA, mB)
    R = retract(M, mA, project=True)
    assert abs(cosine(R.vectors["mB:y"], concepts[0])) < 1e-9


def test_retract_identity_when_disjoint(concepts):
    mA = make_memory("mA", {"x": (concepts[0], 0.0, 1.0)})
    mB = make_memory("mB", {"y": (concepts[1], 0.0, 1.0)})
    assert signature(retract(mA, mB)) == signature(mA)
