from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from memory_algebra import Query, compose, make_memory, retrieve, semantic_distance, signature
from memory_algebra.core import EMPTY

from helpers import DIM, THETA, fold_left, fold_parallel, fold_right, gen_memories


def test_compose_identity():
    v = np.ones(DIM)
    m = make_memory("m", {"a": (v, 0.0, 1.0)})
    assert signature(compose(EMPTY, m)) == signature(m)
    assert signature(compose(m, EMPTY)) == signature(m)


def test_compose_collision_raises():
    m = make_memory("m", {"a": (np.ones(DIM), 0.0, 1.0)})
    with pytest.raises(ValueError):
        compose(m, m)


def test_protocol_3_1_associativity(rng, concepts):
    mems = gen_memories(rng, concepts, 100)
    left = fold_left(mems)
    right = fold_right(mems)
    parallel = fold_parallel(mems)
    assert signature(left) == signature(right)
    assert signature(left) == signature(parallel)
    for m in mems:
        n0 = sorted(m.nodes)[0]
        q = Query(vector=m.vectors[n0], time=m.times[n0], time_scale=5.0, top_k=1, hops=0)
        r_left = retrieve(left, q, THETA).memory
        r_right = retrieve(right, q, THETA).memory
        r_par = retrieve(parallel, q, THETA).memory
        assert semantic_distance(r_left, r_right) < 1e-12
        assert semantic_distance(r_left, r_par) < 1e-12


@st.composite
def memory_lists(draw):
    seed = draw(st.integers(0, 2**31 - 1))
    rng = np.random.default_rng(seed)
    n = draw(st.integers(2, 8))
    mems = []
    for i in range(n):
        k = draw(st.integers(1, 3))
        nodes = {}
        for j in range(k):
            v = rng.standard_normal(DIM)
            v = v / np.linalg.norm(v)
            nodes[f"n{j}"] = (v, float(rng.uniform(0, 10)), float(rng.uniform(0, 1)))
        edges = [(f"n{j}", f"n{j + 1}", "e") for j in range(k - 1)]
        mems.append(make_memory(f"m{i}", nodes, edges))
    return mems


@given(mems=memory_lists())
@settings(max_examples=50, deadline=None)
def test_associativity_property(mems):
    assert signature(fold_left(mems)) == signature(fold_right(mems))
    assert signature(fold_left(mems)) == signature(fold_parallel(mems, block=3))
