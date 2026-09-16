from __future__ import annotations

import networkx as nx

from memory_algebra import Query, cosine, quotient, retrieve
from memory_algebra.builder import make_memory, perturb

from helpers import THETA, fold_left


def test_protocol_3_4_expressivity(rng, concepts):
    eA, eB, eC, eD = concepts[0], concepts[1], concepts[2], concepts[3]
    m1 = make_memory(
        "m1",
        {"a": (eA, 0.0, 1.0), "b": (eB, 0.0, 1.0)},
        [("a", "b", "rel")],
    )
    m2 = make_memory(
        "m2",
        {"b": (perturb(eB, rng, 0.02), 0.0, 1.0), "c": (eC, 0.0, 1.0)},
        [("b", "c", "rel")],
    )
    m3 = make_memory(
        "m3",
        {"c": (perturb(eC, rng, 0.02), 0.0, 1.0), "d": (eD, 0.0, 1.0)},
        [("c", "d", "rel")],
    )
    M = fold_left([m1, m2, m3])
    quot = quotient(M, THETA)
    g = nx.Graph()
    for c in quot.clusters:
        g.add_node(c)
    for cu, cv, _ in quot.edges:
        g.add_edge(cu, cv)
    cl_a = quot.member_cluster["m1:a"]
    cl_d = quot.member_cluster["m3:d"]
    assert nx.has_path(g, cl_a, cl_d)
    assert not g.has_edge(cl_a, cl_d)
    r = retrieve(M, Query(eA, top_k=1, hops=3), THETA)
    assert any(cosine(r.memory.vectors[n], eD) > 0.95 for n in r.memory.nodes)
