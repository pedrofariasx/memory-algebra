from __future__ import annotations

import numpy as np

from memory_algebra import Query, compose, quotient, retrieve
from memory_algebra.builder import make_memory, perturb

from helpers import THETA


def test_protocol_3_3_conflicts(rng, concepts):
    e_joao, e_med, e_eng = concepts[0], concepts[1], concepts[2]
    prof = (e_med + e_eng) / np.linalg.norm(e_med + e_eng)
    mA = make_memory(
        "mA",
        {"joao": (e_joao, 1.0, 1.0), "medico": (e_med, 1.0, 1.0)},
        [("joao", "medico", "profissao")],
    )
    mB = make_memory(
        "mB",
        {"joao": (perturb(e_joao, rng, 0.05), 2.0, 1.0), "engenheiro": (e_eng, 2.0, 1.0)},
        [("joao", "engenheiro", "profissao")],
    )
    M = compose(mA, mB)
    quot = quotient(M, THETA)
    assert any("mA:joao" in c and "mB:joao" in c for c in quot.clusters)
    assert "mA:medico" in M.nodes and "mB:engenheiro" in M.nodes

    qvec = (e_joao + prof) / np.linalg.norm(e_joao + prof)
    r_now = retrieve(M, Query(qvec, time=2.0, time_scale=0.5, top_k=4, hops=0), THETA)
    r_past = retrieve(M, Query(qvec, time=1.0, time_scale=0.5, top_k=4, hops=0), THETA)

    def pos(r, node: str):
        for i, c in enumerate(r.focus):
            if node in c:
                return i
        return None

    assert pos(r_now, "mB:engenheiro") < pos(r_now, "mA:medico")
    assert pos(r_past, "mA:medico") < pos(r_past, "mB:engenheiro")
