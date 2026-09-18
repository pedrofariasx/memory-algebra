"""Phase 3c: Algebra advantage — scenarios where the algebra's hybrid traversal
(similarity quotient + edge hops) is structurally superior to pure KG or pure vector DB.

Key insight: the algebra traverses BOTH explicit edges AND similarity-bridged clusters
in a single hop operation. Neither a pure KG (edges only) nor a pure vector DB (similarity
only) can do this without additional infrastructure.

Experiment: chain A --edge--> B ~~similarity~~ C --edge--> D
- Algebra: A -> B (edge) -> C (similarity bridge via quotient) -> D (edge) = SUCCESS
- Pure KG: A -> B, stuck (no edge B->C) = FAIL
- Pure vector DB: finds B near A, but cannot follow to C/D = FAIL
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory_algebra import (
    EMPTY,
    Query,
    compose,
    cosine,
    make_memory,
    orthogonal_concepts,
    quotient,
    retrieve,
)
from memory_algebra.baselines_honest import SimpleKG, TemporalVectorStore

DIM = 32
THETA = 0.85
N_SEEDS = 30
RHOS = [0.0, 0.8]


def correlated_concepts(n, dim, rho, rng):
    if rho == 0.0:
        return orthogonal_concepts(dim, dim, rng)
    base = rng.standard_normal(dim)
    base /= np.linalg.norm(base) + 1e-12
    concepts = []
    for _ in range(n):
        noise = rng.standard_normal(dim)
        noise /= np.linalg.norm(noise) + 1e-12
        v = rho * base + np.sqrt(1 - rho * rho) * noise
        concepts.append(v / (np.linalg.norm(v) + 1e-12))
    return concepts


def exp_hybrid_bridge(rho, n_seeds=N_SEEDS):
    """A->B (edge), B~C (similarity, no edge), C->D (edge).
    Only the algebra can traverse the full chain."""
    algebra_score = 0
    kg_score = 0
    tvs_score = 0
    total = 0

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        concepts = correlated_concepts(8, DIM, rho, rng)
        e_a, e_b, e_c, e_d = concepts[0], concepts[1], concepts[2], concepts[3]

        b_vec = e_b + 0.02 * rng.standard_normal(DIM)
        b_vec /= np.linalg.norm(b_vec)
        c_vec = b_vec + 0.03 * rng.standard_normal(DIM)
        c_vec /= np.linalg.norm(c_vec)

        mA = make_memory("mA", {"A": (e_a, 1.0, 1.0), "B": (b_vec, 1.0, 1.0)}, [("A", "B", "rel")])
        mB = make_memory("mB", {"C": (c_vec, 1.0, 1.0), "D": (e_d, 1.0, 1.0)}, [("C", "D", "rel")])
        M = compose(mA, mB)

        q = e_a
        q /= np.linalg.norm(q)
        total += 1

        r = retrieve(M, Query(q, top_k=1, hops=3), THETA)
        if any("D" in n for n in r.memory.nodes):
            algebra_score += 1

        kg = SimpleKG()
        kg.add_node("A", e_a)
        kg.add_node("B", b_vec)
        kg.add_node("C", c_vec)
        kg.add_node("D", e_d)
        kg.add_edge("A", "rel", "B")
        kg.add_edge("C", "rel", "D")
        kg_results = kg.search_with_hops(q, k=2, hops=3)
        if any("D" in n for n in kg_results):
            kg_score += 1

        tvs = TemporalVectorStore()
        tvs.insert("A", e_a, 1.0)
        tvs.insert("B", b_vec, 1.0)
        tvs.insert("C", c_vec, 1.0)
        tvs.insert("D", e_d, 1.0)
        tvs_results = tvs.search(q, k=2)
        if any("D" in n for n in tvs_results):
            tvs_score += 1

    return {
        "rho": rho,
        "n_seeds": n_seeds,
        "algebra": f"{algebra_score}/{total}",
        "simple_kg": f"{kg_score}/{total}",
        "temporal_vector_store": f"{tvs_score}/{total}",
        "algebra_pct": round(algebra_score / total, 4),
        "kg_pct": round(kg_score / total, 4),
        "tvs_pct": round(tvs_score / total, 4),
        "algebra_wins": algebra_score > max(kg_score, tvs_score),
    }


def exp_implicit_clustering(rho, n_seeds=N_SEEDS):
    """Facts stored without explicit edges but semantically related.
    The quotient groups them automatically; KG has no edges to traverse."""
    algebra_score = 0
    kg_score = 0
    total = 0

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed + 500)
        concepts = correlated_concepts(6, DIM, rho, rng)
        e_topic = concepts[0]

        facts = []
        for i in range(4):
            v = e_topic + 0.03 * rng.standard_normal(DIM)
            v /= np.linalg.norm(v)
            facts.append((f"fact{i}", (v, 1.0, 1.0)))

        distractors = []
        for i in range(4, 6):
            v = concepts[i] + 0.03 * rng.standard_normal(DIM)
            v /= np.linalg.norm(v)
            distractors.append((f"dist{i}", (v, 1.0, 1.0)))

        all_nodes = dict(facts + distractors)
        M = make_memory("mem", all_nodes, [])

        q = e_topic
        q /= np.linalg.norm(q)
        total += 1

        r = retrieve(M, Query(q, top_k=4, hops=0), THETA)
        n_facts = sum(1 for n in r.memory.nodes if "fact" in n)
        if n_facts >= 3:
            algebra_score += 1

        kg = SimpleKG()
        for name, (v, t, w) in all_nodes.items():
            kg.add_node(name, v)
        kg_results = kg.search_with_hops(q, k=4, hops=1)
        n_facts_kg = sum(1 for n in kg_results if "fact" in n)
        if n_facts_kg >= 3:
            kg_score += 1

    return {
        "rho": rho,
        "n_seeds": n_seeds,
        "algebra": f"{algebra_score}/{total}",
        "simple_kg": f"{kg_score}/{total}",
        "algebra_pct": round(algebra_score / total, 4),
        "kg_pct": round(kg_score / total, 4),
        "algebra_wins": algebra_score > kg_score,
    }


def exp_multi_source_composition(rho, n_seeds=N_SEEDS):
    """Multiple agents contribute partial knowledge. Only composition gives
    a unified view without manual merging. Test: 3 agents each know part of
    a chain; only composed memory can answer the full query."""
    algebra_score = 0
    kg_score = 0
    total = 0

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed + 1000)
        concepts = correlated_concepts(6, DIM, rho, rng)
        e_a, e_b, e_c, e_d = concepts[0], concepts[1], concepts[2], concepts[3]

        b1 = e_b + 0.02 * rng.standard_normal(DIM)
        b1 /= np.linalg.norm(b1)
        c1 = e_c + 0.02 * rng.standard_normal(DIM)
        c1 /= np.linalg.norm(c1)

        agent1 = make_memory("ag1", {"A": (e_a, 1.0, 1.0), "B": (b1, 1.0, 1.0)}, [("A", "B", "knows")])
        agent2 = make_memory("ag2", {"B": (b1, 1.0, 1.0), "C": (c1, 1.0, 1.0)}, [("B", "C", "knows")])
        agent3 = make_memory("ag3", {"C": (c1, 1.0, 1.0), "D": (e_d, 1.0, 1.0)}, [("C", "D", "knows")])

        M = compose(compose(agent1, agent2), agent3)
        q = e_a
        q /= np.linalg.norm(q)
        total += 1

        r = retrieve(M, Query(q, top_k=1, hops=3), THETA)
        if any("D" in n for n in r.memory.nodes):
            algebra_score += 1

        kg = SimpleKG()
        kg.add_node("ag1:A", e_a)
        kg.add_node("ag1:B", b1)
        kg.add_node("ag2:B", b1)
        kg.add_node("ag2:C", c1)
        kg.add_node("ag3:C", c1)
        kg.add_node("ag3:D", e_d)
        kg.add_edge("ag1:A", "knows", "ag1:B")
        kg.add_edge("ag2:B", "knows", "ag2:C")
        kg.add_edge("ag3:C", "knows", "ag3:D")
        kg_results = kg.search_with_hops(q, k=2, hops=3)
        if any("D" in n for n in kg_results):
            kg_score += 1

    return {
        "rho": rho,
        "n_seeds": n_seeds,
        "algebra": f"{algebra_score}/{total}",
        "simple_kg": f"{kg_score}/{total}",
        "algebra_pct": round(algebra_score / total, 4),
        "kg_pct": round(kg_score / total, 4),
        "algebra_wins": algebra_score > kg_score,
    }


def main():
    results = {}
    print("=== Phase 3c: Algebra Advantage ===", flush=True)

    print("\n[1/3] Hybrid Bridge (edge + similarity + edge)", flush=True)
    hybrid = {}
    for rho in RHOS:
        print(f"  rho={rho}...", end=" ", flush=True)
        r = exp_hybrid_bridge(rho)
        hybrid[f"rho_{rho}"] = r
        print(f"algebra={r['algebra']} kg={r['simple_kg']} tvs={r['temporal_vector_store']} wins={r['algebra_wins']}", flush=True)
    results["hybrid_bridge"] = hybrid

    print("\n[2/3] Implicit Clustering (no edges, quotient groups)", flush=True)
    implicit = {}
    for rho in RHOS:
        print(f"  rho={rho}...", end=" ", flush=True)
        r = exp_implicit_clustering(rho)
        implicit[f"rho_{rho}"] = r
        print(f"algebra={r['algebra']} kg={r['simple_kg']} wins={r['algebra_wins']}", flush=True)
    results["implicit_clustering"] = implicit

    print("\n[3/3] Multi-source Composition (3 agents, broken chain)", flush=True)
    multi = {}
    for rho in RHOS:
        print(f"  rho={rho}...", end=" ", flush=True)
        r = exp_multi_source_composition(rho)
        multi[f"rho_{rho}"] = r
        print(f"algebra={r['algebra']} kg={r['simple_kg']} wins={r['algebra_wins']}", flush=True)
    results["multi_source_composition"] = multi

    out = Path(__file__).resolve().parent / "results_phase3_advantage.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out}", flush=True)

    all_win = all(
        v.get("algebra_wins", False)
        for exp in results.values()
        for v in exp.values()
    )
    print(f"ALGEBRA WINS ALL: {all_win}", flush=True)


if __name__ == "__main__":
    main()
