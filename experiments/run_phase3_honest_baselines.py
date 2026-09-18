"""Phase 3b: Honest Baselines — algebra vs TemporalVectorStore and SimpleKG.

Baselines here are production-equivalent:
  - TemporalVectorStore: kNN with timestamp metadata filtering (like Qdrant/Weaviate)
  - SimpleKG: triple-store knowledge graph with hop traversal (like Neo4j)

Both support deletion. The question is whether the algebra still wins.
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
    retrieve,
    retract,
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


def exp_temporal_conflict(rho, n_seeds=N_SEEDS):
    algebra_score = 0
    tvs_score = 0
    kg_score = 0
    total = 0
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        concepts = correlated_concepts(8, DIM, rho, rng)
        e_joao, e_med, e_eng = concepts[0], concepts[1], concepts[2]
        j1 = e_joao + 0.02 * rng.standard_normal(DIM)
        j1 /= np.linalg.norm(j1)
        mA = make_memory("mA", {"joao": (j1, 1.0, 1.0), "medico": (e_med, 1.0, 1.0)}, [("joao", "medico", "profissao")])
        j2 = e_joao + 0.02 * rng.standard_normal(DIM)
        j2 /= np.linalg.norm(j2)
        mB = make_memory("mB", {"joao": (j2, 5.0, 1.0), "engenheiro": (e_eng, 5.0, 1.0)}, [("joao", "engenheiro", "profissao")])
        M = compose(mA, mB)
        q = e_joao + e_med + e_eng
        q /= np.linalg.norm(q)

        r_now = retrieve(M, Query(q, time=5.0, time_scale=0.5, top_k=4, hops=0), THETA)
        r_past = retrieve(M, Query(q, time=1.0, time_scale=0.5, top_k=4, hops=0), THETA)

        def rank_of(retrieval, node):
            for i, c in enumerate(retrieval.focus):
                if node in c:
                    return i
            return 999

        total += 2
        if rank_of(r_now, "mB:engenheiro") < rank_of(r_now, "mA:medico"):
            algebra_score += 1
        if rank_of(r_past, "mA:medico") < rank_of(r_past, "mB:engenheiro"):
            algebra_score += 1

        tvs = TemporalVectorStore()
        tvs.insert("mA:joao", j1, 1.0)
        tvs.insert("mA:medico", e_med, 1.0)
        tvs.insert("mB:joao", j2, 5.0)
        tvs.insert("mB:engenheiro", e_eng, 5.0)

        tvs_now = tvs.search(q, k=4, time=5.0, time_scale=0.5)
        tvs_past = tvs.search(q, k=4, time=1.0, time_scale=0.5)
        eng_now_tvs = next((i for i, n in enumerate(tvs_now) if "engenheiro" in n), 999)
        med_now_tvs = next((i for i, n in enumerate(tvs_now) if "medico" in n), 999)
        eng_past_tvs = next((i for i, n in enumerate(tvs_past) if "engenheiro" in n), 999)
        med_past_tvs = next((i for i, n in enumerate(tvs_past) if "medico" in n), 999)
        if eng_now_tvs < med_now_tvs:
            tvs_score += 1
        if med_past_tvs < eng_past_tvs:
            tvs_score += 1

        kg = SimpleKG()
        kg.add_node("mA:joao", j1)
        kg.add_node("mA:medico", e_med)
        kg.add_node("mB:joao", j2)
        kg.add_node("mB:engenheiro", e_eng)
        kg.add_edge("mA:joao", "profissao", "mA:medico")
        kg.add_edge("mB:joao", "profissao", "mB:engenheiro")

        kg_now = kg.search_with_hops(q, k=4, hops=1)
        kg_past = kg.search_with_hops(q, k=4, hops=1)
        eng_now_kg = next((i for i, n in enumerate(kg_now) if "engenheiro" in n), 999)
        med_now_kg = next((i for i, n in enumerate(kg_now) if "medico" in n), 999)
        eng_past_kg = next((i for i, n in enumerate(kg_past) if "engenheiro" in n), 999)
        med_past_kg = next((i for i, n in enumerate(kg_past) if "medico" in n), 999)
        if eng_now_kg < med_now_kg:
            kg_score += 1
        if med_past_kg < eng_past_kg:
            kg_score += 1

    return {
        "rho": rho,
        "n_seeds": n_seeds,
        "algebra": f"{algebra_score}/{total}",
        "temporal_vector_store": f"{tvs_score}/{total}",
        "simple_kg": f"{kg_score}/{total}",
        "algebra_pct": round(algebra_score / total, 4),
        "tvs_pct": round(tvs_score / total, 4),
        "kg_pct": round(kg_score / total, 4),
    }


def exp_retraction(rho, n_seeds=N_SEEDS):
    algebra_score = 0
    tvs_score = 0
    kg_score = 0
    total = 0
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        concepts = correlated_concepts(6, DIM, rho, rng)
        e_paris, e_lyon, e_france = concepts[0], concepts[1], concepts[2]
        mA = make_memory("mA", {"paris": (e_paris, 2.0, 1.0), "france": (e_france, 2.0, 1.0)}, [("paris", "france", "capital")])
        mB = make_memory("mB", {"lyon": (e_lyon, 5.0, 1.0), "france2": (e_france + 0.01 * rng.standard_normal(DIM), 5.0, 1.0)}, [("lyon", "france2", "capital")])
        M = compose(mA, mB)
        q = e_paris
        q /= np.linalg.norm(q)

        M_retracted = retract(M, mB)
        r = retrieve(M_retracted, Query(q, top_k=4, hops=0), THETA)
        total += 1
        paris_found = any("paris" in n for n in r.memory.nodes)
        lyon_found = any("lyon" in n for n in r.memory.nodes)
        if paris_found and not lyon_found:
            algebra_score += 1

        tvs = TemporalVectorStore()
        tvs.insert("mA:paris", e_paris, 2.0)
        tvs.insert("mA:france", e_france, 2.0)
        tvs.insert("mB:lyon", e_lyon, 5.0)
        tvs.insert("mB:france2", e_france + 0.01 * rng.standard_normal(DIM), 5.0)
        tvs.delete("mB:lyon")
        tvs.delete("mB:france2")
        tvs_results = tvs.search(q, k=4)
        total_tvs_paris = any("paris" in n for n in tvs_results)
        total_tvs_lyon = any("lyon" in n for n in tvs_results)
        if total_tvs_paris and not total_tvs_lyon:
            tvs_score += 1

        kg = SimpleKG()
        kg.add_node("mA:paris", e_paris)
        kg.add_node("mA:france", e_france)
        kg.add_node("mB:lyon", e_lyon)
        kg.add_node("mB:france2", e_france + 0.01 * rng.standard_normal(DIM))
        kg.add_edge("mA:paris", "capital", "mA:france")
        kg.add_edge("mB:lyon", "capital", "mB:france2")
        kg.delete_node("mB:lyon")
        kg.delete_node("mB:france2")
        kg_results = kg.search_with_hops(q, k=4, hops=1)
        kg_paris = any("paris" in n for n in kg_results)
        kg_lyon = any("lyon" in n for n in kg_results)
        if kg_paris and not kg_lyon:
            kg_score += 1

    return {
        "rho": rho,
        "n_seeds": n_seeds,
        "algebra": f"{algebra_score}/{total}",
        "temporal_vector_store": f"{tvs_score}/{total}",
        "simple_kg": f"{kg_score}/{total}",
        "algebra_pct": round(algebra_score / total, 4),
        "tvs_pct": round(tvs_score / total, 4),
        "kg_pct": round(kg_score / total, 4),
    }


def exp_multi_hop(rho, n_seeds=N_SEEDS):
    algebra_score = 0
    kg_score = 0
    total = 0
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        concepts = correlated_concepts(10, DIM, rho, rng)
        e_a, e_b, e_c, e_d = concepts[0], concepts[1], concepts[2], concepts[3]
        mA = make_memory("mA", {"A": (e_a, 1.0, 1.0), "B": (e_b, 1.0, 1.0)}, [("A", "B", "rel")])
        mB = make_memory("mB", {"B": (e_b + 0.01 * rng.standard_normal(DIM), 1.0, 1.0), "C": (e_c, 1.0, 1.0)}, [("B", "C", "rel")])
        mC = make_memory("mC", {"C": (e_c + 0.01 * rng.standard_normal(DIM), 1.0, 1.0), "D": (e_d, 1.0, 1.0)}, [("C", "D", "rel")])
        M = compose(compose(mA, mB), mC)
        q = e_a
        q /= np.linalg.norm(q)

        r = retrieve(M, Query(q, top_k=2, hops=3), THETA)
        total += 1
        found_d = any("D" in n for n in r.memory.nodes)
        if found_d:
            algebra_score += 1

        kg = SimpleKG()
        kg.add_node("A", e_a)
        kg.add_node("B", e_b)
        kg.add_node("C", e_c)
        kg.add_node("D", e_d)
        kg.add_edge("A", "rel", "B")
        kg.add_edge("B", "rel", "C")
        kg.add_edge("C", "rel", "D")
        kg_results = kg.search_with_hops(q, k=2, hops=3)
        kg_found_d = any("D" in n for n in kg_results)
        if kg_found_d:
            kg_score += 1

    return {
        "rho": rho,
        "n_seeds": n_seeds,
        "algebra": f"{algebra_score}/{total}",
        "simple_kg": f"{kg_score}/{total}",
        "algebra_pct": round(algebra_score / total, 4),
        "kg_pct": round(kg_score / total, 4),
    }


def main():
    results = {}
    print("=== Phase 3b: Honest Baselines ===", flush=True)

    print("\n[1/3] Temporal Conflict", flush=True)
    temporal = {}
    for rho in RHOS:
        print(f"  rho={rho}...", end=" ", flush=True)
        r = exp_temporal_conflict(rho)
        temporal[f"rho_{rho}"] = r
        print(f"algebra={r['algebra']} tvs={r['temporal_vector_store']} kg={r['simple_kg']}", flush=True)
    results["temporal_conflict"] = temporal

    print("\n[2/3] Retraction", flush=True)
    retraction = {}
    for rho in RHOS:
        print(f"  rho={rho}...", end=" ", flush=True)
        r = exp_retraction(rho)
        retraction[f"rho_{rho}"] = r
        print(f"algebra={r['algebra']} tvs={r['temporal_vector_store']} kg={r['simple_kg']}", flush=True)
    results["retraction"] = retraction

    print("\n[3/3] Multi-hop", flush=True)
    multi_hop = {}
    for rho in RHOS:
        print(f"  rho={rho}...", end=" ", flush=True)
        r = exp_multi_hop(rho)
        multi_hop[f"rho_{rho}"] = r
        print(f"algebra={r['algebra']} kg={r['simple_kg']}", flush=True)
    results["multi_hop"] = multi_hop

    out = Path(__file__).resolve().parent / "results_phase3_honest.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out}", flush=True)


if __name__ == "__main__":
    main()
