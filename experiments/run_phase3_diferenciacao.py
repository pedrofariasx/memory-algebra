"""Phase 3: Differentiation — prove the algebra beats kNN where kNN is structurally incapable.

Experiments:
  1. temporal_conflict: time-weighted retrieval vs time-blind kNN.
  2. retraction: algebra retracts invalidated facts; kNN keeps everything.
  3. multi_hop_distractors: algebra traverses edges via hops; kNN only reaches top-k neighbours.

Each experiment runs an orthogonal variant (easy) and a correlated variant (hard),
so the comparison is not an artifact of perfectly separable concepts.
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


def knn_retrieve(mems, q_vec, k=4):
    scored = []
    for m in mems:
        for n in m.nodes:
            scored.append((n, cosine(np.asarray(q_vec, dtype=float), np.asarray(m.vectors[n], dtype=float))))
    scored.sort(key=lambda x: -x[1])
    return [n for n, _ in scored[:k]]


def exp_temporal_conflict(rho, n_seeds=N_SEEDS):
    algebra_score = 0
    knn_score = 0
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

        mems_flat = [mA, mB]
        knn_now = knn_retrieve(mems_flat, q, k=4)
        knn_past = knn_retrieve(mems_flat, q, k=4)
        eng_now = next((i for i, n in enumerate(knn_now) if "engenheiro" in n), 999)
        med_now = next((i for i, n in enumerate(knn_now) if "medico" in n), 999)
        eng_past = next((i for i, n in enumerate(knn_past) if "engenheiro" in n), 999)
        med_past = next((i for i, n in enumerate(knn_past) if "medico" in n), 999)
        if eng_now < med_now:
            knn_score += 1
        if med_past < eng_past:
            knn_score += 1

    return {
        "rho": rho,
        "n_seeds": n_seeds,
        "algebra_score": f"{algebra_score}/{total}",
        "knn_score": f"{knn_score}/{total}",
        "algebra_pct": round(algebra_score / total, 4),
        "knn_pct": round(knn_score / total, 4),
        "algebra_wins": algebra_score > knn_score,
    }


def exp_retraction(rho, n_seeds=N_SEEDS):
    """The retracted (old) fact is made highly similar to the query so it actively
    competes for retrieval. Algebra retracts it; kNN keeps it and returns it."""
    algebra_score = 0
    knn_score = 0
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed + 1000)
        concepts = correlated_concepts(6, DIM, rho, rng)
        e_entity, e_attr = concepts[0], concepts[1]
        j = e_entity + 0.02 * rng.standard_normal(DIM)
        j /= np.linalg.norm(j)
        old_fact = e_attr + 0.03 * rng.standard_normal(DIM)
        old_fact /= np.linalg.norm(old_fact)
        new_fact = e_attr + 0.03 * rng.standard_normal(DIM)
        new_fact /= np.linalg.norm(new_fact)
        mA = make_memory("old", {"entity": (j, 1.0, 1.0), "fact": (old_fact, 1.0, 0.3)}, [("entity", "fact", "attr")])
        mB = make_memory("new", {"entity": (j, 2.0, 1.0), "fact_new": (new_fact, 2.0, 1.0)}, [("entity", "fact_new", "attr")])
        M = compose(mA, mB)
        M_retracted = retract(M, mA)
        q = e_entity + e_attr
        q /= np.linalg.norm(q)
        r = retrieve(M_retracted, Query(q, time=2.0, time_scale=0.5, top_k=3, hops=0), THETA)
        if not any("old:fact" in n for n in r.memory.nodes):
            algebra_score += 1
        knn = knn_retrieve([mA, mB], q, k=3)
        if not any("old:fact" in n for n in knn):
            knn_score += 1
    return {
        "rho": rho,
        "n_seeds": n_seeds,
        "algebra_correct": f"{algebra_score}/{n_seeds}",
        "knn_correct": f"{knn_score}/{n_seeds}",
        "algebra_pct": round(algebra_score / n_seeds, 4),
        "knn_pct": round(knn_score / n_seeds, 4),
        "algebra_wins": algebra_score > knn_score,
    }


def exp_multi_hop(rho, n_seeds=N_SEEDS):
    algebra_score = 0
    knn_score = 0
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed + 2000)
        concepts = correlated_concepts(10, DIM, rho, rng)
        eA, eB, eC, eD = concepts[0], concepts[1], concepts[2], concepts[3]
        b1 = eB + 0.02 * rng.standard_normal(DIM)
        b1 /= np.linalg.norm(b1)
        c1 = eC + 0.02 * rng.standard_normal(DIM)
        c1 /= np.linalg.norm(c1)
        m1 = make_memory(f"s{seed}_1", {"a": (eA, 0.0, 1.0), "b": (b1, 0.0, 1.0)}, [("a", "b", "rel")])
        m2 = make_memory(f"s{seed}_2", {"b2": (b1, 0.0, 1.0), "c": (c1, 0.0, 1.0)}, [("b2", "c", "rel")])
        m3 = make_memory(f"s{seed}_3", {"c2": (c1, 0.0, 1.0), "d": (eD, 0.0, 1.0)}, [("c2", "d", "rel")])
        distractors = []
        for k, d in enumerate(concepts[4:10]):
            dm = make_memory(f"dist{seed}_{k}", {"x": (d, 0.0, 1.0)}, [])
            distractors.append(dm)
        M = compose(compose(compose(EMPTY, m1), m2), m3)
        for dm in distractors:
            M = compose(M, dm)
        r = retrieve(M, Query(eA, top_k=1, hops=3), THETA)
        if any("d" in n for n in r.memory.nodes):
            algebra_score += 1
        knn = knn_retrieve([m1, m2, m3] + distractors, eA, k=4)
        if any("d" in n for n in knn):
            knn_score += 1
    return {
        "rho": rho,
        "n_seeds": n_seeds,
        "algebra_correct": f"{algebra_score}/{n_seeds}",
        "knn_correct": f"{knn_score}/{n_seeds}",
        "algebra_pct": round(algebra_score / n_seeds, 4),
        "knn_pct": round(knn_score / n_seeds, 4),
        "algebra_wins": algebra_score > knn_score,
    }


def main():
    results = {
        "temporal_conflict": {"rho_0.0": exp_temporal_conflict(0.0), "rho_0.8": exp_temporal_conflict(0.8)},
        "retraction": {"rho_0.0": exp_retraction(0.0), "rho_0.8": exp_retraction(0.8)},
        "multi_hop_distractors": {"rho_0.0": exp_multi_hop(0.0), "rho_0.8": exp_multi_hop(0.8)},
    }
    out = Path(__file__).resolve().parent / "results_phase3.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("PHASE 3: DIFFERENTIATION (algebra vs kNN)")
    print("=" * 60)
    all_win = True
    for exp_name, variants in results.items():
        print(f"\n[{exp_name}]")
        for variant, res in variants.items():
            print(f"  {variant}: algebra={res.get('algebra_score', res.get('algebra_correct'))} "
                  f"knn={res.get('knn_score', res.get('knn_correct'))} wins={res['algebra_wins']}")
            all_win = all_win and res["algebra_wins"]
    print(f"\nALL EXPERIMENTS algebra > kNN: {all_win}")
    print(f"results saved to {out}")


if __name__ == "__main__":
    main()
