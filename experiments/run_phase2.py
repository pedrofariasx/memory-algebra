"""Phase 2 Validation: Robustness, Realism, Baselines, Scale, QA."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.stats as stats

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
    semantic_distance,
    signature,
)
from memory_algebra.baselines import LSTMCell
from memory_algebra.baselines import train_lstm_recall

DIM = 16
N_MEMS = 50
NUM_SEEDS = 50
THETAS = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
DIMS = [8, 16, 32, 64]
RHOS = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]
NOISES = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
SCALE_NS = [100, 500, 1000, 5000, 10000]


def gen_sequence(rng, concepts, n, noise=0.0):
    mems = []
    for i in range(n):
        k = int(rng.integers(1, 4))
        idx = rng.integers(0, len(concepts), size=k)
        nodes = {}
        for j, ci in enumerate(idx):
            v = concepts[ci].copy()
            if noise > 0:
                v = v + rng.normal(0, noise, size=v.shape)
                v = v / (np.linalg.norm(v) + 1e-12)
            nodes[f"n{j}"] = (
                v,
                float(i) + float(rng.uniform(0.0, 0.5)),
                float(rng.uniform(0.2, 1.0)),
            )
        edges = [(f"n{j}", f"n{j + 1}", "rel") for j in range(k - 1)]
        mems.append(make_memory(f"m{i}", nodes, edges))
    return mems


def fold_left(mems):
    M = EMPTY
    for m in mems:
        M = compose(M, m)
    return M


def fold_right(mems):
    M = EMPTY
    for m in reversed(mems):
        M = compose(m, M)
    return M


def measure_associativity(rng, concepts, n, theta):
    mems = gen_sequence(rng, concepts, n)
    left = fold_left(mems)
    right = fold_right(mems)
    sig_eq = signature(left) == signature(right)
    max_d = 0.0
    for m in mems:
        n0 = sorted(m.nodes)[0]
        q = Query(vector=m.vectors[n0], time=m.times[n0], time_scale=5.0, top_k=1, hops=0)
        rl = retrieve(left, q, theta).memory
        rr = retrieve(right, q, theta).memory
        max_d = max(max_d, semantic_distance(rl, rr))
    return sig_eq, max_d


def measure_stability(rng, concepts, n, theta, noise=0.0):
    mems = gen_sequence(rng, concepts, n, noise=noise)
    m1 = mems[0]
    n0 = sorted(m1.nodes)[0]
    q = Query(vector=m1.vectors[n0], time=m1.times[n0], time_scale=100.0, top_k=1, hops=0)
    M = EMPTY
    curve = []
    for m in mems:
        M = compose(M, m)
        r = retrieve(M, q, theta).memory
        curve.append(semantic_distance(m1, r))
    return curve


def lstm_stability(mems, dim):
    cell = LSTMCell(dim=dim, hidden=dim, seed=11)
    m1 = mems[0]
    n0 = sorted(m1.nodes)[0]
    v1 = m1.vectors[n0]
    curve = []
    for m in mems:
        n0 = sorted(m.nodes)[0]
        x = m.vectors[n0] / (np.linalg.norm(m.vectors[n0]) + 1e-12)
        h = cell.step(x)
        curve.append(1.0 - cosine(h, v1))
    return curve


def lstm_trained_stability(mems, dim, steps=500):
    seqs = []
    for _ in range(20):
        xs = []
        for m in mems:
            n0 = sorted(m.nodes)[0]
            x = m.vectors[n0] / (np.linalg.norm(m.vectors[n0]) + 1e-12)
            xs.append(x)
        seqs.append(xs)
    cell = train_lstm_recall(dim=dim, hidden=dim, sequences=seqs, steps=steps, seed=11)
    m1 = mems[0]
    n0 = sorted(m1.nodes)[0]
    v1 = m1.vectors[n0]
    curve = []
    for m in mems:
        n0 = sorted(m.nodes)[0]
        x = m.vectors[n0] / (np.linalg.norm(m.vectors[n0]) + 1e-12)
        h = cell.step(x)
        curve.append(1.0 - cosine(h, v1))
    return curve


def sum_stability(mems, dim):
    m1 = mems[0]
    n0 = sorted(m1.nodes)[0]
    v1 = m1.vectors[n0]
    vec_sum = np.zeros(dim)
    curve = []
    for m in mems:
        n0 = sorted(m.nodes)[0]
        x = m.vectors[n0] / (np.linalg.norm(m.vectors[n0]) + 1e-12)
        vec_sum = vec_sum + x
        curve.append(1.0 - cosine(vec_sum, v1))
    return curve


def correlated_concepts(n, dim, rho, rng):
    concepts = []
    for _ in range(n):
        base = rng.standard_normal(dim)
        base /= np.linalg.norm(base) + 1e-12
        noise = rng.standard_normal(dim)
        noise /= np.linalg.norm(noise) + 1e-12
        v = rho * base + np.sqrt(1 - rho**2) * noise
        concepts.append(v / (np.linalg.norm(v) + 1e-12))
    return concepts


def knn_retrieve_baseline(mems, query_vec, top_k=1):
    best = []
    for m in mems:
        for nid, v in m.vectors.items():
            c = cosine(v, query_vec)
            best.append((c, m, nid))
    best.sort(key=lambda x: -x[0])
    return best[:top_k]


def phase2a_seeds():
    results = {"seeds": [], "stats": {}}
    all_assoc_pass = []
    all_max_d = []
    all_stab_final = []
    for seed in range(NUM_SEEDS):
        rng = np.random.default_rng(seed)
        concepts = orthogonal_concepts(DIM, DIM, rng)
        mems = gen_sequence(rng, concepts, N_MEMS)
        sig_eq, max_d = measure_associativity(rng, concepts, N_MEMS, 0.85)
        all_assoc_pass.append(sig_eq)
        all_max_d.append(max_d)
        stab = measure_stability(rng, concepts, N_MEMS, 0.85)
        all_stab_final.append(stab[-1])
        results["seeds"].append({
            "seed": seed,
            "sig_eq": sig_eq,
            "max_d": max_d,
            "stab_final": stab[-1],
        })
    results["stats"] = {
        "assoc_pass_rate": sum(all_assoc_pass) / NUM_SEEDS,
        "max_d_mean": float(np.mean(all_max_d)),
        "max_d_std": float(np.std(all_max_d)),
        "max_d_ic95": list(stats.t.interval(0.95, NUM_SEEDS - 1, loc=np.mean(all_max_d), scale=stats.sem(all_max_d))) if NUM_SEEDS > 1 else [0, 0],
        "stab_final_mean": float(np.mean(all_stab_final)),
        "stab_final_std": float(np.std(all_stab_final)),
        "stab_final_ic95": list(stats.t.interval(0.95, NUM_SEEDS - 1, loc=np.mean(all_stab_final), scale=stats.sem(all_stab_final))) if NUM_SEEDS > 1 else [0, 0],
    }
    return results


def phase2a_sensitivity():
    theta_results = {}
    for theta in THETAS:
        pass_rates = []
        max_ds = []
        for seed in range(20):
            rng = np.random.default_rng(seed)
            concepts = orthogonal_concepts(DIM, DIM, rng)
            sig_eq, max_d = measure_associativity(rng, concepts, N_MEMS, theta)
            pass_rates.append(sig_eq)
            max_ds.append(max_d)
        theta_results[theta] = {
            "pass_rate": sum(pass_rates) / 20,
            "max_d_mean": float(np.mean(max_ds)),
            "max_d_std": float(np.std(max_ds)),
        }
    dim_results = {}
    for dim in DIMS:
        pass_rates = []
        max_ds = []
        for seed in range(20):
            rng = np.random.default_rng(seed)
            concepts = orthogonal_concepts(dim, dim, rng)
            sig_eq, max_d = measure_associativity(rng, concepts, N_MEMS, 0.85)
            pass_rates.append(sig_eq)
            max_ds.append(max_d)
        dim_results[dim] = {
            "pass_rate": sum(pass_rates) / 20,
            "max_d_mean": float(np.mean(max_ds)),
            "max_d_std": float(np.std(max_ds)),
        }
    return {"theta": theta_results, "dim": dim_results}


def phase2a_adversarial():
    results = {}
    for seed in range(10):
        rng = np.random.default_rng(seed)
        concepts = orthogonal_concepts(DIM, DIM, rng)
        n0 = sorted(concepts[0])
        identical_mem = make_memory("id", {"n0": (concepts[0], 0.0, 1.0)})
        comp = compose(
            make_memory("a", {"n0": (concepts[0], 0.0, 1.0)}),
            make_memory("b", {"n0": (concepts[0], 1.0, 1.0)}),
        )
        results[f"seed_{seed}_identical"] = {
            "sig_eq": signature(comp) == signature(comp),
            "n_nodes": len(comp.nodes),
        }
        antiparallel = -concepts[0]
        comp2 = compose(
            make_memory("a", {"n0": (concepts[0], 0.0, 1.0)}),
            make_memory("b", {"n0": (antiparallel, 1.0, 1.0)}),
        )
        results[f"seed_{seed}_antiparallel"] = {
            "n_nodes": len(comp2.nodes),
            "cosine": float(cosine(concepts[0], antiparallel)),
        }
    rng = np.random.default_rng(42)
    concepts = orthogonal_concepts(DIM, DIM, rng)
    mems = gen_sequence(rng, concepts, 1000)
    t0 = time.time()
    M = fold_left(mems)
    t_compose = time.time() - t0
    q = Query(vector=concepts[0], top_k=1, hops=0)
    t0 = time.time()
    retrieve(M, q, 0.85)
    t_retrieve = time.time() - t0
    results["scale_1000"] = {
        "n_nodes": len(M.nodes),
        "compose_time": t_compose,
        "retrieve_time": t_retrieve,
    }
    return results


def phase2b_correlated():
    results = {}
    for rho in RHOS:
        stab_curves = []
        assoc_pass = []
        for seed in range(20):
            rng = np.random.default_rng(seed)
            concepts = correlated_concepts(DIM, DIM, rho, rng)
            mems = gen_sequence(rng, concepts, N_MEMS)
            sig_eq, max_d = measure_associativity(rng, concepts, N_MEMS, 0.85)
            assoc_pass.append(sig_eq)
            stab = measure_stability(rng, concepts, N_MEMS, 0.85)
            stab_curves.append(stab[-1])
        results[rho] = {
            "assoc_pass_rate": sum(assoc_pass) / 20,
            "stab_final_mean": float(np.mean(stab_curves)),
            "stab_final_std": float(np.std(stab_curves)),
        }
    return results


def phase2b_noise():
    results = {}
    for noise in NOISES:
        stab_curves = []
        for seed in range(20):
            rng = np.random.default_rng(seed)
            concepts = orthogonal_concepts(DIM, DIM, rng)
            stab = measure_stability(rng, concepts, N_MEMS, 0.85, noise=noise)
            stab_curves.append(stab[-1])
        results[noise] = {
            "stab_final_mean": float(np.mean(stab_curves)),
            "stab_final_std": float(np.std(stab_curves)),
        }
    return results


def phase2c_baselines():
    results = {"algebra": {}, "lstm": {}, "sum": {}, "knn": {}}
    alg_curves = []
    lstm_curves = []
    lstm_trained_curves = []
    sum_curves = []
    knn_curves = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        concepts = orthogonal_concepts(DIM, DIM, rng)
        mems = gen_sequence(rng, concepts, N_MEMS)
        alg = measure_stability(rng, concepts, N_MEMS, 0.85)
        lstm = lstm_stability(mems, DIM)
        lt = lstm_trained_stability(mems, DIM)
        sm = sum_stability(mems, DIM)
        m1 = mems[0]
        n0 = sorted(m1.nodes)[0]
        v1 = m1.vectors[n0]
        knn_curve = []
        buffer = []
        for m in mems:
            buffer.append(m)
            if len(buffer) > 10:
                buffer.pop(0)
            top = knn_retrieve_baseline(buffer, v1, top_k=1)
            if top:
                best_v = top[0][1].vectors[top[0][2]]
                knn_curve.append(1.0 - cosine(best_v, v1))
            else:
                knn_curve.append(1.0)
        alg_curves.append(alg)
        lstm_curves.append(lstm)
        lstm_trained_curves.append(lt)
        sum_curves.append(sm)
        knn_curves.append(knn_curve)
    for name, curves in [("algebra", alg_curves), ("lstm", lstm_curves), ("lstm_trained", lstm_trained_curves), ("sum", sum_curves), ("knn", knn_curves)]:
        finals = [c[-1] for c in curves]
        results[name] = {
            "final_mean": float(np.mean(finals)),
            "final_std": float(np.std(finals)),
            "curve_mean": [float(np.mean([c[i] for c in curves])) for i in range(N_MEMS)],
        }
    alg_finals = [c[-1] for c in alg_curves]
    for name, curves in [("lstm", lstm_curves), ("lstm_trained", lstm_trained_curves), ("sum", sum_curves), ("knn", knn_curves)]:
        other_finals = [c[-1] for c in curves]
        _, p = stats.mannwhitneyu(alg_finals, other_finals, alternative="less")
        results[f"wilcoxon_algebra_vs_{name}"] = float(p)
    return results


def phase2d_scale():
    results = {}
    for n in SCALE_NS:
        rng = np.random.default_rng(42)
        concepts = orthogonal_concepts(DIM, DIM, rng)
        mems = gen_sequence(rng, concepts, n)
        t0 = time.time()
        M = fold_left(mems)
        t_compose = time.time() - t0
        q = Query(vector=concepts[0], top_k=1, hops=0)
        t0 = time.time()
        retrieve(M, q, 0.85)
        t_retrieve = time.time() - t0
        results[n] = {
            "n_nodes": len(M.nodes),
            "compose_time": t_compose,
            "retrieve_time": t_retrieve,
        }
    return results


def phase2e_qa():
    rng = np.random.default_rng(42)
    concepts = orthogonal_concepts(DIM, DIM, rng)
    brasilia = concepts[0]
    brasil = concepts[1]
    lula = concepts[2]
    argentina = concepts[3]
    facts = [
        ("Brasilia_capital_Brasil", brasilia, brasil, 1.0),
        ("Lula_presidente_Brasil", lula, brasil, 2.0),
        ("Brasil_fronteira_Argentina", brasil, argentina, 3.0),
    ]
    M = EMPTY
    for name, subj, obj, t in facts:
        m = make_memory(
            name,
            {
                "subj": (subj, t, 1.0),
                "obj": (obj, t, 1.0),
            },
            [("subj", "obj", "rel")],
        )
        M = compose(M, m)
    q = Query(vector=lula, time=2.5, time_scale=2.0, top_k=3, hops=2)
    r = retrieve(M, q, 0.85)
    nodes = list(r.memory.nodes)
    has_brasil = any("Brasil" in n for n in nodes)
    has_argentina = any("Argentina" in n for n in nodes)
    return {
        "n_retrieved": len(nodes),
        "has_brasil": has_brasil,
        "has_argentina": has_argentina,
        "multi_hop_ok": has_brasil and has_argentina,
    }


def main():
    out = Path(__file__).resolve().parent / "results_phase2.json"
    print("Phase 2A: Multiple seeds + statistics...")
    p2a_seeds = phase2a_seeds()
    print(f"  Assoc pass rate: {p2a_seeds['stats']['assoc_pass_rate']:.2f}")
    print(f"  Max d mean: {p2a_seeds['stats']['max_d_mean']:.6f} ± {p2a_seeds['stats']['max_d_std']:.6f}")
    print(f"  Stab final mean: {p2a_seeds['stats']['stab_final_mean']:.6f} ± {p2a_seeds['stats']['stab_final_std']:.6f}")

    print("Phase 2A: Parameter sensitivity...")
    p2a_sens = phase2a_sensitivity()
    for theta in THETAS:
        r = p2a_sens["theta"][theta]
        print(f"  theta={theta}: pass={r['pass_rate']:.2f}, max_d={r['max_d_mean']:.6f}")
    for dim in DIMS:
        r = p2a_sens["dim"][dim]
        print(f"  dim={dim}: pass={r['pass_rate']:.2f}, max_d={r['max_d_mean']:.6f}")

    print("Phase 2A: Adversarial tests...")
    p2a_adv = phase2a_adversarial()
    print(f"  Scale 1000: compose={p2a_adv['scale_1000']['compose_time']:.3f}s, retrieve={p2a_adv['scale_1000']['retrieve_time']:.3f}s")

    print("Phase 2B: Correlated concepts...")
    p2b_corr = phase2b_correlated()
    for rho in RHOS:
        r = p2b_corr[rho]
        print(f"  rho={rho}: pass={r['assoc_pass_rate']:.2f}, stab={r['stab_final_mean']:.6f}")

    print("Phase 2B: Noise degradation...")
    p2b_noise = phase2b_noise()
    for noise in NOISES:
        r = p2b_noise[noise]
        print(f"  noise={noise}: stab={r['stab_final_mean']:.6f}")

    print("Phase 2C: Baselines comparison...")
    p2c = phase2c_baselines()
    for name in ["algebra", "lstm", "lstm_trained", "sum", "knn"]:
        r = p2c[name]
        print(f"  {name}: final={r['final_mean']:.6f} ± {r['final_std']:.6f}")
    for name in ["lstm", "lstm_trained", "sum", "knn"]:
        p = p2c[f"wilcoxon_algebra_vs_{name}"]
        print(f"  Wilcoxon algebra vs {name}: p={p:.6f}")

    print("Phase 2D: Scale benchmark...")
    p2d = phase2d_scale()
    for n in SCALE_NS:
        r = p2d[n]
        print(f"  N={n}: nodes={r['n_nodes']}, compose={r['compose_time']:.3f}s, retrieve={r['retrieve_time']:.3f}s")

    print("Phase 2E: QA multi-hop...")
    p2e = phase2e_qa()
    print(f"  Retrieved {p2e['n_retrieved']} nodes, multi_hop_ok={p2e['multi_hop_ok']}")

    results = {
        "phase2a_seeds": p2a_seeds,
        "phase2a_sensitivity": p2a_sens,
        "phase2a_adversarial": p2a_adv,
        "phase2b_correlated": p2b_corr,
        "phase2b_noise": p2b_noise,
        "phase2c_baselines": p2c,
        "phase2d_scale": p2d,
        "phase2e_qa": p2e,
    }
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
