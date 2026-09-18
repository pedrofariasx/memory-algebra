"""Chaining analysis: quantify single-linkage transitive closure collapse.

Measures how many connected components the quotient produces for different theta
values, using both synthetic vectors and real sentence embeddings.

Key question: at what theta does the quotient collapse all nodes into a single
component (the chaining problem), and does adaptive theta (mu + k*sigma of the
pairwise similarity distribution) prevent it?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory_algebra import make_memory, quotient


def pairwise_cosines(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = matrix / norms
    sims = unit @ unit.T
    iu = np.triu_indices(len(unit), k=1)
    return sims[iu]


def adaptive_theta(sims: np.ndarray, k: float = 2.0) -> float:
    mu = float(np.mean(sims))
    sigma = float(np.std(sims))
    return float(np.clip(mu + k * sigma, 0.0, 0.999))


def count_components(m, theta: float) -> tuple[int, int]:
    q = quotient(m, theta)
    sizes = [len(c) for c in q.clusters]
    return len(q.clusters), max(sizes) if sizes else 0


def exp_synthetic(n_nodes: int, dim: int, n_concepts: int, noise: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    concepts = rng.standard_normal((n_concepts, dim))
    concepts /= np.linalg.norm(concepts, axis=1, keepdims=True)

    nodes = {}
    for i in range(n_nodes):
        c = concepts[i % n_concepts]
        v = c + noise * rng.standard_normal(dim)
        v /= np.linalg.norm(v)
        nodes[f"n{i}"] = (v, 0.0, 1.0)

    m = make_memory("syn", nodes, [])
    matrix = np.stack([v for v, _, _ in nodes.values()])
    sims = pairwise_cosines(matrix)

    results = {}
    for theta in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]:
        n_comp, max_size = count_components(m, theta)
        results[f"theta_{theta}"] = {"components": n_comp, "max_cluster": max_size}

    theta_adapt = adaptive_theta(sims)
    n_comp, max_size = count_components(m, theta_adapt)
    results["adaptive"] = {
        "theta": round(theta_adapt, 4),
        "components": n_comp,
        "max_cluster": max_size,
    }
    results["similarity_stats"] = {
        "mean": round(float(np.mean(sims)), 4),
        "std": round(float(np.std(sims)), 4),
        "p95": round(float(np.percentile(sims, 95)), 4),
        "p99": round(float(np.percentile(sims, 99)), 4),
    }
    results["expected_components"] = n_concepts
    return results


def exp_real_embeddings(n_sentences: int, n_topics: int, seed: int) -> dict:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    rng = np.random.default_rng(seed)

    topics = [
        "machine learning", "quantum physics", "medieval history", "marine biology",
        "classical music", "organic chemistry", "space exploration", "philosophy of mind",
        "economic theory", "cognitive neuroscience", "ancient architecture", "climate science",
        "linguistics", "number theory", "immunology", "artificial intelligence",
        "volcanology", "game theory", "epistemology", "astrophysics",
    ][:n_topics]

    templates = [
        "The field of {t} has seen major advances in {y}.",
        "Researchers in {t} published new findings about {y}.",
        "A comprehensive review of {t} was published in {y}.",
        "The fundamentals of {t} were revised in {y}.",
        "New methods in {t} emerged during {y}.",
    ]
    years = ["2020", "2021", "2022", "2023", "2024", "2025"]

    sentences = []
    for i in range(n_sentences):
        topic = topics[i % n_topics]
        template = templates[i % len(templates)]
        year = years[rng.integers(len(years))]
        sentences.append(template.format(t=topic, y=year))

    embeddings = model.encode(sentences, normalize_embeddings=True)

    nodes = {}
    for i, emb in enumerate(embeddings):
        nodes[f"s{i}"] = (emb, 0.0, 1.0)

    m = make_memory("real", nodes, [])
    sims = pairwise_cosines(embeddings)

    results = {}
    for theta in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]:
        n_comp, max_size = count_components(m, theta)
        results[f"theta_{theta}"] = {"components": n_comp, "max_cluster": max_size}

    theta_adapt = adaptive_theta(sims)
    n_comp, max_size = count_components(m, theta_adapt)
    results["adaptive"] = {
        "theta": round(theta_adapt, 4),
        "components": n_comp,
        "max_cluster": max_size,
    }
    results["similarity_stats"] = {
        "mean": round(float(np.mean(sims)), 4),
        "std": round(float(np.std(sims)), 4),
        "p95": round(float(np.percentile(sims, 95)), 4),
        "p99": round(float(np.percentile(sims, 99)), 4),
    }
    results["expected_components"] = n_topics
    return results


def main():
    results = {}
    print("=== Chaining Analysis ===", flush=True)

    print("\n[1/2] Synthetic (200 nodes, 10 concepts, noise=0.3)...", flush=True)
    syn = exp_synthetic(n_nodes=200, dim=64, n_concepts=10, noise=0.3, seed=42)
    results["synthetic"] = syn
    print(f"  sim: mean={syn['similarity_stats']['mean']} std={syn['similarity_stats']['std']}", flush=True)
    for k, v in syn.items():
        if k.startswith("theta_") or k == "adaptive":
            print(f"  {k}: components={v['components']} max_cluster={v['max_cluster']}", flush=True)

    print("\n[2/2] Real embeddings (200 sentences, 20 topics)...", flush=True)
    real = exp_real_embeddings(n_sentences=200, n_topics=20, seed=42)
    results["real_embeddings"] = real
    print(f"  sim: mean={real['similarity_stats']['mean']} std={real['similarity_stats']['std']}", flush=True)
    for k, v in real.items():
        if k.startswith("theta_") or k == "adaptive":
            print(f"  {k}: components={v['components']} max_cluster={v['max_cluster']}", flush=True)

    out = Path(__file__).resolve().parent / "results_chaining.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out}", flush=True)


if __name__ == "__main__":
    main()
