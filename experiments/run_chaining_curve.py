"""Chaining curve: giant component fraction vs theta for N in {200, 1000, 5000, 10000}.

Demonstrates that chaining is a percolation phenomenon: as N grows with the same
similarity distribution, the expected number of edges grows as N^2 while nodes grow
as N, so the giant component forms at progressively lower theta. A theta that works
at N=200 has no guarantee of working at N=10000.

Optimization: compute the full pairwise similarity matrix once per seed, sort all
upper-triangle pairs by similarity, then sweep theta descending while incrementally
adding edges to a union-find. This avoids recomputing the N^2 matrix per theta.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

THETAS = np.round(np.arange(0.50, 1.00, 0.01), 2)
N_VALUES = [200, 1000, 5000, 10000]
N_SEEDS = 5


def generate_corpus(n: int) -> list[str]:
    topics = [
        "machine learning", "climate change", "quantum computing", "healthcare",
        "renewable energy", "space exploration", "artificial intelligence",
        "economic policy", "education reform", "cybersecurity",
    ]
    templates = [
        "The impact of {t} on modern society is significant",
        "Recent advances in {t} have transformed the field",
        "Experts disagree about the future of {t}",
        "The relationship between {t} and public policy is complex",
        "New research on {t} reveals unexpected findings",
        "The economic implications of {t} are being studied",
        "Public opinion on {t} has shifted dramatically",
        "The history of {t} spans several decades",
        "International cooperation on {t} is increasing",
        "The ethical dimensions of {t} require careful consideration",
    ]
    sentences = []
    for i in range(n):
        topic = topics[i % len(topics)]
        template = templates[(i // len(topics)) % len(templates)]
        sentences.append(template.format(t=topic))
    return sentences


def sweep_curve(unit: np.ndarray, thetas: np.ndarray) -> dict:
    n = unit.shape[0]
    sims = unit @ unit.T
    iu = np.triu_indices(n, k=1)
    pair_sims = sims[iu]
    mu = float(pair_sims.mean())
    sigma = float(pair_sims.std())

    order = np.argsort(-pair_sims)
    sorted_sims = pair_sims[order]
    sorted_i = iu[0][order]
    sorted_j = iu[1][order]
    del sims, pair_sims, iu, order

    parent = np.arange(n, dtype=np.int64)
    size = np.ones(n, dtype=np.int64)
    max_size = 1

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    curve = {}
    edge_ptr = 0
    total_pairs = len(sorted_sims)

    for theta in sorted(thetas, reverse=True):
        while edge_ptr < total_pairs and sorted_sims[edge_ptr] >= theta:
            a, b = int(sorted_i[edge_ptr]), int(sorted_j[edge_ptr])
            ra, rb = find(a), find(b)
            if ra != rb:
                if size[ra] < size[rb]:
                    ra, rb = rb, ra
                parent[rb] = ra
                size[ra] += size[rb]
                if size[ra] > max_size:
                    max_size = int(size[ra])
            edge_ptr += 1
        curve[str(float(theta))] = max_size / n

    return {"mu": mu, "sigma": sigma, "curve": curve}


def main():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not installed")
        sys.exit(1)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    results = {}

    for n in N_VALUES:
        print(f"\n=== N={n} ===", flush=True)
        seeds_data = []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed)
            sentences = generate_corpus(n)
            embeddings = model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)
            unit = np.asarray(embeddings, dtype=np.float64)

            print(f"  seed={seed}...", end=" ", flush=True)
            data = sweep_curve(unit, THETAS)
            theta_adapt = data["mu"] + 2 * data["sigma"]
            adapt_key = str(round(theta_adapt, 2))
            adapt_frac = data["curve"].get(adapt_key, data["curve"][str(round(theta_adapt, 2))])
            default_frac = data["curve"].get("0.85", 0.0)

            data["n"] = n
            data["seed"] = seed
            data["theta_adapt"] = theta_adapt
            data["giant_frac_at_theta_adapt"] = adapt_frac
            data["giant_frac_at_0.85"] = default_frac
            seeds_data.append(data)
            print(
                f"mu={data['mu']:.3f} sigma={data['sigma']:.3f} "
                f"theta_adapt={theta_adapt:.3f} "
                f"giant(adapt)={adapt_frac:.3f} "
                f"giant(0.85)={default_frac:.3f}",
                flush=True,
            )
        results[str(n)] = seeds_data

    out = Path(__file__).resolve().parent / "results_chaining_curve.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")

    print("\n=== Summary: theta where giant component exceeds 50% ===")
    for n in N_VALUES:
        thresholds = []
        for data in results[str(n)]:
            found = False
            for t_str in sorted(data["curve"].keys(), key=float, reverse=True):
                if data["curve"][t_str] > 0.5:
                    thresholds.append(float(t_str))
                    found = True
                    break
            if not found:
                thresholds.append(1.0)
        print(f"  N={n}: transition at theta ~ {np.mean(thresholds):.3f} +/- {np.std(thresholds):.3f}")


if __name__ == "__main__":
    main()
