"""Phase 4 Scale: Real embeddings at N=10000."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory_algebra import EMPTY, Query, compose_all, make_memory, retrieve
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
THETA = 0.60
N = 10000


def main():
    print(f"Loading {MODEL_NAME}...", flush=True)
    model = SentenceTransformer(MODEL_NAME)

    rng = np.random.default_rng(42)
    sentences = [
        f"Fact number {i} about topic {i % 50}: "
        f"the value is {rng.integers(0, 1000)} and the category is {i % 50}."
        for i in range(N)
    ]

    print(f"Encoding {N} sentences...", flush=True)
    t0 = time.time()
    embeddings = model.encode(sentences, normalize_embeddings=True, batch_size=256, show_progress_bar=True)
    t_encode = time.time() - t0
    print(f"  Encoding: {t_encode:.1f}s, dim={embeddings.shape[1]}", flush=True)

    mems = [
        make_memory(f"m{i}", {"n0": (embeddings[i], float(i), 1.0)})
        for i in range(N)
    ]

    print(f"Composing {N} memories...", flush=True)
    t0 = time.time()
    M = compose_all(mems)
    t_compose = time.time() - t0
    print(f"  Compose: {t_compose:.2f}s, nodes={len(M.nodes)}", flush=True)

    print(f"Retrieving (theta={THETA})...", flush=True)
    t0 = time.time()
    q = Query(vector=embeddings[0], top_k=1, hops=0)
    r = retrieve(M, q, THETA)
    t_retrieve = time.time() - t0
    print(f"  Retrieve: {t_retrieve:.2f}s, nodes_retrieved={len(r.memory.nodes)}", flush=True)

    t0 = time.time()
    q2 = Query(vector=embeddings[N // 2], top_k=1, hops=0)
    r2 = retrieve(M, q2, THETA)
    t_retrieve2 = time.time() - t0
    print(f"  Retrieve (mid): {t_retrieve2:.2f}s, nodes_retrieved={len(r2.memory.nodes)}", flush=True)

    out = Path(__file__).resolve().parent / "results_phase4_scale.json"
    data = {
        "model": MODEL_NAME,
        "theta": THETA,
        "n_memories": N,
        "n_nodes": len(M.nodes),
        "encode_time": t_encode,
        "compose_time": t_compose,
        "retrieve_time_first": t_retrieve,
        "retrieve_time_mid": t_retrieve2,
        "nodes_retrieved": len(r.memory.nodes),
    }
    out.write_text(json.dumps(data, indent=2))
    print(f"\nResults: {out}", flush=True)


if __name__ == "__main__":
    main()
