"""Phase 4 at scale: real embeddings (384-dim) with N=10k facts."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory_algebra import Query, compose_all, make_memory, retrieve

MODEL_NAME = "all-MiniLM-L6-v2"
N_FACTS = 10000
THETA = 0.60


def generate_sentences(n):
    subjects = ["the cat", "the dog", "the bird", "the fish", "the tree",
                "the river", "the mountain", "the city", "the star", "the moon"]
    verbs = ["observes", "creates", "transforms", "connects", "illuminates",
             "protects", "guides", "inspires", "shapes", "reveals"]
    objects = ["a pattern", "a signal", "a structure", "a network", "a system",
               "a process", "a concept", "a model", "a theory", "a framework"]
    sentences = []
    for i in range(n):
        s = subjects[i % len(subjects)]
        v = verbs[(i // len(subjects)) % len(verbs)]
        o = objects[(i // (len(subjects) * len(verbs))) % len(objects)]
        sentences.append(f"{s} {v} {o} number {i}")
    return sentences


def main():
    print(f"Loading model {MODEL_NAME}...", flush=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)

    print(f"Generating {N_FACTS} sentences...", flush=True)
    sentences = generate_sentences(N_FACTS)

    print("Encoding...", flush=True)
    t0 = time.time()
    embeddings = model.encode(sentences, normalize_embeddings=True, batch_size=512)
    t_encode = time.time() - t0
    print(f"  Encoded {embeddings.shape[0]} x {embeddings.shape[1]} in {t_encode:.1f}s", flush=True)

    print("Building memories...", flush=True)
    t0 = time.time()
    mems = [
        make_memory(f"m{i}", {"n0": (embeddings[i], float(i), 1.0)})
        for i in range(N_FACTS)
    ]
    M = compose_all(mems)
    t_compose = time.time() - t0
    print(f"  Composed {len(M.nodes)} nodes in {t_compose:.3f}s", flush=True)

    print("Retrieval test (100 random queries)...", flush=True)
    rng = np.random.default_rng(123)
    query_indices = rng.choice(N_FACTS, size=100, replace=False)
    hits = 0
    t0 = time.time()
    for qi in query_indices:
        q = Query(vector=embeddings[qi], top_k=1, hops=0)
        r = retrieve(M, q, THETA)
        if f"m{qi}:n0" in r.memory.nodes:
            hits += 1
    t_retrieve = time.time() - t0

    accuracy = hits / len(query_indices)
    print(f"  Top-1 accuracy: {accuracy:.2%} ({hits}/{len(query_indices)})", flush=True)
    print(f"  Total retrieve time: {t_retrieve:.2f}s ({t_retrieve/len(query_indices)*1000:.1f}ms/query)", flush=True)

    if accuracy >= 0.95:
        print("PASS: retrieval accuracy >= 95% at N=10k with real embeddings")
        return 0
    else:
        print(f"WARN: accuracy {accuracy:.2%} < 95%")
        return 1


if __name__ == "__main__":
    sys.exit(main())
