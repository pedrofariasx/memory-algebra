"""Phase 5 Scale: Algebra memory layer accuracy at N=100-1000 facts."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory_algebra import EMPTY, Query, compose_all, make_memory, retrieve, retract
from memory_algebra.core import MemoryObject

BASE_URL = "https://api.kilo.ai/api/gateway/v1"
MODEL = "stepfun/step-3.7-flash:free"
THETA = 0.60
SCALES = [50, 100, 200, 500]

CATEGORIES = [
    "geography", "science", "history", "technology", "biology",
    "physics", "chemistry", "mathematics", "literature", "music",
]
ENTITIES = [
    "Paris", "Tokyo", "Brazil", "Egypt", "Canada", "India", "Germany",
    "Australia", "Mexico", "Italy", "Spain", "Russia", "China", "Korea",
]
ATTRIBUTES = [
    "capital", "population", "area", "language", "currency",
    "president", "founded", "largest city", "official name", "continent",
]


def generate_facts(rng, n):
    facts = []
    for i in range(n):
        cat = CATEGORIES[i % len(CATEGORIES)]
        entity = ENTITIES[rng.integers(0, len(ENTITIES))]
        attr = ATTRIBUTES[rng.integers(0, len(ATTRIBUTES))]
        value = f"{entity}_{attr}_{i}"
        text = f"The {attr} of {entity} in {cat} is {value}"
        facts.append({"text": text, "entity": entity, "attr": attr, "value": value, "time": float(i)})
    return facts


def build_memory(facts, embeddings):
    mems = []
    for i, fact in enumerate(facts):
        m = make_memory(f"fact_{i}", {"n": (embeddings[i], fact["time"], 1.0)})
        mems.append(m)
    return compose_all(mems)


def test_retrieval_accuracy(M, emb_model, facts, embeddings, n_queries=10):
    hits = 0
    rng = np.random.default_rng(42)
    indices = rng.choice(len(facts), size=min(n_queries, len(facts)), replace=False)
    for idx in indices:
        q_vec = embeddings[idx]
        q = Query(vector=q_vec, top_k=1, hops=0)
        r = retrieve(M, q, THETA)
        target = f"fact_{idx}:n"
        if target in r.memory.nodes:
            hits += 1
    return hits / len(indices)


def test_temporal_at_scale(M, emb_model, facts, embeddings, n_conflicts=5):
    rng = np.random.default_rng(123)
    correct = 0
    total = 0
    entity_counts = {}
    for i, f in enumerate(facts):
        entity_counts.setdefault(f["entity"], []).append(i)

    for entity, indices in entity_counts.items():
        if len(indices) < 2 or total >= n_conflicts:
            continue
        latest_idx = max(indices, key=lambda i: facts[i]["time"])
        q_vec = embeddings[latest_idx]
        q = Query(vector=q_vec, time=facts[latest_idx]["time"] + 0.5, time_scale=0.5, top_k=3, hops=0)
        r = retrieve(M, q, THETA)
        target = f"fact_{latest_idx}:n"
        if target in r.memory.nodes:
            correct += 1
        total += 1

    return correct / max(total, 1), total


def test_retraction_at_scale(M, facts, embeddings, n_retractions=5):
    rng = np.random.default_rng(456)
    indices = rng.choice(len(facts), size=min(n_retractions, len(facts)), replace=False)
    absent_count = 0
    for idx in indices:
        target_name = f"fact_{idx}"
        target_nodes = frozenset(n for n in M.nodes if n.startswith(target_name))
        if not target_nodes:
            continue
        sub_vectors = {n: M.vectors[n] for n in target_nodes}
        sub_times = {n: M.times[n] for n in target_nodes}
        sub_weights = {n: M.weights[n] for n in target_nodes}
        sub = MemoryObject(nodes=target_nodes, vectors=sub_vectors, edges=frozenset(), times=sub_times, weights=sub_weights)
        M_retracted = retract(M, sub)
        q_vec = embeddings[idx]
        q = Query(vector=q_vec, top_k=3, hops=0)
        r = retrieve(M_retracted, q, THETA)
        if target_nodes.isdisjoint(r.memory.nodes):
            absent_count += 1
    return absent_count / max(len(indices), 1)


def run_scale(n_facts):
    from sentence_transformers import SentenceTransformer

    print(f"\n--- N={n_facts} ---", flush=True)
    rng = np.random.default_rng(42)
    facts = generate_facts(rng, n_facts)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [f["text"] for f in facts]

    t0 = time.time()
    embeddings = model.encode(texts, normalize_embeddings=True, batch_size=128)
    t_encode = time.time() - t0
    print(f"  Encode: {t_encode:.1f}s", flush=True)

    t0 = time.time()
    M = build_memory(facts, embeddings)
    t_compose = time.time() - t0
    print(f"  Compose: {t_compose:.2f}s, nodes={len(M.nodes)}", flush=True)

    t0 = time.time()
    acc = test_retrieval_accuracy(M, model, facts, embeddings, n_queries=10)
    t_retrieve = time.time() - t0
    print(f"  Retrieval accuracy: {acc:.0%} ({t_retrieve:.1f}s)", flush=True)

    t0 = time.time()
    temporal_acc, n_conflicts = test_temporal_at_scale(M, model, facts, embeddings)
    t_temporal = time.time() - t0
    print(f"  Temporal accuracy: {temporal_acc:.0%} ({n_conflicts} conflicts, {t_temporal:.1f}s)", flush=True)

    t0 = time.time()
    retraction_acc = test_retraction_at_scale(M, facts, embeddings)
    t_retract = time.time() - t0
    print(f"  Retraction accuracy: {retraction_acc:.0%} ({t_retract:.1f}s)", flush=True)

    return {
        "n_facts": n_facts,
        "n_nodes": len(M.nodes),
        "encode_time": round(t_encode, 2),
        "compose_time": round(t_compose, 2),
        "retrieval_accuracy": acc,
        "retrieval_time": round(t_retrieve, 2),
        "temporal_accuracy": temporal_acc,
        "temporal_conflicts_tested": n_conflicts,
        "retraction_accuracy": retraction_acc,
    }


def main():
    results = []
    for n in SCALES:
        r = run_scale(n)
        results.append(r)

    out = Path(__file__).resolve().parent / "results_phase5_scale.json"
    data = {
        "model": MODEL,
        "theta": THETA,
        "scales": results,
    }
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"\nResults: {out}", flush=True)


if __name__ == "__main__":
    main()
