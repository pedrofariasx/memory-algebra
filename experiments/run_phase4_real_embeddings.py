"""Phase 4: Real embeddings — validate algebra with sentence-transformers."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory_algebra import EMPTY, Query, compose, compose_all, make_memory, retrieve, signature

MODEL_NAME = "all-MiniLM-L6-v2"
THETA = 0.60

FACTS = [
    ("The capital of France is Paris", "Paris is the capital city of France"),
    ("The capital of Japan is Tokyo", "Tokyo serves as the capital of Japan"),
    ("Water boils at 100 degrees Celsius", "The boiling point of water is 100C"),
    ("The Earth orbits the Sun", "Our planet revolves around the Sun"),
    ("Python is a programming language", "Python is used for software development"),
    ("The Amazon is the largest rainforest", "The Amazon rainforest is the biggest on Earth"),
    ("Light travels faster than sound", "Sound is slower than light"),
    ("The heart pumps blood through the body", "Blood is circulated by the heart"),
    ("Shakespeare wrote Hamlet", "Hamlet was written by Shakespeare"),
    ("The Moon causes ocean tides", "Ocean tides are caused by the Moon"),
    ("DNA carries genetic information", "Genetic information is stored in DNA"),
    ("Ice is the solid form of water", "Water in solid form is called ice"),
    ("The Sun is a star", "Our Sun belongs to the class of stars"),
    ("Beethoven composed nine symphonies", "Nine symphonies were composed by Beethoven"),
    ("Gravity pulls objects toward Earth", "Objects fall due to gravitational pull"),
    ("The Pacific is the largest ocean", "The biggest ocean is the Pacific"),
    ("Newton formulated the laws of motion", "The laws of motion were formulated by Newton"),
    ("Photosynthesis converts sunlight to energy", "Plants turn sunlight into energy via photosynthesis"),
    ("The Nile is the longest river in Africa", "Africa's longest river is the Nile"),
    ("Einstein developed the theory of relativity", "The theory of relativity was developed by Einstein"),
]


def load_embeddings():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    originals = [f[0] for f in FACTS]
    paraphrases = [f[1] for f in FACTS]
    emb_orig = model.encode(originals, normalize_embeddings=True)
    emb_para = model.encode(paraphrases, normalize_embeddings=True)
    return emb_orig, emb_para


def phase4a_associativity(emb_orig):
    mems = [
        make_memory(f"m{i}", {"n0": (emb_orig[i], float(i), 1.0)})
        for i in range(min(10, len(emb_orig)))
    ]
    left = compose(compose(mems[0], mems[1]), mems[2])
    right = compose(mems[0], compose(mems[1], mems[2]))
    assoc_exact = signature(left) == signature(right)
    shuffled = [mems[i] for i in [3, 7, 1, 9, 0, 5, 2, 8, 4, 6]]
    m1 = compose_all(mems)
    m2 = compose_all(shuffled)
    order_invariant = len(m1.nodes) == len(m2.nodes)
    return {"assoc_exact": bool(assoc_exact), "order_invariant": bool(order_invariant), "n_nodes": len(m1.nodes)}


def phase4b_retrieval(emb_orig, emb_para):
    mems = [
        make_memory(f"m{i}", {"n0": (emb_orig[i], float(i), 1.0)})
        for i in range(len(emb_orig))
    ]
    M = compose_all(mems)
    hits = 0
    sims = []
    for i in range(len(emb_para)):
        q = Query(vector=emb_para[i], top_k=1, hops=0)
        r = retrieve(M, q, THETA)
        target = f"m{i}:n0"
        hit = target in r.memory.nodes
        hits += int(hit)
        best_score = max(r.scores.values()) if r.scores else 0.0
        sims.append(float(best_score))
    return {
        "top1_accuracy": hits / len(emb_para),
        "hits": hits,
        "total": len(emb_para),
        "mean_score": float(np.mean(sims)),
    }


def phase4c_stability(emb_orig, emb_para):
    mems = [
        make_memory(f"m{i}", {"n0": (emb_orig[i], float(i), 1.0)})
        for i in range(len(emb_orig))
    ]
    M = EMPTY
    for m in mems:
        M = compose(M, m)
    q = Query(vector=emb_para[0], top_k=1, hops=0)
    r = retrieve(M, q, THETA)
    first_recovered = "m0:n0" in r.memory.nodes
    return {"first_memory_recovered": bool(first_recovered), "n_nodes": len(M.nodes)}


def phase4d_multihop(emb_orig):
    a = make_memory("a", {"n0": (emb_orig[0], 1.0, 1.0)})
    b = make_memory("b", {"n0": (emb_orig[1], 2.0, 1.0)})
    c = make_memory("c", {"n0": (emb_orig[2], 3.0, 1.0)})
    ab = compose(a, b)
    bc = compose(b, c)
    M = compose_all([a, b, c])
    q = Query(vector=emb_orig[0], top_k=1, hops=2)
    r = retrieve(M, q, THETA)
    return {"n_retrieved": len(r.memory.nodes), "hops": 2}


def main():
    out = Path(__file__).resolve().parent / "results_phase4.json"
    print("Loading sentence-transformers model...", flush=True)
    emb_orig, emb_para = load_embeddings()
    print(f"Embeddings: {emb_orig.shape[0]} facts, dim={emb_orig.shape[1]}", flush=True)

    print("Phase 4A: Associativity with real embeddings...", flush=True)
    r4a = phase4a_associativity(emb_orig)
    print(f"  assoc_exact={r4a['assoc_exact']}, order_invariant={r4a['order_invariant']}", flush=True)

    print("Phase 4B: Semantic retrieval (paraphrase -> fact)...", flush=True)
    r4b = phase4b_retrieval(emb_orig, emb_para)
    print(f"  top1_accuracy={r4b['top1_accuracy']:.2%} ({r4b['hits']}/{r4b['total']})", flush=True)

    print("Phase 4C: Stability under sequential composition...", flush=True)
    r4c = phase4c_stability(emb_orig, emb_para)
    print(f"  first_memory_recovered={r4c['first_memory_recovered']}", flush=True)

    print("Phase 4D: Multi-hop traversal...", flush=True)
    r4d = phase4d_multihop(emb_orig)
    print(f"  n_retrieved={r4d['n_retrieved']}", flush=True)

    data = {
        "model": MODEL_NAME,
        "theta": THETA,
        "phase4a_associativity": r4a,
        "phase4b_retrieval": r4b,
        "phase4c_stability": r4c,
        "phase4d_multihop": r4d,
    }
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"\nResults written to {out}", flush=True)


if __name__ == "__main__":
    main()
